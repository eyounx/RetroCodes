import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2
import tensorflow as tf
import json
from detect.backbone.tiny_darknet_fcn import yolo_net, load_from_binary
from detect.util.postprocessing import getboxes

cv2.ocl.setUseOpenCL(False)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class WrapWarpFrame(gym.ObservationWrapper):
    #增加了图像对比的通道
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1*2), dtype=np.uint8)
        self.previous_observation = None
        self.current_observation = None
        # self.diff_observation = None

    def observation(self, frame):
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        if self.current_observation is None:
            self.previous_observation = self.current_observation = frame
        else:
            self.previous_observation = self.current_observation
            self.current_observation = frame

        diff_observation = self.opticalFlow(self.previous_observation, self.current_observation)[0]

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame[:,:,np.newaxis]

        diff_observation = cv2.cvtColor(diff_observation, cv2.COLOR_RGB2GRAY)
        diff_observation = diff_observation[:,:,np.newaxis]
        frame = np.concatenate((frame, diff_observation), axis=2)

        return frame#[:, :, None]
        # return frame

    def opticalFlow(self, prvsimg, nextimg):
        hsv = np.zeros_like(nextimg)
        hsv[..., 1] = 255

        prvs = cv2.cvtColor(prvsimg, cv2.COLOR_BGR2GRAY)
        n = cv2.cvtColor(nextimg, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, n, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        prvs = n
        return [rgb, prvs]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

def make_atari(env_id):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env

def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

class WarpFrameRGB(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(84, 84, 12), dtype=np.uint8)  # hack this part so that the graph is correctly built

    def observation(self, frame):
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        return frame

class WarpFrameRGBYolo(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(84, 84, 24), dtype=np.uint8)  # hack this part so that the graph is correctly built

    def observation(self, frame):
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        return frame


class WarpFrameRGBGreyFlow(gym.ObservationWrapper):
    # 增加了图像对比的通道
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 1 * 4), dtype=np.uint8)
        self.previous_observation = None
        self.current_observation = None
        # self.diff_observation = None
        self.diff_observation_cache = []

    def observation(self, frame):
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        if self.current_observation is None:
            self.previous_observation = self.current_observation = frame
        else:
            self.previous_observation = self.current_observation
            self.current_observation = frame

        diff_observation = self.opticalFlow(self.previous_observation, self.current_observation)[0]

        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # frame = frame[:, :, np.newaxis]
        diff_observation = cv2.cvtColor(diff_observation, cv2.COLOR_RGB2GRAY)
        diff_observation = diff_observation[:, :, np.newaxis]

        if len(self.diff_observation_cache) > 5:
            del(self.diff_observation_cache[0])
        self.diff_observation_cache.append(diff_observation)

        diff_observation = np.array(self.diff_observation_cache).min(axis=0)

        frame = np.concatenate((frame, diff_observation), axis=2)

        return frame  # [:, :, None]
        # return frame

    def opticalFlow(self, prvsimg, nextimg):
        hsv = np.zeros_like(nextimg)
        hsv[..., 1] = 255

        prvs = cv2.cvtColor(prvsimg, cv2.COLOR_BGR2GRAY)
        n = cv2.cvtColor(nextimg, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, n, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        prvs = n
        return [rgb, prvs]

SCALE = 32
GRID_W, GRID_H = 7, 7
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH = GRID_H*SCALE, GRID_W*SCALE, 3

class WrapFrameRGBwithBoundingBox(gym.ObservationWrapper):
  # RGB图像+yolo预测特征图像(白板+不同颜色灰色框)
  def __init__(self, env):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    gym.ObservationWrapper.__init__(self, env)
    self.width = 84
    self.height = 84
    self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1 * 4), dtype=np.uint8)

    self.yolo = None
    self.sess = None

  def build(self):
    config_path = "detect_model/shijc_config_0505.json"
    weights_path = "detect_model/shijc_weights_0505.binary"
    with open(config_path) as config_buffer:
      config = json.load(config_buffer)
    self.config = config


    self.image = tf.placeholder(shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], dtype=tf.float32, name='image_placeholder')
    self.yolo = yolo_net(self.image, False, n_class=len(config['model']['labels']), collections=[tf.GraphKeys.LOCAL_VARIABLES], trainable=False)

    self.weights_path = weights_path

    self.assign_kernel, self.assign_bias = load_from_binary(self.weights_path, offset=0)

  def load(self, sess):
    if self.sess == None:
      self.sess = sess
    # tf.reset_default_graph()
    sess.run(self.assign_kernel + self.assign_bias)


  def observation(self, frame):
    box_channel = np.ones(frame.shape[0:2], dtype=np.int8) * 255
    if self.sess != None :
      org_img = frame
      img = cv2.resize(org_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
      img = img / 255.0
      anchors = np.array(self.config['model']['anchors']).reshape(-1, 2)
      data = self.sess.run(self.yolo, feed_dict={self.image: img})
      boxes = getboxes(data, anchors, nclass=len(self.config['model']['labels']))
      box_channel = self.draw_boxes_2(box_channel, boxes, self.config["model"]["labels"])
    else :
      print("no tf session found")
    box_channel = cv2.resize(box_channel, (self.width, self.height), interpolation=cv2.INTER_AREA)
    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
    box_channel = box_channel[:,:, np.newaxis]

    frame = np.concatenate((frame, box_channel), axis=2)

    return frame

  def draw_boxes_2(self, img, boxes, labels):
    for box in boxes:
      xmin = int((box['x'] - box['w'] / 2) * img.shape[1])
      xmax = int((box['x'] + box['w'] / 2) * img.shape[1])
      ymin = int((box['y'] - box['h'] / 2) * img.shape[0])
      ymax = int((box['y'] + box['h'] / 2) * img.shape[0])

      cv2.rectangle(img, (xmin, ymin), (xmax, ymax), self.label_to_color(labels[box['label']]), 4)

    return img


  def label_to_color(self, label):
    for i in range(len(self.config["model"]['labels'])):
      if label == self.config["model"]['labels'][i] :
        return int(i * 8 + 8)

    return int(255)


# for vec env
import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'build':
            remote.close()
            break
        elif cmd == 'load':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True