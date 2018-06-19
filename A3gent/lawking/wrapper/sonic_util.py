"""
Environments and wrappers for Sonic training.
"""

import gym, random
import numpy as np

from lawking.wrapper.atari_wrapper import WarpFrame, FrameStack, WarpFrameRGB
import gym_remote.client as grc

from collections import deque
import cv2

train_level = [['SonicTheHedgehog-Genesis', 'SpringYardZone.Act3'],
               ['SonicTheHedgehog-Genesis', 'SpringYardZone.Act2'],
               ['SonicTheHedgehog-Genesis', 'GreenHillZone.Act3'],
               ['SonicTheHedgehog-Genesis', 'GreenHillZone.Act1'],
               ['SonicTheHedgehog-Genesis', 'StarLightZone.Act2'],
               ['SonicTheHedgehog-Genesis', 'StarLightZone.Act1'],
               ['SonicTheHedgehog-Genesis', 'MarbleZone.Act2'],
               ['SonicTheHedgehog-Genesis', 'MarbleZone.Act1'],
               ['SonicTheHedgehog-Genesis', 'MarbleZone.Act3'],
               ['SonicTheHedgehog-Genesis', 'ScrapBrainZone.Act2'],
               ['SonicTheHedgehog-Genesis', 'LabyrinthZone.Act2'],
               ['SonicTheHedgehog-Genesis', 'LabyrinthZone.Act1'],
               ['SonicTheHedgehog-Genesis', 'LabyrinthZone.Act3'],
               ['SonicTheHedgehog2-Genesis', 'EmeraldHillZone.Act1'],
               ['SonicTheHedgehog2-Genesis', 'EmeraldHillZone.Act2'],
               ['SonicTheHedgehog2-Genesis', 'ChemicalPlantZone.Act2'],
               ['SonicTheHedgehog2-Genesis', 'ChemicalPlantZone.Act1'],
               ['SonicTheHedgehog2-Genesis', 'MetropolisZone.Act1'],
               ['SonicTheHedgehog2-Genesis', 'MetropolisZone.Act2'],
               ['SonicTheHedgehog2-Genesis', 'OilOceanZone.Act1'],
               ['SonicTheHedgehog2-Genesis', 'OilOceanZone.Act2'],
               ['SonicTheHedgehog2-Genesis', 'MysticCaveZone.Act2'],
               ['SonicTheHedgehog2-Genesis', 'MysticCaveZone.Act1'],
               ['SonicTheHedgehog2-Genesis', 'HillTopZone.Act1'],
               ['SonicTheHedgehog2-Genesis', 'CasinoNightZone.Act1'],
               ['SonicTheHedgehog2-Genesis', 'WingFortressZone'],
               ['SonicTheHedgehog2-Genesis', 'AquaticRuinZone.Act2'],
               ['SonicTheHedgehog2-Genesis', 'AquaticRuinZone.Act1'],
               ['SonicAndKnuckles3-Genesis', 'LavaReefZone.Act2'],
               ['SonicAndKnuckles3-Genesis', 'CarnivalNightZone.Act2'],
               ['SonicAndKnuckles3-Genesis', 'CarnivalNightZone.Act1'],
               ['SonicAndKnuckles3-Genesis', 'MarbleGardenZone.Act1'],
               ['SonicAndKnuckles3-Genesis', 'MarbleGardenZone.Act2'],
               ['SonicAndKnuckles3-Genesis', 'MushroomHillZone.Act2'],
               ['SonicAndKnuckles3-Genesis', 'MushroomHillZone.Act1'],
               ['SonicAndKnuckles3-Genesis', 'DeathEggZone.Act1'],
               ['SonicAndKnuckles3-Genesis', 'DeathEggZone.Act2'],
               ['SonicAndKnuckles3-Genesis', 'FlyingBatteryZone.Act1'],
               ['SonicAndKnuckles3-Genesis', 'SandopolisZone.Act1'],
               ['SonicAndKnuckles3-Genesis', 'SandopolisZone.Act2'],
               ['SonicAndKnuckles3-Genesis', 'HiddenPalaceZone'],
               ['SonicAndKnuckles3-Genesis', 'HydrocityZone.Act2'],
               ['SonicAndKnuckles3-Genesis', 'IcecapZone.Act1'],
               ['SonicAndKnuckles3-Genesis', 'IcecapZone.Act2'],
               ['SonicAndKnuckles3-Genesis', 'AngelIslandZone.Act1'],
               ['SonicAndKnuckles3-Genesis', 'LaunchBaseZone.Act2'],
               ['SonicAndKnuckles3-Genesis', 'LaunchBaseZone.Act1']]

test_level = [['SonicTheHedgehog-Genesis', 'SpringYardZone.Act1'],
              ['SonicTheHedgehog-Genesis', 'GreenHillZone.Act2'],
              ['SonicTheHedgehog-Genesis', 'StarLightZone.Act3'],
              ['SonicTheHedgehog-Genesis', 'ScrapBrainZone.Act1'],
              ['SonicTheHedgehog2-Genesis', 'MetropolisZone.Act3'],
              ['SonicTheHedgehog2-Genesis', 'HillTopZone.Act2'],
              ['SonicTheHedgehog2-Genesis', 'CasinoNightZone.Act2'],
              ['SonicAndKnuckles3-Genesis', 'LavaReefZone.Act1'],
              ['SonicAndKnuckles3-Genesis', 'FlyingBatteryZone.Act2'],
              ['SonicAndKnuckles3-Genesis', 'HydrocityZone.Act1'],
              ['SonicAndKnuckles3-Genesis', 'AngelIslandZone.Act2']]



def make_env_local(stack=True, scale_rew=True, idx=6, frame_wrapper=WarpFrame, reward_type=None):
    """
    Create an environment with some standard wrappers.
    """
    from retro_contest.local import make

    all_level = train_level + test_level

    print(str(idx) + ": start game=" + all_level[idx][0] + ", state=" + all_level[idx][1])

    env = make(game=all_level[idx][0], state=all_level[idx][1])

    return wrap_env(env, stack, scale_rew, frame_wrapper, reward_type)


def make_env(stack=True, scale_rew=True, frame_wrapper=WarpFrame, reward_type=None):
    """
    Create an environment with some standard wrappers.
    """
    env = grc.RemoteEnv('tmp/sock')

    return wrap_env(env, stack, scale_rew, frame_wrapper, reward_type)


def wrap_env(env, stack=True, scale_rew=True, frame_wrapper=WarpFrame, reward_type=None):
    env = SonicDiscretizer(env)

    if scale_rew:
        env = RewardScaler(env)
    env = frame_wrapper(env)

    print('reward_type=%d' % reward_type)
    if reward_type == 30:
        print('choose EpisodeInfoEnvHist')
        env = EpisodeInfoEnvHist(env, reward_type)
    else:
        print('choose EpisodeInfoEnv')
        env = EpisodeInfoEnv(env, reward_type)

    if stack:
        env = FrameStack(env, 4)

    return env


class EpisodeInfoEnv(gym.Wrapper):
    def __init__(self, env, reward_type):
        gym.Wrapper.__init__(self, env)
        self.reward_type = reward_type

        self.episode_reward = 0
        self.episode_step = 0
        self.episode_negbuf = 0
        self.rewardbuf = deque(maxlen=500)

        if self.reward_type:
            self.last_rings = -1
            self.visited = {}

        self.WINDOWSSIZE = 50

    def reset(self, **kwargs):
        self.episode_reward = 0
        self.episode_step = 0
        self.episode_negbuf = 0
        self.rewardbuf.clear()

        if self.reward_type:
            self.last_rings = -1
            self.visited = {}

        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.episode_reward += reward / 0.005
        self.episode_step += 1

        self.rewardbuf.append(self.episode_reward)

        info['episode_reward'] = self.episode_reward
        info['episode_step'] = self.episode_step
        info['old_episode_reward'] = self.rewardbuf[0]

        if done:
            if self.reward_type:
                if self.episode_step < 4490 and self.episode_reward < 8000:  # killed
                    reward -= 0.5

            obs = self.reset()

        # process reward
        not_punish_backward = True

        if not_punish_backward:
            if reward < 0 or self.episode_negbuf < 0:
                self.episode_negbuf += reward
                reward = 0

        if self.reward_type:
            if self.reward_type in [1, 3]:  # use ring / ring_location
                if self.last_rings >= 0:
                    reward += (info['rings'] - self.last_rings) * 0.1
                self.last_rings = info['rings']

            if self.reward_type in [2, 3]:  # use location / ring_location
                curx = int(info['x'] / self.WINDOWSSIZE)
                cury = int(info['y'] / self.WINDOWSSIZE)
                if not (curx, cury) in self.visited:
                    self.visited[(curx, cury)] = True
                    reward += 0.1

        return obs, reward, done, info


class EpisodeInfoEnvHist(gym.Wrapper):
    def __init__(self, env, reward_type):
        gym.Wrapper.__init__(self, env)
        self.reward_type = reward_type

        self.episode_reward = 0
        self.episode_step = 0
        self.episode_negbuf = 0
        self.rewardbuf = deque(maxlen=500)

        if self.reward_type:
            self.history = {}

        self.WINDOWSSIZE = 50

    def reset(self, **kwargs):
        self.episode_reward = 0
        self.episode_step = 0
        self.episode_negbuf = 0
        self.rewardbuf.clear()

        if self.reward_type:
            self.history = {}

        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.episode_reward += reward / 0.005
        self.episode_step += 1

        self.rewardbuf.append(self.episode_reward)

        info['episode_reward'] = self.episode_reward
        info['episode_step'] = self.episode_step
        info['old_episode_reward'] = self.rewardbuf[0]

        if done:
            if self.reward_type:
                if self.episode_step < 4490 and self.episode_reward < 8000:  # killed
                    reward -= 0.5

            obs = self.reset()

        # process reward
        not_punish_backward = True

        if not_punish_backward:
            if reward < 0 or self.episode_negbuf < 0:
                self.episode_negbuf += reward
                reward = 0

        if self.reward_type:
            if self.reward_type in [30]:
                curx = int(info['episode_reward'] / 90)
                tmp = str(int(curx))
                for i in range(3):
                    tmp += "," + str(int(np.sum(obs[:, :, i] / 555555)))
                if not (tmp in self.history):
                    self.history[tmp] = True
                    reward += 0.1

        return obs, reward, done, info


class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B'], [], ['LEFT', 'B'], ['RIGHT', 'B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()


class SonicDiscretizerV2(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B'], [], ['LEFT', 'B'], ['RIGHT', 'B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions) + 1)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, a):
        if a < len(self._actions):
            return self.env.step(self._actions[a].copy())
        else:
            if a == len(self._actions):
                reward = 0
                for i in range(30):
                    obs, r, done, info = self.env.step(self._actions[4].copy())
                    reward += r

                    if done:
                        return obs, reward, done, info

                obs, r, done, info = self.env.step(self._actions[6].copy())
                reward += r

                return obs, reward, done, info


class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.005

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info


import numpy as np
from baselines.common.vec_env import VecEnv

class FaKeSubprocVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        obs = []
        rews = []
        dones = []
        infos = []

        for i in range(self.num_envs):
            obs_tuple, reward,  done, info = self.envs[i].step(self.actions[i])
            # if i == 0:
            #     import time
            #     time.sleep(0.03)
            #     self.envs[i].render()

            obs.append(obs_tuple)
            rews.append(reward)
            dones.append(done)
            infos.append(info)

        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        obs = []
        for i in range(self.num_envs):
            obs_tuple = self.envs[i].reset()
            obs.append(obs_tuple)
        return np.stack(obs)

    def close(self):
        for i in range(self.num_envs):
            self.envs[i].close()