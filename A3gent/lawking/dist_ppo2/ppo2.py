import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance
import json
from detect.backbone.tiny_darknet_fcn import yolo_net, load_from_binary
from detect.util.postprocessing import getboxes
from lawking.wrapper.atari_wrapper import LazyFrames
import random

import cv2
cv2.ocl.setUseOpenCL(False)
import sys

class YoloModel(object):
    def __init__(self):
        self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_DEPTH = 224, 224, 3

    def build(self, num_envs):
        config_path = "detect_model/shijc_config_0505.json"
        weights_path = "detect_model/shijc_weights_0505.binary"
        with open(config_path) as config_buffer:
            config = json.load(config_buffer)
        self.config = config

        self.image = tf.placeholder(shape=[None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_DEPTH],
                                    dtype=tf.float32,
                                    name='image_placeholder')
        self.yolo = yolo_net(self.image, False, n_class=len(config['model']['labels']),
                             collections=[tf.GraphKeys.LOCAL_VARIABLES], trainable=False)

        self.weights_path = weights_path

        self.assign_kernel, self.assign_bias = load_from_binary(self.weights_path, offset=0)

        self.sess = None

        self.label_set = {}
        for i in range(len(self.config["model"]['labels'])):
            self.label_set[self.config["model"]['labels'][i]] = [int(i * 8 + 8), int(i * 8 + 8), int(i * 8 + 8)]

        self.box_channel = np.ones(shape=(num_envs, 224, 224, 3), dtype=np.uint8) * 255
        self.box_channel_small = np.ones(shape=(num_envs, 84, 84, 3), dtype=np.uint8) * 255

        print('yolo built')

    def load(self, sess):
        self.sess = sess
        sess.run(self.assign_kernel + self.assign_bias)
        print('yolo loaded')

    def observation(self, frame):
        self.box_channel[:] = 0

        img = frame / 255.0
        anchors = np.array(self.config['model']['anchors']).reshape(-1, 2)
        data = self.sess.run(self.yolo, feed_dict={self.image: img})

        for i in range(frame.shape[0]):

            boxes = getboxes(data[i], anchors, nclass=len(self.config['model']['labels']))

            labels = self.config["model"]["labels"]
            for box in boxes:
                xmin = int((box['x'] - box['w'] / 2) * img.shape[2])
                xmax = int((box['x'] + box['w'] / 2) * img.shape[2])
                ymin = int((box['y'] - box['h'] / 2) * img.shape[1])
                ymax = int((box['y'] + box['h'] / 2) * img.shape[1])

                self.box_channel[i][ymin:ymax + 1, xmin:xmax + 1, :] = frame[i][ymin:ymax + 1, xmin:xmax + 1, :]
                cv2.rectangle(self.box_channel[i], (xmin, ymin), (xmax, ymax), self.label_set[labels[box['label']]], 5)

            cv2.resize(self.box_channel[i], (84, 84), dst=self.box_channel_small[i], interpolation=cv2.INTER_AREA)

        return self.box_channel_small


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val
    return f


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


class ppo2:

    class Model(object):
        def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                    nsteps, ent_coef, vf_coef, max_grad_norm, scope='model', collections=None, trainable=True, local_model=None, global_step=None):

            self.scope = scope
            self.act_model = policy(ob_space, ac_space, nbatch_act, 1, scope, reuse=False, collections=collections, trainable=trainable)

            if local_model:
                self.step = local_model.act_model.step
                self.value = local_model.act_model.value
                self.step_given_action = local_model.act_model.step_given_action
                self.initial_state = local_model.act_model.initial_state
                self.yolo = local_model.yolo
                self.yolo_load = local_model.yolo.load
                self.yolo_observe = local_model.yolo.observation

            else:
                self.step = self.act_model.step
                self.value = self.act_model.value
                self.step_given_action = self.act_model.step_given_action
                self.initial_state = self.act_model.initial_state

                self.local_variables = tf.local_variables(scope)

                self.yolo = YoloModel()
                self.yolo_build = self.yolo.build

                return

            init_updates = []
            assert len(tf.global_variables(scope)) == len(local_model.local_variables)
            for var, target_var in zip(tf.global_variables(scope), local_model.local_variables):
                print('syn  {} <- {}'.format(target_var.name, var.name))
                init_updates.append(tf.assign(target_var, var).op)
            self.syn = tf.group(*init_updates)

            train_model = policy(ob_space, ac_space, nbatch_train, nsteps, scope, reuse=True)

            A = train_model.pdtype.sample_placeholder([None])
            ADV = tf.placeholder(tf.float32, [None])
            R = tf.placeholder(tf.float32, [None])
            OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
            OLDVPRED = tf.placeholder(tf.float32, [None])
            LR = tf.placeholder(tf.float32, [])
            CLIPRANGE = tf.placeholder(tf.float32, [])

            neglogpac = train_model.pd.neglogp(A)
            entropy = tf.reduce_mean(train_model.pd.entropy())

            vpred = train_model.vf
            vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
            vf_losses1 = tf.square(vpred - R)
            vf_losses2 = tf.square(vpredclipped - R)
            vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
            ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
            pg_losses = -ADV * ratio
            pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
            approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
            loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
            with tf.variable_scope(scope):
                params = tf.trainable_variables(scope)
                for p in params:
                    print(p)

            print('params_grad')
            params_grad = params
            # params_grad = []
            # for p in params:
            #     if len(p.get_shape().as_list()) < 4:
            #         print(p)
            #         params_grad.append(p)

            grads = tf.gradients(loss, params_grad)
            if max_grad_norm is not None:
                grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads = list(zip(grads, params_grad))
            trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)

            if global_step:
                _train = trainer.apply_gradients(grads, global_step=global_step)
            else:
                _train = trainer.apply_gradients(grads)

            def train(sess, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
                advs = returns - values
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)
                td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                        CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
                if states is not None:
                    td_map[train_model.S] = states
                    td_map[train_model.M] = masks
                return sess.run(
                    [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                    td_map
                )[:-1]
            self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

            def save(sess, save_path):
                ps = sess.run(params)
                joblib.dump(ps, save_path)

            def get_restore(path):
                loaded_params = joblib.load(path)
                print("loading...")
                print(len(params))
                print(len(loaded_params))
                for x in params:
                    print(x)
                for x in loaded_params:
                    print(x.shape)
                if len(params) != len(loaded_params):
                    loaded_params = [loaded_params[i] for i in range(1, len(loaded_params))]
                restores = []
                print('len(params)=%d' % (len(params)))
                print('len(loaded_params)=%d' % (len(loaded_params)))
                for p, loaded_p in zip(params, loaded_params):
                    print(p.name)
                    print(p.shape)
                    print(loaded_p.shape)

                    if p.name == 'model/c1/w:0' and p.shape != loaded_p.shape:
                        print("need to do somthing")

                        x = np.zeros(p.shape)
                        x[:, :, 0:loaded_p.shape[2], :] = loaded_p
                        loaded_p = x

                    restores.append(p.assign(loaded_p))

                return restores

            self.restores = []
            self.restores.append(get_restore('cpt/checkpoints/0511_npn_01350_110592000'))

            def load(sess, idx=0):
                sess.run(self.restores[idx])
                print('restored')
                x1 = sess.run('model/v/b:0')
                x2 = sess.run('local_model/v/b:0')
                print('restored value')
                print(x1)
                print(x2)
                # If you want to load weights, also save/load observation scaling inside VecNormalize

            self.train = train
            self.train_model = train_model
            self.save = save
            self.load = load

    class Runner(object):

        def __init__(self, *, env, model, nsteps, gamma, lam):
            self.env = env
            self.model = model
            self.nenv = env.num_envs
            self.obshape = env.observation_space.shape
            print('observation_space shape')
            print(self.obshape)

            self.frames = []
            for i in range(self.nenv):
                self.frames.append(deque([], maxlen=4))

            self.obs = np.zeros((self.nenv, self.obshape[0], self.obshape[1], self.obshape[2]), dtype=model.train_model.X.dtype.name)

            self.gamma = gamma
            self.lam = lam
            self.nsteps = nsteps
            self.states = model.initial_state
            self.dones = [False for _ in range(self.nenv)]

            self.observe(env.reset())

            self.epson_max = 0.1
            self.epson_min = 0.005
            self.epson_delta = (self.epson_max - self.epson_min) / 1e6
            self.epson = self.epson_max

        def observe(self, obs):
            # print('true obs shape')
            # print(obs.shape)
            # print(obs.dtype)

            obs_onestep_merged = np.ones((self.nenv, self.obshape[0], self.obshape[1], 3), dtype=np.uint8) * 255

            # resize RGB
            for i in range(self.nenv):
                obs_onestep_merged[i, :, :, 0:3] = cv2.resize(obs[i], (self.obshape[0], self.obshape[1]), interpolation=cv2.INTER_AREA)

            # if self.model.yolo.sess:
            #     detected = self.model.yolo_observe(obs)
            #
            #     for i in range(obs.shape[0]):
            #         obs_onestep_merged[i, :, :, 3:6] = detected[i]

                # to show the images are correct
                # cv2.imshow('1', obs_onestep_merged[0, :, :, 0:3])
                # cv2.imshow('2', obs_onestep_merged[1, :, :, 0:3])
                #
                # cv2.imshow('1a', obs_onestep_merged[0, :, :, 3:6])
                # cv2.imshow('2a', obs_onestep_merged[0, :, :, 3:6])

                # cv2.waitKey(10)

            for i in range(self.nenv):
                if self.dones[i] or not self.model.yolo.sess:
                    # print('reset frame for env %d' % i)
                    for _ in range(4):
                        self.frames[i].append(obs_onestep_merged[i])
                else:
                    self.frames[i].append(obs_onestep_merged[i])

                # set self.obs
                self.obs[i] = LazyFrames(list(self.frames[i]))

            return

        def run(self, sess):
            print('self.epson=%f' % self.epson)

            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
            mb_states = self.states
            epinfos = []
            for _ in range(self.nsteps):
                actions, values, self.states, neglogpacs = self.model.step(sess, self.obs, self.states, self.dones)

                # random_actions = np.random.randint(self.env.action_space.n, size=actions.shape)
                # actions_r, values_r, self.states, neglogpacs_r = self.model.step_given_action(sess, self.obs, random_actions)
                #
                # idx = np.random.rand(*actions.shape) < self.epson
                # actions[idx] = actions_r[idx]
                # values[idx] = values_r[idx]
                # neglogpacs[idx] = neglogpacs_r[idx]
                #
                # self.epson -= self.epson_delta

                mb_obs.append(self.obs.copy())
                mb_actions.append(actions)
                mb_values.append(values)
                mb_neglogpacs.append(neglogpacs)
                mb_dones.append(self.dones)
                obs, rewards, self.dones, infos = self.env.step(actions)
                self.observe(obs)

                mb_rewards.append(rewards)

                for i in range(self.nenv):
                    done = self.dones[i]
                    info = infos[i]
                    if done:
                        epinfos.append(info)

            #batch of steps to batch of rollouts
            mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_actions = np.asarray(mb_actions)
            mb_values = np.asarray(mb_values, dtype=np.float32)
            mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
            mb_dones = np.asarray(mb_dones, dtype=np.bool)
            last_values = self.model.value(sess, self.obs, self.states, self.dones)

            #discount/bootstrap off value fn
            mb_returns = np.zeros_like(mb_rewards)
            mb_advs = np.zeros_like(mb_rewards)
            lastgaelam = 0
            for t in reversed(range(self.nsteps)):
                if t == self.nsteps - 1:
                    nextnonterminal = 1.0 - self.dones
                    nextvalues = last_values
                else:
                    nextnonterminal = 1.0 - mb_dones[t+1]
                    nextvalues = mb_values[t+1]
                delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
                mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            mb_returns = mb_advs + mb_values
            print('return!')
            return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states, epinfos)


    def build(self, *, policy, env, nsteps, total_timesteps, ent_coef, lr,
                vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
                log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
                save_interval=0, save_dir='cpt', task_index=-1, scope='model', collections=None, trainable=True, local_model=None, global_step=None):

        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.nsteps = nsteps
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.lr = lr

        self.cliprange = cliprange
        self.save_dir = save_dir
        self.task_index = task_index

        if isinstance(lr, float): self.lr = constfn(lr)
        else: assert callable(lr)
        if isinstance(cliprange, float): self.cliprange = constfn(cliprange)
        else: assert callable(cliprange)
        self.total_timesteps = int(total_timesteps)

        self.nenvs = env.num_envs
        self.ob_space = env.observation_space
        self.ac_space = env.action_space
        self.nbatch = self.nenvs * nsteps
        self.nbatch_train = self.nbatch // nminibatches

        self.make_model = lambda : self.Model(policy=policy, ob_space=self.ob_space, ac_space=self.ac_space, nbatch_act=self.nenvs, nbatch_train=self.nbatch_train,
                        nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                        max_grad_norm=max_grad_norm, scope=scope, collections=collections, trainable=trainable, local_model=local_model, global_step=global_step)

        self.model = self.make_model()

        if trainable:
            self.runner = self.Runner(env=env, model=self.model, nsteps=nsteps, gamma=gamma, lam=lam)

    def learn(self, sess):

        max_score = [0, 0]
        self.best_idx = -1

        epinfobuf = deque(maxlen=100)
        tfirststart = time.time()

        nupdates = self.total_timesteps//self.nbatch
        for update in range(1, nupdates+1):
            assert self.nbatch % self.nminibatches == 0
            nbatch_train = self.nbatch // self.nminibatches
            tstart = time.time()
            frac = 1.0 - (update - 1.0) / nupdates
            lrnow = self.lr(frac)
            cliprangenow = self.cliprange(frac)

            # syn to local
            sess.run(self.model.syn)
            x1 = sess.run('model/v/b:0')
            x2 = sess.run('local_model/v/b:0')
            print('syn valid')
            print(x1)
            print(x2)

            syn_ends = time.time()
            obs, returns, masks, actions, values, neglogpacs, states, epinfos = self.runner.run(sess) #pylint: disable=E0632
            epinfobuf.extend(epinfos)

            mblossvals = []
            learnstart = time.time()
            if states is None: # nonrecurrent version
                inds = np.arange(self.nbatch)

                for _ in range(self.noptepochs):
                    np.random.shuffle(inds)
                    for start in range(0, self.nbatch, nbatch_train):
                        end = start + nbatch_train
                        mbinds = inds[start:end]
                        slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mblossvals.append(self.model.train(sess, lrnow, cliprangenow, *slices))

            else: # recurrent version
                assert self.nenvs % self.nminibatches == 0
                envsperbatch = self.nenvs // self.nminibatches
                envinds = np.arange(self.nenvs)
                flatinds = np.arange(self.nenvs * self.nsteps).reshape(self.nenvs, self.nsteps)
                envsperbatch = nbatch_train // self.nsteps
                for _ in range(self.noptepochs):
                    np.random.shuffle(envinds)
                    for start in range(0, self.nenvs, envsperbatch):
                        end = start + envsperbatch
                        mbenvinds = envinds[start:end]
                        mbflatinds = flatinds[mbenvinds].ravel()
                        slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mbstates = states[mbenvinds]
                        mblossvals.append(self.model.train(sess, lrnow, cliprangenow, *slices, mbstates))

            lossvals = np.mean(mblossvals, axis=0)
            tnow = time.time()
            sample_time = int((learnstart - tstart))
            learn_time = int((tnow - learnstart))
            total_time = int((tnow - tstart))
            syn_time = int(syn_ends - tstart)
            fps = int(self.nbatch / (tnow - tstart))

            print(update % self.log_interval == 0 or update == 1)
            if update % self.log_interval == 0 or update == 1:
                ev = explained_variance(values, returns)

                eprewmean = safemean([epinfo['episode_reward'] for epinfo in epinfobuf])
                epstepmean = safemean([epinfo['episode_step'] for epinfo in epinfobuf])

                logger.logkv("serial_timesteps", update * self.nsteps)
                logger.logkv("nupdates", update)
                logger.logkv("total_timesteps", update * self.nbatch)
                logger.logkv("fps", fps)
                logger.logkv("sample_time", sample_time)
                logger.logkv("learn_time", learn_time)
                logger.logkv("total_time", total_time)
                logger.logkv("syn_time", syn_time)
                logger.logkv("explained_variance", float(ev))
                logger.logkv('eprewmean', eprewmean)
                logger.logkv('epstepmean', epstepmean)
                logger.logkv('time_elapsed', tnow - tfirststart)
                logger.logkv('best_idx', self.best_idx)
                for (lossval, lossname) in zip(lossvals, self.model.loss_names):
                    logger.logkv(lossname, lossval)
                logger.dumpkvs()

            if self.task_index == 0 and self.save_interval and (update % self.save_interval == 0 or update == 1) and self.save_dir:
                # save to joblib
                checkdir = osp.join(self.save_dir, 'checkpoints')
                os.makedirs(checkdir, exist_ok=True)
                savepath = osp.join(checkdir, '%.5i'%update)
                print('Saving to', savepath)
                self.model.save(sess, savepath)
                print('Saved to', savepath)

        self.env.close()


