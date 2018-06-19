import gym_remote.exceptions as gre
import tensorflow as tf

from lawking.wrapper.atari_wrapper import WarpFrameRGB, WarpFrameRGBYolo
from lawking.dist_ppo2.ppo2 import ppo2
from lawking.dist_ppo2.policies import CnnPolicy

from lawking.wrapper.sonic_util import FaKeSubprocVecEnv

import argparse

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument("--local", type="bool", nargs='?', const=False, default=False, help="local mode.")
parser.add_argument("--load", type="bool", nargs='?', const=True, default=True, help="loade mode.")
FLAGS, unparsed = parser.parse_known_args()

if FLAGS.local:
    from lawking.wrapper.sonic_util import make_env_local as make_env
else:
    from lawking.wrapper.sonic_util import make_env as make_env


def main():
    agent = ppo2()
    num_env = 1
    env = FaKeSubprocVecEnv([lambda: make_env(stack=False, scale_rew=True, frame_wrapper=WarpFrameRGB, reward_type=30)] * num_env)

    local_agent = ppo2()
    local_agent.build(policy=CnnPolicy,
                      env=env,
                      nsteps=8192,
                      nminibatches=1,
                      lam=0.95,
                      gamma=0.99,
                      noptepochs=4,
                      log_interval=1,
                      ent_coef=0.001,
                      lr=lambda f: f*4e-4,
                      cliprange=lambda f: f*0.2,
                      total_timesteps=int(3e6),
                      save_interval=0,
                      save_dir='cpt',
                      task_index=0,
                      scope='local_model',
                      collections=[tf.GraphKeys.LOCAL_VARIABLES],
                      trainable=False)
    local_agent.model.yolo_build(num_env)

    # Build model...
    global_step = tf.train.get_or_create_global_step()
    agent.build(policy=CnnPolicy,
                env=env,
                nsteps=8192,
                nminibatches=1,
                lam=0.95,
                gamma=0.99,
                noptepochs=4,
                log_interval=1,
                ent_coef=0.001,
                lr=lambda f: f*4e-4,
                cliprange=lambda f: f*0.2,
                total_timesteps=int(3e6),
                save_interval=0,
                save_dir='cpt',
                task_index=0,
                local_model=local_agent.model,
                global_step=global_step)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # pylint: disable=E1101
    with tf.Session(config=config) as mon_sess:
        mon_sess.run(tf.global_variables_initializer())
        mon_sess.run(tf.local_variables_initializer())

        agent.model.load(mon_sess)
        agent.model.yolo_load(mon_sess)

        agent.learn(mon_sess)

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
