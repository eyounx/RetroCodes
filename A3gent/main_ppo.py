import argparse
import sys

import tensorflow as tf

# !/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
"""

import tensorflow as tf
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from lawking.wrapper.sonic_util import make_env_local
from lawking.wrapper.atari_wrapper import WarpFrameRGB, WarpFrameRGBGreyFlow, WrapFrameRGBwithBoundingBox
from lawking.dist_ppo2.ppo2 import ppo2
from lawking.dist_ppo2.policies import CnnPolicy

import logging, os

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s\t\t%(message)s')

FLAGS = tf.flags.FLAGS
# PAI TF used
tf.flags.DEFINE_string("ps_hosts", "", "ps_hosts")
tf.flags.DEFINE_string("worker_hosts", "", "worker_hosts")
tf.flags.DEFINE_string("job_name", "", "job_name")
tf.flags.DEFINE_integer("task_index", "-1", "task_index")
tf.flags.DEFINE_string("tables", "", "tables names")
tf.flags.DEFINE_string("outputs", "", "output tables names")
tf.flags.DEFINE_string("checkpointDir", "", "oss buckets for saving checkpoint")
tf.flags.DEFINE_string("buckets", "", "oss buckets")

tf.flags.DEFINE_string("config", "lawking/json/atari.json", "filename or the json string of the configuration")
tf.flags.DEFINE_boolean("has_json_file", False,
                        "Set true to load json configuration file from the relative path `./json_file.json`")


def main(_):
    logging.info('pid %d job_name=%s, task_index=%d' % (os.getpid(), FLAGS.job_name, FLAGS.task_index))

    print(FLAGS.ps_hosts)
    print(FLAGS.worker_hosts)
    print(FLAGS.job_name)
    print(FLAGS.task_index)

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    seeds = 0
    for ps in ps_hosts + worker_hosts:
        seeds += int(ps.split(':')[1])
    import random
    random.seed(seeds)

    ip_map = dict()

    fix_ps_hosts = []
    for ps in ps_hosts:
        ip_port = ps.split(':')
        if ip_port[0] in ip_map:
            ip_map[ip_port[0]] += 1
            port = ip_map[ip_port[0]]
        else:
            port = random.randint(10000, 20000)
            ip_map[ip_port[0]] = port
        fix_ps_hosts.append(ip_port[0] + ":" + str(port))

    fix_worker_hosts = []
    for worker in worker_hosts:
        ip_port = worker.split(':')
        if ip_port[0] in ip_map:
            ip_map[ip_port[0]] += 1
            port = ip_map[ip_port[0]]
        else:
            port = random.randint(10000, 20000)
            ip_map[ip_port[0]] = port
        fix_worker_hosts.append(ip_port[0] + ":" + str(port))

    cluster = tf.train.ClusterSpec({"ps": fix_ps_hosts, "worker": fix_worker_hosts})

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # config.log_device_placement = True

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             config=config,
                             task_index=FLAGS.task_index,
                             protocol='grpc')

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        agent = ppo2()
        num_env = 10
        env = SubprocVecEnv([lambda: make_env_local(stack=False, scale_rew=True, idx=int(FLAGS.task_index/1), frame_wrapper=WarpFrameRGB, reward_type=30)] * num_env)
        # Assigns ops to the local worker by default.
        # with tf.device(device_name_or_function="/job:" + FLAGS.job_name + ("/task:%d" % FLAGS.task_index)):

        with tf.device("/job:worker/task:%d" % FLAGS.task_index):
            a = tf.Variable([0.], name='a', collections=[tf.GraphKeys.LOCAL_VARIABLES])
            test_op = tf.assign(a, a + 1)
            local_agent = ppo2()
            local_agent.build(policy=CnnPolicy,
                              env=env,
                              nsteps=8192,
                              nminibatches=8 * num_env,
                              lam=0.95,
                              gamma=0.99,
                              noptepochs=4,
                              log_interval=1,
                              ent_coef=0.001,
                              lr=lambda _: 2e-5,
                              cliprange=lambda _: 0.2,
                              total_timesteps=int(1e10),
                              save_interval=20,
                              save_dir='cpt',
                              task_index=FLAGS.task_index,
                              scope='local_model',
                              collections=[tf.GraphKeys.LOCAL_VARIABLES],
                              trainable=False)
            local_agent.model.yolo_build(num_env)

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # Build model...
            global_step = tf.train.get_or_create_global_step()
            agent.build(policy=CnnPolicy,
                        env=env,
                        nsteps=8192,
                        nminibatches=8 * num_env,
                        lam=0.95,
                        gamma=0.99,
                        noptepochs=4,
                        log_interval=1,
                        ent_coef=0.001,
                        lr=lambda _: 2e-5,
                        cliprange=lambda _: 0.2,
                        total_timesteps=int(1e10),
                        save_interval=10,
                        save_dir='cpt',
                        task_index=FLAGS.task_index,
                        local_model=local_agent.model,
                        global_step=global_step)

        # The StopAtStepHook handles stopping after running given steps.
        hooks = [tf.train.StopAtStepHook(last_step=1000000000000)]

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        checkpoint_dir = 'checkpoint'
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               checkpoint_dir=checkpoint_dir,
                                               hooks=hooks,
                                               save_checkpoint_secs=None,
                                               save_summaries_steps=None,
                                               save_summaries_secs=None) as mon_sess:
            agent.model.load(mon_sess)
            agent.model.yolo_load(mon_sess)

            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.

                agent.learn(mon_sess)


if __name__ == "__main__":
    tf.app.run()

