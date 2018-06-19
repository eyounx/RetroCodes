#! /usr/bin/env python

import argparse
import os
import json
import struct
import numpy as np
import tensorflow as tf
from keras import backend as K
from ..backbone.frontend import YOLO

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default='cfg/tiny_darknet.json',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    default='model/tiny_darknet_person_n.h5',
    help='path to pretrained weights')

argparser.add_argument(
    '-o',
    '--output',
    default='model/DetectNet.data',
    help='path to detect network params')


def store_weights(sess, weights_file=None):
    data = open(weights_file, 'wb')
    graph = sess.graph

    print([v.name for v in tf.trainable_variables()])

    total = 0
    nb_conv = 19
    for i in range(1, 20):
        if i < nb_conv:
            kernel = graph.get_tensor_by_name("conv_{}/kernel:0".format(i))
            beta = graph.get_tensor_by_name("norm_{}/beta:0".format(i))
            gamma = graph.get_tensor_by_name("norm_{}/gamma:0".format(i))
            mean = graph.get_tensor_by_name("norm_{}/moving_mean:0".format(i))
            var = graph.get_tensor_by_name("norm_{}/moving_variance:0".format(i))
            beta, gamma, mean, var, kernel = sess.run([beta, gamma, mean, var, kernel])
            # k_bias = np.zeros(kernel.shape[-1])
            # std = np.sqrt(var + 0.001)
            # scale = gamma / std
            # bias = beta - gamma * mean / std
            values = [beta, gamma, mean, var, kernel]
        else:
            i = 23 # hard code
            kernel = graph.get_tensor_by_name("conv_{}/kernel:0".format(i))
            bias = graph.get_tensor_by_name("conv_{}/bias:0".format(i))
            kernel, bias = sess.run([kernel, bias])
            values = [bias, kernel]

        for v in values:
            if len(v.shape) > 1:
                v = np.transpose(v, (3, 2, 0, 1))
            print('{}:{}'.format(i, v.shape))
            v = v.ravel()
            total += v.shape[0]
            ff = 'f' * v.shape[0]
            d = struct.pack(ff, *v)
            data.write(d)
    data.close()
    print('total parameters: {}'.format(total))


def _main_(args):
    config_path = args.conf
    weights_path = args.weights
    output_path = args.output

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    ###############################
    #   Make the model
    ###############################

    yolo = YOLO(architecture=config['model']['architecture'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################

    print weights_path
    yolo.load_weights(weights_path)

    ###############################
    #   Convert Keras model to TF
    ###############################
    sess = K.get_session()
    store_weights(sess, output_path)

    saver = tf.train.Saver(max_to_keep=None)
    saver.save(sess, 'model/tiny_darknet/tiny_darknet_keras', global_step=0)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
