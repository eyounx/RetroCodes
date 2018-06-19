import os
import cv2
import sys
import json
import numpy as np
import tensorflow as tf
from ..backbone.tiny_darknet_fcn import yolo_net, load_from_binary
from ..util.postprocessing import postprocess

SCALE = 32
GRID_W, GRID_H = 7, 7
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH = GRID_H*SCALE, GRID_W*SCALE, 3


def predict(args):
    config_path = args.conf
    weights_path = args.weights
    image_dir = args.input
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    image = tf.placeholder(shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], dtype=tf.float32, name='image_placeholder')
    y = yolo_net(image, False, n_class=len(config['model']['labels']))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    load_from_binary(sess, weights_path, offset=0)

    # saver = tf.train.Saver(max_to_keep=None)
    # saver.save(sess, '../../model/yolo/tiny_darknet/tiny_darknet', global_step=0)

    # kernel = sess.graph.get_tensor_by_name("conv{}/conv2d/kernel:0".format(16))
    # print(sess.run(kernel))

    anchors = np.array(config['model']['anchors']).reshape(-1, 2)

    for root, dirs, files in os.walk(image_dir):
        for f in files:
            org_img = cv2.imread(os.path.join(root, f))
            img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            img = img / 255.0

            data = sess.run(y, feed_dict={image: [img]})
            img, _ = postprocess(data, anchors, config['model']['labels'],
                                 org_img, nclass=len(config['model']['labels']))
            cv2.imwrite(os.path.join(output_dir, f), img)
