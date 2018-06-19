#! /usr/bin/env python

import argparse
import os
import cv2
import json
from ..util.utils import draw_boxes
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
    '-i',
    '--input',
    default='images/validation',
    help='path to an image or an video (mp4 format)')

argparser.add_argument(
    '-o',
    '--output',
    default='images/output',
    help='path to an image or an video (mp4 format)')


def _main_(args):
    config_path = args.conf
    weights_path = args.weights
    image_path = args.input
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

    # for layer in yolo.model.layers:
    #     print(layer.name, layer.input_shape, layer.output_shape)

    ###############################
    #   Load trained weights
    ###############################

    print weights_path
    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes
    ###############################
    out = open(os.path.join(output_path, 'box.txt'), 'w')
    for root, dirs, files in os.walk(image_path):
        for f in files:
            out.write('{}'.format(f))
            image = cv2.imread(os.path.join(root, f))
            boxes = yolo.predict(image)
            image = draw_boxes(image, boxes, config['model']['labels'])
            for box in boxes:
                xmin = int((box.x - box.w / 2) * image.shape[1])
                xmax = int((box.x + box.w / 2) * image.shape[1])
                ymin = int((box.y - box.h / 2) * image.shape[0])
                ymax = int((box.y + box.h / 2) * image.shape[0])
                out.write(',{} {} {} {}'.format(xmin, ymin, xmax, ymax))
            out.write('\n')
            cv2.imwrite(os.path.join(output_path, f), image)
    out.close()

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
