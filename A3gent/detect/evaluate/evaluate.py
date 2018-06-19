#! /usr/bin/env python

import argparse
import os
import cv2
import json
import tqdm
import numpy as np
from tqdm import tqdm
from ..util.preprocessing import parse_annotation
from ..util.utils import draw_boxes
from ..backbone.frontend import YOLO
from ...mean_average_precision.detection_map import DetectionMAP

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

def _main_(args):

    config_path  = args.conf
    weights_path = args.weights

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    ###############################
    #   Make the model
    ###############################

    yolo = YOLO(architecture        = config['model']['architecture'],
                input_size          = config['model']['input_size'],
                labels              = config['model']['labels'],
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################

    print weights_path
    yolo.load_weights(weights_path)

    ###############################
    #   Parse validation images
    ###############################

    valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'],
                                                config['valid']['valid_image_folder'],
                                                config['model']['labels'])

    ###############################
    #   Comupte mAP
    ###############################

    mAP = DetectionMAP(len(config['model']['labels']))
    i = 0
    for image in tqdm(valid_imgs):
        img = cv2.imread(image['filename'])
        boxes = yolo.predict(img)
        # process predict box
        pred_bb = []
        pred_cls = []
        pred_conf = []
        for box in boxes:
            xmin = round((box.x - box.w/2) * img.shape[1])
            xmax = round((box.x + box.w/2) * img.shape[1])
            ymin = round((box.y - box.h/2) * img.shape[0])
            ymax = round((box.y + box.h/2) * img.shape[0])
            pred_bb.append([xmin, ymin, xmax, ymax])
            pred_cls.append(0)    # only one class, fit zero
            pred_conf.append(box.get_score())

        # process ground truth box
        gt_bb = []
        gt_cls = []
        object = image['object']
        for obj in object:
            xmin = float(obj['xmin'])
            xmax = float(obj['xmax'])
            ymin = float(obj['ymin'])
            ymax = float(obj['ymax'])
            gt_bb.append([xmin, ymin, xmax, ymax])
            gt_cls.append(0)     # only one class, fit zero

        pred_bb = np.array(pred_bb)
        pred_cls = np.array(pred_cls)
        pred_conf = np.array(pred_conf)
        gt_bb = np.array(gt_bb)
        gt_cls = np.array(gt_cls)
        mAP.evaluate(pred_bb, pred_cls, pred_conf, gt_bb, gt_cls)
    mAP.savefig()



if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
