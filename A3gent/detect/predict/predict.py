#! /usr/bin/env python

import os
import cv2
import numpy as np
from tqdm import tqdm
from ..util.utils import draw_boxes
from ..backbone.frontend import YOLO
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def predict(args):
	config_path = args.conf
	weights_path = args.weights
	image_path = args.input
	output_path = args.output

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

	print (weights_path)
	yolo.load_weights(weights_path)

	###############################
	#   Predict bounding boxes
	###############################

	if image_path[-4:] == '.mp4':
		video_out = image_path[:-4] + '_detected' + image_path[-4:]

		video_reader = cv2.VideoCapture(image_path)

		nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
		frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
		frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

		video_writer = cv2.VideoWriter(video_out,
							   cv2.VideoWriter_fourcc(*'MPEG'),
							   50.0,
							   (frame_w, frame_h))

		for i in tqdm(range(nb_frames)):
			_, image = video_reader.read()

			boxes = yolo.predict(image)
			image = draw_boxes(image, boxes, config['model']['labels'])

			video_writer.write(np.uint8(image))

		video_reader.release()
		video_writer.release()
	else:
		for root, dirs, files in os.walk(image_path):
			for f in files:
				ext = os.path.splitext(f)[-1]
				if ext not in ('.jpg', '.jpeg', '.png', '.bmp'):
					continue
				name = os.path.join(root, f)
				image = cv2.imread(name)

				image = cv2.resize(image, (500, 500))

				boxes = yolo.predict(image)
				# if len(boxes) > 0:
				# 	print (f)
				# 	# print ([box.classes for box in boxes])
				# 	# for box in boxes:
				# 	#     print (box.classes)
				# 	print ([config['model']['labels'][box.get_label()] for box in boxes] )
				image = draw_boxes(image, boxes, config['model']['labels'])
				cv2.imwrite(os.path.join(output_path, f), image)

		# image = cv2.imread(image_path)
		# boxes = yolo.predict(image)
		# image = draw_boxes(image, boxes, config['model']['labels'])
		# cv2.imwrite(output_path, image)

