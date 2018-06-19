import cv2
import random
import colorsys
import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def filter_boxes(yolo_output, obj_threshold, anchors):

    grid_h, grid_w, nb_box = yolo_output.shape[:3]

    # score = softmax(confidence score) * sigmoid(class probabilities)
    yolo_output[..., 5:] = softmax(yolo_output[..., 5:]) * sigmoid(yolo_output[..., 4][..., np.newaxis])

    # set score to 0, if score < obj_threshold
    yolo_output[..., 5:] *= yolo_output[..., 5:] > obj_threshold

    boxes = []

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = yolo_output[row, col, b, 5:]

                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = yolo_output[row, col, b, :4]

                    x = (col + sigmoid(x)) / grid_w  # center position, unit: image width
                    y = (row + sigmoid(y)) / grid_h  # center position, unit: image height
                    w = anchors[b, 0] * np.exp(w) / grid_w  # unit: image width
                    h = anchors[b, 1] * np.exp(h) / grid_h  # unit: image height
                    label = np.argmax(classes)
                    score = np.max(classes)

                    box = {'x':x, 'y':y, 'w':w, 'h':h, 'classes':classes, 'label':label, 'score':score, 'iou':True}

                    boxes.append(box)

    return boxes


# Count dimensions overlap of two boxes
def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


# Count intersection over union
def iou(box1, box2):
    x1_min = box1['x'] - box1['w'] / 2
    x1_max = box1['x'] + box1['w'] / 2
    y1_min = box1['y'] - box1['h'] / 2
    y1_max = box1['y'] + box1['h'] / 2

    x2_min = box2['x'] - box2['w'] / 2
    x2_max = box2['x'] + box2['w'] / 2
    y2_min = box2['y'] - box2['h'] / 2
    y2_max = box2['y'] + box2['h'] / 2

    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])

    intersect = intersect_w * intersect_h

    union = box1['w'] * box1['h'] + box2['w'] * box2['h'] - intersect

    return float(intersect) / union


# Apply non-max suppression and filter boxes again
def non_max_suppress(boxes, nms_threshold, nb_class):
    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box['classes'][c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i]['classes'][c] == 0:
                continue
            else:
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if iou(boxes[index_i], boxes[index_j]) > nms_threshold:
                        boxes[index_j]['iou'] = False

    # remove the boxes which have too high iou
    boxes = [box for box in boxes if box['iou'] is True]
    return boxes


# Generate list of colours for drowing boxes
def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors


# Draw boxes on image
def draw_boxes(image, boxes, labels, colours):
    for box in boxes:
        xmin = int((box['x'] - box['w'] / 2) * image.shape[1])
        xmax = int((box['x'] + box['w'] / 2) * image.shape[1])
        ymin = int((box['y'] - box['h'] / 2) * image.shape[0])
        ymax = int((box['y'] + box['h'] / 2) * image.shape[0])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colours[box['label']], 2)
        cv2.putText(image,
                    labels[box['label']] + ' ' + str(box['score'].round(2)),
                    (xmin, ymin - 13),
                    cv2.FONT_HERSHEY_SIMPLEX ,
                    2e-3 * image.shape[0],
                    colours[box['label']], 2)

    return image


def postprocess(data, anchors, labels, original_image, nclass=80):
    data = np.squeeze(data)
    boxes = filter_boxes(data, 0.5, anchors)
    boxes = non_max_suppress(boxes, 0.3, nclass)
    colours = generate_colors(labels)
    output_image = draw_boxes(original_image, boxes, labels, colours)
    return output_image, len(boxes)

def getboxes(data, anchors, nclass=80):
    data = np.squeeze(data)
    boxes = filter_boxes(data, 0.5, anchors)
    boxes = non_max_suppress(boxes, 0.3, nclass)
    return boxes