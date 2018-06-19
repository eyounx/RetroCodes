import os
import cv2
import numpy as np
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom


class Target(object):

    def __init__(self):
        self.class_name = None
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

    def init(self, class_name, xmin, xmax, ymin, ymax):
        self.class_name, self.xmin, self.xmax, self.ymin, self.ymax = class_name, xmin, xmax, ymin, ymax


class Annotation(object):

    def __init__(self):
        self.folder = None
        self.filename = None
        self.image_width = None
        self.image_height = None
        self.image_depth = None
        self.targets = []



def pretty_print(x):
    lines = minidom.parseString(x).toprettyxml(indent=' ' * 2).split('\n')
    return '\n'.join([line for line in lines if line.strip()])


def write_xml_annot(dest_dir, xml_params):
    # Write the xml annotation to file
    xml_annot = Element('annotation')

    folder = SubElement(xml_annot, 'folder')
    folder.text = xml_params.folder

    filename = SubElement(xml_annot, 'filename')
    filename.text = xml_params.filename

    size = SubElement(xml_annot, 'size')

    width = SubElement(size, 'width')
    width.text = xml_params.image_width
    height = SubElement(size, 'height')
    height.text = xml_params.image_height
    depth = SubElement(size, 'depth')
    depth.text = xml_params.image_depth

    for target in xml_params.targets:
        object_ = SubElement(xml_annot, 'object')
        class_name = SubElement(object_, 'name')
        class_name.text = target.class_name
        bndbox = SubElement(object_, 'bndbox')
        xmin = SubElement(bndbox, 'xmin')
        xmin.text = target.xmin
        ymin = SubElement(bndbox, 'ymin')
        ymin.text = target.ymin
        xmax = SubElement(bndbox, 'xmax')
        xmax.text = target.xmax
        ymax = SubElement(bndbox, 'ymax')
        ymax.text = target.ymax

    xml_str = pretty_print(tostring(xml_annot))
    basename = os.path.splitext(xml_params.filename)[0]
    with open(os.path.join(dest_dir, basename + '.xml'), 'w') as out:
        out.write(xml_str)


def convert(image_dir, mask_dir, anno_dir):
    cls_dict = {5: 'kilt', 6: 'trousers', 7: 'dress', 16: 'bag'}
    cls_counter = {}
    anno = Annotation()
    anno.folder = 'humanparsing'

    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)

    for root, dirs, files in os.walk(mask_dir):
        for f in files:
            mask = cv2.imread(os.path.join(root, f))
            height, width, depth = mask.shape
            basename = os.path.splitext(f)[0]
            f = basename + '.jpg'
            if not os.path.exists(os.path.join(image_dir, f)):
                raise Exception('image not exist:{}'.format(os.path.join(image_dir, f)))
            anno.filename = f
            anno.image_width = str(width)
            anno.image_height = str(height)
            anno.image_depth = str(depth)

            targets = []
            for cls in cls_dict:
                idx = np.where(mask[:, :, 0] == cls)
                if len(idx[0]) == 0:
                    continue
                t = Target()
                t.class_name = cls_dict[cls]
                t.xmin = str(np.min(idx[1]))
                t.xmax = str(np.max(idx[1]))
                t.ymin = str(np.min(idx[0]))
                t.ymax = str(np.max(idx[0]))
                targets.append(t)

                cnt = cls_counter.get(cls, 0)
                cnt += 1
                cls_counter[cls] = cnt

            if len(targets) > 0:
                anno.targets = targets
                write_xml_annot(anno_dir, anno)

    print('total class statistic:')
    for cls in cls_counter:
        cls_name = cls_dict[cls]
        print('{:<10} = {:<10}'.format(cls_name, cls_counter[cls]))


if __name__ == '__main__':
    image_dir= '../../../data/Game0_Bee/'
    mask_dir = '../../../data/Game0_Bee_an/'
    anno_dir = '../../../data/Annotations/'
    convert(image_dir, mask_dir, anno_dir)




















