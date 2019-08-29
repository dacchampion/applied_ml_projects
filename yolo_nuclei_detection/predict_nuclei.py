#! /usr/bin/env python

import argparse
import json
import os

import cv2

from frontend import YOLO
from utils import draw_boxes

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

argparser = argparse.ArgumentParser(
    description='Infer the localization of nucleus using YOLO_v2 model')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to directory of nuclei images')

argparser.add_argument(
    '-d',
    '--draw',
    help='Flag to indicate if draw the predictions or not (0 or 1)')


def _main_(args):
    config_path = args.conf
    weights_path = args.weights
    nuclei_imgs_path = args.input
    draw_boxes_flg = bool(int(args.draw))

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

    print(weights_path)
    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes
    ###############################
    total_boxes = 0
    detections = {'image': []}
    for image_directory in sorted(os.listdir(nuclei_imgs_path)):
        if image_directory == '.DS_Store':
            continue

        image_dir = "{}/{}/{}".format(nuclei_imgs_path, image_directory, 'images')
        image_file_name = os.listdir(image_dir)[1] if os.listdir(image_dir)[0] == '.DS_Store' else \
            os.listdir(image_dir)[0]
        image_full_path = "{}/{}".format(image_dir, image_file_name)
        image = cv2.imread(image_full_path)
        img_h = image.shape[1]
        img_w = image.shape[0]
        image_vars = {'path': "{}/{}/{}".format(image_directory, 'images', image_file_name), 'height': img_h, 'width': img_w, 'bbox': []}
        boxes = yolo.predict(image)
        for box in boxes:
            xmin = int((box.x - box.w / 2) * img_h)
            xmax = int((box.x + box.w / 2) * img_h)
            ymin = int((box.y - box.h / 2) * img_w)
            ymax = int((box.y + box.h / 2) * img_w)
            image_vars['bbox'].append(
                {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'score': str(box.get_score())})

        detections['image'].append(image_vars)
        total_boxes += len(boxes)
        print("{} boxes are found on image with id {}".format(len(boxes), image_directory))
        if draw_boxes_flg:
            image = draw_boxes(image, boxes, config['model']['labels'])
            cv2.imwrite(image_full_path[:-4] + '_detected' + image_full_path[-4:], image)

    print("Total number of boxes detected: {}".format(total_boxes))
    with open('nuclei_detections.json', 'w') as outfile:
        json.dump(detections, outfile)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
