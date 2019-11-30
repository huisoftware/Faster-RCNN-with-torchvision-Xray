# -*-coding:utf-8-*-
import torch
import torchvision
import argparse
import cv2
import numpy as np
import sys

sys.path.append('./')
import coco_names
import random


def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Detection')

    parser.add_argument('--image_path',default="D:\sysfile\desktop\mlbighomework\output/test\Image\coreless_battery00000001.jpg", type=str, help='image path')
    parser.add_argument('--annotation_path',default="D:\sysfile\desktop\mlbighomework\output/test\Annotation\coreless_battery00000001.txt", type=str, help='annotation path')

    args = parser.parse_args()

    return args


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return (b, g, r)


def main():
    args = get_args()
    print(args.image_path)
    src_img = cv2.imread(args.image_path)
    boxes = []
    names = []
    with open(args.annotation_path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            linPireList = line.split(" ")
            name = linPireList[1]
            names.append(name)
            x = linPireList[2]
            y = linPireList[3]
            w = linPireList[4]
            h = linPireList[5]
            boxe = [int(x),int(y),int(w),int(h)]
            boxes.append(boxe)

    k = 0
    for boxe in boxes:
        k = k +1
        x1, y1, x2, y2 = boxe[0], boxe[1], boxe[2], boxe[3]
        name = names[k-1]
        if "不" in name:
            name = 'coreless'
        else:
            name = 'core'
        print(x1, y1, x2, y2)
        cv2.rectangle(src_img, (x1, y1), (x2, y2), random_color(), thickness=2)
        #cv2.rectangle(src_img, (1, 1), (111, 111), (0, 255, 0), 1)

        cv2.putText(src_img, text=name, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

    cv2.imshow('result', src_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
