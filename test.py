import torch
import torchvision
import argparse
import cv2
import numpy as np
import sys
import os
sys.path.append('./')
import random
#--model_path F:\Masters2019\lh\dl\objectdetection\faster_r_cnn\transdata\output\myout\model_19.pth
def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Detection')

    parser.add_argument('--model_path', type=str, default='.'+os.sep+'model_19.pth', help='model path')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--score', type=float, default=0.01, help='objectness score threshold')
    parser.add_argument('--gpu', type=str, default='0', help='gpu cuda')

    args = parser.parse_args()

    return args


def test(img_path, anno_path):
    args = get_args()

    names = {'0': 'background', '1': 'core', '2': 'coreless'}

    # 创建输出文件
    core_file = open('det_test_带电芯充电宝.txt','w')
    coreless_file = open('det_test_不带电芯充电宝.txt','w')

    # Model creating
    print("Creating model")
    model = torchvision.models.detection.__dict__[args.model](num_classes=3, pretrained=False)
    if args.gpu =='0':
        model = model.cpu()
    else:
        model = model.cuda()

    model.eval()

    # 获取要测试的图像文件夹下所有图像文件的文件名
    img_list = os.listdir(img_path)
    # 遍历所有图像
    for img_file in img_list:
        print(img_file)

        src_img = cv2.imread(os.path.join(img_path,img_file))
        img = cv2.cvtColor(src_img,cv2.COLOR_BGR2RGB)

        if args.gpu == '0':
            img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
        else:
            img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()
        out = model([img_tensor])
        boxes = out[0]['boxes']
        labels = out[0]['labels']
        scores = out[0]['scores']

        for idx in range(boxes.shape[0]):
            if scores[idx] > args.score:
                x1, y1, x2, y2 = boxes[idx][0].detach().numpy(), boxes[idx][1].detach().numpy(), boxes[idx][2].detach().numpy(), boxes[idx][3].detach().numpy()
                name = names.get(str(labels[idx].item()))


                # 按照要求的格式，将结果写入到相应的输出文件中
                # 文件名不带后缀，置信度保留小数点后3位，坐标保留一位小数点
                # 注2： Windows下可能需要修改成 \n\r作为换行符！！！
                str_write = "%s %.3f %.1f %.1f %.1f %.1f\n" % (img_file[:-4], scores[idx].detach().numpy(), x1, y1, x2, y2)

                if name == 'core':
                    core_file.write(str_write)
                elif name == 'coreless':
                    coreless_file.write(str_write)
    
    core_file.close()
    coreless_file.close()


if __name__ == '__main__':
    test('F:\\Masters2019\\lh\\dl\\objectdetection\\faster_r_cnn\\transdata\\output\\test\\Image\\', None)