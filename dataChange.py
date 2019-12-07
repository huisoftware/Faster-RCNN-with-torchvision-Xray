import os
import cv2
import time
import numpy as np
import argparse




import numpy as np
import random
import cv2
import glob
import os
import xml.etree.cElementTree as ET


def random_translate(img, bboxes, p=0.5):
    # 随机平移
    if random.random() < p:
        h_img, w_img, _ = img.shape
        # 得到可以包含所有bbox的最大bbox
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w_img - max_bbox[2]
        max_d_trans = h_img - max_bbox[3]

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (w_img, h_img))

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
    return img, bboxes


def random_crop(img, bboxes, p=0.5):
    # 随机裁剪
    if random.random() < p:
        h_img, w_img, _ = img.shape
        # 得到可以包含所有bbox的最大bbox
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w_img - max_bbox[2]
        max_d_trans = h_img - max_bbox[3]

        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))

        img = img[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
    return img, bboxes



def horizontal_flip(img, bboxes):
    _, w_img, _ = img.shape
    img = img[:, ::-1, :]
    bboxes[:, [0, 2]] = w_img - bboxes[:, [2, 0]]
    return img, bboxes

# 随机垂直反转
def vertical_flip(img, bboxes):
    h_img, _, _ = img.shape
    img = img[::-1, :, :]
    bboxes[:, [1, 3]] = h_img - bboxes[:, [3, 1]]
    return img, bboxes

# 缩小1/2
def resize_1_2(img, bboxes):

    height, width = img.shape[:2]
    # 缩小图片
    size = (int(width * 0.5), int(height * 0.5))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    return img, bboxes/2

def rot90_1(img, bboxes=None):
    # 顺时针旋转90度
    h, w, _ = img.shape
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 1)
    if bboxes is None:
        return new_img
    else:
        # bounding box 的变换: 一个图像的宽高是W,H, 如果顺时90度转换，那么原来的原点(0, 0)到了 (H, 0) 这个最右边的顶点了，
        # 设图像中任何一个转换前的点(x1, y1), 转换后，x1, y1是表示到 (H, 0)这个点的距离，所以我们只要转换回到(0, 0) 这个点的距离即可！
        # 所以+90度转换后的点为 (H-y1, x1), -90度转换后的点为(y1, W-x1)
        bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]
        bboxes[:, [0, 2]] = h - bboxes[:, [0, 2]]
        return new_img, bboxes



def rot90_2(img, bboxes=None):
    h, w, _ = img.shape
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 0)
    if bboxes is None:
        return new_img
    else:
        # bounding box 的变换: 一个图像的宽高是W,H, 如果顺时90度转换，那么原来的原点(0, 0)到了 (H, 0) 这个最右边的顶点了，
        # 设图像中任何一个转换前的点(x1, y1), 转换后，x1, y1是表示到 (H, 0)这个点的距离，所以我们只要转换回到(0, 0) 这个点的距离即可！
        # 所以+90度转换后的点为 (H-y1, x1), -90度转换后的点为(y1, W-x1)
        bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]
        bboxes[:, [1, 3]] = w - bboxes[:, [1, 3]]
        return new_img, bboxes


# 随机对比度和亮度 (概率：0.5)
def random_bright(img, bboxes, p=0.5, lower=0.5, upper=1.5):
    if random.random() < p:
        mean = np.mean(img)
        img = img - mean
        img = img * random.uniform(lower, upper) + mean * random.uniform(lower, upper)  # 亮度
        img = img / 255.
    return img, bboxes


# 随机变换通道
def random_swap(im, bboxes, p=0.5):
    perms = ((0, 1, 2), (0, 2, 1),
             (1, 0, 2), (1, 2, 0),
             (2, 0, 1), (2, 1, 0))
    if random.random() < p:
        swap = perms[random.randrange(0, len(perms))]
        print
        swap
        im[:, :, (0, 1, 2)] = im[:, :, swap]
    return im, bboxes


# 随机变换饱和度
def random_saturation(im, bboxes, p=0.5, lower=0.5, upper=1.5):
    if random.random() < p:
        im[:, :, 1] = im[:, :, 1] * random.uniform(lower, upper)
    return im, bboxes


# 随机变换色度(HSV空间下(-180, 180))
def random_hue(im, bboxes, p=0.5, delta=18.0):
    if random.random() < p:
        im[:, :, 0] = im[:, :, 0] + random.uniform(-delta, delta)
        im[:, :, 0][im[:, :, 0] > 360.0] = im[:, :, 0][im[:, :, 0] > 360.0] - 360.0
        im[:, :, 0][im[:, :, 0] < 0.0] = im[:, :, 0][im[:, :, 0] < 0.0] + 360.0
    return im, bboxes


# # 随机旋转0-90角度
# def random_rotate_image_func(image):
#     # 旋转角度范围
#     angle = np.random.uniform(low=0, high=90)
#     return misc.imrotate(image, angle, 'bicubic')


def random_rot(image, bboxes, angle, center=None, scale=1.0, ):
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    if bboxes is None:
        for i in range(image.shape[2]):
            image[:, :, i] = cv2.warpAffine(image[:, :, i], M, (w, h), flags=cv2.INTER_CUBIC,
                                            borderMode=cv2.BORDER_CONSTANT)
        return image
    else:
        box_x, box_y, box_label, box_tmp = [], [], [], []
        for box in bboxes:
            box_x.append(int(box[0]))
            box_x.append(int(box[2]))
            box_y.append(int(box[1]))
            box_y.append(int(box[3]))
            box_label.append(box[4])
        box_tmp.append(box_x)
        box_tmp.append(box_y)
        bboxes = np.array(box_tmp)
        ####make it as a 3x3 RT matrix
        full_M = np.row_stack((M, np.asarray([0, 0, 1])))
        img_rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

        ###make the bboxes as 3xN matrix
        full_bboxes = np.row_stack((bboxes, np.ones(shape=(1, bboxes.shape[1]))))
        bboxes_rotated = np.dot(full_M, full_bboxes)

        bboxes_rotated = bboxes_rotated[0:2, :]
        bboxes_rotated = bboxes_rotated.astype(np.int32)

        result = []
        for i in range(len(box_label)):
            x1, y1, x2, y2 = bboxes_rotated[0][2 * i], bboxes_rotated[1][2 * i], bboxes_rotated[0][2 * i + 1], \
                             bboxes_rotated[1][2 * i + 1]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
            x1, x2 = min(w, x1), min(w, x2)
            y1, y2 = min(h, y1), min(h, y2)
            one_box = [x1, y1, x2, y2, box_label[i]]
            result.append(one_box)
        return img_rotated, result


def readAnnotations(xml_path):
    et = ET.parse(xml_path)
    element = et.getroot()
    element_objs = element.findall('object')

    results = []
    for element_obj in element_objs:
        result = []
        class_name = element_obj.find('name').text

        obj_bbox = element_obj.find('bndbox')
        x1 = int(round(float(obj_bbox.find('xmin').text)))
        y1 = int(round(float(obj_bbox.find('ymin').text)))
        x2 = int(round(float(obj_bbox.find('xmax').text)))
        y2 = int(round(float(obj_bbox.find('ymax').text)))

        result.append(int(x1))
        result.append(int(y1))
        result.append(int(x2))
        result.append(int(y2))
        result.append(222)

        results.append(result)
    return results


def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn dataChange')
    parser.add_argument('--data_path', type=str, default='D:\sysfile\desktop\mlbighomework', help='data_path')
    parser.add_argument('--data_file', type=str, default='core_500', help='data_file')
    parser.add_argument('--type', type=str, default='0', help='type x 旋转 s 缩放 f 翻转')
    parser.add_argument('--param', type=str, default='0', help='旋转角度1=90 2=270 或 缩放倍数1=0.5 或翻转方向1水平 2垂直')
    args = parser.parse_args()

    return args
# --data_path D:\sysfile\desktop\mlbighomework\output\ --data_file test --type x --param 1
if __name__ == "__main__":
    args = get_args()
    image_path = args.data_path+os.sep+args.data_file+os.sep+'Image'+os.sep
    annotation_path = args.data_path+os.sep+args.data_file+os.sep+'Annotation'+os.sep

    changedatafile = args.data_file + "_" + args.type + "_" + args.param

    images = os.listdir(image_path)  # 采用listdir来读取所有文件
    for images_name in images:  # 循环读取每个文件名
        file_name = images_name[:-4]
        annotation_name = file_name+'.txt'
        oldImageAllPath = image_path+os.sep+images_name
        oldAnnotationAllPath = annotation_path+os.sep+annotation_name



        newImagePath = args.data_path+os.sep+changedatafile+os.sep+'Image'
        newAnnotationPath = args.data_path+os.sep+changedatafile+os.sep+'Annotation'
        if not os.path.exists(newImagePath):
            os.makedirs(newImagePath)
        if not os.path.exists(newAnnotationPath):
            os.makedirs(newAnnotationPath)
        newImageAllPath = newImagePath+os.sep+file_name + "_" + args.type + "_" + args.param+'.jpg'
        newAnnotationAllPath = newAnnotationPath+os.sep+file_name + "_" + args.type + "_" + args.param+'.txt'

        src_img = cv2.imread(oldImageAllPath)
        boxes = []
        names = []
        with open(oldAnnotationAllPath, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                linPireList = line.split(" ")
                name = linPireList[1]
                names.append(name)
                x = linPireList[2]
                y = linPireList[3]
                w = linPireList[4]
                h = linPireList[5]
                boxe = [int(x), int(y), int(w), int(h)]
                boxes.append(boxe)

        if args.type == 'x':
            # 旋转90
            if args.param == '1':
                img, boxes = rot90_1(src_img, np.array(boxes))

            # 旋转270
            elif args.param == '2':
                img, boxes = rot90_2(src_img, np.array(boxes))

            else:
                print("不支持的参数")
                break
        elif args.type == 's':
            # 缩放0.5
            img, boxes = resize_1_2(src_img, np.array(boxes))
        elif args.type == 'f':
            # 水平翻转
            if args.param == '1':
                img, boxes = horizontal_flip(src_img, np.array(boxes))

            # 垂直翻转
            elif args.param == '2':
                img, boxes = vertical_flip(src_img, np.array(boxes))
            else:
                print("不支持的参数")
            break
        else:
            print("不支持的参数")
            break
        cv2.imwrite(newImageAllPath,img)
        with open(oldAnnotationAllPath, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            i = 0
            newlines = []
            for line in lines:
                linPireList = line.split(" ")
                newline = linPireList[0]+" "+linPireList[1]+" "+str(boxes[i][0])+" "+str(boxes[i][1])+" "+str(boxes[i][2])+" "+str(boxes[i][3])+"\n"
                i = i+1
                newlines.append(newline)
            with open(newAnnotationAllPath, 'w', encoding='UTF-8') as f1:
                f1.writelines(newlines)



