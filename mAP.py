import torch
import os
import argparse
import cv2
import numpy as np
import pickle
import torchvision
from torchvision import transforms
from train import get_dataset,get_transform
import shutil
import time
from PIL import Image
import gc
loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage()

names = {'0': 'background', '1': 'core', '2': 'coreless'}
labelmap = ['core', 'coreless']

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn mAP')

    parser.add_argument('--model_path', type=str, default='./result/model_19.pth', help='model path')
    parser.add_argument('--data_path', type=str, default='./test.jpg', help='image path')
    parser.add_argument('--cache_path', type=str, default='./test.jpg', help='image path')
    parser.add_argument('--out_path', type=str, default='./test.jpg', help='image path')

    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    #parser.add_argument('--score', type=float, default=0.8, help='objectness score threshold')
    parser.add_argument('--gpu', type=str, default='0', help='gpu cuda')

    args = parser.parse_args()

    return args


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename, imgpath):
    """ Parse a PASCAL VOC xml file """
    # tree = ET.parse(filename)
    # filename = filename[:-3] + 'txt'
    if filename=='core_battery00000191':
        print(filename)
    filename1 = filename
    filename = filename+".txt"
    # filename = filename.replace('.xml', '.txt')
    #
    # # imagename0 = filename.replace('Anno_core_coreless_battery_sub_2000_500', 'cut_Image_core_coreless_battery_sub_2000_500')
    #
    # img_fold_name = imgpath.split('/')[-2]
    # imagename0 = filename.replace('Anno_core_coreless_battery_sub_2000_500', 'img_fold_name')
    #
    #
    # imagename1 = imagename0.replace('.txt', '.jpg')  # jpg form
    # imagename2 = imagename0.replace('.txt', '.jpg')
    imagename1 = imgpath+os.sep+filename1+'.jpg'

    objects = []
    img = cv2.imread(imagename1)
    height, width, channels = img.shape
    fpath = annopath + os.sep + filename
    with open(fpath, "r", encoding='utf-8') as f1:
        dataread = f1.readlines()
        for annotation in dataread:
            obj_struct = {}
            temp = annotation.split()
            name = temp[1]
            if name != '带电芯充电宝' and name != '不带电芯充电宝':
                continue
            if name == '带电芯充电宝':
                name = 'core'
            if name == '不带电芯充电宝':
                name = 'coreless'
            xmin = int(temp[2])
            # 只读取V视角的
            if int(xmin) > width:
                continue
            if xmin < 0:
                xmin = 1
            ymin = int(temp[3])
            if ymin < 0:
                ymin = 1
            xmax = int(temp[4])
            if xmax > width:
                xmax = width - 1
            ymax = int(temp[5])
            if ymax > height:
                ymax = height - 1
            ##name
            obj_struct['name'] = name
            obj_struct['pose'] = 'Unspecified'
            obj_struct['truncated'] = 0
            obj_struct['difficult'] = 0
            obj_struct['bbox'] = [float(xmin) - 1,
                          float(ymin) - 1,
                          float(xmax) - 1,
                          float(ymax) - 1]
            objects.append(obj_struct)



    '''
    for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text.lower().strip()
    obj_struct['pose'] = obj.find('pose').text
    obj_struct['truncated'] = int(obj.find('truncated').text)
    obj_struct['difficult'] = int(obj.find('difficult').text)
    bbox = obj.find('bndbox')
    obj_struct['bbox'] = [float(bbox.find('xmin').text) - 1,
                          float(bbox.find('ymin').text) - 1,
                          float(bbox.find('xmax').text) - 1,
                          float(bbox.find('ymax').text) - 1]
    objects.append(obj_struct)
    '''
    return objects



def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        # print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                name = dataset.coco.dataset["images"][im_ind]["file_name"][:-4]
                dets = all_boxes[cls_ind + 1][im_ind]
                if dets == []:
                    continue
                dets = np.array(dets)
                # the VOCdevkit expects a1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(name, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir='output', use_07=False):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    # print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(
            filename, annopath, imgpath, imagesetfile, cls, cachedir,
            ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print("mAP, {:.4f}, core_AP, {:.4f}, coreless_AP, {:.4f}".format(np.mean(aps), aps[0], aps[1]))


#rec:召回率 prec:准确率；召回率越高，准确率越低
def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    # Y轴查准率p,X轴召回率r,取11个点,如[r(0.0),p(0)],[r(0.1),p(1)],...,[r(1.0),p(10)],ap=(p(0)+p(1)+...+p(10))/11
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            # 召回率rec中大于阈值t的数量;等于0表示超过了最大召回率,对应的p设置为0
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                # 召回率大于t时精度的最大值 ???
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        # 计算PR曲线向下包围的面积
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imgpath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images

    files = os.listdir(imagesetfile)

    imagenames = []
    for file_name in files:  # 循环读取每个文件名
        imagenames.append(file_name[:-4])

    # with open(imagesetfile, 'r') as f:
    #     lines = f.readlines()
    # imagenames = [x.strip() for x in lines]

    '''
    imagenames = []
    listdir = os.listdir(osp.join('%s' % args.SIXray_root, 'Annotation'))
    for name in listdir:
        imagenames.append(osp.splitext(name)[0])
    '''

    if not os.path.isfile(cachefile):
        # print('not os.path.isfile')
        # load annots
        recs = {}
        for imagename in imagenames:
            recs[imagename] = parse_rec(imagename, imgpath)
            '''
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
            '''
        # save
        # print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # print('no,no,no')
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # print (recs)
    # print (classname)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for i, imagename in enumerate(imagenames):
        R = [obj for obj in recs[imagename] if obj['name'] == classname]

        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # print (class_recs)

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        # np.argsort()排序默认从小到大,所以这里将置信度取负
        sorted_ind = np.argsort(-confidence)
        # 按照置信度排序,置信度高的排在前面;
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        # 检测结果保存时,每行一个bbox,所以一张图像多个bbox的情况就被分成了多行;这里image_ids中存在多行同为一个图像的情况
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            # R表示当前帧图像上所有的GT bbox的信息
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                # bb表示测试集中某一个检测出来的框的四个坐标，BBGT表示和bb同一图像上的所有检测框，取其中IOU最大的作为检测框的ground-true
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:# 判断是否被检测过(如果之前有置信度更高的bbox匹配上了这个BBGT,那么就表示检测过了)
                        tp[d] = 1.#预测为正，实际为正
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.#预测为正，实际为负
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        # 召回率
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        # 精准率,查准率
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

def test_net(data_path, cache_path, out_path, model, gpu, dataset, im_size=300, thresh=0.05):
    # //
    # //
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap) + 1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = out_path
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        img, target = dataset[i]

        #im_det = tensor_to_PIL(img)
        #im_gt = tensor_to_PIL(img)

        _t['im_detect'].tic()

        input = []
        input.append(img)
        gc.collect()
        out = model(input)
        detections = out[0]

        print(i)

        boxes = out[0]['boxes']
        labels = out[0]['labels']
        scores = out[0]['scores']

        detect_time = _t['im_detect'].toc(average=False)

        try:
            x1, y1, x2, y2 = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]
            cls_dets1 = [x1, y1, x2, y2, scores[0]]
            all_boxes[labels[0].item()][i].append(cls_dets1)
        except IndexError:
            x1, y1, x2, y2 = 0, 0, 0, 0
            cls_dets1 = [x1, y1, x2, y2, 0]
            all_boxes[1][i].append(cls_dets1)


        # for idx in range(boxes.shape[0]):
        #     x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
        #     cls_dets1 = [x1, y1, x2, y2, scores[idx]]
        #     all_boxes[labels[idx].item()][i].append(cls_dets1)

            # if not all_boxes[labels[idx].item()][i]:
            #     all_boxes[labels[idx].item()][i].append(cls_dets1)
            # else:
            #     cls_dets = []
            #     cls_dets.append(cls_dets1)
            #     all_boxes[labels[idx].item()][i] = cls_dets
                        #
        # for idx in range(boxes.shape[0]):
        #     if scores[idx] >= args.score:
        #         x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
        #         name = names.get(str(labels[idx].item()))
        #         # cv2.rectangle(img,(x1,y1),(x2,y2),colors[labels[idx].item()],thickness=2)
        #         cv2.rectangle(src_img, (x1, y1), (x2, y2), (0, 255, 255), thickness=2)
        #         cv2.putText(src_img, text=name, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #                     fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))

        # for j in range(1, detections.size(1)):
        #     dets = detections[0, j, :]
        #     mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
        #     dets = torch.masked_select(dets, mask).view(-1, 5)
        #     if dets.size(0) == 0:
        #         continue
        #     boxes = dets[:, 1:]
        #     boxes[:, 0] *= w
        #     boxes[:, 2] *= w
        #     boxes[:, 1] *= h
        #     boxes[:, 3] *= h
        #     scores = dets[:, 0].cpu().numpy()
        #     cls_dets = np.hstack((boxes.cpu().numpy(),
        #                           scores[:, np.newaxis])).astype(np.float32,
        #                                                          copy=False)
        #     cls_dets=[x1, y1, x2, y2,scores[idx]]
        #     all_boxes[j][i] = cls_dets

            # for item in cls_dets:
            #     if item[4] > thresh:
            #         chinese = labelmap[j - 1] + str(round(item[4], 2))
            #         if chinese[0] == '带':
            #             chinese = 'P_Battery_Core' + chinese[6:]
            #         else:
            #             chinese = 'P_Battery_No_Core' + chinese[7:]
            #         cv2.rectangle(im_det, (item[0], item[1]), (item[2], item[3]), (0, 0, 255), 2)
            #         cv2.putText(im_det, chinese, (int(item[0]), int(item[1]) - 5), 0,
            #                     0.6, (0, 0, 255), 2)
        # real = 0
        # if gt[0][4] == 3:
        #     real = 0
        # else:
        #     real = 1
        #
        # for item in gt:
        #     if real == 0:
        #         print('this pic dont have the obj:', dataset.ids[i])
        #         break
        #     chinese = labelmap[int(item[4])]
        #     # print(chinese+'gt\n\n')
        #     if chinese[0] == '带':
        #         chinese = 'P_Battery_Core'
        #     else:
        #         chinese = 'P_Battery_No_Core'
        #     cv2.rectangle(im_det, (int(item[0] * w), int(item[1] * h)), (int(item[2] * w), int(item[3] * h)),
        #                   (0, 255, 255), 2)
        #     cv2.putText(im_det, chinese, (int(item[0] * w), int(item[1] * h) - 5), 0, 0.6, (0, 255, 255), 2)
            # print(labelmap[int(item[4])])

        # print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))

        # cv2.imwrite('/media/trs2/wuzhangjie/SSD/eval/Xray20190723/Attention/base_battery_core_bs8_V/det_images/{0}_det.jpg'.format(dataset.ids[i]), im_det)

        # cv2.imwrite('/media/dsg3/shiyufeng/eval/Xray20190723/battery_2cV_version/20epoch_network/{0}_gt.jpg'.format(dataset.ids[i]), im_gt)
        # cv2.imwrite( '/media/dsg3/husheng/eval/{0}_det.jpg'.format(dataset.ids[i]), im_det)
        # cv2.imwrite( '/media/dsg3/husheng/eval/{0}_gt.jpg'.format(dataset.ids[i]), im_gt)


    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    # print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir)

#
# def reset_args(EPOCH):
#     global args
#     args.trained_model = "/media/trs2/wuzhangjie/SSD/weights/Xray20190723/2019-10-18_16-23-15Xray0723_bat_core_coreless_bs8_V_resume140/ssd300_Xray20190723_{:d}.pth".format(
#         EPOCH)
#     saver_root = '/media/trs2/wuzhangjie/SSD/eval/Xray20190723/Attention/base_battery_core_coreless_bs8_V/'
#     if not os.path.exists(saver_root):
#         os.mkdir(saver_root)
#     args.save_folder = saver_root + '{:d}epoch_500/'.format(EPOCH)
#
#     if not os.path.exists(args.save_folder):
#         os.mkdir(args.save_folder)
#     else:
#         shutil.rmtree(args.save_folder)
#         os.mkdir(args.save_folder)
#
#     global devkit_path
#     devkit_path = args.save_folder

# --model_path D:\sysfile\desktop\mlbighomework\model_19.pth --data_path D:\sysfile\desktop\mlbighomework\output --cache_path D:\sysfile\desktop\mlbighomework\cache --out_path D:\sysfile\desktop\mlbighomework\output2 --gpu 0
#
if __name__ == '__main__':
    args = get_args()

    global devkit_path
    devkit_path = args.out_path
    global set_type
    set_type = "myXray"
    global annopath
    annopath = args.data_path+os.sep+"test"+os.sep+"Annotation"
    global imgpath
    imgpath = args.data_path+os.sep+"test"+os.sep+"Image"
    global imagesetfile
    imagesetfile = args.data_path+os.sep+"test"+os.sep+"Image"

    do_python_eval(args.out_path, use_07=False)

    # # Model creating
    # print("Creating model")
    # model = torchvision.models.detection.__dict__[args.model](num_classes=len(names), pretrained=False)
    # if args.gpu == '0':
    #     model = model.cpu()
    # else:
    #     model = model.cuda()
    #
    # model.eval()
    # if args.gpu == '0':
    #     save = torch.load(args.model_path,map_location='cpu')
    # else:
    #     save = torch.load(args.model_path)
    #
    # model.load_state_dict(save['model'])
    #
    # dataset, _ = get_dataset("myselfXray", "test", get_transform(train=False), data_path=args.data_path)
    # with torch.no_grad():
    #     test_net(args.data_path, args.cache_path, args.out_path, model, args.gpu, dataset)


