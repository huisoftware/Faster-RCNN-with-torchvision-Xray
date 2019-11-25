import torch
import os
import argparse
import cv2
import numpy as np
import pickle

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

    filename = filename.replace('.xml', '.txt')

    # imagename0 = filename.replace('Anno_core_coreless_battery_sub_2000_500', 'cut_Image_core_coreless_battery_sub_2000_500')

    img_fold_name = imgpath.split('/')[-2]
    imagename0 = filename.replace('Anno_core_coreless_battery_sub_2000_500', 'img_fold_name')


imagename1 = imagename0.replace('.txt', '.jpg')  # jpg form
imagename2 = imagename0.replace('.txt', '.jpg')
objects = []
# 还需要同时打开图像，读入图像大小
img = cv2.imread(imagename1)
if img is None:
    img = cv2.imread(imagename2)
height, width, channels = img.shape
with open(filename, "r", encoding='utf-8') as f1:
    dataread = f1.readlines()
    for annotation in dataread:
        obj_struct = {}
        temp = annotation.split()
        name = temp[1]
        if name != '带电芯充电宝' and name != '不带电芯充电宝':
            continue
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
                dets = all_boxes[cls_ind + 1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects a1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
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
            filename, annopath, imgpath, args.imagesetfile, cls, cachedir,
            ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print("mAP, {:.4f}, core_AP, {:.4f}, coreless_AP, {:.4f}".format(np.mean(aps), aps[0], aps[1]))


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
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
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

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
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename), imgpath)
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
    for imagename in imagenames:
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
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
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
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap

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
        im, gt, h, w, og_im = dataset.pull_item(i)

        im_det = og_im.copy()
        im_gt = og_im.copy()

        _t['im_detect'].tic()

        input = []
        if args.gpu == '0':
            img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
        else:
            img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()
        input.append(img_tensor)
        out = model(input)
        detections = out[0]

        detect_time = _t['im_detect'].toc(average=False)

        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            # print(boxes)
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets

            # print(all_boxes)
            for item in cls_dets:
                # print(item)
                # print(item[5])
                if item[4] > thresh:
                    # print(item)
                    chinese = labelmap[j - 1] + str(round(item[4], 2))
                    # print(chinese+'det\n\n')
                    if chinese[0] == '带':
                        chinese = 'P_Battery_Core' + chinese[6:]
                    else:
                        chinese = 'P_Battery_No_Core' + chinese[7:]
                    cv2.rectangle(im_det, (item[0], item[1]), (item[2], item[3]), (0, 0, 255), 2)
                    cv2.putText(im_det, chinese, (int(item[0]), int(item[1]) - 5), 0,
                                0.6, (0, 0, 255), 2)
        real = 0
        if gt[0][4] == 3:
            real = 0
        else:
            real = 1

        for item in gt:
            if real == 0:
                print('this pic dont have the obj:', dataset.ids[i])
                break
            chinese = labelmap[int(item[4])]
            # print(chinese+'gt\n\n')
            if chinese[0] == '带':
                chinese = 'P_Battery_Core'
            else:
                chinese = 'P_Battery_No_Core'
            cv2.rectangle(im_det, (int(item[0] * w), int(item[1] * h)), (int(item[2] * w), int(item[3] * h)),
                          (0, 255, 255), 2)
            cv2.putText(im_det, chinese, (int(item[0] * w), int(item[1] * h) - 5), 0, 0.6, (0, 255, 255), 2)
            # print(labelmap[int(item[4])])

        # print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))

        # cv2.imwrite('/media/trs2/wuzhangjie/SSD/eval/Xray20190723/Attention/base_battery_core_bs8_V/det_images/{0}_det.jpg'.format(dataset.ids[i]), im_det)

        # cv2.imwrite('/media/dsg3/shiyufeng/eval/Xray20190723/battery_2cV_version/20epoch_network/{0}_gt.jpg'.format(dataset.ids[i]), im_gt)
        # cv2.imwrite( '/media/dsg3/husheng/eval/{0}_det.jpg'.format(dataset.ids[i]), im_det)
        # cv2.imwrite( '/media/dsg3/husheng/eval/{0}_gt.jpg'.format(dataset.ids[i]), im_gt)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    # print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir)


def reset_args(EPOCH):
    global args
    args.trained_model = "/media/trs2/wuzhangjie/SSD/weights/Xray20190723/2019-10-18_16-23-15Xray0723_bat_core_coreless_bs8_V_resume140/ssd300_Xray20190723_{:d}.pth".format(
        EPOCH)
    saver_root = '/media/trs2/wuzhangjie/SSD/eval/Xray20190723/Attention/base_battery_core_coreless_bs8_V/'
    if not os.path.exists(saver_root):
        os.mkdir(saver_root)
    args.save_folder = saver_root + '{:d}epoch_500/'.format(EPOCH)

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    else:
        shutil.rmtree(args.save_folder)
        os.mkdir(args.save_folder)

    global devkit_path
    devkit_path = args.save_folder


if __name__ == '__main__':
    args = get_args()
    input = []


    # Model creating
    print("Creating model")
    model = torchvision.models.detection.__dict__[args.model](num_classes=len(names), pretrained=False)
    if args.gpu == '0':
        model = model.cpu()
    else:
        model = model.cuda()

    model.eval()

    save = torch.load(args.model_path)
    model.load_state_dict(save['model'])

    dataset_test, _ = get_dataset("myselfXray", "test", get_transform(train=False), data_path=args.data_path)


    test_net(args.data_path, args.cache_path, args.out_path, model, args.gpu, dataset)
