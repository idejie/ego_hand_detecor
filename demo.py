# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dis import dis
import imageio
import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image
from model.utils.viz_hand_obj import *
import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
# from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, vis_detections_filtered_objects_PIL, vis_detections_filtered_objects # (1) here add a function to viz
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
from PIL import Image, ImageDraw, ImageFont
import tqdm
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/res101.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="models")
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="images")
  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save results',
                      default="images_det")
  parser.add_argument('--cuda', dest='cuda', 
                      help='whether use CUDA',
                      default=True, type=bool)
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=8, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=132028, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      default=True)
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)
  parser.add_argument('--thresh_hand',
                      type=float, default=0.5,
                      required=False)
  parser.add_argument('--thresh_obj', default=0.5,
                      type=float,
                      required=False)
  parser.add_argument('--p',type=str,required=True)
  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def create_gif(image_list, gif_name):
 
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    # Save them as frames into a gif 
    imageio.mimsave(gif_name, frames, 'GIF', duration = 0.1)



def calculate_center(bb):
    return [(bb[0] + bb[2])/2, (bb[1] + bb[3])/2]
def filter_object(obj_dets, hand_dets):
    filtered_object = []
    object_cc_list = []
    for j in range(obj_dets.shape[0]):
        object_cc_list.append(calculate_center(obj_dets[j,:4]))
    object_cc_list = np.array(object_cc_list)
    img_obj_id = []

    for i in range(hand_dets.shape[0]):
        # if hand_dets[i, 5] <= 0:
        #     img_obj_id.append(-1)
        #     continue
        hand_cc = np.array(calculate_center(hand_dets[i,:4]))
        point_cc = np.array([(hand_cc[0]+hand_dets[i,6]*10000*hand_dets[i,7]), (hand_cc[1]+hand_dets[i,6]*10000*hand_dets[i,8])])
        dist = np.sum((object_cc_list - point_cc)**2,axis=1)
        dist_min = np.argsort(dist)
        c=0
        indx = dist_min[c]
        
        while indx in img_obj_id and c<len(dist_min)-1:
          c+=1
          indx = dist_min[c]
        img_obj_id.append(indx)
    
    return img_obj_id

def vis_detections_PIL(im, class_name, dets, thresh=0.8, font_path='lib/model/utils/times_b.ttf'):
    """Visual debugging of detections."""
    
    image = Image.fromarray(im).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, size=30)
    width, height = image.size
    
    for hand_idx, i in enumerate(range(np.minimum(10, dets.shape[0]))):
        bbox = list(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, 4]
        lr = dets[i, -1]
        state = dets[i, 5]
        if score > thresh:
            image = draw_hand_mask(image, draw, hand_idx, bbox, score, lr, state, width, height, font)
            
    return image

def vis_detections_filtered_objects_PIL(im, obj_dets, hand_dets, thresh_hand=0.8, thresh_obj=0.01, font_path='lib/model/utils/times_b.ttf'):

    # convert to PIL
    im = im[:,:,::-1]
    image = Image.fromarray(im).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, size=30)
    width, height = image.size 

    if (obj_dets is not None) and (hand_dets is not None):
        
        for obj_idx, i in enumerate(range(np.minimum(10, obj_dets.shape[0]))):
            bbox = list(int(np.round(x)) for x in obj_dets[i, :4])
            score = obj_dets[i, 4]
            image = draw_obj_mask(image, draw, obj_idx, bbox, score, width, height, font)

        for hand_idx, i in enumerate(range(np.minimum(10, hand_dets.shape[0]))):
            bbox = list(int(np.round(x)) for x in hand_dets[i, :4])
            score = hand_dets[i, 4]
            lr = hand_dets[i, -1]
            state = hand_dets[i, 5]
            if True:
                # viz hand by PIL
                image = draw_hand_mask(image, draw, hand_idx, bbox, score, lr, state, width, height, font)

                # if state > 0: # in contact hand

                obj_cc, hand_cc =  calculate_center(obj_dets[i,:4]), calculate_center(bbox)
                # viz line by PIL
                if lr == 0:
                    side_idx = 0
                elif lr == 1:
                    side_idx = 1
                draw_line_point(draw, side_idx, (int(hand_cc[0]), int(hand_cc[1])), (int(obj_cc[0]), int(obj_cc[1])))


        

    elif hand_dets is not None:
        image = vis_detections_PIL(im, 'hand', hand_dets, thresh_hand, font_path)
        
    return image


side_map = {'l':'Left', 'r':'Right'}
side_map2 = {0:'Left', 1:'Right'}
side_map3 = {0:'L', 1:'R'}
state_map = {0:'No Contact', 1:'Self Contact', 2:'Another Person', 3:'Portable Object', 4:'Stationary Object'}
state_map2 = {0:'N', 1:'S', 2:'O', 3:'P', 4:'F'}


if __name__ == '__main__':

  args = parse_args()
  prefix = '/S5/MIPL/yangdj/ego4d_data/v1/frames/'

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  cfg.USE_GPU_NMS = args.cuda
  np.random.seed(cfg.RNG_SEED)

  # load model
  model_dir = args.load_dir + "/" + args.net + "_handobj_100K" + "/" + args.dataset
  if not os.path.exists(model_dir):
    raise Exception('There is no input directory for loading network from ' + model_dir)
  load_name = os.path.join(model_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) 
  args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]'] 

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  if args.cuda > 0:
    checkpoint = torch.load(load_name)
  else:
    checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']

  print('load model successfully!')


  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)
  box_info = torch.FloatTensor(1) 

  # ship to cuda
  if args.cuda > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  with open('all_frames_final.txt') as f:
      todo_frames = json.load(f)
  index = args.p
  s,e = index.split(':')
  s,e = int(s),int(e)
  # part = 3
  todo_list= list(todo_frames.items())

  # s = part*s
  

  # t = part*e
  print('starting ... ',s,e)
  todo_list = todo_list[s:e]

    # imglist = os.listdir(args.image_dir)
    # num_images = len(imglist)

  print('Loaded Photo: {} video.'.format(len(todo_list)))

  
  with torch.no_grad():
    if args.cuda > 0:
      cfg.CUDA = True

    if args.cuda > 0:
      fasterRCNN.cuda()

    fasterRCNN.eval()

    # start = time.time()
    # max_per_image = 100
    thresh_hand = args.thresh_hand 
    thresh_obj = args.thresh_obj
    vis = args.vis


    webcam_num = args.webcam_num
 

    c = -1
    for video_name, frames in todo_list:
      c+=1
      pre_dir = os.path.join(prefix,video_name)
      npz_d = os.path.join('/S5/MIPL/yangdj/ego4d_data/hand_object_new/', video_name)
      if not os.path.exists(npz_d):
        os.makedirs(npz_d, exist_ok=True)
      for frame in tqdm.tqdm(frames,desc=f'{s+c}/{e} |  {video_name}'):
        if os.path.exists(os.path.join(npz_d, f'{frame:010d}.npz')):
          continue
        im_file = os.path.join(pre_dir, f'{frame:010d}.jpg')

        if not os.path.exists(im_file):
          with open('failed.txt','a') as f:
            f.write(f'{video_name}/{frame:010d}.jpg\n')
          print(f'{im_file} not exists',video_name,frame)
          continue
        im_in = cv2.imread(im_file)
        # bgr
        im = im_in

        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        with torch.no_grad():
          im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
          im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
          gt_boxes.resize_(1, 1, 5).zero_()
          num_boxes.resize_(1).zero_()
          box_info.resize_(1, 1, 5).zero_() 

        

          pooled_feat,rois, cls_prob, bbox_pred, \
          rpn_loss_cls, rpn_loss_box, \
          RCNN_loss_cls, RCNN_loss_bbox, \
          rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info) 

          scores = cls_prob.data
          boxes = rois.data[:, :, 1:5]

          # extact predicted params
          contact_vector = loss_list[0][0] # hand contact state info
          offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
          lr_vector = loss_list[2][0].detach() # hand side info (left/right)

          # get hand contact 
          _, contact_indices = torch.max(contact_vector, 2)
          contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

          # get hand side 
          lr = torch.sigmoid(lr_vector) >= 0.5
          lr = lr.squeeze(0).float()
          if cfg.TEST.BBOX_REG:
              # Apply bounding-box regression deltas
              box_deltas = bbox_pred.data
              if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
              # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                  + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                  + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                  + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                  + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

              pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
              pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
          else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

          pred_boxes /= im_scales[0]

          scores = scores.squeeze()
          pred_boxes = pred_boxes.squeeze()
        if vis:
            im2show = np.copy(im)
        obj_dets, hand_dets = None, None
        
        for j in xrange(1, len(pascal_classes)):
            # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
            min_conf = 0.3
            
            inds = torch.nonzero(scores[:,j]>min_conf).view(-1)
            while inds.numel() <50 and min_conf>=0:
              min_conf -=5e-2
              inds = torch.nonzero(scores[:,j]>min_conf).view(-1)
            
            # if there is det
            if inds.numel() > 0:
              cls_scores = scores[:,j][inds]
              _, order = torch.sort(cls_scores, 0, True)
              if args.class_agnostic:
                cls_boxes = pred_boxes[inds, :]
              else:
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
              
              
              if pascal_classes[j] == 'targetobject':
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
                cls_dets = cls_dets[order]
                nms_min = cfg.TEST.NMS

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
                cls_dets = cls_dets[order]
                cls_feats = pooled_feat[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], nms_min)
                while len(keep)<2 and nms_min>=0:
                  nms_min -=5e-2
                  keep = nms(cls_boxes[order, :], cls_scores[order], nms_min)
                  
                cls_dets = cls_dets[keep.view(-1).long()]
                cls_feats = pooled_feat[keep.view(-1).long()]
                obj_dets = cls_dets.cpu().numpy()
                obj_feats = cls_feats.cpu().numpy()
              if pascal_classes[j] == 'hand':
                nms_min = cfg.TEST.NMS
                hands = {'left':None,'right':None}
                
                while (hands['left'] is None or hands['right'] is None) and nms_min>=0:
                  cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
                  cls_dets = cls_dets[order]
                  cls_feats = pooled_feat[order]
                  keep = nms(cls_boxes[order, :], cls_scores[order], nms_min)
                  cls_dets = cls_dets[keep.view(-1).long()]
                  for i in range(len(cls_dets)):

                    hand_lr = cls_dets[i, -1]
                    if hand_lr>0 and hands['left'] is None:
                        hands['left'] = i
                    if hand_lr<=0 and hands['right'] is None:
                        hands['right'] = i
                  nms_min -=5e-2
                if hands['left'] is None:
                  hands['left'] = hands['right']
                if hands['right'] is None:
                  hands['right'] = hands['left']
                cls_feats = torch.stack([cls_feats[hands['left']],cls_feats[hands['right']]])
                cls_dets = torch.stack([cls_dets[hands['left']],cls_dets[hands['right']]])
                # cls_dets = cls_dets[keep.view(-1).long()]
                # keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                # cls_dets = cls_dets[keep.view(-1).long()]
                hand_dets = cls_dets.cpu().numpy()
                hand_feats = cls_feats.cpu().numpy()
        img_obj_id = filter_object(obj_dets, hand_dets)

        obj_dets = obj_dets[img_obj_id]
        obj_feats = obj_feats[img_obj_id]
        np.savez_compressed(
          os.path.join(npz_d, f'{frame:010d}'),
          obj_dets=obj_dets, 
          obj_feats=obj_feats, 
          hand_dets=hand_dets,
          hand_feat=hand_feats, 
          pooled_feat=pooled_feat.cpu().numpy(),
          rois=rois.cpu().numpy(),


          bbox_pred = bbox_pred.cpu().numpy(),
          pred_boxes = pred_boxes.cpu().numpy(),
          contact_vector = contact_vector.cpu().numpy(),
          offset_vector = offset_vector.cpu().numpy(),
          lr_vector = lr_vector.cpu().numpy(),


          blobs = blobs,
          im_scales=im_scales


            
        )


        # if vis:
        #   im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, 0.1, 0.2)
        #   folder_name = args.save_dir
        #   os.makedirs(folder_name, exist_ok=True)
        #   result_path = os.path.join(folder_name, imglist[num_images][:-4] + "_det.png")
        #   im2show.save(result_path)

    # if vis:
      # image_list = glob.glob(os.path.join(folder_name,'*_det.png'))
      # gif_name = 'created_gif.gif'
      # create_gif(image_list, gif_name)