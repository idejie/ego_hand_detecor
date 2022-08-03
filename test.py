
side_map = {'l':'Left', 'r':'Right'}
side_map2 = {0:'Left', 1:'Right'}
side_map3 = {0:'L', 1:'R'}
state_map = {0:'No Contact', 1:'Self Contact', 2:'Another Person', 3:'Portable Object', 4:'Stationary Object'}
state_map2 = {0:'N', 1:'S', 2:'O', 3:'P', 4:'F'}
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from model.utils.viz_hand_obj import *
import cv2

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


npz = '/S5/MIPL/yangdj/ego4d_data/hand_object_new/9c59e912-2340-4400-b2df-7db3d4066723/0000000075'
video_id,image_id = npz.split('/')[-2:]
print('/S5/MIPL/yangdj/ego4d_data/v1/frames/'+video_id+'/'+image_id+'.jpg')
image = cv2.imread('/S5/MIPL/yangdj/ego4d_data/v1/frames/'+video_id+'/'+image_id+'.jpg')
npz = np.load(npz+'.npz')
obj_dets = npz['obj_dets']
obj_feats = npz['obj_feats']
print(obj_feats[0]==obj_feats[1])
print(obj_dets[0]==obj_dets[1])
print(obj_dets.shape)
hand_dets = npz['hand_dets']
print(hand_dets.shape)
im2show = vis_detections_filtered_objects_PIL(image, obj_dets, hand_dets, 0.1, 0.2)
im2show.save('test.png')

