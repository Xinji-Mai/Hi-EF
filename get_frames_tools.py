import cv2
import os
import numpy as np
from numpy.random import randint
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import random
import cv2
import torchvision.transforms as transforms
from PIL import Image
import natsort
import numpy as np
import models
from models.common import DetectMultiBackend
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)
from utils.torch_utils import select_device, smart_inference_mode
from pathlib import Path
import numpy as np

meta_o = pd.read_csv('/home/et23-maixj/mxj/DFER_Datasets/SIRV_final/sample.csv')
meta_p = pd.read_csv('/home/et23-maixj/mxj/DFER_Datasets/SIRV_final/annotation.csv')
ori_path = '/home/et23-maixj/mxj/DFER_Datasets/SIRV_final/video'

save_path = '/home/et23-maixj/mxj/DFER_Datasets/SIRV_final/Image/Pose'
csv_path = '/home/et23-maixj/mxj/DFER_Datasets/SIRV_final/process.csv'
meta_sample = []
meta_text = []

for idx, row in meta_o.iterrows():
    inPath = row[0]
    clip1 = row[1]
    clip2 = row[2]
    clip3 = row[3]
    clip4 = row[4]
    temp = [inPath, clip1, clip2, clip3, clip4]
    meta_sample.append(temp)

for idx, row in meta_p.iterrows():
    inPath = row[0]
    text = row[1]
    temp = [inPath, text]
    meta_text.append(temp)

def _get_video(vi_path,dirt,file,num):
    vid = cv2.VideoCapture(vi_path)
    count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if count < 34: #小于34全取；33到60，2步1个；59到80，3步1个；80以上，4步1个
        offsets1 = list(range(1, count)) 
    elif(33 < count < 60): 
        offsets1 = list(range(1, count, 2))
    elif(59 < count < 80): 
        offsets1 = list(range(1, count, 3)) 
    else: 
        offsets1 = list(range(1, count, 4)) 
    
    vipa = '/home/et23-maixj/mxj/DFER_Datasets/SIRV_final/processon/ori/' + dirt
    if not os.path.exists(vipa):  # 文件夹不存在，则创建
        os.makedirs(vipa)
        
    vipaa = vipa + '/' + file

    if not os.path.exists(vipaa):  # 文件夹不存在，则创建
        os.makedirs(vipaa)
    
    indices = offsets1
    i=1
    
    for seg_ind in indices:
        p = int(seg_ind)
        vid.set(cv2.CAP_PROP_FPS, p)
        ret,seg_img = vid.read()
        name = vipaa + '/' + '{:0>5}.jpg'.format(i)
        cv2.imwrite(name,seg_img)
        i = i + 1

num = 1
files = []
# folder_path = './videos_without_audio/'
folder_path = '/home/et23-maixj/mxj/DFER_Datasets/SIRV_final/video/'
for root, dirts, files in os.walk(folder_path):
    for dirt in dirts:
        dirt_name = root + dirt
        for root1, dirts1, files1 in os.walk(dirt_name):
            for filename in files1:
                _get_video(os.path.join(root1, filename),dirt,filename.split('.mp4')[0],num)

    # for file in files:
    #     _get_video(os.path.join(root, file),file.split('.mp4')[0],num)
    #     num = num + 1
    