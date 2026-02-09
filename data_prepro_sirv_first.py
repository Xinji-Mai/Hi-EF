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
import multiprocessing as mp
import os
import warnings
import multiprocessing as mp

import tqdm
import librosa

import numpy as np
import pandas as pd

import torch.utils.data as td
import torchvision as tv
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from utils import transforms as u_transforms
from ignite_trainer import _utils

#第1阶段，3推3

class data_prepro(Dataset):
    def __init__(self, 
                 root='/path/to/DFER_Datasets',
                 ):
        self.dataset_name = "SIRV_final"
        super(data_prepro, self).__init__()
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.f_image_dir = os.path.join(self.dataset_dir, "processon/face")
        self.o_image_dir = os.path.join(self.dataset_dir, "processon/ori")
        self.p_image_dir = os.path.join(self.dataset_dir, "preprocess/person")

        self.audio_dir = os.path.join(self.dataset_dir, "audio")
        self.emo = {}
        self.text = {}

        mxlsx = pd.read_csv(os.path.join(self.dataset_dir,'sample_train.csv'))
        axlsx = pd.read_csv(os.path.join(self.dataset_dir,'annotation.csv'))
        classnames = {"angry":0, "disgust":1, "fear":2, "happy":3, "neutral":4, "sad":5, "surprise":6}
        for nidx, nrow in axlsx.iterrows():
            key = nrow[0]
            value = nrow
            if type(value[7]) == str:
                self.emo[key] = classnames[value[7]]
            else:
                self.emo[key] = value[7]
            self.text[key] = str(value[1])

        self.sample_rate = 22050
        self.audio_data = {}
        self.load_audio_data()

        
        self.label_34 = []

        last = None
        for idx, row in mxlsx.iterrows():
            clip3 = row[3]
            clip4 = row[4] #ccc
            if type(self.emo[clip3]) == int and clip3 != last:
                self.label_34.append(clip3)
            if type(self.emo[clip4]) == int:#ccc
                self.label_34.append(clip4) #ccc
            last = clip4
            #第1阶段，只训第3
            
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.transforms_audio = transforms.Compose([
            u_transforms.ToTensor1D(),
            u_transforms.RandomPadding(out_len = 110250, train = False),
            u_transforms.RandomCrop(out_len = 110250, train = False)
            ])

    def __len__(self):
        return len(self.label_34)
    
    @staticmethod
    def _load_worker(ident:str, idx: int, filename: str, sample_rate: Optional[int] = None) -> Tuple[int, int, np.ndarray]:
        wav, sample_rate = librosa.load(filename, sr=sample_rate, mono=True)
        # 计算目标长度，假设为5秒
        target_length = 3 * sample_rate
        # 使用trim函数去除静音部分，并获取索引范围
        wav, index = librosa.effects.trim(wav, top_db=20)
        # 如果去除静音后的长度小于目标长度，则在末尾填充零
        if len(wav) < target_length:
            wav = np.pad(wav, (0, target_length - len(wav)), mode='constant')
        # 如果去除静音后的长度大于目标长度，则从中间截取一段
        elif len(wav) > target_length:
            # 计算中间位置
            mid = (index[0] + index[1]) // 2
            # 计算起始位置和结束位置
            start = mid - target_length // 2
            end = mid + target_length // 2
            # 截取音频信号
            wav = wav[start:end]

        if wav.ndim == 1:
            wav = wav[:, np.newaxis]

        wav = wav.T * 32768.0

        return ident, idx, sample_rate, wav.astype(np.float32)

    
    def load_audio_data(self):
        items_to_load = []
        lists = os.listdir(self.audio_dir)
        tidx = 0
        for listz in lists:
            temp_list = os.path.join(self.audio_dir, listz)
            audio_lists = os.listdir(temp_list)
            for au in audio_lists:
                ident = listz + '/' + au.split('.mp3')[0]
                au_path = os.path.join(temp_list, au)
                temp = ident, tidx, au_path, self.sample_rate
                items_to_load.append(temp)
                tidx = tidx + 1

        for ident, idx, au_path, sample_rate in items_to_load:
            ident, idx, sample_rate, wav = self._load_worker(ident=ident, idx=idx, filename=au_path, sample_rate=sample_rate)
            self.audio_data[ident] = [wav, sample_rate]

    def __getitem__(self, idx):
        ident = self.label_34[idx]
        daudio = self.transforms_audio(self.audio_data[ident][0])
        dtext = self.text[ident]
        label = self.emo[ident]
        if type(label) != int:
            label = 4
        f_path = os.path.join(self.f_image_dir,ident)
        o_path = os.path.join(self.o_image_dir,ident)
        if os.path.exists(o_path):
            o_frames_3, valid_list_o = self.get_all_video_frame(o_path)
        else:
            o_frames_3 = torch.zeros(16,3,224,224)
            valid_list_o = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        if os.path.exists(f_path):
            f_frames_3, valid_list_f = self.get_all_video_frame(f_path)
        else:
            f_frames_3 = torch.zeros(16,3,224,224)
            valid_list_f = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        return f_frames_3, o_frames_3, daudio, dtext, label, valid_list_o, valid_list_f
    

    def get_all_video_frame(self, orignal_path):
        video_x = list()
        img_lists = os.listdir(orignal_path)
        img_lists = natsort.natsorted(img_lists)

        img_lists = os.listdir(orignal_path)

        img_lists = [f for f in img_lists if f.endswith(".jpg")]
        img_count = len(img_lists)

        # print('img_count',img_count)
        AllFrames = 16
        valid_list = torch.ones(AllFrames)
        # pred = self.get_pred_img(img_count,orignal_path,img_lists)
        if(img_count < AllFrames):
            # print('###################')
            # print(img_count)
            img_first = Image.new("RGB", (112, 168))
            img_first_t = self.transform(img_first)
            for i in range(img_count):
                path_first_image = os.path.join(orignal_path, img_lists[i])
                img = cv2.imread(path_first_image)
                get_frame_same_size = np.zeros((112, 168, 3))
                height_scale = 110 / img.shape[0]
                width_scale = 160 / img.shape[1]
                if (height_scale < 1 or width_scale < 1):
                    scale_min = min(height_scale, width_scale)
                    img = cv2.resize(img, None, fx=scale_min, fy=scale_min, interpolation=cv2.INTER_CUBIC)
                    get_frame_same_size[: img.shape[0], : img.shape[1], : img.shape[2]] = img
                else:
                    get_frame_same_size[: img.shape[0], : img.shape[1], : img.shape[2]] = img

                get_frame_same_size = get_frame_same_size.astype(np.uint8)
                get_frame_same_size = cv2.cvtColor(get_frame_same_size, cv2.COLOR_BGR2RGB)
                img_first = Image.fromarray(get_frame_same_size)
                img_first_t = self.transform(img_first)
                video_x.append(img_first_t)
            addFrameNumber = AllFrames - img_count
            for i in range(addFrameNumber):
                video_x.append(img_first_t)
                valid_list[len(video_x)-1] = 0
            video_x = torch.stack(video_x, dim=0)
        else:
            for i in range(img_count - AllFrames, img_count):
                path_first_image = os.path.join(orignal_path, img_lists[i])
                img = cv2.imread(path_first_image)
                get_frame_same_size = np.zeros((112, 168, 3))
                height_scale = 110 / img.shape[0]
                width_scale = 160 / img.shape[1]
                if (height_scale < 1 or width_scale < 1):
                    scale_min = min(height_scale, width_scale)
                    img = cv2.resize(img, None, fx=scale_min, fy=scale_min, interpolation=cv2.INTER_CUBIC)
                    get_frame_same_size[: img.shape[0], : img.shape[1], : img.shape[2]] = img
                else:
                    get_frame_same_size[: img.shape[0], : img.shape[1], : img.shape[2]] = img

                get_frame_same_size = get_frame_same_size.astype(np.uint8)
                get_frame_same_size = cv2.cvtColor(get_frame_same_size, cv2.COLOR_BGR2RGB)
                img_first = Image.fromarray(get_frame_same_size)
                img_first_t = self.transform(img_first)
                video_x.append(img_first_t)
                #3 224 224
            video_x = torch.stack(video_x, dim=0)
        # 3 16 224 224
        return video_x,valid_list
    
    def get_label_to_cate(self):
            return self.cls_num_list
    

