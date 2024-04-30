import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import librosa
import numpy as np
from typing import Tuple, Optional
import natsort
from typing import Optional
from utils import transforms as u_transforms
from ignite_trainer import _utils
import cv2

class DataPreprocessing(Dataset):
    """
    Dataset class for data preprocessing.
    
    Attributes:
    - root: Directory containing datasets.
    - sample_rate: Sample rate for audio data.
    - dataset_name: Name of the dataset.
    - transform: Transformation for image data.
    - transforms_audio: Transformation for audio data.
    """
    
    def __init__(self, root: str):
        self.dataset_name = "DatasetName"  # Use a placeholder or generic name
        super().__init__()
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.f_image_dir = os.path.join(self.dataset_dir, "face")
        self.s_image_dir = os.path.join(self.dataset_dir, "origin")
        self.p_image_dir = os.path.join(self.dataset_dir, "person")

        self.audio_dir = os.path.join(self.dataset_dir, "audio")
        self.emo = {}
        self.text = {}
        self.sample = {}

        annotation_file = os.path.join(self.dataset_dir, 'annotation.csv')
        axlsx = pd.read_csv(annotation_file)
        classnames = {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "neutral": 4, "sad": 5, "surprise": 6}
        for nidx, nrow in axlsx.iterrows():
            key = nrow[0]
            self.emo[key] = classnames.get(nrow[7], nrow[7])
            self.text[key] = str(nrow[1])

        # Processing additional CSV files and populating relevant attributes omitted for brevity

        self.sample_rate = 22050
        self.audio_data = {}
        self.load_audio_data()

        # Transformation for image data
        self.transforms_image = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # Transformation for audio data
        self.transforms_audio = transforms.Compose([
            u_transforms.ToTensor1D(),
            u_transforms.RandomPadding(out_len = 110250, train = False),
            u_transforms.RandomCrop(out_len = 110250, train = False)
        ])

    def __len__(self) -> int:
        return len(self.label_3)

    @staticmethod
    def _load_worker(ident:str, idx: int, filename: str, sample_rate: Optional[int] = None) -> Tuple[int, int, np.ndarray]:
        # load audio data
        
        wav, sample_rate = librosa.load(filename, sr=sample_rate, mono=True)
        target_length = 3 * sample_rate
        wav, index = librosa.effects.trim(wav, top_db=20)
        if len(wav) < target_length:
            wav = np.pad(wav, (0, target_length - len(wav)), mode='constant')
        elif len(wav) > target_length:
            mid = (index[0] + index[1]) // 2
            start = mid - target_length // 2
            end = mid + target_length // 2
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
        ident_1 = self.label_1[idx]
        au1 = self.transforms_audio(self.audio_data[ident_1][0])
        tt1 = self.text[ident_1]
        ident_2 = self.label_2[idx]
        au2 = self.transforms_audio(self.audio_data[ident_2][0])
        tt2 = self.text[ident_2]
        ident_3 = self.label_3[idx]
        au3 = self.transforms_audio(self.audio_data[ident_3][0])
        tt3 = self.text[ident_3]
        
        ident_4 = self.label_4[idx]
        label_4 = self.emo[ident_4]
        label_3 = self.emo[ident_3]
        if type(label_4) != int:
            label_4 = 4

        if type(label_3) != int:
            label_3 = 4

        f_path_1 = os.path.join(self.f_image_dir,ident_1)
        s_path_1 = os.path.join(self.s_image_dir,ident_1)
        f_path_2 = os.path.join(self.f_image_dir,ident_2)
        s_path_2 = os.path.join(self.s_image_dir,ident_2)
        f_path_3 = os.path.join(self.f_image_dir,ident_3)
        s_path_3 = os.path.join(self.s_image_dir,ident_3)


        if os.path.exists(s_path_1):
            sf1, vsf1 = self.get_all_video_frame(s_path_1)
        else:
            sf1 = torch.zeros(16,3,224,224)
            vsf1 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if os.path.exists(f_path_1):
            ff1, vff1 = self.get_all_video_frame(f_path_1)
        else:
            ff1 = torch.zeros(16,3,224,224)
            vff1 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        if os.path.exists(s_path_2):
            sf2, vsf2 = self.get_all_video_frame(s_path_2)
        else:
            sf2 = torch.zeros(16,3,224,224)
            vsf2 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if os.path.exists(f_path_2):
            ff2, vff2 = self.get_all_video_frame(f_path_2)
        else:
            ff2 = torch.zeros(16,3,224,224)
            vff2 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        if os.path.exists(s_path_3):
            sf3, vsf3 = self.get_all_video_frame(s_path_2)
        else:
            sf3 = torch.zeros(16,3,224,224)
            vsf3 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if os.path.exists(f_path_3):
            ff3, vff3 = self.get_all_video_frame(f_path_3)
        else:
            ff3 = torch.zeros(16,3,224,224)
            vff3 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        return ff1, sf1, ff2, sf2, ff3, sf3, au1, tt1, au2, tt2, au3, tt3, label_3, label_4, vsf1, vff1, vsf2, vff2, vsf3, vff3


    def get_all_video_frame(self, orignal_path):
        video_x = list()
        img_lists = os.listdir(orignal_path) 
        img_lists = natsort.natsorted(img_lists)

        img_lists = os.listdir(orignal_path)

        img_lists = [f for f in img_lists if f.endswith(".jpg")]
        img_count = len(img_lists)

        AllFrames = 16
        valid_list = torch.ones(AllFrames)

        if(img_count < AllFrames):
            img_first = Image.new("RGB", (112, 168))
            img_first_t = self.transforms_image(img_first)
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
                img_first_t = self.transforms_image(img_first)
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
                img_first_t = self.transforms_image(img_first)
                video_x.append(img_first_t)
            video_x = torch.stack(video_x, dim=0)
        return video_x,valid_list
    
    def get_label_to_cate(self):
            return self.cls_num_list
