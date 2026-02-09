import torch
import torch.nn as nn
from pytorch_lightning import Trainer
import torch
import argparse
from torch.utils.data import DataLoader
from model import AICLIP
import argparse
import numpy as np
from collections import defaultdict
import contextlib
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn
import os
import torch
from PIL import Image
import torch
import argparse
from torch.utils.data import DataLoader

from model_sirv_first_va import AICLIP
import torchvision as tv
from typing import Type
from typing import Union
from typing import Optional
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
import numpy as np
import os
import math
import argparse
import numpy as np
from collections import defaultdict
import contextlib
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn
import torchvision.transforms as transforms
from clip.simple_tokenizer import SimpleTokenizer
from clip import clip
from PIL import Image

from data_prepro_sirv_first_test import data_prepro
devices=[2]

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4"
USE_CUDA = torch.cuda.is_available()
device_ids_parallel = [2]
device = torch.device("cuda:{}".format(device_ids_parallel[0]) if USE_CUDA else "cpu")


train_ratio, valid_ratio, test_ratio = 0.7, 0.15, 0.15
BATCH_TRAIN = 32
BATCH_TEST = 32
WORKERS_TRAIN = 8
WORKERS_TEST = 8
EPOCHS = 100
LOG_INTERVAL = 50

topk=1

# 定义和初始化您的模型，确保它和保存checkpoint时的一致
model = AICLIP()
dataset = data_prepro(root = '/path/to/DFER_Datasets')
len_dataset = dataset.__len__()

# 加载checkpoint文件，它是一个字典，包含了模型的参数和其他信息
checkpoint = torch.load("sec_checkpoint/mmmm_va_rr.ckpt", map_location = device) # 这里假设您的checkpoint文件名是best-checkpoint.ckpt，您可以根据您的实际文件名来修改


# 将模型的参数加载到您的模型中
model.load_state_dict(checkpoint["state_dict"])

test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1) 


# 设置模型为评估模式，这样可以关闭一些影响推理结果的功能，比如dropout和batch normalization
model.eval()

trainer = Trainer(
    accelerator="gpu", 
    devices=devices,
)

trainer.test(model,test_loader)


