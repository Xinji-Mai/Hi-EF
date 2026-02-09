import torch
import argparse
from torch.utils.data import DataLoader
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
import io
import os
import glob
import json
import sys
import time
import math
import signal
import argparse
import numpy as np
from collections import defaultdict
import contextlib
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn
from head import make_classifier_head, get_zero_shot_weights
from logit import LogitHead
from copy import deepcopy
import hashlib
import os
import urllib
import warnings
from typing import Union, List
from collections import OrderedDict
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
import torch.nn as nn
import clip
from clip.model import CLIP
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import torchmetrics
from esresnet import ESResNeXtFBSP
from audioclip import AudioCLIP
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from partial_model import get_text_encoder
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import wandb
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import pandas as pd
from sklearn.metrics import classification_report
torch.set_default_dtype(torch.float32) # 设置全局的默认数据类型为torch.float32
# torch.set_default_tensor_type(torch.cuda.FloatTensor) # 设置全局的默认张量类型为CUDA浮点张量
n_fft = 2048
hop_length = 561
win_length = 1654
window = 'blackmanharris'
normalized = True,
onesided = True,
spec_height = -1
spec_width = -1
apply_attention: bool = True
multilabel: bool = True
pretrained: Union[bool, str] = True
embed_dim = 527
logit = 4.60517
lr = 0.0002
# lr = 0.01
weight_decay = 0.000001
ADAM_BETAS = (0.9, 0.999)
_tokenizer = _Tokenizer()
fea_size = 768
clip_name = 'ViT-B/32'

if clip_name == 'ViT-B/32':
    fea_size = 512
elif clip_name == 'ViT-L/14':
    fea_size = 768

wandb.init(
            # set the wandb project where this run will be logged
            project="b",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": 0.0002,
            "architecture": "CLIP",
            "dataset": "SIRV",
            "epochs": 60,
            }
        )
class QKVTransformerClip(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[QKVResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        return self.resblocks((q, kv))[0]
    
class QKVResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_11 = LayerNorm(d_model)
        self.ln_12 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head

    def attention(self, q: torch.Tensor, kv: torch.Tensor,):
        return self.attn(q, kv, kv, need_weights=False)[0]

    def forward(self, para_tuple: tuple):
        # x: torch.Tensor, attn_mask: torch.Tensor
        # print(para_tuple)
        q, kv = para_tuple
        x = kv + self.attention(self.ln_11(q), self.ln_12(kv))
        x = x + self.mlp(self.ln_2(x))
        return (q, x)

class TextTensorDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, label_tensor, eot_indices):
        super(TextTensorDataset, self).__init__()
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        self.eot_indices = eot_indices
    
    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index], self.eot_indices[index]

    def __len__(self):
        return self.input_tensor.size(0)

class CapEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class TransformerClip(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return self.resblocks((x, attn_mask))[0]

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module): 
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head) #PyTorch提供的nn.MultiheadAttention类，这个类可以自动根据输入的序列计算QKV三个子空间的向量，
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor):
        attn_mask_ = attn_mask.repeat_interleave(self.n_head, dim=0)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, para_tuple: tuple):
        # x: torch.Tensor, attn_mask: torch.Tensor
        # print(para_tuple)
        x, attn_mask = para_tuple
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)

class PositionalEncoding(nn.Module):
    # 初始化函数，需要指定嵌入维度和最大序列长度
    def __init__(self):
        super().__init__()
        self.frame_position_embeddings = nn.Embedding(16,fea_size)
    def forward(self, x):
        return self.frame_position_embeddings(x)
    
class selfattention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)
        self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim = -1)
    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        #input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        #input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        #q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(attn_matrix)#经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out = out.view(*input.shape)
 
        return self.gamma * out + input
#自定义clip 重新计算交叉熵损失
class CustomCLIP(nn.Module):
    def device(self):
        return self.visual.conv1.weight.device

    def __init__(self, classnames, clip_model):
        super().__init__()
        self.cap_encoder = CapEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        transformer_width = fea_size
        transformer_heads = transformer_width // 64
        transformer_layers = 12
        self.transformer_heads = transformer_heads
        dtype = clip_model.dtype
        self.transshape = nn.Linear(527, 512)
        
        self.f_pencoding = PositionalEncoding()
        self.o_pencoding = PositionalEncoding()
        self.p_pencoding = PositionalEncoding()
        self.f_transformerClip = TransformerClip(width=transformer_width, layers=transformer_layers,
                                        heads=transformer_heads, )
        self.o_transformerClip = TransformerClip(width=transformer_width, layers=transformer_layers,
                                        heads=transformer_heads, )
        self.p_transformerClip = TransformerClip(width=transformer_width, layers=transformer_layers,
                                        heads=transformer_heads, )
        
        self.seq_transformerClip = TransformerClip(width=transformer_width, layers=transformer_layers,
                                        heads=transformer_heads, )
        
        self.cross_transformerClip = TransformerClip(width=transformer_width, layers=transformer_layers,
                                        heads=transformer_heads, )
        
        self.a_transformerClip = QKVTransformerClip(width=transformer_width, layers=transformer_layers,
                                        heads=transformer_heads, )
        
        self.b_transformerClip = QKVTransformerClip(width=transformer_width, layers=transformer_layers,
                                        heads=transformer_heads, )

        self.model = clip_model
        self.image_encoder = clip_model.encode_image
        self.logit_scale =  clip_model.logit_scale
        self.audio_encoder = ESResNeXtFBSP(
            n_fft=n_fft,#FFT的窗口大小
            hop_length=hop_length,#FFT的跳跃长度
            win_length=win_length,#FFT每个窗口的实际长度
            window=window,#FFT使用的窗函数
            normalized=normalized,#对频谱图进行归一化处理
            onesided=onesided,#是否只保留频谱图的一半
            spec_height=spec_height,#频谱图的高度
            spec_width=spec_width,#频谱图的宽度，也就是时间维度的大小
            num_classes=embed_dim,#音频编码器输出的特征向量的维度，也就是类别数
            apply_attention=apply_attention,#是否在音频编码器中使用注意力机制
            pretrained=False,
        )

        temp = 'ESRNXFBSP.pt'

        labels = [0,1,2,3,4,5,6]
        labels = torch.tensor(labels)
        text_encoder = get_text_encoder(
            text_layer_idx = 0,
            clip_model = clip_model
        )
        with torch.no_grad():
            prompts = ["a photo of a " + name for name in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(clip_model.logit_scale.device)
            text_features, eot_indices = text_encoder.feature_extractor(prompts)
        # text_features, eot_indices = text_encoder.feature_extractor(self.tokenized_prompts)
        
        text_dataset = TextTensorDataset(text_features,labels,eot_indices)

        head, num_classes, in_features = make_classifier_head(
                classifier_head = 'adapter',
                clip_encoder = clip_name,
                classifier_init = 'fewshot',
                zeroshot_dataset = text_dataset,
                text_encoder = text_encoder
            )
        
        self.logit_head = LogitHead(
                        head,
                        logit_scale=logit,
                    )


        self.audio_encoder.load_state_dict(torch.load(
                    temp,
                    # map_location='cpu'
                ), strict=False)
        
    def _mean_pooling_for_similarity_visual(self, visual_output, valid_list,):
        video_mask_un = valid_list.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def forward(self, f_frames = None, o_frames = None, p_frames = None, cap = None, audio = None, label = None, valid_list_o = None, valid_list_f = None, valid_list_p = None,):
        cap = None
        image_encoder = self.image_encoder
        audio_encoder = self.audio_encoder
        cap_encoder = self.cap_encoder

        logit_scale = self.logit_scale.exp()
        f_image_feature_flag = False
        o_image_feature_flag = False
        p_image_feature_flag = False

        image_feature_flag = True
        cap_feature_flag = False
        audio_feature_flag = False

        a_transformerClip = self.a_transformerClip
        b_transformerClip = self.b_transformerClip

        if f_frames != None:
            f_image_feature_flag = True
            b, timestep, channel, h, w = f_frames.shape
            f_frames = f_frames.view(b*timestep,channel,h,w)
            f_video_features = image_encoder(f_frames.type(self.dtype))
            bs_pair = valid_list_f.size(0)
            #12*16 512
            f_video_features = f_video_features.view(bs_pair, timestep, f_video_features.size(-1))
            f_visual_output = f_video_features
            
            visual_output_original = f_visual_output
            seq_length = f_visual_output.size(1)#获取序列长度L
            position_ids = torch.arange(seq_length, dtype=torch.long, device=f_visual_output.device)
            #把position_ids扩展成和visual_output相同的批次大小N
            position_ids = position_ids.unsqueeze(0).expand(f_visual_output.size(0), -1)
            
            f_frame_position_embeddings = self.f_pencoding(position_ids)

            #获取位置编码向量，存储在frame_position_embeddings中
            f_visual_output = f_visual_output + f_frame_position_embeddings
            #把位置编码向量和视频帧特征向量相加
            extended_video_mask = (1.0 - valid_list_f.unsqueeze(1)) * -1000000.0
            #指示哪些视频帧是有效的，哪些是无效的
            extended_video_mask = extended_video_mask.expand(-1, valid_list_f.size(1), -1)
            f_visual_output = f_visual_output.permute(1, 0, 2)  # NLD -> LND 调整visual_output的维度顺序
            f_visual_output = self.f_transformerClip(f_visual_output, extended_video_mask)
            #对输入数据进行自注意力（self-attention）计算，并输出新的特征向量
            f_visual_output = f_visual_output.permute(1, 0, 2)  # LND -> NLD
            f_visual_output = f_visual_output + visual_output_original

            
            # f_visual_output = f_visual_output / f_visual_output.norm(dim=-1, keepdim=True)
            f_visual_output = self._mean_pooling_for_similarity_visual(f_visual_output, valid_list_f)
            # f_image_feature = f_visual_output / f_visual_output.norm(dim=-1, keepdim=True)
            f_image_feature = f_visual_output
        else:
            raise ValueError("face frames are None")
        o_image_feature = torch.zeros_like(f_image_feature)
        p_image_feature = torch.zeros_like(f_image_feature)

        if o_frames != None:
            o_image_feature_flag = True
            b, timestep, channel, h, w = o_frames.shape
            o_frames = o_frames.view(b*timestep,channel,h,w)
            o_video_features = image_encoder(o_frames.type(self.dtype))
            bs_pair = valid_list_o.size(0)
            #12*16 512
            o_video_features = o_video_features.view(bs_pair, timestep, o_video_features.size(-1))
            o_visual_output = f_video_features
            
            visual_output_original = o_visual_output
            seq_length = o_visual_output.size(1)#获取序列长度L
            position_ids = torch.arange(seq_length, dtype=torch.long, device=o_visual_output.device)
            #把position_ids扩展成和visual_output相同的批次大小N
            position_ids = position_ids.unsqueeze(0).expand(o_visual_output.size(0), -1)
            
            o_frame_position_embeddings = self.o_pencoding(position_ids)
            #获取位置编码向量，存储在frame_position_embeddings中
            o_visual_output = o_visual_output + o_frame_position_embeddings
            #把位置编码向量和视频帧特征向量相加
            extended_video_mask = (1.0 - valid_list_o.unsqueeze(1)) * -1000000.0
            #指示哪些视频帧是有效的，哪些是无效的
            extended_video_mask = extended_video_mask.expand(-1, valid_list_o.size(1), -1)
            o_visual_output = o_visual_output.permute(1, 0, 2)  # NLD -> LND 调整visual_output的维度顺序
            o_visual_output = self.o_transformerClip(o_visual_output, extended_video_mask)
            #对输入数据进行自注意力（self-attention）计算，并输出新的特征向量
            o_visual_output = o_visual_output.permute(1, 0, 2)  # LND -> NLD
            o_visual_output = o_visual_output + visual_output_original

            o_visual_output = self._mean_pooling_for_similarity_visual(o_visual_output, valid_list_o)
            o_image_feature = o_visual_output
        else:
            o_image_feature = torch.zeros_like(f_image_feature)

        if p_frames != None:
            p_image_feature_flag = True
            b, timestep, channel, h, w = p_frames.shape
            p_frames = p_frames.view(b*timestep,channel,h,w)
            p_video_features = image_encoder(p_frames.type(self.dtype))
            bs_pair = valid_list_p.size(0)
            #12*16 512
            p_video_features = p_video_features.view(bs_pair, timestep, p_video_features.size(-1))
            p_visual_output = p_video_features
            
            visual_output_original = p_visual_output
            seq_length = p_visual_output.size(1)#获取序列长度L
            position_ids = torch.arange(seq_length, dtype=torch.long, device=p_visual_output.device)
            #把position_ids扩展成和visual_output相同的批次大小N
            position_ids = position_ids.unsqueeze(0).expand(p_visual_output.size(0), -1)
            
            p_frame_position_embeddings = self.p_pencoding(position_ids)
            #获取位置编码向量，存储在frame_position_embeddings中
            p_visual_output = p_visual_output + p_frame_position_embeddings
            #把位置编码向量和视频帧特征向量相加
            extended_video_mask = (1.0 - valid_list_p.unsqueeze(1)) * -1000000.0
            #指示哪些视频帧是有效的，哪些是无效的
            extended_video_mask = extended_video_mask.expand(-1, valid_list_p.size(1), -1)
            p_visual_output = p_visual_output.permute(1, 0, 2)  # NLD -> LND 调整visual_output的维度顺序
            p_visual_output = self.p_transformerClip(p_visual_output, extended_video_mask)
            #对输入数据进行自注意力（self-attention）计算，并输出新的特征向量
            p_visual_output = p_visual_output.permute(1, 0, 2)  # LND -> NLD
            p_visual_output = p_visual_output + visual_output_original

            
            # p_visual_output = p_visual_output / p_visual_output.norm(dim=-1, keepdim=True)
            p_visual_output = self._mean_pooling_for_similarity_visual(p_visual_output, valid_list_p)
            p_image_feature = p_visual_output
        else:
            p_image_feature = torch.zeros_like(f_image_feature) #8 512
        

        image_part = 0
        image_feature = torch.stack([f_image_feature, o_image_feature, p_image_feature], dim=0) #3 8 512 type batchsize feature

        if f_image_feature_flag and o_image_feature_flag and p_image_feature_flag:
            valid = [1,1,1]
            image_part = 3
        elif f_image_feature_flag and o_image_feature_flag:
            valid = [1,1,0]
            image_part = 2
        elif f_image_feature_flag and p_image_feature_flag:
            valid = [1,0,1]
            image_part = 2
        elif o_image_feature is not None and p_image_feature is not None:
            valid = [0,1,1]   
            image_part = 2
        elif f_image_feature is not None:
            valid = [1,0,0]   
            image_part = 1
        else:
            image_feature = None

        image_feature_ori = image_feature

        valid = torch.tensor(valid)
        valid = valid.repeat(b,1).to(image_feature.device)
        mask = (1.0 - valid.unsqueeze(1)) * -1000000.0
        mask = mask.expand(-1, valid.size(1), -1).to(image_feature.device)
        
        f_image_feature = f_image_feature.unsqueeze(0).repeat(3,1,1)
        image_feature = a_transformerClip(f_image_feature,image_feature_ori) + image_feature_ori
        image_feature = image_feature.permute(1, 0, 2)
        image_feature = self._mean_pooling_for_similarity_visual(image_feature, valid) #变为8 512

        # image_feature = self.seq_transformerClip(image_feature, mask) + image_feature_ori
        # image_feature = image_feature.permute(1, 0, 2)  # NLD -> LND 调整visual_output的维度顺序 变为8 3 512
        # image_feature = self._mean_pooling_for_similarity_visual(image_feature, valid)
        # image_feature = a_transformerClip(f_image_feature,image_feature)
        audio_feature = torch.zeros_like(image_feature)

        if cap is not None and label is not None:
            cap_feature_flag = True
            cap_t = []
            for cap_i in cap:
                cap_i = str(cap_i).replace("_", " ")
                prompt_i = clip.tokenize(cap_i)
                cap_t.append(prompt_i)

            cap_tokenized_prompts = torch.cat([p for p in cap_t]).to(self.model.logit_scale.device)

            with torch.no_grad():
                cap_embedding = self.model.token_embedding(cap_tokenized_prompts).type(self.model.dtype)
            
            # prompt = clip.tokenize(cap)
            # with torch.no_grad():
            #       embedding = self.model.token_embedding(prompt.to('cpu')).type(self.model.dtype)
            cap_feature = cap_encoder(cap_embedding,cap_tokenized_prompts)
            # cap_feature = cap_feature / cap_feature.norm(dim=-1, keepdim=True)
        else:
            cap_feature = torch.zeros_like(image_feature)

        if audio is not None and label is not None:
            audio_feature_flag = True
            audio_feature = audio_encoder(audio)
        else:
            audio_feature = torch.zeros_like(image_feature)

        audio_feature = self.transshape(audio_feature)


        if cap_feature != None and image_feature !=None:
            cap_feature = cap_feature.to(image_feature.device)

        model_num = 0
        feature = torch.stack([image_feature, cap_feature, audio_feature],dim=0)

        if image_feature_flag and cap_feature_flag and audio_feature_flag:
            valid = [1,1,1]
            model_num = 3
        elif image_feature_flag and cap_feature_flag:
            valid = [1,1,0]
            model_num = 2
        elif image_feature_flag and audio_feature_flag:
            valid = [1,0,1]
            model_num = 2
        elif audio_feature_flag and cap_feature_flag:
            valid = [0,1,1]
            model_num = 2
        elif audio_feature_flag:
            valid = [0,0,1]
            model_num = 1
        elif cap_feature_flag:
            valid = [0,1,0]
            model_num = 1
        elif image_feature_flag:
            valid = [1,0,0]
            model_num = 1
        else:
            raise ValueError("audio_feature, image_feature and cap_feature are None")
        
        feature_ori = feature
        valid = torch.tensor(valid)
        valid = valid.repeat(b,1).to(feature.device)
        mask = (1.0 - valid.unsqueeze(1)) * -1000000.0
        mask = mask.expand(-1, valid.size(1), -1).to(feature.device)
        # feature = self.cross_transformerClip(feature, mask) + feature_ori
        
        image_feature = image_feature.unsqueeze(0).repeat(3,1,1)
        feature = b_transformerClip(image_feature,feature_ori) + feature_ori
        feature = feature.permute(1, 0, 2)  # NLD -> LND 调整visual_output的维度顺序 变为8 3 512
        feature = self._mean_pooling_for_similarity_visual(feature, valid)

        feature = feature / feature.norm(dim=-1, keepdim=True)
        logits = self.logit_head(feature)

        loss = None

        if label != None:
            loss = F.cross_entropy(logits, label)
            
        return loss,logits,model_num,image_part

class AICLIP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        classnames = {"Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"}
        # classnames = {'anger','disgust','fear','happiness','neutral','sadness','surprise', #change
        #               'contempt','anxiety','helplessness','disappointment'}
        
        self.war = torchmetrics.Accuracy(task='multiclass', num_classes=7)
        self.recall = torchmetrics.Recall(task='multiclass',average='none', num_classes=7)
        clip_model, preprocess = clip.load(clip_name,device = self.device)

        self.model = CustomCLIP(classnames, clip_model)
        # print(self.model)
        # name_to_update = ["prompt_learner",]
        name_to_update = ["f_transformerClip","o_transformerClip","p_transformerClip","f_pencoding","o_pencoding","p_pencoding","seq_transformerClip","cross_transformerClip",'logit_head',"a_transformerClip","b_transformerClip", "transshape"]
        
        
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
            for targ in name_to_update:
                if targ in name:
                    param.requires_grad_(True)

        # # 遍历模型中的所有参数
        # print("not train param:")
        # for name, param in self.model.named_parameters():
        #     # 检查参数是否可训练
        #     if not param.requires_grad:
        #         # 打印参数的名称和值
        #         print(name)
        
    @property
    def learning_rate(self):
        # 获取优化器
        optimizer = self.optimizers()
        # 返回第一个参数组的学习率
        return optimizer.param_groups[0]["lr"]

    def parse_batch_train(self, batch):
        f_frames_3, o_frames_3, daudio, dtext, label, valid_list_o, valid_list_f = batch
        return f_frames_3, o_frames_3, daudio, dtext, label, valid_list_o, valid_list_f
    
    def forward(self,batch):
        #self, f_frames, o_frames, p_frames, cap, audio = None, label = None, valid_list = None, 
        audio = None
        cap = None
        f_frames = None
        o_frames = None
        p_frames = None
        label = None
        valid_list_f = None
        valid_list_o = None
        valid_list_p = None

        # f_frames, o_frames, p_frames, cap, label, valid_list = self.parse_batch_train(batch) #change
        f_frames, o_frames, audio, cap, label, valid_list_o, valid_list_f = self.parse_batch_train(batch)

        loss,logits,model_num,image_part = self.model(f_frames, o_frames, p_frames, cap, audio, label, valid_list_o, valid_list_f, valid_list_p)
        preds = torch.argmax(logits, dim=1)
        acc = self.war(preds, label)
        self.recall.update(preds, label)
        recall_per_class = self.recall.compute()
        uar = torch.mean(recall_per_class)
        return loss, preds, acc, uar

    def training_step(self, batch, batch_nb):
        loss, preds, war, uar = self.forward(batch)
        
        self.log('train_loss', loss, prog_bar=True,sync_dist=True)
        self.log('train_war', war, prog_bar=True, sync_dist=True)
        self.log('train_uar', uar, prog_bar=True, sync_dist=True)
        self.log("learning_rate", self.learning_rate, prog_bar=True, sync_dist=True)
        wandb.log({"train_loss": loss, "train_war": war,"train_uar": uar, "learning_rate":self.learning_rate})

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, preds, war, uar = self.forward(batch)
        self.log('val_loss', loss, prog_bar=True,sync_dist=True)
        self.log('val_war', war, prog_bar=True, sync_dist=True)
        self.log('val_uar', uar, prog_bar=True, sync_dist=True)
        wandb.log({"val_loss": loss, "val_war": war,"val_uar": uar})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
        self.parameters(), lr=lr, weight_decay=weight_decay, betas=ADAM_BETAS
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.001)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def test_step(self, batch, batch_idx):
        f_frames, o_frames, audio, cap, label, valid_list_o, valid_list_f = self.parse_batch_train(batch)
        loss, preds, war, uar = self.forward(batch)

        # 转换为numpy数组
        y_pred = preds.cpu().numpy()
        y_true = label.cpu().numpy()
        # 获取每个类别的TP、FP和FN
        report = classification_report(y_true, y_pred, output_dict=True)
        # 初始化WAR和UAR
        war_0, war_1, war_2, war_3, war_4, war_5, war_6 = 0, 0, 0, 0, 0, 0, 0

        if '0' in report:
            war_0 = report['0']['recall']
        if '1' in report:
            war_1 = report['1']['recall']
        if '2' in report:
            war_2 = report['2']['recall']
        if '3' in report:
            war_3 = report['3']['recall']
        if '4' in report:
            war_4 = report['4']['recall']
        if '5' in report:
            war_5 = report['5']['recall']
        if '6' in report:
            war_6 = report['6']['recall']

        self.log('test_rec_0', war_0, prog_bar=True,sync_dist=True)
        self.log('test_rec_1', war_1, prog_bar=True,sync_dist=True)
        self.log('test_rec_2', war_2, prog_bar=True,sync_dist=True)
        self.log('test_rec_3', war_3, prog_bar=True,sync_dist=True)
        self.log('test_rec_4', war_4, prog_bar=True,sync_dist=True)
        self.log('test_rec_5', war_5, prog_bar=True,sync_dist=True)
        self.log('test_rec_6', war_6, prog_bar=True,sync_dist=True)
        
        
        self.log('test_loss', loss, prog_bar=True,sync_dist=True)
        self.log('test_war', war, prog_bar=True, sync_dist=True)
        self.log('test_uar', uar, prog_bar=True, sync_dist=True)
        wandb.log({"test_war_0": war_0, "test_war_1": war_1,"test_war_2": war_2,'test_war_3':war_3,'test_war_4':war_4,'test_war_5':war_5, 'test_war_6': war_6,'test_war':war,'test_uar': uar})

