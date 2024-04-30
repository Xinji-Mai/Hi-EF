#face only
'''
Aum Sri Sai Ram

Resnet50 models

'''
import torch.nn as nn
import math,os
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from collections import OrderedDict
import math
import torchmetrics
import os
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.nn.modules.utils import _triple
import pytorch_lightning as pl
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from dfer_models.ConfusionMatrix import display_cls_confusion_matrix
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
lr = 0.0002
# lr = 0.01
weight_decay = 0.000001
ADAM_BETAS = (0.9, 0.999)


# from models.LSTM import Rnn
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class Rnn(nn.Module):
	def __init__(self, in_dim, hidden_dim, n_layer):
		super(Rnn, self).__init__()
		self.n_layer = n_layer
		self.hidden_dim = hidden_dim
		self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
	def forward(self,x):
		out, _ = self.lstm(x)
		out = out[:,-1,:]
		return out



__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# def channelReshape(out):
#     conv1 = nn.Conv2d((out.shape)[1], 128, kernel_size=1, stride=1, bias=False)
#     out = conv1(out)
#     return out


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def norm_angle(angle):
    norm_angle = sigmoid(10 * (abs(angle) / 0.7853975 - 1))
    return norm_angle

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class baseResNet(nn.Module):

    def __init__(self, block, layers, num_classes=7, end2end=True, at_type=''):
        self.inplanes = 64
        self.end2end = end2end
        super(baseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Rnn = Rnn(512, 512, 3)
        for param in self.parameters():
            param.requires_grad = True

        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.6)
        self.alpha = nn.Sequential(nn.Linear(512, 1),
                                   nn.Sigmoid())

        self.beta = nn.Sequential(nn.Linear(1024, 1),
                                  nn.Sigmoid())

               
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, f_frames):

        input = f_frames
        input = input.permute(0, 2, 1, 3, 4)
        b, c, h, w, s = input.size()

        vs = []
        alphas = []

        for i in range(1, s, 2):
            x = input[:, :, :, :, i]
            ##stem layer
            out_1_1 = self.relu(self.bn1(self.conv1(x)))  # bs,112,112,64
            out_1 = self.maxpool(out_1_1)  # bs,56,56,64

            ##layers:
            out_2_3 = self.layer1(out_1)  # bs,64, 56,56,
            out_3_4 = self.layer2(out_2_3)  # bs,128, 28,28
            out_4_6 = self.layer3(out_3_4)  # bs,256, 14,14
            out_5_3 = self.layer4(out_4_6)  # bs,512, 7,7
            f = self.avgpool(out_5_3)
            f = f.squeeze(3).squeeze(2)   # f[1, 512, 1, 1] ---> f[1, 512]
            vs.append(f)

        vs_stack = torch.stack(vs, dim=2)
        vs_stack = vs_stack.permute(0,2,1)
        # print('vs_stack.shape',vs_stack.shape)
        out = self.Rnn(vs_stack)
        # print('out.shape',out.shape)
        
        return out

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

class TransformerClip(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return self.resblocks((x, attn_mask))[0]

class ResNet(pl.LightningModule):

    def __init__(self, block, layers, num_classes=7, end2end=True, at_type=''):
        self.inplanes = 64
        self.end2end = end2end
        super(ResNet, self).__init__()
        self.confusion_matrix = np.zeros([7,7])
        self.model_f = baseResNet(block, layers, num_classes, end2end, at_type)
        self.model_o = baseResNet(block, layers, num_classes, end2end, at_type)
        self.model_p = baseResNet(block, layers, num_classes, end2end, at_type)

        self.beta = nn.Sequential(nn.Linear(1024, 1),
                                  nn.Sigmoid())

        self.pred_fc1 = nn.Linear(512, num_classes)
        self.pred_fc2 = nn.Linear(1024, num_classes)

        self.war = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.recall = torchmetrics.Recall(task='multiclass',average='none', num_classes=7)
        self.transformerClip = TransformerClip(width=512, layers=12,
                                        heads=8, )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def forward(self, batch):
        f_frames_1, o_frames_1, p_frames_1, f_frames_2, o_frames_2, p_frames_2, f_frames_3, o_frames_3, p_frames_3, label_3, label_4 = self.parse_batch_train(batch)

        feature_1_f = self.model_f(f_frames_1)
        feature_2_f = self.model_f(f_frames_2)
        feature_3_f = self.model_f(f_frames_3)
        feature_1_o = self.model_o(o_frames_1)
        feature_2_o = self.model_o(o_frames_2)
        feature_3_o = self.model_o(o_frames_3)
        feature_1_p = self.model_p(p_frames_1)
        feature_2_p = self.model_p(p_frames_2)
        feature_3_p = self.model_p(p_frames_3)
        
        b,f = feature_1_f.shape

        valid = [1,1,1]
        valid = torch.tensor(valid)
        valid = valid.repeat(b,1).to(feature_1_f.device)
        mask = (1.0 - valid.unsqueeze(1)) * -1000000.0
        mask = mask.expand(-1, valid.size(1), -1).to(feature_1_f.device)

        image_feature_1 = torch.stack([feature_1_f, feature_1_o, feature_1_p], dim=0)
        image_feature_1 = self.transformerClip(image_feature_1,mask) + image_feature_1
        image_feature_1 = image_feature_1.permute(1, 0, 2)
        image_feature_1 = self._mean_pooling_for_similarity_visual(image_feature_1, valid) #变为8 512

        image_feature_2 = torch.stack([feature_2_f, feature_2_o, feature_2_p], dim=0)
        image_feature_2 = self.transformerClip(image_feature_2,mask) + image_feature_2
        image_feature_2 = image_feature_2.permute(1, 0, 2)
        image_feature_2 = self._mean_pooling_for_similarity_visual(image_feature_2, valid) #变为8 512


        image_feature_3 = torch.stack([feature_3_f, feature_3_o, feature_3_p], dim=0)
        image_feature_3 = self.transformerClip(image_feature_3,mask) + image_feature_3
        image_feature_3 = image_feature_3.permute(1, 0, 2)
        image_feature_3 = self._mean_pooling_for_similarity_visual(image_feature_3, valid) #变为8 512

        out = image_feature_1 + image_feature_2 + image_feature_3

        logits = self.pred_fc1(out)

        loss = None

        if label_4 != None:
            loss = F.cross_entropy(logits, label_4)
        preds = torch.argmax(logits, dim=1)
        acc = self.war(preds, label_4)
        self.recall.update(preds, label_4)
        recall_per_class = self.recall.compute()
        uar = torch.mean(recall_per_class)
        
        return loss, preds, acc, uar
    
    def _mean_pooling_for_similarity_visual(self, visual_output, valid_list,):
        video_mask_un = valid_list.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out
    
    @property
    def learning_rate(self):
        # 获取优化器
        optimizer = self.optimizers()
        # 返回第一个参数组的学习率
        return optimizer.param_groups[0]["lr"]

    def parse_batch_train(self, batch):
        f_frames_1, o_frames_1, p_frames_1, f_frames_2, o_frames_2, p_frames_2, f_frames_3, o_frames_3, p_frames_3, label_3, label_4 = batch
        return f_frames_1, o_frames_1, p_frames_1, f_frames_2, o_frames_2, p_frames_2, f_frames_3, o_frames_3, p_frames_3, label_3, label_4
    
    
    def training_step(self, batch, batch_nb):
        loss, preds, war, uar = self.forward(batch)
        
        self.log('train_loss', loss, prog_bar=True,sync_dist=True)
        self.log('train_war', war, prog_bar=True, sync_dist=True)
        self.log('train_uar', uar, prog_bar=True, sync_dist=True)
        self.log("learning_rate", self.learning_rate, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, preds, war, uar = self.forward(batch)
        self.log('val_loss', loss, prog_bar=True,sync_dist=True)
        self.log('val_war', war, prog_bar=True, sync_dist=True)
        self.log('val_uar', uar, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
        self.parameters(), lr=lr, weight_decay=weight_decay, betas=ADAM_BETAS
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.001)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def on_test_epoch_end(self) -> None:
        print(self.confusion_matrix)
            # 'FERV39k'  Heatmap
        labels_7 = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

        test_number_7 = [1487, 467, 431, 1473, 1393, 638, 1958]
        name_7 = 'FERV39K (7 classes)'
        display_cls_confusion_matrix(self.confusion_matrix, labels_7, test_number_7, name_7,'r18lstm_trans_add')

        return super().on_test_epoch_end()
    
    def test_step(self, batch, batch_idx):
        f_frames_1, o_frames_1, p_frames_1, f_frames_2, o_frames_2, p_frames_2, f_frames_3, o_frames_3, p_frames_3, label_3, label_4 = self.parse_batch_train(batch)
        loss, preds, war, uar = self.forward(batch)

        # 转换为numpy数组
        y_pred = preds.cpu().numpy()
        y_true = label_4.cpu().numpy()
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
        uar = report['macro avg']['recall']
        war = report['accuracy']

        batch_size = len(y_pred)
        for i in range(batch_size):
            self.confusion_matrix[y_true[i],y_pred[i]] += 1

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
        self.log('test_uar', uar, prog_bar=True, sync_dist=True)

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def main():
    USE_CUDA = torch.cuda.is_available()
    device_ids_parallel = [0,1,2,3]
    device = torch.device("cuda:{}".format(device_ids_parallel[0]) if USE_CUDA else "cpu")
    # device ='cpu'
    model = resnet18(end2end=True, num_classes=7)
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids_parallel)
    print('net #', count_parameters(model))


    for i in range(1000):
    # x = torch.rand(32, 3, 16, 112, 112).to(device)
        x = torch.rand(32, 3, 112, 112, 8).to(device)

        pred_score = model(x)

        print('pred_score.size()', pred_score.size())

if __name__=='__main__':
    main()


