


from sklearn.metrics import classification_report
from dfer_models.ConfusionMatrix import display_cls_confusion_matrix
import math
import torchmetrics
import os
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.nn.modules.utils import _triple
import pytorch_lightning as pl

lr = 0.003
# lr = 0.01
weight_decay = 0.000001
ADAM_BETAS = (0.9, 0.999)

fea_size = 768
clip_name = 'ViT-B/32'

class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)


        self.temporal_spatial_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.bn(self.temporal_spatial_conv(x))
        x = self.relu(x)
        return x


class SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2when producing the residual
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)


class SpatioTemporalResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock,
                 downsample=False):

        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class R3DNet(pl.LightningModule):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.
        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """
    def __init__(self, layer_sizes, num_classes=7, block_type = SpatioTemporalResBlock):
        super(R3DNet, self).__init__()
        self.confusion_matrix = np.zeros([7,7])
        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv1 = SpatioTemporalConv(3, 64, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)
        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)
        # self.linear = nn.Linear(512, num_classes)
        self.dropout2 = nn.Dropout(0.6)
        self.linear = nn.Linear(512, 7)
        self.war = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.recall = torchmetrics.Recall(task='multiclass',average='none', num_classes=7)
        

    def forward(self, batch):
        f_frames, label = self.parse_batch_train(batch)
        x = f_frames
        x = x.permute(0, 1, 4, 2, 3)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # print('x.shape', x.shape)
        x = self.conv5(x)
        x = self.pool(x)

        x = x.view(-1, 512)
        x =  self.dropout2(x)
        logits = self.linear(x)

        loss = None

        if label != None:
            loss = F.cross_entropy(logits, label)
        preds = torch.argmax(logits, dim=1)
        acc = self.war(preds, label)
        self.recall.update(preds, label)
        recall_per_class = self.recall.compute()
        uar = torch.mean(recall_per_class)

        return loss, preds, acc, uar

    @property
    def learning_rate(self):
        # 获取优化器
        optimizer = self.optimizers()
        # 返回第一个参数组的学习率
        return optimizer.param_groups[0]["lr"]

    def parse_batch_train(self, batch):
        f_frames, label = batch
        return f_frames, label
    
    
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
        
        # print('model_num:',model_num)
        # print('image_part:',image_part)
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
        display_cls_confusion_matrix(self.confusion_matrix, labels_7, test_number_7, name_7,'r3dr18')

        return super().on_test_epoch_end()

    def test_step(self, batch, batch_idx):
        f_frames, label = self.parse_batch_train(batch)
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
    def __load_pretrained_weights(self):
        s_dict = self.state_dict()
        for name in s_dict:
            print(name)
            print(s_dict[name].size())

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for the conv layer of the net.
    """
    b = [model.res3d]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the fc layer of the net.
    """
    b = [model.linear]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k



def r3d_18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = R3DNet([2, 2, 2, 2], block_type = SpatioTemporalResBlock, **kwargs)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = R3DNet([3, 4, 6, 3],num_classes=7,block_type = SpatioTemporalResBlock, **kwargs)
    return model



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


