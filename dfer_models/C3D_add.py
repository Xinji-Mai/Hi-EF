#face only clip add
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

lr = 0.0002
# lr = 0.01
weight_decay = 0.000001
ADAM_BETAS = (0.9, 0.999)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# def confusion_max(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""

#     confusion_matrix = np.zeros([7, 7])
#     maxk = max(topk)
#     batch_size = target.size(0)
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.view(-1)
#     for i in range(batch_size):
#         # print('target[i],pred[i]',target[i],pred[i])
#         # print('target[i].cpu().numpy()',target[i].cpu().numpy())
#         # print('pred[i].cpu().numpy()',pred[i].cpu().numpy())
#         confusion_matrix[target[i].cpu().numpy(),pred[i].cpu().numpy()] += 1
#     return confusion_matrix

class baseC3D(nn.Module):
    """
    The C3D network.
    """
    def __init__(self, num_classes=7):
        super(baseC3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.war = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.recall = torchmetrics.Recall(task='multiclass',average='none', num_classes=7)

        self.__init_weight()

        

        # if pretrained:
        #     self.__load_pretrained_weights()

    def forward(self, f_frames):

        # f_i = torch.split(f_frames,1,dim=0)
        # for ii in f_i:
        #     ii = torch.squeeze(ii)
        #     save_image(ii, '/home/et23-maixj/mxj/SIRV_baseline/icache/0.jpg')
        
        x = f_frames
        x = x.permute(0,2,1,3,4)# [1, 3, 16, 112, 112]

        x = self.relu(self.conv1(x))
        x = self.pool1(x)    # [1, 64, 16, 56, 56]

        x = self.relu(self.conv2(x))
        x = self.pool2(x)   # [1, 128, 8, 28, 28]

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x) # [1, 256, 4, 14, 14]

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x) # [1, 512, 2, 7, 7]

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x) # [1, 512, 1, 4, 4]

        # print('x.shape', x.shape)


        x = x.view(-1, 8192)


        return x
    
    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    

class C3D(pl.LightningModule):

    def __init__(self, num_classes=7):
        super(C3D, self).__init__()
        self.confusion_matrix = np.zeros([7,7])
        self.model = baseC3D()
        self.fc8 = nn.Linear(4096, num_classes)
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.war = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.recall = torchmetrics.Recall(task='multiclass',average='none', num_classes=7)


    def forward(self, batch):
        f_frames_1,f_frames_2,f_frames_3,label = self.parse_batch_train(batch)
        feature_1 = self.model(f_frames_1)
        feature_2 = self.model(f_frames_2)
        feature_3 = self.model(f_frames_3)
        
        x = feature_1 + feature_2 + feature_3

        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)

        # avg_prec = accuracy(logits, label, topk=(1,))
        # top1 = AverageMeter()
        # top1.update(avg_prec[0], f_frames.size(0))
        # confusion_matrix = confusion_max(logits, label, topk=(1,))

        loss = None
        uar = None
        acc = None
        preds = None
        preds = torch.argmax(logits, dim=1)

        if label != None:
            loss = F.cross_entropy(logits, label)
            acc = self.war(preds, label)
            self.recall.update(preds, label)
            recall_per_class = self.recall.compute()
            uar = torch.mean(recall_per_class)
            # print(label)
            # print(preds)


        return loss, preds, acc, uar


    @property
    def learning_rate(self):
        # 获取优化器
        optimizer = self.optimizers()
        # 返回第一个参数组的学习率
        return optimizer.param_groups[0]["lr"]

    def parse_batch_train(self, batch):
        f_frames_1,f_frames_2,f_frames_3,label = batch
        return f_frames_1,f_frames_2,f_frames_3,label
    
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, threshold=0.001)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def on_test_epoch_end(self) -> None:
        print(self.confusion_matrix)
            # 'FERV39k'  Heatmap
        labels_7 = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

        test_number_7 = [1487, 467, 431, 1473, 1393, 638, 1958]
        name_7 = 'FERV39K (7 classes)'
        display_cls_confusion_matrix(self.confusion_matrix, labels_7, test_number_7, name_7,'c3d_add')

        return super().on_test_epoch_end()

    def test_step(self, batch, batch_idx):
        f_frames_1,f_frames_2,f_frames_3,label = self.parse_batch_train(batch)
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


    

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == "__main__":
    USE_CUDA = torch.cuda.is_available()
    device_ids_parallel = [0]
    device = torch.device("cuda:{}".format(device_ids_parallel[0]) if USE_CUDA else "cpu")
    # device ='cpu'
    model = C3D(num_classes=7)
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids_parallel)

    print('net #', count_parameters(model))
    # x = torch.rand(2, 3, 224, 224, 16).to(device)
    # x = torch.rand(1, 3, 224, 224, 16).to(device)
    for i in range(1000):
        # x = torch.rand(32, 3, 16, 112, 112).to(device)
        x = torch.rand(32, 3, 112, 168, 8).to(device)

        pred_score = model(x)
        print(pred_score.size())


    # inputs = torch.rand(1, 3, 16, 112, 112)
    # net = C3D(num_classes=101)
    #
    # outputs = net.forward(inputs)
    # print(outputs.size())