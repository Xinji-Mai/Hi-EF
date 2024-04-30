import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
AVAI_HEADS = ['linear', 'adapter']

#跨模态适应
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4, residual_ratio=0.2):
        super(Adapter, self).__init__()
        self.residual_ratio = residual_ratio
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        a = self.fc(x)
        x = self.residual_ratio * a + (1 - self.residual_ratio) * x
        return x

def get_text_dataset_per_class(text_dataset):
    print("Building text dataset per class...")
    text_dataset_per_class = {}
    for text, text_label, eot_indices in tqdm(text_dataset):
        text_label = int(text_label)
        if text_label not in text_dataset_per_class:
            text_dataset_per_class[text_label] = []
        text_dataset_per_class[text_label].append([text, eot_indices])
    num_of_templates = len(text_dataset_per_class[text_label])
    for text_label in text_dataset_per_class:
        assert len(text_dataset_per_class[text_label]) == num_of_templates
    return text_dataset_per_class, num_of_templates

#获得文本数据中每个类别对应的文本特征
# def get_text_dataset_per_class(text_dataset):
#     print("Building text dataset per class...")
#     text_dataset_per_class = {}
#     for text, text_label in tqdm(text_dataset):
#         text_label = int(text_label)
#         if text_label not in text_dataset_per_class:
#             text_dataset_per_class[text_label] = []
#         text_dataset_per_class[text_label].append(text)
#     num_of_templates = len(text_dataset_per_class[text_label])
#     for text_label in text_dataset_per_class:
#         assert len(text_dataset_per_class[text_label]) == num_of_templates
#     return text_dataset_per_class, num_of_templates

#根据给定的文本数据集，计算每个类别对应的文本特征的平均值，并将其作为权重（weights）返回
# def get_zero_shot_weights(text_dataset, num_classes, in_features, text_encoder, device="cuda"):
#     with torch.no_grad():
#         text_dataset_per_class, _ = get_text_dataset_per_class(text_dataset)
#         weights = torch.zeros(num_classes, in_features)
#         for label in range(num_classes):
#             texts = None
#             # eot_indices = None
#             for i in range(len(text_dataset_per_class[label])):
#                 text = text_dataset_per_class[label][i]
#                 text = text.unsqueeze(0)
#                 # eot_indice = eot_indice.unsqueeze(0).to(device)
#                 if texts is None:
#                     texts = text
#                     # eot_indices = eot_indice
#                 else:
#                     texts = torch.cat([texts, text], dim=0)
#                     # eot_indices = torch.cat([eot_indices, eot_indice], dim=0)
#             text_features = text_encoder(texts)
#             text_features = text_features.mean(dim=0)
#             weights[label] = text_features
#         # normalize the weights
#         weights.data = torch.nn.functional.normalize(weights, dim=1)
#     return weights

def get_zero_shot_weights(text_dataset, num_classes, in_features, text_encoder, device="cuda"):
    with torch.no_grad():
        text_dataset_per_class, _ = get_text_dataset_per_class(text_dataset)
        weights = torch.zeros(num_classes, in_features)
        for label in range(num_classes):
            texts = None
            eot_indices = None
            for i in range(len(text_dataset_per_class[label])):
                text, eot_indice = text_dataset_per_class[label][i]
                text = text.unsqueeze(0)
                eot_indice = eot_indice.unsqueeze(0)
                if texts is None:
                    texts = text
                    eot_indices = eot_indice
                else:
                    texts = torch.cat([texts, text], dim=0)
                    eot_indices = torch.cat([eot_indices, eot_indice], dim=0)
            text_features = text_encoder(texts, eot_indices)
            text_features = text_features.mean(dim=0)
            weights[label] = text_features
        # normalize the weights
        weights.data = torch.nn.functional.normalize(weights, dim=1)
    return weights

#创建一个图像分类器的头部（head），用于对图像进行分类
def make_classifier_head(classifier_head,
                         clip_encoder,
                         classifier_init,
                         zeroshot_dataset,
                         text_encoder,
                         bias=False):
    assert classifier_head in AVAI_HEADS
    if clip_encoder == 'ViT-B/32':
        in_features = 512
    elif clip_encoder == 'RN50':
        in_features = 1024
    elif clip_encoder == 'ViT-L/14':
        in_features = 768

    num_classes = int(zeroshot_dataset.label_tensor.max()) + 1

    linear_head = nn.Linear(in_features, num_classes, bias=bias)
    if classifier_init == 'zeroshot':
        # assert zeroshot_dataset.input_tensor.shape[1] == in_features
        linear_head.weight.data = get_zero_shot_weights(
            zeroshot_dataset, num_classes, in_features, text_encoder)
    
    if classifier_head == 'linear':
        head = linear_head
    elif classifier_head == 'adapter':
        adapter = Adapter(in_features, residual_ratio=0.2)
        head = nn.Sequential(
            adapter,
            linear_head
        )
    else:
        raise ValueError(f"Invalid head: {classifier_head}")
    return head, num_classes, in_features