import torch
import argparse
from torch.utils.data import DataLoader
import torchvision as tv #noqa
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
from pytorch_lightning import loggers

# from dfer_models.C3D import C3D #c3d
# from dfer_models.R3D_r18 import r3d_18 #Res18 + c3d
# from dfer_models.I3D import InceptionI3d #i3d
from dfer_models.P3D import P3D199
# from dfer_models.vgg16_lstm import vgg16 #vgg16 + lstm
# from dfer_models.Two_VGG13_LSTM import TwoVGG13LSTM

# from model_39k import AICLIP

# model = AICLIP()

# model = C3D(num_classes=7)
# model = r3d_18()
# model = InceptionI3d(num_classes=7)

model = P3D199(num_classes=7)
# model = vgg16(num_classes=7)
# model = TwoVGG13LSTM(num_classes=7)

from data_prepro_39k import data_prepro


devices = [0]



os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4"
USE_CUDA = torch.cuda.is_available()
device_ids_parallel = [0]
device = torch.device("cuda:{}".format(device_ids_parallel[0]) if USE_CUDA else "cpu")


checkpoint = torch.load("sec_checkpoint/p3d.ckpt", map_location = device) # 这里假设您的checkpoint文件名是best-checkpoint.ckpt，您可以根据您的实际文件名来修改 14ep

# 将模型的参数加载到您的模型中
model.load_state_dict(checkpoint["state_dict"])

num_workers = 2
train_ratio, valid_ratio, test_ratio = 0.7, 0.15, 0.15
BATCH_TRAIN = 32
BATCH_TEST = 32
WORKERS_TRAIN = 2
WORKERS_TEST = 2
EPOCHS = 60
LOG_INTERVAL = 50
SAVED_MODELS_PATH = os.path.join(os.path.expanduser('~'), 'saved_models')
# 设置矩阵乘法的精度为中等或高等
torch.set_float32_matmul_precision('medium')

def arg_selector(arg_cmd: Optional[Any], arg_conf: Optional[Any], arg_const: Any) -> Any:
    if arg_cmd is not None:
        return arg_cmd
    else:
        if arg_conf is not None:
            return arg_conf
        else:
            return arg_const

def main(args):

    if args.random_seed >= 0:  # 固定随机种子
        print("Setting fixed seed: {}".format(args.random_seed))
        args.suffix = '{}r-{}'.format(
            '{}_'.format(args.suffix) if args.suffix is not None else '',
            args.random_seed
        )
    if args.batch_test is None:
        args.batch_test = args.batch_train

    configs_found = list(sorted(glob.glob(os.path.expanduser(args.config))))
    for config_path in configs_found:
        config = json.load(open(config_path))
        config = defaultdict(None, config)

        experiment_name = config['Setup']['name']

        # batch_train = int(arg_selector(
        #     args.batch_train, config['Setup']['batch_train'], BATCH_TRAIN
        # ))
        batch_test = int(arg_selector(
            args.batch_test, config['Setup']['batch_test'], BATCH_TEST
        ))
        workers_train = arg_selector(
            args.workers_train, config['Setup']['workers_train'], WORKERS_TRAIN
        )
        workers_test = arg_selector(
            args.workers_test, config['Setup']['workers_test'], WORKERS_TEST
        )
        epochs = arg_selector(
            args.epochs, config['Setup']['epochs'], EPOCHS
        )
        log_interval = arg_selector(
            args.log_interval, config['Setup']['log_interval'], LOG_INTERVAL
        )
        saved_models_path = arg_selector(
            args.saved_models_path, config['Setup']['saved_models_path'], SAVED_MODELS_PATH
        )

        model_class = config['Model']['class']
        model_args = config['Model']['args']

        if 'Scheduler' in config:
            scheduler_class = config['Scheduler']['class']
            scheduler_args = config['Scheduler']['args']
        else:
            scheduler_class = None
            scheduler_args = None

        dataset_class = config['Dataset']['class']
        dataset_args = config['Dataset']['args']

        run(
            experiment_name=experiment_name,
            model_class=model_class,
            model_args=model_args,
            dataset_class=dataset_class,
            dataset_args=dataset_args,
            batch_train=BATCH_TRAIN,
            batch_test=BATCH_TEST,
            workers_train=workers_train,
            workers_test=workers_test,
            epochs=epochs,
            log_interval=log_interval,
            saved_models_path=saved_models_path,
            scheduler_class=scheduler_class,
            scheduler_args=scheduler_args,
            model_suffix=config['Setup']['suffix'],
            setup_suffix=args.suffix,
        )


def run(experiment_name: str,
        model_class: str,
        model_args: Dict[str, Any],
        dataset_class: str,
        dataset_args: Dict[str, Any],
        batch_train: int,
        batch_test: int,
        workers_train: int,
        workers_test: int,
        epochs: int,
        log_interval: int,
        saved_models_path: str,
        scheduler_class: Optional[str] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        model_suffix: Optional[str] = None,
        setup_suffix: Optional[str] = None,
        skip_train_val: bool = False
        ):

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        experiment_name = f'{experiment_name}-trainOn{num_gpus}'
    
    # # Set up the data loader
    dataset = data_prepro(root=dataset_args['root'])
    # text_dataset = text_data_prepro(root = dataset_args['root'])
    len_dataset = dataset.__len__()
    print('len_dataset:', len_dataset)

    train_len = int(train_ratio * len_dataset)  # 训练集长度
    valid_len = math.ceil(valid_ratio * len_dataset)  # 验证集长度
    test_len = len_dataset - train_len - valid_len  # 测试集长度

    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths=[
                                                            train_len, valid_len, test_len], generator=torch.Generator().manual_seed(0))

    batch_train_size = batch_train  # 定义批次大小
    batch_test_size = batch_test  # 定义批次大小
    shuffle = True  # 定义是否打乱数据
    train_loader = DataLoader(
        train_dataset, batch_size=batch_train_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_train_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_test_size, shuffle=False, num_workers=num_workers)
    # 创建DataLoader实例，传入dataset实例和其他参数

    checkpoint_callback = ModelCheckpoint(
        dirpath="sec_checkpoint",
        filename="p3d",
        save_top_k=1,  # 保存最好的1个checkpoint
        verbose=True,
        monitor="val_loss",  # 根据验证集损失来判断最好的checkpoint
        mode="min"  # 最小化验证集损失
    )

    
    trainer = Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=devices,
        callbacks=[checkpoint_callback],  # 是否启用checkpoint回调
        strategy='ddp_find_unused_parameters_true'
    )

    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=val_loader,ckpt_path='sec_checkpoint/p3d.ckpt')
    trainer.test(model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default='protocols/aiclip-data_prepro.json', required=False)
    parser.add_argument('-b', '--batch-train', type=int, required=False)
    parser.add_argument('-B', '--batch-test', type=int, required=False)
    parser.add_argument('-w', '--workers-train', type=int, required=False)
    parser.add_argument('-W', '--workers-test', type=int, required=False)
    parser.add_argument('-e', '--epochs', type=int, required=False)
    parser.add_argument('-R', '--random-seed', type=int,
                        default=-1, required=False)
    parser.add_argument('-s', '--suffix', type=str, required=False)
    parser.add_argument('-L', '--log-interval', type=int, required=False)
    parser.add_argument('-M', '--saved-models-path', type=str, required=False)
    args = parser.parse_args()
    main(args)
