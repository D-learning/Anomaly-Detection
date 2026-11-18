#!/usr/bin/python
# -*- coding: utf-8 -*-
from textwrap import indent
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import json
from collections import OrderedDict
from PIL import Image
import csv
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import ImageFolder
import copy
from utils import *
import datasets.toyset as toyset
from torch.utils.data import Subset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


OBJ_DICT = {'mvtec':['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',  'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
            'loco':['breakfast_box', 'juice_bottle', 'pushpins',  'screw_bag', 'splicing_connectors'],
            'btad':['01', '02', '03'],
            'visa':['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'],
            'mpdd':['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']}


_DATASETS = {
    "mvtec": ["datasets.mvtec", "mvtecDataset"],
    "loco": ["datasets.loco", "LocoDataset"],
    "btad": ["datasets.btad", "BTADDataset"],
    "visa": ["datasets.visa", "VISADataset"],
    "realiad": ["datasets.realiad", "RealIADDataset"],
    "mpdd": ["datasets.mpdd", "MPDDDataset"]
}

PATH_DICT = {'mvtec': '/mnt/c/Users/weiya/mvtec_anomaly_detection',
                'loco': '/home/yoon/data/mvtec_loco',
                'btad': '/home/yoon/data/btad',
                'visa': '/mnt/c/Users/weiya/VisA',
                'realiad': '/home/yoon/data/Real-IAD',
                'mpdd': '/home/yoon/data/mpdd'
                }    


def get_dataloaders(config):

    dataset_info = _DATASETS[config.dataset]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])
    dataset_path = getattr(config, "data_path", None) or PATH_DICT[config.dataset]

    obj_list = OBJ_DICT[config.dataset]

    train_dataset_list = []
    val_dataset_list = []
    test_dataset_list = []

    for obj in obj_list:
        full_train_dataset = dataset_library.__dict__[dataset_info[1]](
            dataset_path,
            classname=obj,
            resize=config.resize,
            train_val_split=config.train_val_split,
            imagesize=config.image_size,
            split=dataset_library.DatasetSplit.TRAIN,
            seed=config.seed,
            augment=config.augment,
        )

        train_size = int((1-config.val_ratio)*len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        rng = torch.Generator().manual_seed(config.seed)
        train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset,[train_size,val_size],rng)



        test_dataset = dataset_library.__dict__[dataset_info[1]](
            dataset_path,
            classname=obj,
            resize=config.resize,
            imagesize=config.image_size,
            split=dataset_library.DatasetSplit.TEST,
            seed=config.seed,
        )

        train_dataset_list.append(train_dataset)
        val_dataset_list.append(val_dataset)
        test_dataset_list.append(test_dataset)


    trainloader_list = []
    valloader_list = []

    offset_idx = 0
    for num_cls in config.incremental_class_nums:
        get_train_list = ConcatDataset(train_dataset_list[offset_idx:offset_idx+num_cls])
        get_val_list = ConcatDataset(val_dataset_list[offset_idx:offset_idx+num_cls])

        train_dataloader = torch.utils.data.DataLoader(
                get_train_list,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=True,
            )
        val_dataloader = torch.utils.data.DataLoader(
                    get_val_list,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=config.num_workers,
                    pin_memory=True,
                )

        trainloader_list.append(train_dataloader)
        valloader_list.append(val_dataloader)
        offset_idx += num_cls


    get_test_list = ConcatDataset(test_dataset_list)
    test_dataloader = torch.utils.data.DataLoader(
        get_test_list,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        )


    return trainloader_list, valloader_list, test_dataloader




def get_realiad_datasets(config):
    import datasets.realiad as realiad
    import yaml

    full_trainset_list = []
    full_testset_list = []

    with open('./datasets/configs/fuiad_0.0_config.yaml', "r") as file:
        dset_config = yaml.safe_load(file)['dataset']

    train_meta_files = dset_config['train']['meta_file']
    test_meta_files = dset_config['test']['meta_file']

    for train_metas in train_meta_files:
        train_dataset = realiad.RealIADDataset(PATH_DICT['realiad'], config, train_metas, training=True)
        full_trainset_list.append(train_dataset)
    for test_metas in test_meta_files:
        test_dataset = realiad.RealIADDataset(PATH_DICT['realiad'], config, test_metas, training=False)
        full_testset_list.append(test_dataset)


    trainset_list = []
    testset_list = []

    offset_idx = 0
    for num_cls in config.incremental_class_nums:
        trainset_list.append(ConcatDataset(full_trainset_list[offset_idx:offset_idx+num_cls]))
        testset_list.append(ConcatDataset(full_testset_list[offset_idx:offset_idx+num_cls]))
        offset_idx += num_cls


    return trainset_list, testset_list




def get_datasets(config):

    dataset_info = _DATASETS[config.dataset]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])
    dataset_path = getattr(config, "data_path", None) or PATH_DICT[config.dataset]

    obj_list = OBJ_DICT[config.dataset]

    full_trainset_list = []
    full_testset_list = []

    for obj in obj_list:
        train_dataset = dataset_library.__dict__[dataset_info[1]](
            dataset_path,
            classname=obj,
            resize=config.resize,
            train_val_split=1.0,
            imagesize=config.image_size,
            split=dataset_library.DatasetSplit.TRAIN,
            seed=config.seed,
            augment=config.augment,
        )

        test_dataset = dataset_library.__dict__[dataset_info[1]](
            dataset_path,
            classname=obj,
            resize=config.resize,
            imagesize=config.image_size,
            split=dataset_library.DatasetSplit.TEST,
            seed=config.seed,
        )

        full_trainset_list.append(train_dataset)
        full_testset_list.append(test_dataset)

    trainset_list = []
    testset_list = []

    offset_idx = 0
    for num_cls in config.incremental_class_nums:
        trainset_list.append(ConcatDataset(full_trainset_list[offset_idx:offset_idx+num_cls]))
        testset_list.append(ConcatDataset(full_testset_list[offset_idx:offset_idx+num_cls]))
        offset_idx += num_cls

    return trainset_list, testset_list



def get_task_shuffled_datasets(config):

    dataset_info = _DATASETS[config.dataset]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])
    dataset_path = getattr(config, "data_path", None) or PATH_DICT[config.dataset]

    obj_list = OBJ_DICT[config.dataset]

    full_trainset_list = []
    full_testset_list = []

    for obj in obj_list:
        train_dataset = dataset_library.__dict__[dataset_info[1]](
            dataset_path,
            classname=obj,
            resize=config.resize,
            train_val_split=1.0,
            imagesize=config.image_size,
            split=dataset_library.DatasetSplit.TRAIN,
            seed=config.seed,
            augment=config.augment,
        )

        test_dataset = dataset_library.__dict__[dataset_info[1]](
            dataset_path,
            classname=obj,
            resize=config.resize,
            imagesize=config.image_size,
            split=dataset_library.DatasetSplit.TEST,
            seed=config.seed,
        )

        full_trainset_list.append(train_dataset)
        full_testset_list.append(test_dataset)

    trainset_list = []
    testset_list = []

    offset_idx = 0
    for num_cls in config.incremental_class_nums:
        task_trainset = ConcatDataset(full_trainset_list[offset_idx:offset_idx+num_cls])
        total_size = len(task_trainset)
        indices = list(range(total_size))
        random.shuffle(indices)

        idx_start = 0
        for _ in range(config.num_task_split-1):
            idx_end = idx_start + total_size // config.num_task_split
            train_idx = indices[idx_start:idx_end]
            trainset_list.append(Subset(task_trainset, train_idx))
            idx_start = idx_end
        train_idx = indices[idx_start:]
        trainset_list.append(Subset(task_trainset, train_idx))

        testset_list.append(ConcatDataset(full_testset_list[offset_idx:offset_idx+num_cls]))
        offset_idx += num_cls

    random.shuffle(trainset_list)
    return trainset_list, testset_list

def get_cross_datasets(config):

    tmp_config = copy.deepcopy(config)
    tmp_config.dataset = 'mvtec'
    tmp_config.incremental_class_list = '15-0-0'
    tmp_config.incremental_class_nums = [15]
    mvtec_train_list, mvtec_test_list = get_datasets(tmp_config)


    tmp_config = copy.deepcopy(config)
    tmp_config.dataset = 'visa'
    tmp_config.incremental_class_list = '12-0-0'
    tmp_config.incremental_class_nums = [12]
    visa_train_list, visa_test_list = get_datasets(tmp_config)

    if config.dataset == 'mvtec_visa':
        trainset_list = mvtec_train_list + visa_train_list
        testset_list = mvtec_test_list + visa_test_list
    else:
        trainset_list = visa_train_list + mvtec_train_list
        testset_list = visa_test_list + mvtec_test_list

    return trainset_list, testset_list



def get_onecls_dataset(config):

    dataset_info = _DATASETS[config.dataset]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])
    dataset_path = getattr(config, "data_path", None) or PATH_DICT[config.dataset]

    obj = config.object

    trainset = dataset_library.__dict__[dataset_info[1]](
        dataset_path,
        classname=obj,
        resize=config.resize,
        train_val_split=config.train_val_split,
        imagesize=config.image_size,
        split=dataset_library.DatasetSplit.TRAIN,
        seed=config.seed,
        augment=config.augment,
    )

    train_size = int((1-config.val_ratio)*len(trainset))
    val_size = len(trainset) - train_size
    rng = torch.Generator().manual_seed(config.seed)
    trainset, valset = torch.utils.data.random_split(trainset,[train_size,val_size],rng)


    testset = dataset_library.__dict__[dataset_info[1]](
        dataset_path,
        classname=obj,
        resize=config.resize,
        imagesize=config.image_size,
        split=dataset_library.DatasetSplit.TEST,
        seed=config.seed,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        )



    return trainset, valset, testloader





def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.0)
    parser.add_argument("--gpus", nargs='+', default=[0])



    #### model arguments
    parser.add_argument("--bbone_name", "-b", type=str, default='wideresnet50')
    parser.add_argument('--layers_to_extract_from', "-le", nargs='+', default=['layer2', 'layer3'])
    # Parameters feature embedder
    parser.add_argument("--pretrain_embed_dimension", type=int, default=1024)
    parser.add_argument("--target_embed_dimension", type=int, default=1024)
    parser.add_argument("--preprocessing", type=str, default="mean")
    parser.add_argument("--aggregation", type=str, default="mean")
    # Sampler params
    parser.add_argument("--max_coreset_size", type=int, default=10000)
    parser.add_argument("--greedy_ratio", type=float, default=0.1)
    parser.add_argument("--greedy_proj_dim", type=int, default=128)

    # Nearest-Neighbour Anomaly Scorer parameters.
    parser.add_argument("--anomaly_score_num_nn", type=int, default=1)
    # Patch-parameters.
    parser.add_argument("--patchsize", type=int, default=3)
    parser.add_argument("--patchstride", type=int, default=1)
    parser.add_argument("--patchscore", type=str, default="max")
    parser.add_argument("--patchoverlap", type=float, default=0.0)
    parser.add_argument("--patchsize_aggregate", "-pa", type=int, default=[])
    # NN on GPU.
    parser.add_argument("--faiss_on_gpu", type=str2bool, default='True')
    parser.add_argument("--faiss_num_workers", type=int, default=8)



    ## dataset arguments
    parser.add_argument('--dataset', type=str, default='mvtec',choices=['mvtec', 'loco','visa','btad'])
    parser.add_argument("--classes", nargs='+')
    parser.add_argument("--train_val_split", type=float, default=1)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--resize", default=256, type=int)
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--augment", type=str2bool, default='True')


    ## incremental arguments
    parser.add_argument('--incremental_class_list', type=str, default='14-1-1')


    return parser.parse_args()




if __name__ == '__main__':
    config = get_argparse()
    trainloader_list, valloader_list, testloader = get_dataloaders(config)

    for data in testloader:
        print(data[0]['mask'])


