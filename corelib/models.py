
import torch
import torch.nn as nn
from typing import Optional
import corelib.backbones
import corelib.utils
import corelib.sampler
import torch.nn.functional as F
import numpy as np 
from torchvision import transforms
import os
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_recall_curve, \
    average_precision_score
import metrics
import corelib.contcore as contcore
import corelib.sampler 
import corelib



def get_contcore_model(
    backbone_name: str,
    layers_to_extract_from: list, 
    feature_dim: int,
    reduce_feature_dim: bool,
    sampling_threshold: int,
    chunk_size: int,
    image_size: int,
    
    sampling_ratio: float,
    coreset_size: int,
    project_dim: int,
    approximate_ratio: float,
    metric: str,

    save_path: str,
    config
    ):
    device = config.device
    if "dinov2" in backbone_name:
        feature_extractor = torch.nn.ModuleDict({})
        feature_extractor['patchifier'] = corelib.utils.PatchMaker(config.patchsize, stride=config.patchstride)
        feature_extractor['feature_aggregator'] = corelib.backbones.load(backbone_name)

    else:
        backbone = corelib.backbones.load(backbone_name).to(device).eval()
        feature_extractor = torch.nn.ModuleDict({})
        feature_extractor['patchifier'] = corelib.utils.PatchMaker(config.patchsize, stride=config.patchstride)
        feature_extractor['feature_aggregator'] = corelib.utils.FeatureAggregator(
                backbone, layers_to_extract_from, device
        )
        feature_dimensions = feature_extractor['feature_aggregator'].feature_dimensions([3, image_size, image_size])
        feature_extractor['preprocessing'] = corelib.utils.Preprocessing(
            feature_dimensions, config.pretrain_embed_dimension
        )
        feature_extractor["preadapt_aggregator"] = corelib.utils.Aggregator(
            target_dim=feature_dim
        ).to(device)
        

    sampler = corelib.sampler.MetaGreedyCoresetSampler(
        device=device, 
        sampling_ratio=sampling_ratio, 
        feature_dim=feature_dim, 
        coreset_size=coreset_size,
        reduce_feature_dim=reduce_feature_dim,
        project_dim=project_dim,
        approximate_ratio=approximate_ratio,
        metric=metric,
    )
    model = contcore.ContCore(
        backbone_name=backbone_name,
        feature_extractor=feature_extractor,
        layers_to_extract_from=layers_to_extract_from,
        feature_dim=feature_dim,
        sampling_threshold=sampling_threshold,
        sampler=sampler,

        chunk_size=chunk_size,
        image_size=image_size,
        save_path=save_path,
        device=device,
    )

    return model


