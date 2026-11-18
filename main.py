#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
without dist mat (iterative)
build task coreset (tc)
base-task meta-greedy sampling : with dist mat
"""

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
import torch.nn as nn
import torch.nn.functional as F
import json
from collections import OrderedDict
from datasets.common import *
from utils import *
import copy
import corelib.contcore as contcore
import corelib.utils
import time
from datetime import datetime, timezone
import subprocess
import uuid
import corelib.models as models
import corelib

def main():
    config = get_argparse()
    inc_cls_num = list(map(int, config.incremental_class_list.split('-')))
    if len(inc_cls_num) != 3 or inc_cls_num[0] <= 0 or inc_cls_num[1] < 0 or inc_cls_num[2] < 0:
        raise ValueError("Task schedule must be A-B-N with A>0, B>=0, N>=0")
    config.incremental_class_nums = [inc_cls_num[0]] + [inc_cls_num[1]] * inc_cls_num[2]
    if any(n <= 0 for n in config.incremental_class_nums) or sum(config.incremental_class_nums) > len(OBJ_DICT[config.dataset]):
        raise ValueError("Task schedule has empty tasks or exceeds dataset classes")
    if os.environ.get('FULL15_VARIANT') and os.environ.get('FULL15_SMOKE')!='1' and sum(config.incremental_class_nums)!=15:
        raise ValueError('Full15 formal runs must cover exactly fifteen classes')
    if os.environ.get('G_VARIANT'):
        if os.environ['G_VARIANT'] not in ('G0','G1','G2','G3'):raise ValueError('Unknown G variant')
        if config.dataset!='mvtec' or os.environ.get('RB_VARIANT')!='R2' or os.environ.get('OPT_VARIANT')!='Rhier':raise ValueError('R2-next requires mvtec/R2/Rhier')
        if os.environ.get('FULL15_VARIANT'):raise ValueError('G and F variants cannot be combined')
        if sum(config.incremental_class_nums)!=15 and os.environ.get('G_SMOKE')!='1':raise ValueError('Formal G protocols require full15')
    config.device = corelib.utils.set_torch_device([int(x) for x in config.gpus.split(",") if x.strip()])

    torch.set_num_threads(int(os.environ.get("CONTCORE_NUM_THREADS", "4")))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    trainset_list, testset_list = get_datasets(config)

    device = config.device

    model = corelib.models.get_contcore_model(
        backbone_name=config.backbone_name,
        image_size=config.image_size,
        layers_to_extract_from=config.layers_to_extract_from, 
        feature_dim=config.feature_dim,
        reduce_feature_dim=config.reduce_feature_dim,
        sampling_threshold=config.sampling_threshold,
        chunk_size=config.chunk_size,        

        sampling_ratio=config.sampling_ratio,
        coreset_size=config.coreset_size,
        project_dim=config.project_dim,
        approximate_ratio=config.approximate_ratio,
        metric=config.metric,

        save_path=config.save_path,
        config=config
    )
    model = model.to(device)

    ## train
    accumulated_testset = []
    train_time = 0
    for step, (trainset, testset) in enumerate(zip(trainset_list, testset_list)):
        # Match task sampling randomness across variants; refinement must not perturb the next task.
        torch.manual_seed(config.seed + step)
        np.random.seed(config.seed + step)
        random.seed(config.seed + step)

        ###################
        ###    train    ###
        ###################
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size = config.batch_size,
            shuffle = True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        if os.environ.get('G_SMOKE')=='1':
            # Explicit smoke only: first16 normal training images per class. Full test sets retained.
            trainset=ConcatDataset([torch.utils.data.Subset(ds,list(range(min(16,len(ds))))) for ds in trainset.datasets])
            train_loader=torch.utils.data.DataLoader(trainset,batch_size=config.batch_size,shuffle=True,num_workers=config.num_workers)
        start = time.time()

        for data in train_loader:
            images = data['image']
            sample_ids = data['sample_id']
            model.update(images, sample_ids, data.get('classlabel'))

        model._sample_task_coresets()
        model._meta_greedy_sampling()

        end = time.time()
        train_time += (end - start)
        model.save_model(step=step+1)
        if hasattr(model.sampler, "last_optimization_diagnostics"):
            with open(os.path.join(config.save_path, f"refinement_s{step+1}.json"), "w") as file:
                json.dump(model.sampler.last_optimization_diagnostics, file, indent=2)
            if os.environ.get('OPT_VARIANT','').startswith(('D','M','R')):
                with open(os.path.join(config.save_path,f'compression_s{step+1}.json'),'w') as file:
                    json.dump(getattr(model.sampler,'last_merge_diagnostics',{}),file,indent=2)

        ###################
        ###    test     ###
        ###################

        if getattr(model,'cseries',None) is not None:
            model.cseries.route_counts={'total':0,'accepted':0,'fallback':0,'predicted_tasks':{}}
        model.build_faiss(config.faiss_on_gpu, config.faiss_num_workers, config.anomaly_score_num_nn, backend=config.nn_backend)
        accumulated_testset.append(testset)
        concat_testset = ConcatDataset(accumulated_testset)
        test_loader = torch.utils.data.DataLoader(
            concat_testset,
            batch_size = config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        scores_all = []
        segmentations_all = []
        labels_gt_all = []
        masks_gt_all = []
        class_all = []
        image_paths_all = []
        mask_paths_all = []
        alt_predictions={name:dict(scores=[],masks=[]) for name in getattr(model,'diagnostic_banks',{})}
        neighbour_rows=[]
        route_rows=[]
        for data in test_loader:
            image = data['image']
            labels_gt = data['is_anomaly'].numpy().tolist()
            masks_gt = data['mask'].numpy()
            cls = data['classname']
            image_paths = data['image_path']
            mask_paths = data['mask_path']

            scores, segmentations  = model.inference(image)
            if model.g_router is not None:
                # Ground truth read only AFTER prediction, exclusively for evaluation.
                for i,predicted in enumerate(model.g_router.last_classes.cpu().tolist()):
                    route_rows.append(dict(classname=cls[i],true_class=int(data['classlabel'][i]),predicted_class=predicted,
                        is_anomaly=int(labels_gt[i]),path=image_paths[i]))
            for name,(alt_scores,alt_masks) in getattr(model,'last_diagnostic_predictions',{}).items():
                alt_predictions[name]['scores'].extend(alt_scores)
                alt_predictions[name]['masks'].extend(alt_masks)
            if alt_predictions:
                diagnostic=model.last_neighbour_diagnostic
                for i in range(len(image)):
                    neighbour_rows.append(dict(classname=cls[i],is_anomaly=labels_gt[i],path=image_paths[i],
                        **{k:v[i] for k,v in diagnostic.items()}))

            scores_all.extend(scores)
            segmentations_all.extend(segmentations)
            labels_gt_all.extend(labels_gt)
            masks_gt_all.extend(masks_gt)
            class_all.extend(cls)
            image_paths_all.extend(image_paths)
            mask_paths_all.extend(mask_paths)
            
        results = model.get_metric_results(scores_all, segmentations_all, labels_gt_all, masks_gt_all, class_all)
        model._save_metrics(results, f'results_s{step+1}.csv')
        for name,values in alt_predictions.items():
            alt_results=model.get_metric_results(values['scores'],values['masks'],labels_gt_all,masks_gt_all,class_all)
            model._save_metrics(alt_results,f'diagnostic_{name}_s{step+1}.csv')
        if route_rows:
            with open(os.path.join(config.save_path,f'route_s{step+1}.json'),'w') as file:json.dump(route_rows,file)
        if neighbour_rows:
            with open(os.path.join(config.save_path,f'neighbours_s{step+1}.json'),'w') as file:
                json.dump(neighbour_rows,file)
        if getattr(model,'cseries',None) is not None:
            with open(os.path.join(config.save_path,f'auxiliary_s{step+1}.json'),'w') as file:
                json.dump(dict(memory=model.cseries_memory,routing=model.cseries.route_counts),file,indent=2)
        model.diagnostic_banks={}
        status = {"last_evaluated_step": step + 1, "total_steps": len(trainset_list),
                  "complete": step + 1 == len(trainset_list),
                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                  "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        with open(os.path.join(config.save_path, 'status.json'), 'w') as file:
            json.dump(status, file, indent=2)



    with open(os.path.join(config.save_path, f'time.txt'), 'w') as file:
        file.write(f'sampling time: {train_time}')



def get_argparse():
    parser = argparse.ArgumentParser()

    # model arguments
    # feature extractor 
    ## timm based args
    parser.add_argument("--backbone_name", "-b", type=str, default='wideresnet50')
    parser.add_argument('--layers_to_extract_from', "-le", nargs='+', default=['layer2', 'layer3'])
    parser.add_argument("--pretrain_embed_dimension", type=int, default=1024)
    parser.add_argument("--feature_dim", type=int, default=1024)

    ## Dinov2 based args
    # parser.add_argument("--backbone_name", "-b", type=str, default='dinov2reg_vit_base_14')
    # parser.add_argument('--layers_to_extract_from', "-le", nargs='+', default=['11'])
    # parser.add_argument("--pretrain_embed_dimension", type=int, default=768)
    # parser.add_argument("--feature_dim", type=int, default=768)


    # non-dino model arguments (from PatchCore)
    # these args are for timm based patchcore sampling, not for DINOv2
    parser.add_argument("--preprocessing", type=str, default="mean")
    parser.add_argument("--patchsize", type=int, default=3)
    parser.add_argument("--patchstride", type=int, default=1)
    parser.add_argument("--patchscore", type=str, default="max")
    parser.add_argument("--patchoverlap", type=float, default=0.0)
    parser.add_argument("--patchsize_aggregate", "-pa", type=int, default=[])

    # NN Anomaly Scorer parameters.
    parser.add_argument("--anomaly_score_num_nn", type=int, default=1)
    # Faiss
    parser.add_argument("--faiss_on_gpu", type=str2bool, default='False')
    parser.add_argument("--faiss_num_workers", type=int, default=8)

    # Sampler
    parser.add_argument('--metric', type=str, default='L2',choices=['L2', 'cosine'])
    parser.add_argument("--coreset_size", type=int, default=20000) 
    parser.add_argument("--sampling_threshold", type=int, default=20000) ## num features when sampling starts
    parser.add_argument("--sampling_ratio", type=float, default=0.01) ## sampling ratio
    parser.add_argument("--approximate_ratio", type=float, default=0.75) ## approximation ratio for inter-task sampling
    parser.add_argument("--max_chunk_size", type=str2bool, default='True') 
    parser.add_argument("--chunk_size", type=int, default=100000)
    parser.add_argument("--project_dim", type=int, default=128) ## project dimension to lower dimension
    parser.add_argument("--reduce_feature_dim", type=str2bool, default=True) ## FLAG: lower dimension projection for inter-task sampling


    ## dataset 
    parser.add_argument('--dataset', type=str, default='mvtec',choices=['mvtec','visa'])
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--resize", default=256, type=int)
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--augment", type=str2bool, default='True')

    ## misc
    parser.add_argument("--train", type=str2bool, default='True')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--incremental_class_list", type=str, default="1-1-14")

    parser.add_argument("--gpus", type=str, default="0")

    parser.add_argument("--nn_backend", choices=["faiss", "torch"], default="faiss")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Dataset root; overrides datasets/common.py PATH_DICT")
    parser.add_argument("--output_root", type=str, default="./results")
    config = parser.parse_args()
    if config.coreset_size < 2 or not 0 < config.sampling_ratio < 1 or not 0 <= config.approximate_ratio <= 1:
        parser.error("coreset_size >=2, sampling_ratio in (0,1), approximate_ratio in [0,1] required")

    config.save_path = os.path.join(config.output_root, config.dataset, config.incremental_class_list, f"Coreset{config.coreset_size}_SampRatio{config.sampling_ratio}_Apprx{config.approximate_ratio}")
    method = "baseline" if os.environ.get("USE_RSC", "0") != "1" else (
        "rsc_h1" if os.environ.get("USE_H1", "1") == "1" else "rsc")
    if os.environ.get("OPT_VARIANT"):
        method = os.environ.get('FULL15_VARIANT',os.environ.get("RB_VARIANT",os.environ["OPT_VARIANT"]))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    method=os.environ.get('G_VARIANT',method)
    config.save_path = os.path.join(config.save_path, f"{method}_seed{config.seed}_{run_id}")
    config.model_path = os.path.join(config.save_path, 'model')
    config.metrics_path = os.path.join(config.save_path, 'metrics')
    config.seg_path = os.path.join(config.save_path, 'segmentations')
    os.makedirs(config.model_path, exist_ok=True)
    os.makedirs(config.metrics_path, exist_ok=True)
    os.makedirs(config.seg_path, exist_ok=True)


    metadata = dict(vars(config))
    metadata["environment"] = {k: os.environ.get(k) for k in
        ("USE_RSC", "USE_H1", "H1_ALPHA", "H1_BETA", "CUDA_VISIBLE_DEVICES", "OPT_VARIANT", "RB_VARIANT", "FULL15_VARIANT", "FULL15_SMOKE", "G_VARIANT", "G_SMOKE")}
    metadata["refinement_ablation"] = {
        "variant": os.environ.get("OPT_VARIANT", "legacy"),
        "R_modules": {
            "Rdiag": "M3 exact reproduction; old/union/merge/current scores at9,10,13,15",
            "Rfix": "only original-node union-find fix; otherwise M3",
            "Rresp": "Rfix plus task-owned coverage-ball contraction and matching restoration target; no cross-task transfer; weighted source importance",
            "Rhier": "Rresp plus one nearest-neighbour graph parent level over child medoids; no target cluster-count search",
            "Rctrl": "Rhier4720 prototype capacity control",
            "Rimage": "Rctrl plus16 hard training patch features/task as4 images x4patch sparse bags; squared-L2 max over bag, birth-score stability in restoration; not full image replay",
            "budget": "Rdiag/Rfix/Rresp/Rhier4960, Rctrl/Rimage4720; all mode metadata and bags byte-audited vs5000",
            "limits": "task-local transfer is a structural restriction, no explicit task quota; ball summaries approximate responsibility and do not store all lost historical modes; sparse bags do not guarantee test score ranking"
        },
        "mode_modules": {
            "M0": "D1 capacity4960 reference; no new metadata",
            "modes": "task NN connected components at median NN edge length; frozen mode IDs/birth scales",
            "M1": "local projection16 candidates; original-space contraction; prefer no mode disappearance; carry triangle-bound radius",
            "M2": "M1 plus repeated hardest-normal-image mode association; lexicographic priority, no loss weight",
            "M3": "M2 plus mode-restoration swaps; inherited max32,8 additions,16 deletions; lexicographic missing/critical/worst/mean coverage",
            "budget": "4960 retrieval prototypes; mode IDs,scales,bounds,importance and scalar counters byte audited vs5000",
            "limits": "NN components are proxy modes; no augmentation consistency implemented; no guarantee of test ranking; no separate historical feature bank"
        },
        "dseries_protocol": {"fixed_projection": "merge only; stateless seed19042 Gaussian / sqrt128; original global RNG draws retained",
            "compression": "D2/D2ctrl/D3: backward deletion,16 nominated candidates, exact FP32 original-space query distances, task-macro .5mean+.5CVaR.9",
            "queries": "D2:task-balanced old256 + up to64 different current training images; D3:independent history16/task + current64",
            "persistent_budget": "5000 prototype-equivalent bytes incl queries; D3 and D2ctrl use4720 prototypes",
            "diagnostics": "D0 s8/s9/s13: old,union,merge-before-refinement,current; Dgreedy full merge coverage greedy",
            "retention": "no D-series weights saved; original A2 checkpoint retained"},
        "cseries_protocol": {"routing": "predicted image task; cosine global centers; margin .10; radius 2*q95; min bank 8; global fallback",
            "memory_budget": "core+queries+centers+normalization tensors <= 5000*(1024*4+idbytes+tasklabelbytes)",
            "query_protocol": "16 unique normal training images/task, nonprototype patch, random32 search; local seed 20261106+task",
            "protection": "mean drift + .5*(Huber deadzone .05 short + .5 long); historical mean cannot increase",
            "checkpoint_retention": "C0 final A2 only; all other C experiments metrics and diagnostics only"},
        "historical_query_budget": 256, "current_query_budget_A4": 128,
        "persistent_extra_query_features": 240 if method in ("C2","C3","D3") else 0, "prototype_budget": config.coreset_size,
        "deletion_candidates_dynamic": 8 if method in ("B2","B12") else 16, "addition_candidates": 8,
        "max_swaps": 8 if method in ("B2","B12") else 32, "alpha": .90, "beta": .50,
        "coverage_floor_fraction": .15 if method in ("B1","B12") else 0.,
        "coverage_floor_applied": "merge mandatory FPS seeds and refinement removal guard",
        "current_weight": .50, "current_tolerance": .01, "worst_tolerance": .01,
        "task_labels": "explicit task provenance carried with selected source rows; not image sample IDs"}
    metadata["metric_protocol"] = {
        "version": 2, "scale": "percent", "average": "class_macro",
        "aupro": "PRO@FPR0.30, all background pixels, 8-connectivity, 4096 histogram bins",
        "f1": "test-set maximum; not a deployment threshold", "scores": "raw", "task_rng": "reset seed + zero-based stage before each task"}
    if os.environ.get('RB_VARIANT'):
        metadata['refinement_ablation'] = {
            'variant': os.environ['RB_VARIANT'], 'base': 'original Rhier',
            'R0': 'original task-owned hierarchical responsibility memory',
            'R1': 'R0 + exact task sampler source row class labels',
            'R2': 'R1 + class-local child/parent modes and same-class responsibility',
            'R3': 'R2 + one actual centre representative/class before original task FPS',
            'R4': 'R3 + class worst/mean risk objective and per-swap nonregression guard',
            'R5': 'R4 + 4 images x 4 hard normal patches/class; birth and previous-stage score references',
            'prototype_budget': config.coreset_size,
            'checkpoint_retention': 'no new model checkpoints',
            'inference': 'shared global bank; no test class routing',
            'risk_guard_scope': 'restoration swaps only; not a guarantee on test FM or compression',
            'schedule_semantics': '[A] + [B] * N, original class order, includes leather'}
    if os.environ.get('FULL15_VARIANT'):
        metadata['refinement_ablation']={
            'variant':os.environ['FULL15_VARIANT'],
            'F0':'original Rhier; exact source labels only for bookkeeping',
            'F1':'R2 class-local hierarchical responsibility memory',
            'F2':'F1 + per-class candidate FPS, mode survival, class coverage minimax compression',
            'F3':'F1 + precompression normal/synthetic query score preservation',
            'F4':'F2 + score preservation',
            'prototype_budget':config.coreset_size,
            'persistent_byte_limit':'5000*(feature_dim*4+sample_id_bytes+task_label_bytes)',
            'normal_queries_per_class':16 if os.environ['FULL15_VARIANT'] in ('F3','F4') else 0,
            'synthetic_queries_per_class':4 if os.environ['FULL15_VARIANT'] in ('F3','F4') else 0,
            'synthetic_protocol':'radial feature perturbation of normal hard patches; not real anomalies',
            'teacher':'previous bank before merge for old classes; uncompressed union for current class',
            'birth_reference':'final bank at class birth; kept unchanged subsequently',
            'inference':'original shared global bank, no test class routing',
            'checkpoint_retention':'metrics/config/source/diagnostics only; no model checkpoints',
            'seed_scope':'single-seed ablation; no stability claim',
            'class_order':list(OBJ_DICT[config.dataset]),
            'class_counts':[int(config.incremental_class_list.split('-')[0])]+[int(config.incremental_class_list.split('-')[1])]*int(config.incremental_class_list.split('-')[2])}
    if os.environ.get('G_VARIANT'):
        metadata['refinement_ablation']={
            'variant':os.environ['G_VARIANT'],'base':'F1/R2',
            'G0':'R2 unchanged, rerun control',
            'G1':'per-class birth compression to final quota; old rows frozen; global inference',
            'G2':'R2 unchanged memory; image descriptor predicted top1 routing and class-local scoring',
            'G3':'reserved memory + predicted routing',
            'total_classes_known':15,'prototype_limit':config.coreset_size,
            'quota':'K//15 for all classes, +1 for first K%15 class IDs; unarrived slots remain unused',
            'router':'4 deterministic FPS exemplars of normalized mean patch descriptors/class; normal train only',
            'router_input':'image features only; no test class input, no oracle',
            'scoring':'original squared-L2 nearest-neighbour mean and original image/segmentation aggregation',
            'persistent_limit':'same 5000-prototype-equivalent bytes as F-series, including router',
            'checkpoint_retention':'no new weights','scope':'seed42 structural ablation; no test-driven parameter selection'}
    source_root = os.path.dirname(os.path.abspath(__file__))
    def git_output(args):
        try:
            return subprocess.check_output(["git"] + args, cwd=source_root, stderr=subprocess.DEVNULL).decode("utf-8", "replace")
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    metadata["git_commit"] = git_output(["rev-parse", "HEAD"]).strip()
    metadata["command"] = __import__("sys").argv
    # Include untracked method files (e.g. rsc.py), absent from git diff.
    import zipfile
    from pathlib import Path
    with zipfile.ZipFile(os.path.join(config.save_path, "source_snapshot.zip"), "w") as archive:
        source_files = list(Path(source_root).glob("*.py"))
        for subdir in ("corelib", "datasets"):
            source_files.extend((Path(source_root) / subdir).rglob("*.py"))
        for path in source_files:
            archive.write(str(path), str(path.relative_to(source_root)))
    with open(os.path.join(config.save_path, "config.json"), "w") as file:
        json.dump(metadata, file, indent=2)
    with open(os.path.join(config.save_path, "source.diff"), "w") as file:
        file.write(git_output(["diff", "HEAD"]))
    with open(os.path.join(config.save_path, "status.json"), "w") as file:
        json.dump({"last_evaluated_step": 0, "complete": False}, file)
    print(f"Run output: {config.save_path}", flush=True)
    return config



if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    main()
