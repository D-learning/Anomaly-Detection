#!/usr/bin/python
# -*- coding: utf-8 -*-
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
from torchvision.datasets import ImageFolder
import copy
from datasets import *
from sklearn.cluster import KMeans
import torch.nn.functional as F
import glob
from sklearn import metrics
import pickle as pkl
import logging

LOGGER = logging.getLogger(__name__)


OBJ_DICT = {'mvtec':['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',  'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
            'loco':['breakfast_box', 'juice_bottle', 'pushpins',  'screw_bag', 'splicing_connectors'],
            'btad':['01', '02', '03'],
            'visa':['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']}

PATH_DICT = {'mvtec': '/mnt/c/Users/weiya/mvtec_anomaly_detection',
            'loco': '/home/yoon/data/mvtec_loco',
            'btad' : '/home/yoon/data/btad',
            'visa' : '/mnt/c/Users/weiya/VisA'}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def update_json(json_path, k, v):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        data[k] = v
    else:
        data = OrderedDict()
        data[k] = v
    with open(json_path, 'w') as f:
        json.dump(data, f)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        

def _default(obj):
    """Custom JSON serializer for adding support for numpy data types."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    raise TypeError(
        f"Object of type '{obj.__class__.__name__}' is not JSON serializable"
    )

def _save_dict_as_json(data: dict, file_path: str):
    """Helper function to save data as JSON."""
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4, default=_default)


def _save_dict_as_pickle(data: dict, file_path: str):
    """Helper function to save data as pickle."""
    with open(file_path, "wb") as file:
        pkl.dump(data, file)

def save_dict(data: dict, file_path: str):
    """Saves a dictionary to a file, choosing format based on file extension."""
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    if file_path.endswith(".json"):
        _save_dict_as_json(data, file_path)
    elif file_path.endswith((".pkl", ".pickle")):
        _save_dict_as_pickle(data, file_path)
    else:
        raise ValueError(
            "Unsupported file format. Please use .json or .pkl/.pickle extension."
        )


def _load_dict_from_json(file_path: str):
    """Helper function to load data from a JSON file."""
    with open(file_path) as file:
        return json.load(file)


def _load_dict_from_pickle(file_path: str):
    """Helper function to load data from a pickle file."""
    with open(file_path, "rb") as file:
        return pkl.load(file)


def load_dict(file_path: str):
    """Loads a dictionary from a file, choosing format based on file extension."""
    if file_path.endswith(".json"):
        return _load_dict_from_json(file_path)
    elif file_path.endswith((".pkl", ".pickle")):
        return _load_dict_from_pickle(file_path)
    else:
        raise ValueError(
            "Unsupported file format. Please use .json or .pkl/.pickle extension."
        )


def dump(save_dir, output):
    filename = output["filename"]
    pred_pix = output["pred_pix"].cpu().numpy()  # B x 1 x H x W
    mask = output["mask"].cpu().numpy()  # B x 1 x H x W
    height = output["height"]
    width = output["width"]
    clsname = output["clsname"]
    file_dir, filename = os.path.split(filename)
    _, subname = os.path.split(file_dir)
    filename = "{}_{}_{}".format(clsname, subname, filename)
    filename, _ = os.path.splitext(filename)
    save_file = os.path.join(save_dir, filename + ".npz")
    np.savez(
        save_file,
        filename=filename,
        pred_pix=pred_pix,
        mask=mask,
        height=height,
        width=width,
        clsname=clsname
    )


def merge_together(save_dir):
    npz_file_list = glob.glob(os.path.join(save_dir, "*.npz"))
    fileinfos = []
    preds = []
    masks = []
    for npz_file in npz_file_list:
        npz = np.load(npz_file)
        fileinfos.append(
            {
                "filename": str(npz["filename"]),
                "height": npz["height"],
                "width": npz["width"],
                "clsname": str(npz["clsname"]),
            }
        )
        preds.append(npz["pred_pix"])
        masks.append(npz["mask"])
    preds = np.concatenate(np.asarray(preds), axis=0).squeeze()  # N x H x W
    masks = np.concatenate(np.asarray(masks), axis=0).squeeze()  # N x H x W
    return fileinfos, preds, masks



class EvalImage:
    def __init__(self, data_meta, **kwargs):
        self.preds = self.encode_pred(data_meta.preds, **kwargs)
        self.masks = self.encode_mask(data_meta.masks)
        self.preds_good = sorted(self.preds[self.masks == 0], reverse=True)
        self.preds_defe = sorted(self.preds[self.masks == 1], reverse=True)
        self.num_good = len(self.preds_good)
        self.num_defe = len(self.preds_defe)

    @staticmethod
    def encode_pred(preds):
        raise NotImplementedError

    def encode_mask(self, masks):
        N, _, _ = masks.shape
        masks = (masks.reshape(N, -1).sum(axis=1) != 0).astype(np.int32)  # (N, )
        return masks

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc

class EvalImageMax(EvalImage):
    @staticmethod
    def encode_pred(preds, avgpool_size=[16,16]):
        N, _, _ = preds.shape
        print(preds.shape)
        # preds = torch.tensor(preds[:, None, ...]).cuda()  # N x 1 x H x W
        # preds = (
        #     F.avg_pool2d(preds, avgpool_size, stride=1).cpu().numpy()
        # )  # N x 1 x H x W
        return preds.reshape(N, -1).max(axis=1)  # (N, )

class EvalPerPixelAUC:
    def __init__(self, data_meta):
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc




class EvalDataMeta:
    def __init__(self, preds, masks):
        self.preds = preds  # N x H x W
        self.masks = masks  # N x H x W


def performances(fileinfos, preds, masks):
    ret_metrics = {}
    clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
    for clsname in clsnames:
        preds_cls = []
        masks_cls = []
        for fileinfo, pred, mask in zip(fileinfos, preds, masks):
            if fileinfo["clsname"] == clsname:
                preds_cls.append(pred[None, ...])
                masks_cls.append(mask[None, ...])
        preds_cls = np.concatenate(np.asarray(preds_cls), axis=0)  # N x H x W
        masks_cls = np.concatenate(np.asarray(masks_cls), axis=0)  # N x H x W
        data_meta = EvalDataMeta(preds_cls, masks_cls)
        # auc
        eval_max = EvalImageMax(data_meta)
        max_auc = eval_max.eval_auc()
        eval_pix = EvalPerPixelAUC(data_meta)
        pix_auc = eval_pix.eval_auc()
        ret_metrics["{}_max_auc".format(clsname)] = max_auc * 100
        ret_metrics["{}_pix_auc".format(clsname)] = pix_auc * 100

    for metric in ['max', 'pix']:
        evalvalues = [
            ret_metrics["{}_{}_auc".format(clsname, metric)]
            for clsname in clsnames
        ]
        mean_auc = np.mean(np.array(evalvalues))
        ret_metrics["{}_{}_auc".format("mean", metric)] = mean_auc

    return ret_metrics


def performances_all(fileinfos, preds, masks):
    ret_metrics = {}
    clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
    preds_cls = []
    masks_cls = []
    for clsname in clsnames:
        for fileinfo, pred, mask in zip(fileinfos, preds, masks):
            if fileinfo["clsname"] == clsname:
                preds_cls.append(pred[None, ...])
                masks_cls.append(mask[None, ...])
    preds_cls = np.concatenate(np.asarray(preds_cls), axis=0)  # N x H x W
    masks_cls = np.concatenate(np.asarray(masks_cls), axis=0)  # N x H x W
    data_meta = EvalDataMeta(preds_cls, masks_cls)

    # auc
    eval_max = EvalImageMax(data_meta)
    max_auc = eval_max.eval_auc()
    eval_pix = EvalPerPixelAUC(data_meta)
    pix_auc = eval_pix.eval_auc()
    ret_metrics["{}_max_auc".format('all')] = max_auc * 100
    ret_metrics["{}_pix_auc".format('all')] = pix_auc * 100


    for metric in ['max', 'pix']:
        evalvalues = [
            ret_metrics["{}_{}_auc".format('all', metric)]
        ]
        mean_auc = np.mean(np.array(evalvalues))
        ret_metrics["{}_{}_auc".format("mean", metric)] = mean_auc

    return ret_metrics



def log_metrics(ret_metrics, save_dir):
    clsnames = set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()])
    clsnames = list(clsnames - set(["mean"])) + ["mean"]
    data_list = []

    # auc
    auc_keys = [k for k in ret_metrics.keys() if "auc" in k]
    evalnames = list(set([k.rsplit("_", 2)[1] for k in auc_keys]))
    data_list.append(["clsname"] + evalnames)

    for clsname in clsnames:
        clsvalues = [
            ret_metrics["{}_{}_auc".format(clsname, evalname)]
            for evalname in evalnames
        ]
        data_list.append([clsname] + clsvalues)

    with open(os.path.join(save_dir, 'results.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_list)


def log_metrics_all(ret_metrics, save_dir):
    clsnames = set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()])
    clsnames = list(clsnames - set(["mean"])) + ["mean"]
    data_list = []

    # auc
    auc_keys = [k for k in ret_metrics.keys() if "auc" in k]
    evalnames = list(set([k.rsplit("_", 2)[1] for k in auc_keys]))
    data_list.append(["clsname"] + evalnames)

    for clsname in clsnames:
        clsvalues = [
            ret_metrics["{}_{}_auc".format(clsname, evalname)]
            for evalname in evalnames
        ]
        data_list.append([clsname] + clsvalues)

    with open(os.path.join(save_dir, 'results_all.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_list)




def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
    file_name = 'results.csv'
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, file_name)
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics

