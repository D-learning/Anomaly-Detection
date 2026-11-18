import os
from enum import Enum

import PIL
import torch
from torchvision import transforms

_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class ToyDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for mvtec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        self.data_to_iterate = self.data_to_iterate[:2]

        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)
        self.transform_std=IMAGENET_STD
        self.transform_mean=IMAGENET_MEAN
        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate




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


