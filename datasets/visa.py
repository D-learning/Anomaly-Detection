import os
from enum import Enum

import PIL
import torch
from torchvision import transforms
import csv

_CLASSNAMES = ['candle', 
               'capsules', 
               'cashew', 
               'chewinggum', 
               'fryum', 
               'macaroni1', 
               'macaroni2', 
               'pcb1', 
               'pcb2', 
               'pcb3', 
               'pcb4', 
               'pipe_fryum']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLS2LABEL = {v:k for k,v in enumerate(_CLASSNAMES)}
LABEL2CLS = {k:v for k,v in enumerate(_CLASSNAMES)}

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class VISADataset(torch.utils.data.Dataset):
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
        rotate_degrees=0,
        translate=0,
        brightness_factor=0,
        contrast_factor=0,
        saturation_factor=0,
        gray_p=0,
        h_flip_p=0,
        v_flip_p=0,
        scale=0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the mvtec data folder.
            classname: [str or None]. Name of mvtec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            # transforms.RandomRotation(rotate_degrees, transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(rotate_degrees, 
                                    translate=(translate, translate),
                                    scale=(1.0-scale, 1.0+scale),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path, sample_id = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        cls_label = CLS2LABEL[classname]

        if self.split == DatasetSplit.TEST and mask_path != 'none':
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
            mask = (mask > 0).type(torch.float32)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "classlabel": cls_label,
            "sample_id": sample_id,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != 'normal'),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
            "mask_path": mask_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}
    
        for classname in self.classnames_to_use:
            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            imgpaths_per_class[classname]['normal'] = []
            maskpaths_per_class[classname]['normal'] = []
            if self.split == DatasetSplit.TEST:
                imgpaths_per_class[classname]['anomaly'] = []
                maskpaths_per_class[classname]['anomaly'] = []

            with open(os.path.join(self.source, '1cls.csv'), 'r') as f:
                csv_reader = csv.reader(f)
                for i, row in enumerate(csv_reader):
                    if i == 0:
                        continue
                    if row[0] == classname:
                        path = os.path.join(self.source, row[3])
                        if self.split.value == 'train' and row[1] == 'train':
                            imgpaths_per_class[classname]['normal'].append(path)
                            maskpaths_per_class[classname]['normal'].append('none')
                        elif self.split.value == 'test' and row[1] == 'test':
                            label = row[2]
                            if label == 'normal':
                                imgpaths_per_class[classname][label].append(path)
                                maskpaths_per_class[classname][label].append('none')
                            elif label == 'anomaly':
                                mask_path = os.path.join(self.source, row[4])
                                imgpaths_per_class[classname][label].append(path)
                                maskpaths_per_class[classname][label].append(mask_path)



        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        sample_id = 0
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != 'normal':
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append('none')
                    data_tuple.append(sample_id)
                    data_to_iterate.append(data_tuple)
                    sample_id += 1

        return imgpaths_per_class, data_to_iterate
