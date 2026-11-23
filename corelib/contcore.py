import torch
import torch.nn as nn
from typing import Optional
import corelib.utils
from corelib.sampler import BaseSampler

import torch.nn.functional as F
import numpy as np 
from torchvision import transforms
import os
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_recall_curve, \
    average_precision_score
import metrics


class ContCore(nn.Module):
    def __init__(self, 
                 backbone_name: str,
                 feature_extractor: nn.Module,
                 layers_to_extract_from: Optional[str],
                 feature_dim: int,
                 sampling_threshold: int,
                 sampler: BaseSampler,
                 chunk_size: int,
                 image_size: int,
                 save_path: str,
                 device,
        ):
        super(ContCore, self).__init__()
        self.backbone_name = backbone_name
        self.feature_extractor = feature_extractor
        self.sampling_threshold = sampling_threshold

        self.sampler = sampler
        self.layers_to_extract_from = layers_to_extract_from
        self.image_size = image_size
        self.device = device
        self.feature_dim = feature_dim

        self.cseries_variant=os.environ.get('OPT_VARIANT','')
        self.cseries=None
        if self.cseries_variant in ('C0','C1ctrl','C1','C2ctrl','C2','C3','D0','D1','Dgreedy','D2','D2ctrl','D3','M0','M1','M2','M3','Rdiag','Rfix','Rresp','Rhier','Rctrl','Rimage'):
            from corelib.cseries import ProtectionState
            self.cseries=ProtectionState('C2' if self.cseries_variant=='D3' else self.cseries_variant,feature_dim)
        self.chunk_size = self._get_max_chunk_size(chunk_size)
        self.features = []
        self.sample_ids = []
        self.class_labels = []
        self.coreset_features = None
        self.coreset_sample_ids = None
        self.optimization_class_labels = None
        self.task_coreset_class_labels = None
        self.raw_task_class_labels = None
        self.task_coreset_features = None

        self.save_model_path = os.path.join(save_path, 'model')
        self.save_metrics_path = os.path.join(save_path, 'metrics')
        self.save_seg_path = os.path.join(save_path, 'segmentations')

        os.makedirs(self.save_model_path, exist_ok=True)
        os.makedirs(self.save_metrics_path, exist_ok=True)
        os.makedirs(self.save_seg_path, exist_ok=True)

    def _get_max_chunk_size(self, chunk_size: int):
        max_chunk_size = corelib.utils.auto_get_max_query_size_pdist(
            fea_dim=self.feature_dim,
            device_type=self.device.type,
            device_index=self.device.index,
            max_usage=0.25,
        )
        if chunk_size > max_chunk_size:
            chunk_size = max_chunk_size
        return chunk_size
    
    def _get_feature_shape(self):
        dummy_img = torch.zeros((1, 3, self.image_size, self.image_size)).to(self.device)
        dummy_sample_id = torch.zeros((1))
        features, _ = self._forward(dummy_img, dummy_sample_id, detach=False, accumulate_features=False)
        return features.squeeze().shape

    @torch.no_grad()
    def _extract_features(self,
                     images: torch.Tensor,
                     sample_ids: torch.Tensor=None,
                     return_patch_shape: bool=False,
        ):
        if isinstance(images, dict):
            images = images["image"]
        images = images.to(torch.float).to(self.device)
        sample_ids = sample_ids.to(self.device)

        if "dinov2" in self.backbone_name:
            features = self.feature_extractor['feature_aggregator'].prepare_tokens(images)
            batch_size = features.shape[0]
            for i, blk in enumerate(self.feature_extractor['feature_aggregator'].blocks):
                if i <= int(self.layers_to_extract_from[-1]):
                    features = blk(features)
                else:
                    continue
            if 'reg' in self.backbone_name:
                features = features[:, 1 + self.feature_extractor['feature_aggregator'].num_register_tokens:, :]

            fea_shape = int(np.sqrt(features.reshape(batch_size, -1, self.feature_dim).shape[1]))
            patch_shapes = features.reshape(fea_shape, fea_shape, -1).shape[:2]

            sample_ids = torch.broadcast_to(sample_ids[:,None], features.shape[:2]).reshape(-1)
            features = features.reshape(-1, features.shape[-1])
            features = F.normalize(features, dim=-1)
        else:
            features = self.feature_extractor['feature_aggregator'](images)

            features = [features[layer] for layer in self.layers_to_extract_from]

            features = [
                self.feature_extractor['patchifier'].patchify(x, return_spatial_info=True) for x in features
            ]
            patch_shapes = [x[1] for x in features]
            features = [x[0] for x in features]
            ref_num_patches = patch_shapes[0]

            for i in range(1, len(features)):
                _features = features[i]
                patch_dims = patch_shapes[i]

                _features = _features.reshape(
                    _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
                )
                _features = _features.permute(0, -3, -2, -1, 1, 2)
                perm_base_shape = _features.shape
                _features = _features.reshape(-1, *_features.shape[-2:])
                _features = F.interpolate(
                    _features.unsqueeze(1),
                    size=(ref_num_patches[0], ref_num_patches[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                _features = _features.squeeze(1)
                _features = _features.reshape(
                    *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
                )
                _features = _features.permute(0, -2, -1, 1, 2, 3)
                _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
                features[i] = _features

            patch_shapes = patch_shapes[0]
            sample_ids = torch.broadcast_to(sample_ids[:,None], features[0].shape[:2]).reshape(-1)

            features = [x.reshape(-1, *x.shape[-3:]) for x in features]
            features = self.feature_extractor["preprocessing"](features)
            features = self.feature_extractor["preadapt_aggregator"](features)

        if return_patch_shape:
            return features, sample_ids, patch_shapes

        return features, sample_ids

    def _forward(self, 
                 images: torch.Tensor, 
                 sample_ids: torch.Tensor,
                 class_labels: torch.Tensor = None, 
                 accumulate_features: bool = True,  
                 return_patch_shape: bool = False):
        if isinstance(images, dict):
            images = images["image"]

        images = images.to(torch.float).to(self.device)
        sample_ids = sample_ids.to(self.device)

        features, sample_ids = self._extract_features(images, sample_ids, return_patch_shape)

        if accumulate_features:
            self.features.append(features)
            self.sample_ids.append(sample_ids)
            if class_labels is not None:
                self.class_labels.append(class_labels.to(self.device).repeat_interleave(sample_ids.numel() // class_labels.numel()))

        return features, sample_ids

    def _sample_task_coresets(self):
        features = torch.cat(self.features, dim=0)
        class_labels = torch.cat(self.class_labels, dim=0) if self.class_labels else None
        sample_ids = torch.cat(self.sample_ids, dim=0).reshape(-1)
        self.features = []
        self.sample_ids = []
        self.class_labels = []
        # P0-06: 暂存纯净特征
        self.raw_task_features = features
        self.raw_task_sample_ids = sample_ids
        self.raw_task_class_labels = class_labels
        if getattr(self,'cseries',None) is not None:
            self.cseries.train_center(features,sample_ids)
        self.sampler.raw_class_labels=class_labels
        self.task_coreset_features, self.task_coreset_sample_ids = self.sampler.run(features, sample_ids, task_coreset=self.task_coreset_features, base_coreset=self.coreset_features)
        import os
        if os.environ.get('RB_VARIANT','R0')=='R0' and not os.environ.get('FULL15_VARIANT'):
            # Original reverse lookup, chunked to avoid materialising K x all patches.
            nearest=torch.cat([torch.cdist(q.float(),features.float()).argmin(1) for q in self.task_coreset_features.split(32)])
        else:nearest=self.sampler.last_task_source_indices
        self.task_coreset_class_labels = class_labels[nearest].long() if class_labels is not None else torch.zeros(len(nearest),device=features.device,dtype=torch.long)
        print(f"num. of task coreset features after sampling: {len(self.task_coreset_features)}")
        self.features = []
        self.sample_ids = []
        
    def _meta_greedy_sampling(self):
        print("Meta Greedy Sampling")
        next_task=getattr(self.sampler,'optimization_task_index',-1)+1
        self.diagnostic_banks={}
        if (self.cseries_variant in ('C0','D0') and next_task+1 in (8,9,13)) or (self.cseries_variant=='Rdiag' and next_task+1 in (9,10,13,15)):
            self.diagnostic_banks['old']=self.coreset_features.clone()
            self.diagnostic_banks['union']=torch.cat((self.coreset_features,self.task_coreset_features))
            oldlabels=self.sampler.optimization_task_labels
            self.diagnostic_labels={
                'old':oldlabels.clone(),
                'union':torch.cat((oldlabels,torch.full((len(self.task_coreset_features),),next_task,device=self.device,dtype=torch.long)))}
        if getattr(self,'cseries',None) is not None:
            self.sampler.external_protection=None if self.cseries_variant.startswith('D') else self.cseries.protection_data()
        self.sampler.merge_raw_features=self.raw_task_features
        self.sampler.merge_raw_ids=self.raw_task_sample_ids
        self.sampler.merge_independent_queries=self.cseries.protection_data() if self.cseries_variant=='D3' else None
        self.coreset_features, self.coreset_sample_ids = self.sampler.meta_greedy_sampling(
            self.task_coreset_features, 
            self.task_coreset_sample_ids,
            base_coreset=self.coreset_features,
            base_sample_ids=self.coreset_sample_ids,
            raw_task_features=getattr(self, 'raw_task_features', None),
            raw_task_ids=getattr(self, 'raw_task_sample_ids', None),
                task_class_labels=getattr(self, 'task_coreset_class_labels', None),
                base_class_labels=getattr(self.sampler, 'optimization_class_labels', None),
        )
        if (self.cseries_variant=='D0' and next_task+1 in (8,9,13)) or (self.cseries_variant=='Rdiag' and next_task+1 in (9,10,13,15)):
            self.diagnostic_banks['merge']=self.sampler.merge_bank_for_diagnostic
            self.diagnostic_labels['merge']=self.sampler.merge_labels_for_diagnostic
            self.sampler.merge_bank_for_diagnostic=None
            self.sampler.merge_labels_for_diagnostic=None
        self.sampler.merge_raw_features=None
        self.sampler.merge_raw_ids=None
        self.sampler.merge_independent_queries=None
        self.task_coreset_features = None
        self.task_coreset_sample_ids = None
        self.task_coreset_class_labels = None
        self.sampler.external_protection=None
        if getattr(self,'cseries',None) is not None:
            self.cseries.register_queries(self.raw_task_features,self.raw_task_sample_ids,self.coreset_features,next_task)
            self.cseries_memory=self.cseries.memory_audit(self.coreset_features,self.coreset_sample_ids,self.sampler.optimization_task_labels)
            if hasattr(self.sampler,'mode_state'):
                self.cseries_memory=self.sampler.mode_state.audit(self.coreset_features,self.coreset_sample_ids,self.sampler.optimization_task_labels)
        self.raw_task_features = None
        self.raw_task_sample_ids = None
        self.raw_task_class_labels=None
        self.sampler.raw_class_labels=None
        print(f"num. of coreset features after sampling: {len(self.coreset_features)}")

    def update(
        self,
        images: torch.Tensor,
        sample_ids: torch.Tensor,
        class_labels: torch.Tensor = None,
    ):
        self._forward(images, sample_ids, class_labels=class_labels, accumulate_features=True)

    def inference(self, image):
        self.feature_extractor.eval()
        scores, masks = self.predict(image)
        return scores, masks

    def predict(self, images):
        images = images.to(torch.float).to(self.device)
        batch_size = images.shape[0]
        sample_ids = torch.zeros(batch_size).to(torch.float).to(self.device) 

        with torch.no_grad():
            features, _, patch_shapes = self._extract_features(images, sample_ids, return_patch_shape=True)
            feature_tensor=features
            self.last_diagnostic_predictions={}
            if getattr(self,'cseries',None) is not None and self.cseries.routing:
                patch_scores=self.cseries.route(features.reshape(batch_size,-1,self.feature_dim),
                    self.coreset_features,self.sampler.optimization_task_labels).flatten().cpu().numpy()
            else:
                features = np.asarray(features.cpu().numpy())
                patch_scores = self.anomaly_scorer.predict([features])[0]
            if getattr(self,'diagnostic_banks',{}):
                from corelib.cseries import nearest
                current_dist,current_source=nearest(feature_tensor,self.coreset_features,self.sampler.optimization_task_labels)
                current_source=current_source.reshape(batch_size,-1)
                current_dist=current_dist.reshape(batch_size,-1)
                for name,bank in self.diagnostic_banks.items():
                    alt,alt_source=nearest(feature_tensor,bank,self.diagnostic_labels[name])
                    alt=alt.reshape(batch_size,-1);alt_source=alt_source.reshape(batch_size,-1)
                    score=alt.max(1).values.cpu().numpy()
                    mask=self.anomaly_segmentor.convert_to_segmentation(alt.reshape(batch_size,*patch_shapes).cpu().numpy())
                    self.last_diagnostic_predictions[name]=(score,mask)
                    if name=='union':
                        self.last_neighbour_diagnostic={
                            'new_task_nn_fraction':(alt_source==self.sampler.optimization_task_index).float().mean(1).cpu().tolist(),
                            'source_switch_fraction':(alt_source!=current_source).float().mean(1).cpu().tolist(),
                            'union_mean_score':alt.mean(1).cpu().tolist(),
                            'current_mean_score':current_dist.mean(1).cpu().tolist()}
            image_scores=patch_scores
            image_scores = self.feature_extractor['patchifier'].unpatch_scores(
                image_scores, batchsize=batch_size
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.feature_extractor['patchifier'].score(image_scores)

            patch_scores = self.feature_extractor['patchifier'].unpatch_scores(
                patch_scores, batchsize=batch_size
            )
            patch_scores = patch_scores.reshape(batch_size, patch_shapes[0], patch_shapes[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        scores = [x for x in image_scores]
        masks = [x for x in masks]
        return scores, masks   
    
    def build_faiss(self, faiss_on_gpu, faiss_num_workers, anomaly_score_num_nn, backend="faiss"):
        if backend == "torch":
            from corelib.exact_nn import TorchExactNN
            nn_method = TorchExactNN(self.device, metric=self.sampler.metric)
        elif backend == "faiss":
            nn_method = corelib.utils.FaissNN(faiss_on_gpu, faiss_num_workers)
        else:
            raise ValueError(f"Unknown nearest-neighbour backend: {backend}")
        self.anomaly_scorer = corelib.utils.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )
        self.anomaly_scorer.fit(
            detection_features=[self.coreset_features.detach().cpu().numpy()]
        )
        self.anomaly_segmentor = corelib.utils.RescaleSegmentor(
            device=self.device, target_size=self.image_size
        )

    def save_model(self, step):
        # C experiments retain metrics/diagnostics; only the C0 A2 final bank persists.
        if getattr(self,'cseries',None) is not None and (self.cseries_variant!='C0' or step!=15):
            return
        coreset_path = os.path.join(self.save_model_path, f'model_s{step}.pt')
        os.makedirs(os.path.dirname(coreset_path), exist_ok=True)
        ckpt = {"coreset": self.coreset_features.detach().clone(), 
                "sample_ids": self.coreset_sample_ids}
        if hasattr(self.sampler, "optimization_task_labels"):
            ckpt["prototype_task_labels"] = self.sampler.optimization_task_labels.detach().clone()
            ckpt["refinement_source_indices"] = self.sampler.last_optimization_source_indices.detach().clone()
        torch.save(ckpt, coreset_path)
        print(f"Saved the coreset at {coreset_path}!")

    def load_model(self):
        ckpt = torch.load(os.path.join(self.save_model_path, 'model.pt'))
        self.coreset_features = ckpt["coreset"].cpu()
        self.coreset_sample_ids = ckpt["sample_ids"]
        print(f"loaded coreset")
        return ckpt

    def load_model_step(self, step):
        ckpt = torch.load(os.path.join(self.save_model_path, f'model_s{step}.pt'))
        self.coreset_features = ckpt["coreset"].cpu()
        self.coreset_sample_ids = ckpt["sample_ids"]
        print(f"loaded coreset")
        return ckpt

    def get_metric_results(self, scores, segmentations, labels_gt, masks_gt, classes):
        result_dict = {}
        lengths = [len(x) for x in (scores, segmentations, labels_gt, masks_gt, classes)]
        if not lengths[0] or len(set(lengths)) != 1:
            raise ValueError(f"Evaluation inputs must have equal nonzero lengths: {lengths}")
        for score, seg, lbl, msk_gt, cls in zip(scores, segmentations, labels_gt, masks_gt, classes):
            if not cls in result_dict.keys():
                result_dict[cls] = {'scores':[score],
                                        'segmentations':[seg],
                                        'labels_gt':[lbl],
                                        'masks_gt':[msk_gt]}
            else:
                result_dict[cls]['scores'].append(score)
                result_dict[cls]['segmentations'].append(seg)
                result_dict[cls]['labels_gt'].append(lbl)
                result_dict[cls]['masks_gt'].append(msk_gt)

        result_agg = []
        for k, v in result_dict.items():
            scores = v['scores']
            segmentations = v['segmentations']
            labels_gt = v['labels_gt']
            masks_gt = v['masks_gt']
            auroc_img, auroc_pix, f1_img, f1_pix, ap_img, ap_pix, aupro = self._evaluate(scores, segmentations, labels_gt, masks_gt)
            per_cls_dict = {
                "classname" : k,
                "auroc_img" : auroc_img * 100,
                "auroc_pix" : auroc_pix* 100,
                "f1_img" : f1_img* 100,
                "f1_pix" : f1_pix* 100,
                "ap_img" : ap_img* 100,
                "ap_pix" : ap_pix* 100,
                "aupro" : aupro* 100,
            }
            result_agg.append(per_cls_dict)

        return result_agg

    def _evaluate(self, scores, segmentations, labels_gt, masks_gt):
        # Rank metrics use raw, finite scores: no min/max division is needed.
        return metrics.evaluate_anomaly_detection(
            scores, segmentations, labels_gt, masks_gt
        )

    def _save_metrics(self, result_agg, filename):
        result_metric_names = list(result_agg[-1].keys())[1:]
        result_classnames = [results["classname"] for results in result_agg]
        result_scores = [list(results.values())[1:] for results in result_agg]
        means = corelib.utils.compute_and_store_final_results(
            self.save_metrics_path,
            result_scores,
            column_names=result_metric_names,
            row_names=result_classnames,
            file_name=filename
        )
        print(f"Metrics saved: {filename}; {means}", flush=True)

    def _save_segmentations(self, segmentations, image_paths, mask_paths):
        segmentations = np.array(segmentations)
        min_scores = (
            segmentations.reshape(-1)
            .min(axis=-1)
            .reshape(-1, 1, 1)
        )
        max_scores = (
            segmentations.reshape(-1)
            .max(axis=-1)
            .reshape(-1, 1, 1)
        )
        segmentations = (segmentations - min_scores) / (max_scores - min_scores)
        
        def image_transform(image):
            in_std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
            in_mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
            transform_img = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=in_mean, std=in_std),
            ]
            transform_img = transforms.Compose(transform_img)
            image = transform_img(image)
            return np.clip((image.numpy() * in_std + in_mean) * 255, 0, 255).astype(np.uint8)

        def mask_transform(mask):
            transform_mask = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
            transform_mask = transforms.Compose(transform_mask)            
            return transform_mask(mask).numpy()

        corelib.utils.plot_segmentation_images(
            self.save_seg_path,
            image_paths,
            segmentations,
            mask_paths,
            image_transform=image_transform,
            mask_transform=mask_transform,
        )
