import abc
from typing import Union, List
from tqdm import tqdm
import gc

import torch
import numpy as np
import faiss
from numpy.linalg import pinv
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EmpiricalCovariance
import corelib.automl as automl
from collections import defaultdict
import torch.nn.functional as F
import time


class IdentitySampler:
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        return features


class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        if not 0 < percentage < 1:
            raise ValueError("Percentage value not in (0, 1).")
        self.percentage = percentage
        self._status = defaultdict(float)

    def get_status(self):
        return self._status


    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)



class MetaGreedyCoresetSampler(BaseSampler):
    def __init__(
        self,
        device: torch.device,
        sampling_ratio: float = 0.01,
        feature_dim: int = 1024,
        coreset_size: int = 40000,
        reduce_feature_dim: bool= True,
        project_dim: int = 128,
        approximate_ratio: float = 1.0,
        metric: str = 'L2',
    ):
        super().__init__(percentage=sampling_ratio)

        self.feature_dim = feature_dim
        self.reduce_feature_dim = reduce_feature_dim
        self.project_dim = project_dim
        self.metric = metric
        self.device = device
        self.coreset_size = coreset_size
        if coreset_size is None:
            self.coreset_size = self._get_coreset_size(feature_dim=feature_dim)

        self.approximate_ratio = approximate_ratio

        self._max_coreset_buffer = torch.zeros(
            (coreset_size * 2, feature_dim), requires_grad=False, device=device,
        )  
        self._max_coreset_sample_ids = torch.zeros(
            (coreset_size * 2), requires_grad=False, device=device,
        )  
        self._max_task_coreset_buffer = torch.zeros(
            (coreset_size * 2, feature_dim), requires_grad=False, device=device,
        )  
        self._max_task_coreset_sample_ids = torch.zeros(
            (coreset_size * 2), requires_grad=False, device=device,
        )  

        print(f"coreset size: {coreset_size}")


    def _empty_task_coreset_buffer(self,):
        self._max_task_coreset_buffer = torch.zeros(
            (self.coreset_size * 2, self.feature_dim), requires_grad=False, device=self.device,
        )  
        self._max_task_coreset_sample_ids = torch.zeros(
            (self.coreset_size * 2), requires_grad=False, device=self.device,
        )  

    def _get_coreset_size(
        self, 
        feature_dim: int=1024
    ):
        coreset_size = automl.get_max_len_feas(
            feature_dim,
            device=self.device.type,
            max_usage=0.5,
            memory_type="available",
            device_index=self.device.index,
        )
        return coreset_size

    def _adjust_max_coreset_size(self, feature_dim, max_coreset_size):

        max_coreset_size_limit = automl.get_max_num_feas_for_pairwise_dist_matrix(
            fea_dim=feature_dim,
            device_type="cpu",
            max_usage=0.50,
        )

        if max_coreset_size < max_coreset_size_limit:
            return max_coreset_size

        max_coreset_size = max_coreset_size_limit

        return max_coreset_size

    def run(
            self, 
            features, 
            sample_ids, 
            task_coreset=None,
            base_coreset=None,
        ):

        if self.percentage == 1:
            return features, None
        
        assert isinstance(features, torch.Tensor)
        assert isinstance(task_coreset, (torch.Tensor, type(None)))

        if task_coreset is not None:
            print(f"task coreset size: {len(task_coreset)}")
            n_task_coreset = task_coreset.shape[0]
            self._max_task_coreset_buffer[:n_task_coreset] = task_coreset.clone()
        else:
            n_task_coreset = 0

        reduced_features, reduced_task_coreset = self._reduce_features(
            features, base_features=task_coreset
        )

        if task_coreset is not None:raise RuntimeError('R-balanced requires one full-task sampling call')
        sample_indices = self._greedy_sample_indices(reduced_features, reduced_task_coreset)
        
        self.last_task_source_indices=torch.as_tensor(sample_indices,device=features.device,dtype=torch.long)
        sampled_features = features[sample_indices]
        sampled_ids = sample_ids[sample_indices]

        n_samples = len(sample_indices)
        n_samples_to_delete = n_task_coreset + n_samples - self.coreset_size

        if n_samples_to_delete > 0:
            coreset_features, coreset_sample_ids = self._remove_and_add_to_task_coreset(
                sampled_features, sampled_ids, n_task_coreset
            )
            sample_indices = None
        else:
            coreset_features, coreset_sample_ids = self._add_to_task_coreset(
                sampled_features, sampled_ids, n_task_coreset
            )

        return coreset_features.clone(), coreset_sample_ids.clone()


    def meta_greedy_sampling(
            self, 
            task_coreset,
            task_sample_ids,
            base_coreset=None,
            base_sample_ids=None,
            raw_task_features=None,
            raw_task_ids=None,
            task_class_labels=None,
            base_class_labels=None
        ):
        # H1-only metadata: row indices in [old memory; current task coreset].
        # Never replace image-level sample_ids or change the sampling choices.
        self._last_base_source_indices = None
        self.last_h1_diagnostics = None
        if base_coreset is not None:
            print(f"base coreset size: {len(base_coreset)}")
            n_base_coreset = base_coreset.shape[0]
            self._max_coreset_buffer[:n_base_coreset] = base_coreset.clone()
            if base_sample_ids is not None:
                self._max_coreset_sample_ids[:n_base_coreset] = base_sample_ids.clone()
        else:
            n_base_coreset = 0
        
        import os
        mode_variant=os.environ.get('OPT_VARIANT','')
        if mode_variant in ('M1','M2','M3','Rdiag','Rfix','Rresp','Rhier','Rctrl','Rimage'):
            from corelib.mode_memory import ModeMemory
            if not hasattr(self,'mode_state'):
                if mode_variant=='Rfix':
                    from corelib.fixed_mode_memory import FixedModeMemory
                    self.mode_state=FixedModeMemory()
                elif mode_variant in ('Rresp','Rhier','Rctrl','Rimage'):
                    from corelib.responsibility_memory import make_memory
                    self.mode_state=make_memory(mode_variant)
                else:self.mode_state=ModeMemory()
            self.mode_state.class_ids = torch.cat((base_class_labels,task_class_labels)) if base_class_labels is not None else task_class_labels
            self.mode_state.raw_classes=getattr(self,"raw_class_labels",None)
            self.mode_state.prepare(base_coreset,task_coreset,raw_task_features,raw_task_ids,
                                    getattr(self,'optimization_task_index',-1)+1)
        n_samples = len(task_coreset)
        if os.environ.get('G_VARIANT') in ('G1','G3'):
            return self.mode_state.merge_reserved(self,base_coreset,base_sample_ids,task_coreset,task_sample_ids,task_class_labels)
        n_samples_to_delete = n_base_coreset + n_samples - self.coreset_size

        if n_samples_to_delete > 0:
            coreset_features, coreset_sample_ids = self._remove_and_add_to_base_coreset(
                task_coreset, task_sample_ids, n_base_coreset
            )
        else:
            coreset_features, coreset_sample_ids = self._add_to_base_coreset(
                task_coreset, task_sample_ids, n_base_coreset
            )
        import os
        optimization = os.environ.get("OPT_VARIANT", "")
        if optimization:
            current_task=getattr(self,"optimization_task_index",-1)+1
            self.last_merge_diagnostics=getattr(self,"pending_merge_diagnostics",{})
            self.pending_merge_diagnostics={}
            if (optimization=='D0' and current_task+1 in (8,9,13)) or (optimization=='Rdiag' and current_task+1 in (9,10,13,15)):
                self.merge_bank_for_diagnostic=coreset_features.clone()
                old_labels=getattr(self,"optimization_task_labels",torch.empty(0,device=self.device,dtype=torch.long))
                pool_labels=torch.cat((old_labels,torch.full((len(task_coreset),),current_task,device=self.device,dtype=torch.long)))
                self.merge_labels_for_diagnostic=pool_labels[self._last_base_source_indices].clone()
            from corelib.refinement import refine
            if optimization in ('M1','M2','M3','Rdiag','Rfix','Rresp','Rhier','Rctrl','Rimage'):
                def refine(**kw):
                    source=kw['initial_source_indices'].clone()
                    pool=kw['task_memory'] if kw['old_memory'] is None else torch.cat((kw['old_memory'],kw['task_memory']))
                    ids=kw['task_ids'] if kw['old_ids'] is None else torch.cat((kw['old_ids'],kw['task_ids']))
                    labels=torch.cat((kw['old_task_labels'],torch.full((len(kw['task_memory']),),kw['current_task'],device=pool.device,dtype=torch.long)))
                    old_cls=kw.get('old_class_labels')
                    cur_cls=kw.get('current_class_labels')
                    if old_cls is None: old_cls=torch.empty(0,device=pool.device,dtype=torch.long)
                    if cur_cls is None: cur_cls=torch.empty(0,device=pool.device,dtype=torch.long)
                    class_ids=torch.cat((old_cls,cur_cls))
                    self.mode_state.class_ids=class_ids
                    if hasattr(self.mode_state,"register_current"):self.mode_state.register_current(pool[source])
                    restoration=dict(swaps=0,history=[])
                    if optimization in ('M3','Rdiag','Rfix','Rresp','Rhier','Rctrl','Rimage') and kw['old_memory'] is not None:
                        source,restoration=self.mode_state.restore(pool,source,len(kw['old_memory']))
                    self.mode_state.finish(source)
                    self.mode_state.class_ids=None
                    self.mode_state.raw_classes=None
                    self.optimization_class_labels=class_ids[source].clone()
                    diag=dict(module=optimization,task=kw['current_task']+1,budget=len(source),
                        swaps=restoration['swaps'],restoration=restoration,
                        mode_audit=self.mode_state.audit(pool[source],ids[source],labels[source]))
                    return pool[source].clone(),ids[source].clone(),source,labels[source].clone(),diag
            old_labels = getattr(self, "optimization_task_labels", torch.empty(0,device=self.device,dtype=torch.long))
            current_task = getattr(self, "optimization_task_index", -1) + 1
            self.optimization_task_index = current_task
            (coreset_features, coreset_sample_ids, source_indices,
             self.optimization_task_labels, diagnostics) = refine(
                old_memory=base_coreset, old_ids=base_sample_ids,
                task_memory=task_coreset, task_ids=task_sample_ids,
                initial_source_indices=self._last_base_source_indices,
                old_task_labels=old_labels, current_task=current_task,
                current_features=raw_task_features,
                        current_class_labels=task_class_labels,
                        old_class_labels=base_class_labels,
                protection=getattr(self,"external_protection",None),
                variant="A2" if optimization.startswith(("C","D","M","R")) else optimization, num_anchors=256,
                max_swaps=8 if optimization in ("B2","B12") else 32,
                deletion_candidates=8 if optimization in ("B2","B12") else 16, addition_candidates=8,
                current_queries=128, alpha=.90, beta=.50,
                current_weight=.50, current_tolerance=.01, worst_tolerance=.01,
            )
            self.last_optimization_diagnostics = diagnostics
            self.last_optimization_source_indices = source_indices
            print(f"[OPT-{optimization}] {diagnostics}", flush=True)
        elif os.environ.get("USE_RSC", "0") == "1" and base_coreset is not None and base_sample_ids is not None:
            from corelib.rsc import rsc_refine, h1_refine
            # P0-06 修复：使用传进来的纯净特征
            raw_features = raw_task_features if raw_task_features is not None else task_coreset
            raw_ids = raw_task_ids if raw_task_ids is not None else task_sample_ids
            
            refine_kwargs = dict(
                old_memory=base_coreset,
                old_ids=base_sample_ids,
                initial_memory=coreset_features,
                initial_ids=coreset_sample_ids,
                current_task_features=raw_features,
                current_task_ids=raw_ids,
                num_anchors=256,
                max_swaps=32,
            )
            # Existing USE_RSC switch is preserved. H1 is on by default inside
            # that switch; USE_H1=0 runs the supplied legacy RSC unchanged.
            if os.environ.get("USE_H1", "1") == "1":
                coreset_features, coreset_sample_ids, diagnostics = h1_refine(
                    **refine_kwargs,
                    initial_source_indices=self._last_base_source_indices,
                    alpha=float(os.environ.get("H1_ALPHA", "0.90")),
                    beta=float(os.environ.get("H1_BETA", "0.50")),
                )
                self.last_h1_diagnostics = diagnostics
                print(
                    f"[RSC+H1] refined coreset: {coreset_features.shape}; "
                    f"alpha={diagnostics['alpha']}, beta={diagnostics['beta']}, "
                    f"swaps={diagnostics['swaps']}, "
                    f"mean={diagnostics['mean_before']:.6f}->{diagnostics['mean_after']:.6f}, "
                    f"risk={diagnostics['risk_before']:.6f}->{diagnostics['risk_after']:.6f}, "
                    f"reason={diagnostics['reason']}"
                )
            else:
                coreset_features, coreset_sample_ids = rsc_refine(
                    **refine_kwargs, initial_source_indices=self._last_base_source_indices
                )
                print(f"[RSC] refined coreset: {coreset_features.shape}")

        self._empty_task_coreset_buffer()
        return coreset_features, coreset_sample_ids

    def _add_to_task_coreset(self, sampled_features, sampled_ids, n_base_coreset):
        n_samples = len(sampled_features)
        self._max_task_coreset_buffer[n_base_coreset : n_base_coreset + n_samples] = (
            sampled_features.clone()
        )
        self._max_task_coreset_sample_ids[n_base_coreset : n_base_coreset + n_samples] = (
            sampled_ids.clone()
        )
        coreset_features = self._max_task_coreset_buffer[: n_base_coreset + n_samples].clone()
        coreset_sample_ids = self._max_task_coreset_sample_ids[:n_base_coreset + n_samples].clone()

        print(f"The num. of newly added features to task coreset: {n_samples}")
        return coreset_features, coreset_sample_ids


    def _add_to_base_coreset(self, sampled_features, sampled_ids, n_base_coreset):
        n_samples = len(sampled_features)
        self._max_coreset_buffer[n_base_coreset : n_base_coreset + n_samples] = (
            sampled_features.clone()
        )
        self._max_coreset_sample_ids[n_base_coreset : n_base_coreset + n_samples] = (
            sampled_ids.clone()
        )
        coreset_features = self._max_coreset_buffer[: n_base_coreset + n_samples].clone()
        coreset_sample_ids = self._max_coreset_sample_ids[:n_base_coreset + n_samples].clone()

        print(f"The num. of newly added features to base coreset: {n_samples}")

        self._last_base_source_indices = torch.arange(
            n_base_coreset + n_samples, device=coreset_features.device
        )
        return coreset_features, coreset_sample_ids



    def _remove_and_add_to_task_coreset(self, sampled_features, sampled_ids, n_base_coreset):
        print("Meta-greedy sampling task coreset...")
        n_samples = len(sampled_features)

        self._max_task_coreset_buffer[n_base_coreset:n_base_coreset + n_samples] = sampled_features.clone()
        self._max_task_coreset_sample_ids[n_base_coreset:n_base_coreset + n_samples] = sampled_ids.clone()
        
        all_features = self._max_task_coreset_buffer[:n_base_coreset + n_samples].clone()
        all_sample_ids = self._max_task_coreset_sample_ids[:n_base_coreset + n_samples].clone()

        if self.reduce_feature_dim:
            features, _ = self._reduce_features(all_features)
        else:
            features = all_features

        most_greedy_indices = approx_greedy_sample(
            all_features=features,
            n_samples=self.coreset_size,
            device=self.device,
            approximation_rate=self.approximate_ratio,
            metric=self.metric,
        )
        self._max_task_coreset_buffer[: self.coreset_size] = all_features[most_greedy_indices].clone()
        self._max_task_coreset_sample_ids[: self.coreset_size] = all_sample_ids[most_greedy_indices].clone()

        _num_features_added_to_coreset = np.sum(
            np.array(most_greedy_indices) >= n_base_coreset
        )
        print(
            f"The num. of newly added features to task coreset is {_num_features_added_to_coreset}"
        )

        return all_features[most_greedy_indices].clone(), all_sample_ids[most_greedy_indices].clone()


    def _remove_and_add_to_base_coreset(self, sampled_features, sampled_ids, n_base_coreset):
        import os
        print("Meta-greedy sampling base coreset...")
        n_samples = len(sampled_features)

        self._max_coreset_buffer[n_base_coreset:n_base_coreset + n_samples] = sampled_features.clone()
        self._max_coreset_sample_ids[n_base_coreset:n_base_coreset + n_samples] = sampled_ids.clone()
        
        all_features = self._max_coreset_buffer[:n_base_coreset + n_samples].clone()
        all_sample_ids = self._max_coreset_sample_ids[:n_base_coreset + n_samples].clone()

        if self.reduce_feature_dim:
            features, _ = self._reduce_features(all_features, fixed_merge=os.environ.get("OPT_VARIANT","") in ("D1","Dgreedy","D2","D2ctrl","D3","M0","M1","M2","M3","Rdiag","Rfix","Rresp","Rhier","Rctrl","Rimage"))
        else:
            features = all_features

        import os
        quota_kwargs = {}
        if os.environ.get("OPT_VARIANT") in ("B1", "B12"):
            from corelib.quota import task_floors, coverage_seeds
            old_labels=getattr(self,"optimization_task_labels",torch.empty(0,device=self.device,dtype=torch.long))
            current_task=getattr(self,"optimization_task_index",-1)+1
            pool_labels=torch.cat((old_labels,torch.full((n_samples,),current_task,device=self.device,dtype=torch.long)))
            floors=task_floors(pool_labels,self.coreset_size,current_task+1,.15)
            quota_kwargs["required_indices"]=coverage_seeds(features,pool_labels,floors)

        variant=os.environ.get("OPT_VARIANT","")
        if variant in ('M1','M2','M3','Rdiag','Rfix','Rresp','Rhier','Rctrl','Rimage'):
            chosen,self.pending_merge_diagnostics=self.mode_state.compress(
                all_features,features,self.coreset_size,use_images=variant in ('M2','M3','Rdiag','Rfix','Rresp','Rhier','Rctrl','Rimage'))
            most_greedy_indices=chosen.cpu().numpy()
        elif variant in ("D2","D2ctrl","D3"):
            from corelib.coverage_compression import compress
            current_task=getattr(self,"optimization_task_index",-1)+1
            old_labels=self.optimization_task_labels
            pool_labels=torch.cat((old_labels,torch.full((n_samples,),current_task,device=self.device,dtype=torch.long)))
            most_greedy_indices,self.pending_merge_diagnostics=compress(
                all_features,features,pool_labels,n_base_coreset,self.merge_raw_features,
                self.merge_raw_ids,self.coreset_size,current_task,
                independent=getattr(self,"merge_independent_queries",None) if variant=='D3' else None)
        else:
            most_greedy_indices = approx_greedy_sample(
                all_features=features,
                n_samples=self.coreset_size,
                device=self.device,
                approximation_rate=0. if variant=='Dgreedy' else self.approximate_ratio,
                metric=self.metric,
                **quota_kwargs,
            )
        self._max_coreset_buffer[: self.coreset_size] = all_features[most_greedy_indices].clone()
        self._max_coreset_sample_ids[: self.coreset_size] = all_sample_ids[most_greedy_indices].clone()

        _num_features_added_to_coreset = np.sum(
            np.array(most_greedy_indices) >= n_base_coreset
        )
        print(
            f"The num. of newly added features to base coreset is {_num_features_added_to_coreset}"
        )

        self._last_base_source_indices = torch.as_tensor(
            most_greedy_indices, dtype=torch.long, device=all_features.device
        ).clone()
        return all_features[most_greedy_indices].clone(), all_sample_ids[most_greedy_indices].clone()

    def _reduce_features(
        self, features: torch.Tensor, base_features: torch.Tensor = None, fixed_merge=False
    ):
        if features.shape[1] == self.project_dim:
            return features, base_features
        mapper = torch.nn.Linear(
            features.shape[1], self.project_dim, bias=False
        )
        _ = mapper.to(self.device)
        if fixed_merge:
            # Preserve original global RNG consumption, replace only merge geometry.
            generator=torch.Generator(device=self.device).manual_seed(19042)
            with torch.no_grad():
                mapper.weight.copy_(torch.randn(mapper.weight.shape,device=self.device,generator=generator)/self.project_dim**.5)
        features = features.to(self.device)
        with torch.no_grad():
            reduced_features = mapper(features)

        if base_features is not None:
            base_features = base_features.to(self.device)
            with torch.no_grad():
                reduced_base_features = mapper(base_features)
        else:
            reduced_base_features = None

        return reduced_features, reduced_base_features

    def _class_seeds(self,features,n_samples):
        import os
        if int(os.environ.get('RB_VARIANT','R0')[1:])<3:return None
        labels=self.raw_class_labels
        seeds=[]
        for c in labels.unique(sorted=True):
            rows=torch.where(labels==c)[0]
            # Actual source row nearest class centre, no synthetic prototype.
            centre=features[rows].mean(0)
            seeds.append(int(rows[(features[rows]-centre).square().sum(1).argmin()]))
        if len(seeds)>n_samples:raise RuntimeError('Task budget smaller than number of classes')
        return seeds

    def _greedy_sample_indices(self, features, base_coreset):

        print("Greedy sampling...")

        n_samples = max(1, int(len(features) * self.percentage))
        n_samples = min(n_samples, self.coreset_size)

        import os
        if os.environ.get('G_VARIANT') in ('G1','G3'):
            from corelib.full15_memory import class_candidates
            return class_candidates(features,self.raw_class_labels,n_samples,self.device)
        if os.environ.get('FULL15_VARIANT') in ('F2','F4'):
            from corelib.full15_memory import class_candidates
            if base_coreset is not None:raise ValueError('Full15 samples each task once')
            return class_candidates(features,self.raw_class_labels,n_samples,self.device)

        sample_indices = iterative_greedy_sampling(
            features=features,
            n_samples=n_samples,
            device=self.device,
            coreset=base_coreset,
            required_indices=self._class_seeds(features,n_samples),
        )

        return sample_indices




def greedy_sample_indices(
    features: torch.Tensor,
    n_samples: int,
    device: torch.device,
    f2f_pdist_mat: torch.Tensor = None,
    base_coreset: torch.Tensor = None,
    metric = "L2",
) -> list:
    n_feas = len(features)
    if n_samples == 0:
        return []
    if n_samples == n_feas:
        return torch.arange(n_samples).tolist()
    else:
        assert n_samples < n_feas

    if f2f_pdist_mat is None:
        f2f_pdist_mat = compute_pdist(features, features, device=device, metric=metric)

    assert n_feas == f2f_pdist_mat.shape[0] and n_feas == f2f_pdist_mat.shape[1]

    if base_coreset is None:
        f2c_dists = torch.norm(f2f_pdist_mat, dim=1)
    else:
        f2c_pdist_mat = torch.cdist(features, base_coreset)
        f2c_dists = f2c_pdist_mat.min(dim=-1).values
    f2c_dists = f2c_dists.to(device)
    f2f_pdist_mat = f2f_pdist_mat.to(device)
    sample_indices = []

    for _ in tqdm(
        range(n_samples), leave=False, total=n_samples, desc="Greedy sampling..."
    ):
        selected_idx = torch.argmax(f2c_dists).item()  # this operaiont is extremely light, hence no need to be async
        sample_indices.append(selected_idx)

        dists_feas_to_selected_fea = f2f_pdist_mat[:, selected_idx]
        f2c_dists = torch.minimum(f2c_dists, dists_feas_to_selected_fea)
        f2c_dists[sample_indices] = -float("inf")

    torch.cuda.empty_cache()
    return sample_indices


def brute_greedy_sample_indices(
    features: torch.Tensor,
    n_samples: int,
    device: torch.device,
    base_coreset: torch.Tensor = None,
    metric = "L2",
) -> list:

    n_feas = len(features)
    if n_samples == 0:
        return []
    if n_samples == n_feas:
        return torch.arange(n_samples).tolist()
    else:
        assert n_samples < n_feas

    if base_coreset is None:
        sample_indices = [np.random.randint(len(features))]
        min_distances = torch.norm(features - features[sample_indices[0]], dim=1)
    else:
        batch_size = 10000
        offset = 0
        min_val = torch.inf
        feature_chunks = torch.split(features, batch_size)
        for batch in feature_chunks:
            f2c_pdist_mat = compute_pdist(batch, base_coreset, device=device, metric=metric)
            f2c_dist, _ = f2c_pdist_mat.min(dim=-1)
            val, idx = torch.min(f2c_dist, dim=0)
            if val < min_val:
                min_val = val
                min_idx = idx + offset
            offset += batch_size
        sample_indices = [min_idx.item()]
        min_distances = torch.norm(features - features[sample_indices[0]], dim=1)

    for _ in tqdm(range(1, n_samples), desc="brute greedy sampling", leave=False):
        current_distances = torch.norm(features - features[sample_indices[-1]], dim=1)
        min_distances = torch.minimum(min_distances, current_distances)
        sample_indices.append(torch.argmax(min_distances, dim=0).item())
        current_distances = current_distances.cpu()
        del current_distances
        gc.collect()
        torch.cuda.empty_cache()

    return sample_indices


def compute_pdist(
    X: torch.Tensor,
    Y: torch.Tensor,
    device: torch.device,
    batch_size= None,
    metric="L2",
):
    if batch_size is None:
        # auto batch size
        batch_size = automl.auto_get_max_query_size_pdist(  # FIXME:
            fea_dim=X.shape[-1],
            n_support=len(Y),
            max_usage=0.1,
            device_type=device.type,
            device_index=device.index,
        )
        batch_size = max(min(256, batch_size), 128)

    if metric == "L2":
        _compute_pdist = _compute_pdist_l2
    elif metric == 'cosine':
        _compute_pdist = _compute_pdist_cosine
    else:
        raise ValueError("metric incorrectly set")
    return _compute_pdist(X=X, Y=Y, device=device, batch_size=batch_size)


def _compute_pdist_l2(
    X: torch.Tensor, Y: torch.Tensor, device: torch.device, batch_size: int = 128
):
    X_pinned = X.pin_memory()

    Yt = Y.to(device, non_blocking=True).T
    Yt_norm = (Yt.pow(2)).sum(dim=0, keepdim=True)

    i0_idxes = range(0, len(X), batch_size)
    Z_slices = [[] for _ in range(len(i0_idxes))]

    for n, i0 in tqdm(
        enumerate(i0_idxes), desc="Computing pdist...", total=len(i0_idxes), leave=False
    ):
        i1 = min(len(X), i0 + batch_size)

        Z_slices[n] = pdist_l2(
            X_pinned[i0:i1].to(device, non_blocking=True), Yt, Yt_norm
        ).to("cpu", non_blocking=True)

    return torch.cat(Z_slices)


def pdist_l2(X: torch.Tensor, Yt: torch.Tensor, Yt_norm: torch.Tensor) -> torch.Tensor:
    X_times_X = (X.pow(2)).sum(dim=1, keepdim=True)
    X_times_Y = X.mm(Yt)
    distance_matrix = -2 * X_times_Y + X_times_X + Yt_norm
    return distance_matrix


def _compute_pdist_cosine(
    X: torch.Tensor, Y: torch.Tensor, device: torch.device, batch_size: int = 128
):
    X_pinned = X.pin_memory()
    Y = Y.to(device, non_blocking=True)

    i0_idxes = range(0, len(X), batch_size)
    Z_slices = [[] for _ in range(len(i0_idxes))]

    for n, i0 in tqdm(
        enumerate(i0_idxes), desc="Computing pdist...", total=len(i0_idxes), leave=False
    ):
        i1 = min(len(X), i0 + batch_size)
        Z_slices[n] = pdist_cosine(
            X_pinned[i0:i1].to(device, non_blocking=True), Y
        ).to("cpu", non_blocking=True)

    return torch.cat(Z_slices)

def pdist_cosine(x1, x2):
    x1 = x1 / torch.norm(x1, p=2, dim=1, keepdim=True)
    x2 = x2 / torch.norm(x2, p=2, dim=1, keepdim=True)
    cos_sim = torch.matmul(x1, x2.t()).clip(-1, 1)
    cos_dist = 1 - cos_sim
    return cos_dist


@torch.no_grad()
def approx_greedy_sample(
    all_features: torch.Tensor,
    n_samples: int,
    device: torch.device,
    approximation_rate=0.75,
    metric = "L2",
    required_indices=None,
):

    assert 0 <= approximation_rate <= 1.0

    n_feas = all_features.shape[0]
    if not 0 <= n_samples <= n_feas:
        raise ValueError("Requested sample count is outside feature pool")
    if n_samples == 0:
        return []
    if n_samples == n_feas:
        return list(range(n_feas))

    fea_pdist = torch.cdist(all_features, all_features)

    if required_indices is not None and len(required_indices):
        required_indices = torch.as_tensor(required_indices, device=all_features.device, dtype=torch.long)
        if len(required_indices.unique()) != len(required_indices) or len(required_indices)>n_samples:
            raise ValueError("Invalid mandatory prototype source rows")
        if required_indices.min()<0 or required_indices.max()>=n_feas:
            raise ValueError("Mandatory source row outside pool")
        n_pre = max(len(required_indices), int(approximation_rate*n_samples))
        novelty=fea_pdist.fill_diagonal_(float("inf")).min(1).values
        fea_pdist.fill_diagonal_(0.)
        novelty[required_indices]=-float("inf")
        extra=novelty.topk(n_pre-len(required_indices),largest=True,sorted=False).indices
        pre=torch.cat((required_indices,extra))
        available=torch.ones(n_feas,device=all_features.device,dtype=torch.bool)
        available[pre]=False
        remaining=torch.where(available)[0]
        post=greedy_sample_indices(features=all_features[remaining],n_samples=n_samples-len(pre),
            device=device,f2f_pdist_mat=fea_pdist[remaining][:,remaining],
            base_coreset=all_features[pre],metric=metric)
        return pre.tolist()+remaining[post].tolist()

    if approximation_rate == 0.0:
        sample_indices = greedy_sample_indices(
            features=all_features,
            f2f_pdist_mat=fea_pdist,
            n_samples=n_samples,
            device=device,
            metric=metric,
        )
        sampled_features = all_features[sample_indices]
    else:
        n_approx = int(approximation_rate * n_samples)
        n_approx = max(n_approx, 1)

        assert n_approx <= n_samples

        dist_fea_to_feaset_wo_itself = (
            fea_pdist.fill_diagonal_(float("inf"))
            .topk(k=1, largest=False, dim=-1)
            .values[:, 0]
        )
        fea_pdist.fill_diagonal_(0.0)

        approx_pre_sampled_idices = dist_fea_to_feaset_wo_itself.topk(
            k=n_approx, largest=True, sorted=False
        ).indices
        approx_pre_coreset = all_features[approx_pre_sampled_idices]

        mask = torch.ones(n_feas, dtype=torch.bool, device=all_features.device)
        mask[approx_pre_sampled_idices] = False
        idxes_to_post_sample = mask.nonzero(as_tuple=False).flatten()
        feas_to_greedy_sample = all_features[idxes_to_post_sample]

        _post_sample_indices = greedy_sample_indices(
            features=feas_to_greedy_sample,
            n_samples=n_samples - len(approx_pre_coreset),
            device=device,
            f2f_pdist_mat=fea_pdist[idxes_to_post_sample][:, idxes_to_post_sample],
            base_coreset=approx_pre_coreset,
            metric=metric,
        )

        post_sample_features = feas_to_greedy_sample[_post_sample_indices]
        pose_sample_feature_indices = idxes_to_post_sample[_post_sample_indices]

        sampled_features = torch.cat([approx_pre_coreset, post_sample_features], dim=0)

        sample_indices = (
            approx_pre_sampled_idices.tolist() + pose_sample_feature_indices.tolist()
        )

    return sample_indices



@torch.no_grad()
def iterative_greedy_sampling(
    features, 
    n_samples, 
    device, 
    coreset=None, 
    number_of_starting_points=10,
    required_indices=None
):
    number_of_starting_points = np.clip(
        number_of_starting_points, None, len(features)
    )
    features = features.to(device)
    start_points = np.random.choice(len(features), number_of_starting_points, replace=False).tolist()

    if coreset is None:
        coreset = features.clone()

    approximate_distance_matrix = torch.cdist(coreset, features[start_points])
    approximate_coreset_anchor_distances = torch.mean(
        approximate_distance_matrix, axis=-1
    ).reshape(-1, 1)
    sample_indices = []
    if required_indices is not None:
        for row in required_indices:
            sample_indices.append(row)
            distance=torch.cdist(coreset,features[row:row+1])
            approximate_coreset_anchor_distances=torch.minimum(approximate_coreset_anchor_distances,distance)
        approximate_coreset_anchor_distances[sample_indices]=-float('inf')

    for _ in tqdm(range(n_samples-len(sample_indices)), desc="Subsampling..."): 
        dist, select_idx = torch.max(approximate_coreset_anchor_distances, dim=0)
        sample_indices.append(select_idx.item())
        coreset_select_distance = torch.cdist(coreset, features[select_idx : select_idx + 1])
        approximate_coreset_anchor_distances = torch.cat(
            [approximate_coreset_anchor_distances, coreset_select_distance],
            dim=-1,
        )
        approximate_coreset_anchor_distances = torch.min(
            approximate_coreset_anchor_distances, dim=1
        ).values.reshape(-1, 1)
        approximate_coreset_anchor_distances[sample_indices] = -float("inf")

    return sample_indices
