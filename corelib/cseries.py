"""Predicted image routing and independent normal-query protection.

No test labels enter routing or training. All persistent tensors are byte-audited.
"""
import torch
from corelib.exact_nn import TorchExactNN


@torch.no_grad()
def nearest(queries, bank, labels=None, chunk=512):
    values=[];sources=[]
    norms=(bank*bank).sum(1)
    for q in queries.split(chunk):
        d=((q*q).sum(1,keepdim=True)+norms[None]-2*q@bank.T).clamp_min_(0)
        v,i=d.min(1);values.append(v);sources.append(i if labels is None else labels[i])
    return torch.cat(values),torch.cat(sources)


class ProtectionState:
    def __init__(self, variant, feature_dim):
        self.variant=variant;self.feature_dim=feature_dim
        self.centers=[];self.radii=[];self.queries=[];self.groups=[];self.birth=[];self.scales=[]
        self.route_counts={'total':0,'accepted':0,'fallback':0,'predicted_tasks':{}}

    @property
    def routing(self):return self.variant in ('C1','C3')

    @property
    def protection(self):return self.variant in ('C2','C3')

    @torch.no_grad()
    def train_center(self, features, sample_ids):
        if not self.routing:return
        _,inverse=torch.unique(sample_ids,return_inverse=True)
        count=int(inverse.max())+1
        pooled=torch.zeros(count,features.shape[1],device=features.device)
        pooled.index_add_(0,inverse,features)
        pooled/=torch.bincount(inverse,minlength=count)[:,None]
        pooled=torch.nn.functional.normalize(pooled,dim=1)
        center=torch.nn.functional.normalize(pooled.mean(0),dim=0)
        distances=1-pooled@center
        self.centers.append(center.clone())
        self.radii.append(torch.quantile(distances,.95).clamp_min(1e-5).clone())

    @torch.no_grad()
    def register_queries(self, raw, sample_ids, bank, task):
        if not self.protection:return
        # One query per selected image where possible, with no global RNG use.
        generator=torch.Generator(device=raw.device).manual_seed(20261106+task)
        unique=torch.unique(sample_ids)
        order=unique[torch.randperm(len(unique),generator=generator,device=raw.device)]
        picks=[]
        for image_id in order:
            indices=torch.where(sample_ids==image_id)[0]
            permutation=torch.randperm(len(indices),generator=generator,device=raw.device)[:min(32,len(indices))]
            candidates=indices[permutation]
            distances,_=nearest(raw[candidates],bank)
            valid=torch.where(distances>1e-5)[0]
            for position in valid:
                candidate=candidates[position]
                # FP32 matrix distances can make an exact self-match slightly positive.
                if not bool((bank==raw[candidate]).all(1).any()):
                    picks.append(candidate);break
            if len(picks)==16:break
        if len(picks)<16:
            raise RuntimeError('Insufficient independent image queries; do not silently shrink protocol')
        query=raw[torch.stack(picks)].clone()
        distances,_=nearest(query,bank)
        distances=distances.sqrt()
        self.queries.append(query)
        self.groups.append(torch.full((len(query),),task,device=raw.device,dtype=torch.long))
        self.birth.append(distances.clone())
        self.scales.append(torch.full_like(distances,float(distances.median().clamp_min(.001))))

    def protection_data(self):
        if not self.queries:return None
        return dict(queries=torch.cat(self.queries),groups=torch.cat(self.groups),
                    birth=torch.cat(self.birth),scales=torch.cat(self.scales))

    def tensors(self):
        return dict(centers=self.centers,radii=self.radii,queries=self.queries,
                    groups=self.groups,birth=self.birth,scales=self.scales)

    def memory_audit(self, bank, ids, labels):
        extra=sum(t.numel()*t.element_size() for seq in self.tensors().values() for t in seq)
        core=sum(t.numel()*t.element_size() for t in (bank,ids,labels))
        limit=5000*(self.feature_dim*4+ids.element_size()+labels.element_size())
        if core+extra>limit:raise RuntimeError(f'Persistent memory budget exceeded: {core+extra}>{limit}')
        return dict(prototype_count=len(bank),core_bytes=core,auxiliary_bytes=extra,
                    persistent_total_bytes=core+extra,baseline_limit_bytes=limit,
                    query_count=sum(len(q) for q in self.queries),task_centers=len(self.centers))

    @torch.no_grad()
    def route(self, patch_features, bank, labels):
        # Input is B x patches x dim. It has no ground-truth task argument.
        global_scores,_=nearest(patch_features.flatten(0,1),bank)
        result=global_scores.reshape(patch_features.shape[:2]).clone()
        if len(self.centers)<2:return result
        pooled=torch.nn.functional.normalize(patch_features.mean(1),dim=1)
        centers=torch.stack(self.centers)
        scores=(1-pooled@centers.T).clamp_min(0)
        distances,top=scores.topk(2,largest=False,dim=1)
        for i in range(len(pooled)):
            chosen=int(top[i,0]);self.route_counts['total']+=1
            key=str(chosen)
            self.route_counts['predicted_tasks'][key]=self.route_counts['predicted_tasks'].get(key,0)+1
            # Fixed training-only confidence thresholds: reject ambiguous/OOD routes.
            confident=(distances[i,1]-distances[i,0])>=.1*distances[i,1].clamp_min(1e-5)
            confident=bool(confident and distances[i,0]<=2*self.radii[chosen])
            subset=bank[labels==chosen]
            if confident and len(subset)>=8:
                result[i]=nearest(patch_features[i],subset)[0]
                self.route_counts['accepted']+=1
            else:self.route_counts['fallback']+=1
        return result
