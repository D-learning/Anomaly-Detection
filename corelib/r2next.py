"""R2 follow-up: fixed lifetime class quotas and image-predicted routing.

No anomaly/test labels enter either module. Final class count is known (15).
Old memory rows are frozen; predicted routing can still change when classes arrive.
"""
import torch
import torch.nn.functional as F
from corelib.responsibility_memory import BalancedMemory


def lifetime_quota(class_id,budget,total_classes=15):
    if budget<total_classes or not 0<=class_id<total_classes:
        raise ValueError('Need at least one prototype per final class')
    return budget//total_classes + int(class_id<budget%total_classes)


class ReservedMemory(BalancedMemory):
    def __init__(self):
        super().__init__('Rhier');self.level=2

    @torch.no_grad()
    def merge_reserved(self,sampler,old,old_ids,new,new_ids,new_classes):
        """R2 compression operates independently at each class birth, never on old rows."""
        task=getattr(sampler,'optimization_task_index',-1)+1
        n=0 if old is None else len(old)
        pool=new if old is None else torch.cat((old,new))
        ids=new_ids if old is None else torch.cat((old_ids,new_ids))
        allclasses=self.class_ids.clone()
        oldclasses=allclasses[:n].unique()
        if torch.isin(new_classes.unique(),oldclasses).any():
            raise ValueError('Reserved memory requires disjoint class arrivals')
        chosen=[torch.arange(n,device=pool.device)];reports=[]
        # Match the original fixed merge projection for R2 geometry; only current classes compete.
        projected,_=sampler._reduce_features(pool,fixed_merge=True) if sampler.reduce_feature_dim else (pool,None)
        for c in new_classes.unique(sorted=True).tolist():
            rows=torch.where(allclasses==c)[0]
            capacity=lifetime_quota(c,sampler.coreset_size)
            keep=min(len(rows),capacity)
            if len(rows)>keep:
                local=BalancedMemory('Rhier');local.level=2
                local.pending={k:v[rows].clone() for k,v in self.pending.items()}
                local.task=task
                selected,diag=local.compress(pool[rows],projected[rows],keep)
                selected=rows[selected]
            else:selected=rows;diag={'deletions':0}
            chosen.append(selected)
            reports.append(dict(class_id=c,quota=capacity,candidates=len(rows),selected=len(selected),compression=diag))
        source=torch.cat(chosen)
        assert len(source)==len(source.unique()) and len(source)<=sampler.coreset_size
        assert torch.equal(source[:n],torch.arange(n,device=pool.device))
        old_tasks=getattr(sampler,'optimization_task_labels',torch.empty(0,device=pool.device,dtype=torch.long))
        tasklabels=torch.cat((old_tasks,torch.full((len(new),),task,device=pool.device,dtype=torch.long)))
        self.finish(source)
        sampler.optimization_task_index=task
        sampler.optimization_task_labels=tasklabels[source].clone()
        sampler.optimization_class_labels=allclasses[source].clone()
        sampler._last_base_source_indices=source.clone()
        sampler.last_optimization_source_indices=source.clone()
        sampler.last_merge_diagnostics=dict(module='reserved_R2',classes=reports,old_rows=n,old_rows_preserved=n,
            final_budget=sampler.coreset_size,unused_reserved_slots=sampler.coreset_size-len(source))
        sampler.last_optimization_diagnostics=dict(module='reserved_R2',task=task+1,budget=len(source),swaps=0,
            old_rows_bitwise_preserved=old is None or bool(torch.equal(pool[source][:n],old)),
            mode_audit=self.audit(pool[source],ids[source],tasklabels[source]))
        self.class_ids=None;self.raw_classes=None;sampler.pending_merge_diagnostics={}
        sampler._empty_task_coreset_buffer()
        print('[R2-next]',sampler.last_optimization_diagnostics,flush=True)
        return pool[source].clone(),ids[source].clone()


class ImageRouter:
    def __init__(self):
        self.centres={};self.last_classes=None

    @staticmethod
    def descriptor(patches):
        return F.normalize(patches.mean(-2),dim=-1)

    @torch.no_grad()
    def register(self,features,image_ids,class_ids):
        for c in class_ids.unique(sorted=True).tolist():
            if c in self.centres:raise ValueError('Router requires disjoint class arrivals')
            mask=class_ids==c;part=features[mask];ids=image_ids[mask]
            # Image IDs repeat between classes; split by class BEFORE grouping by image.
            unique,inverse=torch.unique(ids,sorted=True,return_inverse=True)
            sums=torch.zeros(len(unique),features.shape[1],device=features.device,dtype=features.dtype)
            sums.index_add_(0,inverse,part)
            counts=torch.bincount(inverse,minlength=len(unique))
            desc=F.normalize(sums/counts[:,None],dim=1)
            mean=F.normalize(desc.mean(0),dim=0)
            chosen=[int(((desc-mean)**2).sum(1).argmin())]
            dist=((desc-desc[chosen[0]])**2).sum(1)
            for _ in range(1,min(4,len(desc))):
                dist[chosen]=-float('inf');i=int(dist.argmax());chosen.append(i)
                dist=torch.minimum(dist,((desc-desc[i])**2).sum(1))
            self.centres[c]=desc[chosen].clone()

    @torch.no_grad()
    def predict_class(self,patches):
        if not self.centres:raise RuntimeError('Router not trained')
        classes=sorted(self.centres)
        desc=self.descriptor(patches)
        distance=torch.stack([torch.cdist(desc,self.centres[c]).square().min(1).values for c in classes],1)
        return torch.tensor(classes,device=patches.device)[distance.argmin(1)]

    @torch.no_grad()
    def score(self,patches,bank,class_ids,k=1):
        self.last_classes=self.predict_class(patches)
        scores=torch.empty(patches.shape[:2],device=patches.device,dtype=patches.dtype)
        # Match TorchExactNN float32 squared-L2 and neighbour mean, chunking query rows.
        for c in self.last_classes.unique().tolist():
            image_rows=torch.where(self.last_classes==c)[0];memory=bank[class_ids==c]
            if len(memory)<k:raise ValueError('Insufficient class bank for requested kNN')
            query=patches[image_rows].flatten(0,1);parts=[]
            norm=memory.square().sum(1)
            for q in query.split(512):
                squared=(q.square().sum(1,keepdim=True)+norm[None,:]-2*(q@memory.T)).clamp_min_(0)
                parts.append(squared.topk(k,largest=False,sorted=True).values.mean(1))
            scores[image_rows]=torch.cat(parts).reshape(len(image_rows),patches.shape[1])
        return scores

    def audit(self,memory):
        result=dict(memory)
        extra=sum(t.numel()*t.element_size()+8+4096 for t in self.centres.values())
        result['persistent_total_bytes']+=extra;result['auxiliary_bytes']+=extra
        result.update(router_bytes=extra,router_classes=len(self.centres),router_training='normal images only',test_class_input=False)
        if result['persistent_total_bytes']>result['baseline_limit_bytes']:raise RuntimeError('Router exceeds common byte ceiling')
        return result
