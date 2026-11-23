"""R2 follow-up: fixed lifetime class quotas and immutable historical prototypes.

No anomaly/test labels enter either module. Final class count is known (15).
Old memory rows are frozen; inference uses the shared global bank.
"""
import torch
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
