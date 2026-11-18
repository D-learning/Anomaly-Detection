"""Mode-aware local contraction. All state counts toward persistent memory.

Modes are connected components of task prototype nearest-neighbour edges no
longer than the task median NN distance. No category quota or test labels.
Projection proposes 16 local replacements, original space checks distortion.
Contraction carries an upper-bound radius to avoid silently forgetting chains.
"""
import numpy as np
import torch


@torch.no_grad()
def pairwise(x,y,chunk=256):
    return torch.cat([torch.cdist(q.float(),y.float(),compute_mode='donot_use_mm_for_euclid_dist') for q in x.split(chunk)])


def lex_improves(new,old):
    """Numerical tolerance, not a trade-off weight or acceptance hyperparameter."""
    for i,(a,b) in enumerate(zip(new,old)):
        tolerance=0 if i==0 else 1e-6*(1+abs(b))
        if a<b-tolerance:return True
        if a>b+tolerance:return False
    return False


class FixedModeMemory:
    def __init__(self):
        self.meta=None
        self.next_mode=0
        self.total_birth_modes=0
        self.retired_modes=0
        self.pending=None

    @torch.no_grad()
    def prepare(self,old,new,raw,raw_ids,task):
        n=0 if old is None else len(old)
        if self.meta is not None and len(self.meta['mode'])!=n:raise RuntimeError('Mode provenance mismatch')
        d=pairwise(new,new);d.fill_diagonal_(float('inf'))
        if len(new)>1:
            nearest,indices=d.min(1)
            threshold=nearest.median().clamp_min(1e-6)
            edges=torch.where(nearest<=threshold)[0].cpu().tolist()
            ix=indices.cpu().numpy();parents=np.arange(len(new))
            def find(a):
                while parents[a]!=a:
                    parents[a]=parents[parents[a]];a=parents[a]
                return a
            for a in edges:
                u=int(a);v=int(ix[u]);a=find(u);b=find(v);parents[max(a,b)]=min(a,b)
            roots=np.array([find(i) for i in range(len(new))]);_,labels=np.unique(roots,return_inverse=True)
            scale=nearest.clamp_min(float(threshold))
        else:
            labels=np.zeros(1,dtype=np.int64);scale=torch.ones(1,device=new.device)
        modes=torch.as_tensor(labels,device=new.device,dtype=torch.long)+self.next_mode
        count=int(modes.unique().numel());self.next_mode+=count;self.total_birth_modes+=count
        # Training image association: nearest prototype of each image's hardest
        # normal patch. A pattern must occur in >1 distinct image to be protected.
        _,inverse=torch.unique(raw_ids,return_inverse=True)
        values=[];owners=[]
        for q in raw.split(256):
            v,i=pairwise(q,new).min(1);values.append(v);owners.append(i)
        values=torch.cat(values);owners=torch.cat(owners)
        image_count=int(inverse.max())+1
        winners=[]
        for image in range(image_count):
            rows=torch.where(inverse==image)[0]
            winners.append(owners[rows[values[rows].argmax()]])
        wins=torch.bincount(torch.stack(winners),minlength=len(new)).float()
        # Sum repeated image maxima per mode; no raw patch caches persist.
        importance=torch.zeros_like(wins)
        for mode in modes.unique():
            mask=modes==mode;support=wins[mask].sum()
            importance[mask]=torch.where(support>1,support/image_count,torch.zeros_like(support))
        fresh=dict(mode=modes,birth_scale=scale.float(),bound=torch.zeros(len(new),device=new.device),
                   importance=importance.float())
        if n:
            fresh={k:torch.cat((self.meta[k],v)) for k,v in fresh.items()}
        self.pending=fresh
        self.pending_pool=new if old is None else torch.cat((old,new))
        self.reference_bound=fresh['bound'].clone()
        self.task=task;self.old_count=n
        self.image_count=image_count
        return fresh

    @torch.no_grad()
    def compress(self,pool,projected,budget,use_images=False):
        data={k:v.cpu().numpy().copy() for k,v in self.pending.items()}
        raw=pairwise(pool,pool).cpu().numpy();np.fill_diagonal(raw,np.inf)
        pd=pairwise(projected,projected).cpu().numpy();np.fill_diagonal(pd,np.inf)
        order=np.argsort(pd,axis=1,kind='stable').astype(np.int32);del pd
        n=len(pool);alive=np.ones(n,dtype=bool);rows=np.arange(n)
        # Advance local neighbour lists after deletion, never reuse dead rows.
        local=order[:,:min(16,n-1)].copy();cursor=np.full(n,min(16,n-1),dtype=np.int32)
        counts=np.bincount(data['mode'],minlength=self.next_mode)
        mode=data['mode'];scale=data['birth_scale'];bound=data['bound'];importance=data['importance']
        eliminated=0;cross=0
        for step in range(n-budget):
            bad=~alive[local]
            for row in np.where(bad.any(1)&alive)[0]:
                keep=local[row,~bad[row]].tolist()
                while len(keep)<local.shape[1] and cursor[row]<n-1:
                    item=int(order[row,cursor[row]]);cursor[row]+=1
                    if alive[item]:keep.append(item)
                # At least budget>=16 in formal runs; small tests pad safely.
                local[row]=np.resize(np.asarray(keep,dtype=np.int32),local.shape[1])
            valid=alive[local]
            edge=raw[rows[:,None],local]
            new_bound=np.maximum(bound[local],bound[:,None]+edge)
            overflow=np.maximum(new_bound-scale[local],0)/scale[local]
            # Prefer same-mode substitutes, but permit cross-mode contraction.
            crossmode=(mode[:,None]!=mode[local])
            orphan=crossmode & (counts[mode][:,None]==1)
            critical=np.maximum(importance[:,None]-importance[local],0) if use_images else np.zeros_like(edge)
            # Lexicographic per-row proposal, then global minimum contraction.
            rank=np.lexsort((local,edge,overflow,critical,orphan,~valid),axis=1)[:,0]
            target=local[rows,rank]
            active=np.where(alive)[0]
            pick=np.lexsort((active,edge[active,rank[active]],overflow[active,rank[active]],
                             critical[active,rank[active]],orphan[active,rank[active]]))[0]
            delete=int(active[pick]);to=int(target[delete])
            assert delete!=to and alive[to]
            if mode[delete]!=mode[to]:
                cross+=1
                if counts[mode[delete]]==1:eliminated+=1
            bound[to]=max(bound[to],bound[delete]+raw[delete,to])
            importance[to]=max(importance[to],importance[delete])
            counts[mode[delete]]-=1;alive[delete]=False
        selected=np.flatnonzero(alive)
        diag=dict(module='local_mode_contraction',task=self.task+1,prototype_budget=budget,
                  deletions=n-budget,mode_disappearances=eliminated,cross_mode_contractions=cross,
                  mode_count_before=int(np.unique(mode).size),mode_count_after=int(np.unique(mode[selected]).size),
                  image_association=use_images,current_images=self.image_count,
                  mean_bound=float(bound[selected].mean()),max_bound=float(bound[selected].max()))
        # Metadata for deleted/restored rows remains available only within merge.
        self.pending['bound']=torch.as_tensor(bound,device=pool.device)
        self.pending['importance']=torch.as_tensor(importance,device=pool.device)
        return torch.as_tensor(selected,device=pool.device,dtype=torch.long),diag

    @torch.no_grad()
    def restore(self,pool,source,old_count,max_swaps=32):
        """Restore vanished modes first; then reduce worst mode birth-scale drift.

        Candidate limits match existing16 deletion/8 addition. Query references
        are current union representatives, not an extra historical feature bank.
        """
        meta=self.pending;mode=meta['mode'];scale=meta['birth_scale'];weight=meta['importance']
        full=pairwise(pool,pool);full.fill_diagonal_(float('inf'))
        # Include a selected witness itself for representation coverage here;
        # unlike D4 LOO, this measures whether the mode has a surviving source.
        full.fill_diagonal_(0)
        initial=source.clone();history=[]
        unique,inverse=mode.unique(sorted=True,return_inverse=True)
        mode_weight=torch.zeros(len(unique),device=pool.device).scatter_reduce_(0,inverse,weight,reduce='amax')
        def scores(coverage,counts):
            if coverage.ndim==1:coverage=coverage[:,None]
            values=torch.zeros(len(unique),coverage.shape[1],device=pool.device)
            values.scatter_reduce_(0,inverse[:,None].expand_as(coverage),coverage,reduce='amax')
            lost=(counts==0).sum(0)
            critical=(values*mode_weight[:,None]).max(0).values
            return torch.stack((lost,critical,values.max(0).values,values.mean(0)),dim=1)
        def objective(s):
            coverage=full[:,s].min(1).values/scale
            counts=torch.bincount(inverse[s],minlength=len(unique))[:,None]
            return tuple(scores(coverage,counts)[0].cpu().tolist())
        before=objective(source)
        for _ in range(max_swaps):
            live=torch.zeros(len(pool),device=pool.device,dtype=torch.bool);live[source]=True
            missing=torch.where(~live)[0]
            if not len(missing):break
            cover=full[:,source].min(1).values/scale
            priority=cover[missing]*(1+weight[missing])
            modecounts=torch.bincount(inverse[source],minlength=len(unique))
            lost_mask=modecounts[inverse[missing]]==0
            addorder=np.lexsort((missing.cpu().numpy(),-priority.cpu().numpy(),~lost_mask.cpu().numpy()))[:8]
            adds=missing[torch.as_tensor(addorder.copy(),device=pool.device)]
            v,ix=full[:,source].topk(2,largest=False)
            cost=torch.zeros(len(source),device=pool.device).scatter_add_(0,ix[:,0],(v[:,1]-v[:,0])/scale)
            singleton=modecounts[inverse[source]]==1
            removals=np.lexsort((np.arange(len(source)),cost.cpu().numpy(),weight[source].cpu().numpy(),singleton.cpu().numpy()))[:16]
            removes=torch.as_tensor(removals.copy(),device=pool.device)
            without=torch.where(ix[:,:1]==removes[None,:],v[:,1:2],v[:,:1])
            trials=torch.minimum(without[:,None,:],full[:,adds,None]).flatten(1)/scale[:,None]
            counts=torch.bincount(inverse[source],minlength=len(unique))[:,None].expand(-1,trials.shape[1]).clone()
            ai=torch.arange(len(adds),device=pool.device).repeat_interleave(len(removes))
            ri=torch.arange(len(removes),device=pool.device).repeat(len(adds))
            col=torch.arange(trials.shape[1],device=pool.device)
            counts[inverse[source[removes[ri]]],col]-=1
            counts[inverse[adds[ai]],col]+=1
            sc=scores(trials,counts).cpu().numpy()
            order=np.lexsort((col.cpu().numpy(),sc[:,3],sc[:,2],sc[:,1],sc[:,0]))
            best=tuple(sc[order[0]].tolist())
            if not lex_improves(best,before):break
            pos=int(order[0]);slot=int(removes[ri[pos]]);uid=int(adds[ai[pos]])
            source[slot]=uid;before=objective(source)
            history.append(dict(slot=slot,source=uid,objective=before))
        return source,dict(swaps=len(history),history=history,before=objective(initial),after=objective(source))

    def finish(self,source):
        before=set(self.pending['mode'].cpu().tolist());after=set(self.pending['mode'][source].cpu().tolist())
        self.retired_modes+=len(before-after)
        # Reassign all pre-merge coverage balls to final representatives. This
        # keeps a conservative triangle-inequality bound even after restoration.
        d=pairwise(self.pending_pool,self.pending_pool[source]);distance,owner=d.min(1)
        propagated=torch.zeros(len(source),device=d.device)
        propagated.scatter_reduce_(0,owner,self.reference_bound+distance,reduce='amax')
        self.meta={k:v[source].clone() for k,v in self.pending.items()}
        self.meta['bound']=torch.maximum(self.meta['bound'],propagated)
        self.pending=None;self.pending_pool=None;self.reference_bound=None

    def audit(self,bank,ids,labels):
        extra=sum(v.numel()*v.element_size() for v in self.meta.values()) if self.meta else 0
        # Lifecycle/task counters conservatively included at 8 bytes each.
        extra+=64
        core=sum(v.numel()*v.element_size() for v in (bank,ids,labels))
        limit=5000*(bank.shape[1]*4+ids.element_size()+labels.element_size())
        if core+extra>limit:raise RuntimeError('Mode metadata exceeds unified memory budget')
        return dict(persistent_total_bytes=core+extra,baseline_limit_bytes=limit,auxiliary_bytes=extra,
                    prototype_count=len(bank),birth_modes=self.total_birth_modes,retired_modes=self.retired_modes,
                    surviving_modes=int(self.meta['mode'].unique().numel()))
