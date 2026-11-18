"""Class coverage and score-preserving compression, normal training data only.

F2/F4 partition task sampling before the first lossy selection. F3/F4 use
normal image patch queries and labelled-as-synthetic feature perturbations.
Queries are proxies, never ground-truth anomaly examples or test images.
"""
import numpy as np
import torch
from corelib.mode_memory import pairwise
from corelib.responsibility_memory import BalancedMemory, continuous_improves


def lexorder(columns, ids):
    return np.lexsort(tuple([ids]+list(reversed(columns))))


def class_candidates(features, labels, count, device):
    """Class-balanced candidate FPS, without forcing a global centre seed."""
    from corelib.sampler import iterative_greedy_sampling
    classes=labels.unique(sorted=True)
    if count<len(classes):raise ValueError('Not enough candidate slots for classes')
    capacity=np.array([int((labels==c).sum()) for c in classes],dtype=np.int64)
    quota=np.minimum(capacity,count//len(classes));remaining=count-int(quota.sum())
    # Water filling with a deterministic population tie-break for remainder slots.
    while remaining:
        eligible=np.where(quota<capacity)[0]
        if not len(eligible):raise ValueError('Candidate count exceeds raw rows')
        order=np.lexsort((eligible,-capacity[eligible],quota[eligible]))
        take=eligible[order[:min(remaining,len(eligible))]];quota[take]+=1;remaining-=len(take)
    chosen=[]
    for c,n in zip(classes,quota):
        rows=torch.where(labels==c)[0]
        local=iterative_greedy_sampling(features[rows],int(n),device,coreset=None)
        chosen.append(rows[torch.as_tensor(local,device=device)])
    return torch.cat(chosen).tolist()


class Full15Memory(BalancedMemory):
    def __init__(self,variant,coverage=False,scoring=False):
        super().__init__('Rhier');self.level=2
        self.full_variant=variant;self.coverage=coverage;self.scoring=scoring
        self.queries=[]

    @torch.no_grad()
    def prepare(self,old,new,raw,raw_ids,task):
        fresh=super().prepare(old,new,raw,raw_ids,task)
        self._fresh_query_start=len(self.queries)
        if self.scoring:
            n=self.old_count;newcls=self.class_ids[n:]
            for c in newcls.unique(sorted=True):
                if any(q['class_id']==int(c) for q in self.queries):
                    raise ValueError('Full15 protocol requires disjoint class arrivals')
                mask=self.raw_classes==c;part=raw[mask];ids=raw_ids[mask]
                image_ids,counts=torch.unique_consecutive(ids,return_counts=True)
                if len(image_ids)!=len(image_ids.unique()):
                    order=torch.argsort(ids,stable=True);part=part[order];ids=ids[order]
                    image_ids,counts=torch.unique_consecutive(ids,return_counts=True)
                counts=counts.cpu().tolist();starts=np.cumsum([0]+counts)
                if len(counts)<4:raise ValueError('Need four distinct normal images/class')
                descriptors=torch.stack([part[a:b].mean(0) for a,b in zip(starts,starts[1:])])
                centre=descriptors.mean(0);chosen=[int((descriptors-centre).square().sum(1).argmin())]
                minimum=(descriptors-descriptors[chosen[0]]).square().sum(1)
                while len(chosen)<4:
                    minimum[chosen]=-float('inf');i=int(minimum.argmax());chosen.append(i)
                    minimum=torch.minimum(minimum,(descriptors-descriptors[i]).square().sum(1))
                local_bank=new[newcls==c];patches=[];synthetic=[]
                scale=fresh['birth_scale'][n:][newcls==c].median().square().clamp_min(1e-6)
                for i in chosen:
                    image=part[starts[i]:starts[i+1]]
                    if len(image)<4:raise ValueError('Need four normal patches/image')
                    d=pairwise(image,local_bank).min(1).values
                    ranks=torch.linspace(0,len(image)-1,4,device=new.device).long()
                    q=image[d.argsort(stable=True)[ranks]].clone();patches.append(q)
                    direction=q[-1]-centre
                    if float(direction.norm())<1e-8:
                        direction=torch.zeros_like(direction);direction[0]=1
                    synthetic.append(q[-1]+direction/direction.norm()*scale.sqrt())
                features=torch.cat((torch.cat(patches),torch.stack(synthetic))).float().clone()
                self.queries.append(dict(class_id=int(c),features=features,
                    birth=pairwise(features,self.pending_pool).min(1).values.square(),
                    scale=scale.clone(),image_ids=image_ids[chosen].clone()))
            self._q=torch.cat([q['features'] for q in self.queries])
            self._birth=torch.stack([q['birth'] for q in self.queries]).cpu().numpy()
            self._scale=torch.stack([q['scale'] for q in self.queries]).cpu().numpy()
            previous=[]
            for i,q in enumerate(self.queries):
                teacher=old if i<self._fresh_query_start else self.pending_pool
                previous.append(pairwise(q['features'],teacher).min(1).values.square())
            self._previous=torch.stack(previous).cpu().numpy()
            self._qd=pairwise(self._q,self.pending_pool).square().cpu().numpy()
        return fresh

    def register_current(self,bank):
        # Registered before compression in prepare, not after a lossy merge.
        return None

    def _query_losses(self,distances):
        """Per-class two-sided patch, image-score and synthetic-gap drift."""
        values=np.asarray(distances)
        if values.ndim==1:values=values[:,None]
        values=values.reshape(len(self.queries),20,-1)
        scale=self._scale[:,None]
        risk=np.zeros((len(self.queries),values.shape[-1]),dtype=np.float32)
        image=values[:,:16].reshape(len(self.queries),4,4,-1).max(2)
        gap=values[:,16:]-image
        for ref in (self._birth,self._previous):
            patch_error=np.abs(values-ref[:,:,None]).max(1)/scale
            ref_image=ref[:,:16].reshape(len(self.queries),4,4).max(2)
            image_error=np.abs(image-ref_image[:,:,None]).max(1)/scale
            gap_error=np.abs(gap-(ref[:,16:]-ref_image)[:,:,None]).max(1)/scale
            risk=np.maximum(risk,np.maximum(patch_error,np.maximum(image_error,gap_error)))
        return risk

    @staticmethod
    def _nearest_two(qd,alive):
        rows=np.flatnonzero(alive);sub=qd[:,rows]
        if len(rows)<2:raise ValueError('Score memory needs at least two prototypes')
        positions=np.argpartition(sub,1,axis=1)[:,:2]
        values=np.take_along_axis(sub,positions,1)
        order=np.argsort(values,axis=1,kind='stable')
        positions=np.take_along_axis(positions,order,1)
        return np.take_along_axis(sub,positions,1),rows[positions]

    @torch.no_grad()
    def compress(self,pool,projected,budget,use_images=True):
        data={k:v.cpu().numpy().copy() for k,v in self.pending.items()}
        raw=pairwise(pool,pool).cpu().numpy();np.fill_diagonal(raw,np.inf)
        pd=pairwise(projected,projected).cpu().numpy();np.fill_diagonal(pd,np.inf)
        task=data['owner_task'];cls=data['owner_class'];mode=data['mode']
        same=(task[:,None]==task[None,:])&(cls[:,None]==cls[None,:]);pd[~same]=np.inf
        order=np.argsort(pd,axis=1,kind='stable').astype(np.int32);del pd
        n=len(pool);rows=np.arange(n);alive=np.ones(n,dtype=bool)
        width=min(16,n-1);local=order[:,:width].copy();cursor=np.full(n,width,dtype=np.int32)
        bound=data['bound'];scale=data['birth_scale'];mass=data['support_mass'];imp=data['importance']
        mode_count=np.bincount(mode,minlength=self.next_mode)
        classes=np.unique(cls);class_index=np.searchsorted(classes,cls)
        query_values=query_owners=None
        if self.scoring:query_values,query_owners=self._nearest_two(self._qd,alive)
        lost=0
        for step in range(n-budget):
            bad=~alive[local]
            for row in np.where(bad.any(1)&alive)[0]:
                keep=local[row,~bad[row]].tolist()
                while len(keep)<width and cursor[row]<n:
                    item=int(order[row,cursor[row]]);cursor[row]+=1
                    if alive[item] and item!=row:keep.append(item)
                if not keep:raise RuntimeError('No contraction candidates')
                local[row]=np.resize(np.asarray(keep,dtype=np.int32),width)
            valid=alive[local]&same[rows[:,None],local]&(local!=rows[:,None])
            edge=raw[rows[:,None],local];radius=np.maximum(bound[local],bound[:,None]+edge)
            expansion=np.maximum(radius-scale[local],0)/scale[local]
            distortion=edge/scale[:,None];critical=distortion*imp[:,None]
            orphan=(mode[:,None]!=mode[local])&(mode_count[mode][:,None]==1)
            if self.coverage and np.count_nonzero(mode_count)<=budget:valid &= ~orphan
            local_columns=[orphan, radius/scale[local],critical,distortion,edge] if self.coverage else [expansion,critical,distortion,edge]
            rank=np.lexsort(tuple([local]+list(reversed(local_columns))+[~valid]),axis=1)[:,0]
            active=np.where(alive&valid[rows,rank])[0]
            if not len(active):raise RuntimeError('Mode/class floor incompatible with budget')
            target=local[rows,rank]
            if self.coverage:
                current=np.zeros(len(classes),dtype=np.float32)
                np.maximum.at(current,class_index[alive],bound[alive]/scale[alive])
                updated=np.maximum(current[class_index],radius[rows,rank]/scale[target])
                trial_worst=np.maximum(current.max(),updated)
                trial_mean=(current.sum()-current[class_index]+updated)/len(classes)
                columns=[orphan[rows,rank],trial_worst,trial_mean,critical[rows,rank],distortion[rows,rank],edge[rows,rank]]
            else:columns=[expansion[rows,rank],critical[rows,rank],distortion[rows,rank],edge[rows,rank]]
            geometry_order=lexorder([x[active] for x in columns],active)
            if self.scoring:
                candidates=np.unique(np.concatenate((active[geometry_order[:32]],query_owners[:,0])))
                candidates=candidates[np.isin(candidates,active)]
                trial=np.where(query_owners[:,:1]==candidates[None,:],query_values[:,1:2],query_values[:,:1])
                risk=self._query_losses(trial)
                pick=lexorder([risk.max(0),risk.mean(0)]+[x[candidates] for x in columns],candidates)[0]
                delete=int(candidates[pick])
            else:delete=int(active[geometry_order[0]])
            to=int(target[delete]);assert same[delete,to] and delete!=to
            if mode_count[mode[delete]]==1 and mode[delete]!=mode[to]:lost+=1
            mode_count[mode[delete]]-=1
            total=int(mass[to])+int(mass[delete]);imp[to]=(imp[to]*mass[to]+imp[delete]*mass[delete])/total
            mass[to]=total;bound[to]=max(bound[to],bound[delete]+raw[delete,to]);alive[delete]=False
            if self.scoring:
                affected=(query_owners==delete).any(1)
                if affected.any():
                    vals,owners=self._nearest_two(self._qd[affected],alive)
                    query_values[affected]=vals;query_owners[affected]=owners
        selected=np.flatnonzero(alive)
        for key,value in [('bound',bound),('importance',imp),('support_mass',mass)]:self.pending[key]=torch.as_tensor(value,device=pool.device)
        diag=dict(module=self.full_variant,task=self.task+1,deletions=n-budget,
            modes_retired_in_compression=lost,mode_count_after=int(np.unique(mode[selected]).size),
            cross_class_transfers=0,coverage=self.coverage,scoring=self.scoring)
        if self.scoring:
            diag['score_proxy_class_ids']=[q['class_id'] for q in self.queries]
            diag['score_proxy_before_compression']=self._query_losses(self._qd.min(1))[:,0].tolist()
            diag['score_proxy_after_compression']=self._query_losses(query_values[:,0])[:,0].tolist()
        return torch.as_tensor(selected,device=pool.device),diag

    @torch.no_grad()
    def restore(self,pool,source,old_count,max_swaps=32):
        ref=self.reference_meta;classes=ref['owner_class'];tasks=ref['owner_task']
        full=pairwise(pool,pool)
        full.masked_fill_((classes[:,None]!=classes[None,:])|(tasks[:,None]!=tasks[None,:]),float('inf'))
        radius=ref['bound'];scale=ref['birth_scale'];unique,inverse=ref['mode'].unique(sorted=True,return_inverse=True)
        mode_tasks=torch.stack([tasks[ref['mode']==g][0] for g in unique])
        def coverage(values):
            if values.ndim==1:values=values[:,None]
            growth=(values+radius[:,None])/scale[:,None]
            return torch.stack([growth[classes==c].max(0).values for c in classes.unique(sorted=True)])
        def geometry(values):
            if values.ndim==1:values=values[:,None]
            if self.coverage:
                risk=coverage(values)
                return torch.stack((risk.max(0).values,risk.mean(0)),1)
            from corelib.responsibility_memory import task_macro
            growth=((values+radius[:,None]-scale[:,None])/scale[:,None]).clamp_min(0)
            modes=torch.zeros(len(unique),growth.shape[1],device=pool.device)
            modes.scatter_reduce_(0,inverse[:,None].expand_as(growth),growth,reduce='amax')
            return torch.stack((modes.max(0).values,task_macro(modes,mode_tasks),task_macro(growth*ref['importance'][:,None],tasks)),1)
        def query_state(s):
            selected=s.cpu().numpy();alive=np.zeros(len(pool),bool);alive[selected]=True
            return self._nearest_two(self._qd,alive)
        def objective(s):
            g=geometry(full[:,s].min(1).values)[0].cpu().tolist()
            if not self.scoring:return tuple(g)
            q=self._query_losses(self._qd[:,s.cpu().numpy()].min(1))[:,0]
            return tuple([float(q.max()),float(q.mean())]+g)
        initial=source.clone();before=objective(source);history=[];rejected=0
        for step in range(max_swaps):
            live=torch.zeros(len(pool),device=pool.device,dtype=torch.bool);live[source]=True
            missing=torch.where(~live)[0]
            if not len(missing):break
            v,ix=full[:,source].topk(2,largest=False)
            priority=(v[:,0]+radius)/scale
            if self.scoring:
                qv,qo=query_state(source);current_risk=self._query_losses(qv[:,0])
                added=self._query_losses(np.minimum(qv[:,:1],self._qd[:,missing.cpu().numpy()]))
                order=lexorder([added.max(0),added.mean(0),-priority[missing].cpu().numpy()],missing.cpu().numpy())[:8]
            else:order=np.argsort(-priority[missing].cpu().numpy(),kind='stable')[:8]
            adds=missing[torch.as_tensor(order.copy(),device=pool.device)];pairs=[];trials=[]
            for uid in adds.tolist():
                eligible=torch.where((classes[source]==classes[uid])&(tasks[source]==tasks[uid]))[0]
                if not len(eligible):continue
                without=torch.where(ix[:,:1]==eligible[None,:],v[:,1:2],v[:,:1])
                g=geometry(without).cpu().numpy()
                order=lexorder([g[:,k] for k in range(g.shape[1])],eligible.cpu().numpy())[:16]
                for j in order:
                    pairs.append((int(eligible[int(j)]),uid));trials.append(torch.minimum(without[:,int(j)],full[:,uid]))
            if not pairs:break
            trial=torch.stack(trials,1);g=geometry(trial).cpu().numpy();feasible=np.ones(len(pairs),bool)
            if self.coverage:
                current=coverage(v[:,0]);safe=(coverage(trial)<=current+1e-6*(1+current)).all(0)
                feasible &= safe.cpu().numpy()
            if self.scoring:
                deleted=source[torch.tensor([x[0] for x in pairs],device=pool.device)].cpu().numpy()
                added=np.array([x[1] for x in pairs])
                without=np.where(qo[:,:1]==deleted[None,:],qv[:,1:2],qv[:,:1])
                qrisk=self._query_losses(np.minimum(without,self._qd[:,added]))
                feasible &= (qrisk<=current_risk+1e-6*(1+current_risk)).all(0)
                cols=[qrisk.max(0),qrisk.mean(0)]+[g[:,k] for k in range(g.shape[1])]
            else:cols=[g[:,k] for k in range(g.shape[1])]
            rejected+=int((~feasible).sum());valid=np.flatnonzero(feasible)
            if not len(valid):break
            i=int(valid[lexorder([c[valid] for c in cols],valid)[0]])
            candidate=tuple(float(c[i]) for c in cols)
            if not continuous_improves(candidate,before):break
            slot,uid=pairs[i];source[slot]=uid;before=objective(source)
            history.append(dict(slot=slot,source=uid,objective=before))
        diag=dict(swaps=len(history),history=history,before=objective(initial),after=objective(source),rejected_nonregression=rejected)
        if self.scoring:
            diag['score_proxy_before_restore']=self._query_losses(self._qd[:,initial.cpu().numpy()].min(1))[:,0].tolist()
            diag['score_proxy_after_restore']=self._query_losses(self._qd[:,source.cpu().numpy()].min(1))[:,0].tolist()
        return source,diag

    @torch.no_grad()
    def finish(self,source):
        if self.scoring:
            bank=self.pending_pool[source]
            for q in self.queries[self._fresh_query_start:]:q['birth']=pairwise(q['features'],bank).min(1).values.square()
            self._q=None;self._qd=None;self._previous=None;self._birth=None;self._scale=None
        super().finish(source)

    def audit(self,bank,ids,labels):
        result=super().audit(bank,ids,labels)
        extra=sum(v.numel()*v.element_size() for q in self.queries for v in q.values() if torch.is_tensor(v))+len(self.queries)*4096
        result['auxiliary_bytes']+=extra;result['persistent_total_bytes']+=extra
        if result['persistent_total_bytes']>result['baseline_limit_bytes']:raise RuntimeError('Full15 query state exceeds unified byte budget')
        result.update(module=self.full_variant,normal_query_features=len(self.queries)*16,synthetic_query_features=len(self.queries)*4,
                      query_images=len(self.queries)*4,test_class_routing=False)
        return result


def make_full15(variant):
    from corelib.responsibility_original import ResponsibilityMemory
    if variant=='F0':
        m=ResponsibilityMemory('Rhier');m.level=0;return m
    if variant=='F1':
        m=BalancedMemory('Rhier');m.level=2;return m
    if variant not in ('F2','F3','F4'):raise ValueError(variant)
    return Full15Memory(variant,coverage=variant in ('F2','F4'),scoring=variant in ('F3','F4'))
