"""Task-owned coverage balls, hierarchical proxy modes and sparse image bags.

All data come from normal training features. Bags approximate image scores;
they are not complete replay images and cannot guarantee test ranking.
"""
import numpy as np
import torch
from corelib.fixed_mode_memory import FixedModeMemory
from corelib.mode_memory import pairwise,lex_improves


def components(neighbours):
    parent=np.arange(len(neighbours))
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        return a
    for node,neighbour in enumerate(neighbours):
        a=find(node);b=find(int(neighbour));parent[max(a,b)]=min(a,b)
    return np.unique([find(i) for i in range(len(parent))],return_inverse=True)[1]


def task_macro(values,tasks):
    return torch.stack([values[tasks==t].mean(0) for t in tasks.unique()]).mean(0)


class BalancedMemory(FixedModeMemory):
    def __init__(self,variant):
        super().__init__();self.variant=variant
        self.bags=[];self.bag_birth=[];self.bag_scale=[];self.bag_tasks=[]

    @torch.no_grad()
    def prepare(self,old,new,raw,raw_ids,task):
        n=0 if old is None else len(old)
        cls=self.class_ids[n:].to(torch.int16)
        if self.raw_classes is None:raise RuntimeError('Missing raw class provenance')
        fresh=None
        from corelib.responsibility_original import ResponsibilityMemory
        for c in cls.unique(sorted=True):
            rows=torch.where(cls==c)[0];mask=self.raw_classes==c
            local=ResponsibilityMemory('Rhier')
            meta=local.prepare(None,new[rows],raw[mask],raw_ids[mask],task)
            meta['mode']+=self.next_mode
            self.next_mode+=local.next_mode;self.total_birth_modes+=local.total_birth_modes
            if fresh is None:
                fresh={k:torch.empty(len(new),device=new.device,dtype=v.dtype) for k,v in meta.items()}
            for k,v in meta.items():fresh[k][rows]=v
        fresh['owner_class']=cls
        if n:fresh={k:torch.cat((self.meta[k],v)) for k,v in fresh.items()}
        self.pending=fresh;self.pending_pool=new if old is None else torch.cat((old,new))
        self.reference_meta={k:v.clone() for k,v in fresh.items()}
        self.reference_bound=fresh['bound'].clone();self.task=task;self.old_count=n
        self.current_raw=raw if self.level==5 else None
        self.current_raw_ids=raw_ids if self.level==5 else None
        if self.level==5 and self.bags:
            self.bag_previous=self._bag_scores(self._bag_distances(old).min(1).values).clone()
        else:self.bag_previous=None
        return fresh

    def _bag_scores(self,distances):
        if distances.ndim==1:distances=distances[:,None]
        return distances.reshape(-1,4,distances.shape[-1]).max(1).values

    @torch.no_grad()
    def register_current(self,bank):
        if self.level!=5 or self.current_raw is None:return
        for c in self.raw_classes.unique(sorted=True):
            mask=self.raw_classes==c;count=len(self.bags)
            self.register_bags(self.current_raw[mask],self.current_raw_ids[mask],bank)
            self.bag_tasks[-1].fill_(int(c))
            birth=self.bag_birth[-1][:,None]
            self.bag_previous=birth if self.bag_previous is None else torch.cat((self.bag_previous,birth))
        self.current_raw=None;self.current_raw_ids=None

    @torch.no_grad()
    def compress(self,pool,projected,budget,use_images=True):
        data={k:v.cpu().numpy().copy() for k,v in self.pending.items()}
        raw=pairwise(pool,pool).cpu().numpy();np.fill_diagonal(raw,np.inf)
        pd=pairwise(projected,projected).cpu().numpy();np.fill_diagonal(pd,np.inf)
        task=data['owner_task'];same=task[:,None]==task[None,:]
        cls=data['owner_class'];same &= cls[:,None]==cls[None,:]
        pd[~same]=np.inf
        order=np.argsort(pd,axis=1,kind='stable').astype(np.int32);del pd
        n=len(pool);rows=np.arange(n);alive=np.ones(n,dtype=bool)
        width=min(16,n-1);local=order[:,:width].copy();cursor=np.full(n,width,dtype=np.int32)
        bound=data['bound'];scale=data['birth_scale'];mass=data['support_mass'];imp=data['importance']
        max_modes_before=int(np.unique(data['mode']).size)
        for step in range(n-budget):
            bad=~alive[local]
            for row in np.where(bad.any(1)&alive)[0]:
                keep=local[row,~bad[row]].tolist()
                while len(keep)<width and cursor[row]<n:
                    item=int(order[row,cursor[row]]);cursor[row]+=1
                    if alive[item] and item!=row:keep.append(item)
                if not keep:raise RuntimeError('No contraction candidate')
                local[row]=np.resize(np.asarray(keep,dtype=np.int32),width)
            valid=alive[local]&same[rows[:,None],local]&(local!=rows[:,None])
            edge=raw[rows[:,None],local]
            radius=np.maximum(bound[local],bound[:,None]+edge)
            expansion=np.maximum(radius-scale[local],0)/scale[local]
            # Responsibility may never cross a historical task. Equal coverage
            # cost favours preserving important local appearance relationships.
            distortion=edge/scale[:,None]
            critical=distortion*imp[:,None]
            rank=np.lexsort((local,edge,distortion,critical,expansion,~valid),axis=1)[:,0]
            active=np.where(alive&valid[rows,rank])[0]
            if not len(active):raise RuntimeError('No within-owner contraction; budget infeasible')
            pick=np.lexsort((active,edge[active,rank[active]],distortion[active,rank[active]],
                             critical[active,rank[active]],expansion[active,rank[active]]))[0]
            delete=int(active[pick]);to=int(local[delete,rank[delete]])
            assert task[delete]==task[to] and delete!=to
            total=int(mass[to])+int(mass[delete])
            imp[to]=(imp[to]*mass[to]+imp[delete]*mass[delete])/total
            mass[to]=total;bound[to]=max(bound[to],bound[delete]+raw[delete,to]);alive[delete]=False
        selected=np.flatnonzero(alive)
        for key,value in [('bound',bound),('importance',imp),('support_mass',mass)]:
            self.pending[key]=torch.as_tensor(value,device=pool.device)
        return torch.as_tensor(selected,device=pool.device,dtype=torch.long),dict(
            module=self.variant,task=self.task+1,deletions=n-budget,
            mode_count_before=max_modes_before,mode_count_after=int(np.unique(data['mode'][selected]).size),
            cross_task_transfers=0,mean_bound=float(bound[selected].mean()),max_bound=float(bound[selected].max()))

    def _bag_distances(self,pool):
        if not self.bags:return None
        q=torch.cat(self.bags).flatten(0,1)
        return pairwise(q,pool).square()

    def _bag_error(self,patch_distances):
        # Four hard patches from each of four real normal training images/task.
        if patch_distances.ndim==1:patch_distances=patch_distances[:,None]
        scores=patch_distances.reshape(-1,4,patch_distances.shape[-1]).max(1).values
        birth=torch.cat(self.bag_birth)[:,None];scale=torch.cat(self.bag_scale)[:,None]
        long_error=((scores-birth)/scale).abs()
        if self.bag_previous is None:return long_error
        return torch.cat((long_error,((scores-self.bag_previous)/scale).abs()),0)

    @torch.no_grad()
    def restore(self,pool,source,old_count,max_swaps=32):
        meta=self.reference_meta;tasks=meta['owner_task'];scale=meta['birth_scale'];radius=meta['bound']
        importance=meta['importance'];full=pairwise(pool,pool)
        # Full diagonal remains zero: radius is added, so a self-match does NOT
        # erase the responsibility already inherited from earlier tasks.
        full.masked_fill_((tasks[:,None]!=tasks[None,:]) | (meta['owner_class'][:,None]!=meta['owner_class'][None,:]),float('inf'))
        groups=meta['mode'];unique,inverse=groups.unique(sorted=True,return_inverse=True)
        mode_tasks=torch.stack([tasks[groups==g][0] for g in unique])
        ext=self._bag_distances(pool)
        ext_initial=None if ext is None else self._bag_error(ext[:,source].min(1).values)[:,0]
        bag_tasks=None if ext is None else torch.cat(self.bag_tasks)
        if ext is not None and self.bag_previous is not None:bag_tasks=bag_tasks.repeat(2)
        def class_risk(values):
            if values.ndim==1:values=values[:,None]
            growth=((values+radius[:,None]-scale[:,None])/scale[:,None]).clamp_min(0)
            return torch.stack([growth[meta['owner_class']==c].max(0).values for c in meta['owner_class'].unique(sorted=True)])
        class_before=class_risk(full[:,source].min(1).values)[:,0]
        rejected_class=0
        def score(values):
            if values.ndim==1:values=values[:,None]
            growth=((values+radius[:,None]-scale[:,None])/scale[:,None]).clamp_min(0)
            mode_max=torch.zeros((len(unique),growth.shape[1]),device=pool.device)
            mode_max.scatter_reduce_(0,inverse[:,None].expand_as(growth),growth,reduce='amax')
            critical=task_macro(growth*importance[:,None],tasks)
            geometry=torch.stack((mode_max.max(0).values,task_macro(mode_max,mode_tasks),critical),1)
            if self.level>=4:
                risk=class_risk(values)
                geometry=torch.cat((risk.max(0).values[:,None],risk.mean(0)[:,None],geometry),1)
            return geometry
        def objective(s):
            geometry=score(full[:,s].min(1).values)[0]
            if ext is not None:
                err=self._bag_error(ext[:,s].min(1).values)[:,0]
                return tuple(torch.cat((err.max().reshape(1),task_macro(err[:,None],bag_tasks).reshape(1),geometry)).cpu().tolist())
            return tuple(geometry.cpu().tolist())
        initial=source.clone();before=objective(source);history=[];rejected=0;rejected_geometry=0
        initial_geometry=score(full[:,source].min(1).values)[0]
        for step in range(max_swaps):
            live=torch.zeros(len(pool),device=pool.device,dtype=torch.bool);live[source]=True
            missing=torch.where(~live)[0]
            if not len(missing):break
            v,ix=full[:,source].topk(2,largest=False)
            priority=((v[:,0]+radius-scale)/scale).clamp_min(0)
            if ext is not None:
                ev,ei=ext[:,source].topk(2,largest=False)
                candidates=ext[:,missing]
                added=self._bag_error(torch.minimum(ev[:,:1],candidates))
                rank=np.lexsort((missing.cpu().numpy(),-priority[missing].cpu().numpy(),added.mean(0).cpu().numpy(),added.max(0).values.cpu().numpy()))[:8]
            else:rank=np.lexsort((missing.cpu().numpy(),-priority[missing].cpu().numpy()))[:8]
            adds=missing[torch.as_tensor(rank.copy(),device=pool.device)]
            # Select removals per proposed addition's owner, so swapping never
            # silently transfers a task's coverage responsibility to a new task.
            trials=[];pairs=[]
            for uid in adds.tolist():
                eligible=torch.where((tasks[source]==tasks[uid]) & (meta['owner_class'][source]==meta['owner_class'][uid]))[0]
                if not len(eligible):continue
                without=torch.where(ix[:,:1]==eligible[None,:],v[:,1:2],v[:,:1])
                removal_score=score(without).cpu().numpy()
                rank=np.lexsort(tuple([eligible.cpu().numpy()]+[removal_score[:,i] for i in range(removal_score.shape[1]-1,-1,-1)]))[:16]
                for j in rank:
                    slot=int(eligible[int(j)]);pairs.append((slot,uid))
                    trials.append(torch.minimum(without[:,int(j)],full[:,uid]))
            if not trials:break
            trial=torch.stack(trials,1);geom=score(trial)
            feasible=torch.ones(len(pairs),device=pool.device,dtype=torch.bool)
            if self.level>=4:
                current_risk=class_risk(v[:,0])
                safe_class=(class_risk(trial)<=current_risk+1e-6*(1+current_risk)).all(0)
                rejected_class+=int((~safe_class).sum());feasible &= safe_class
            if ext is not None:
                slot=torch.tensor([p[0] for p in pairs],device=pool.device);uid=torch.tensor([p[1] for p in pairs],device=pool.device)
                ew=torch.where(ei[:,:1]==slot[None,:],ev[:,1:2],ev[:,:1])
                err=self._bag_error(torch.minimum(ew,ext[:,uid]))
                feasible &= (err<=ext_initial[:,None]+1e-6*(1+ext_initial[:,None])).all(0)
                rejected+=int((~feasible).sum())
                safe_geometry=(geom<=initial_geometry[None,:]+1e-6*(1+initial_geometry[None,:])).all(1)
                rejected_geometry+=int((feasible&~safe_geometry).sum())
                feasible &= safe_geometry
                sc=torch.cat((err.max(0).values[:,None],task_macro(err,bag_tasks)[:,None],geom),1)
            else:sc=geom
            array=sc.cpu().numpy();valid=np.where(feasible.cpu().numpy())[0]
            if not len(valid):break
            order=np.lexsort(tuple([valid]+[array[valid,i] for i in range(array.shape[1]-1,-1,-1)]))
            pos=int(valid[order[0]]);best=tuple(array[pos].tolist())
            # All objectives continuous here; no mode-count first coordinate.
            if not continuous_improves(best,before):break
            slot,uid=pairs[pos];source[slot]=uid;before=objective(source)
            history.append(dict(slot=slot,source=uid,objective=before))
        return source,dict(swaps=len(history),history=history,before=objective(initial),after=objective(source),
            cross_task_transfers=0,image_bags=0 if ext is None else len(ext_initial),rejected_image=rejected,
            rejected_geometry=rejected_geometry,rejected_class=rejected_class,
            class_ids=meta['owner_class'].unique(sorted=True).cpu().tolist(),
            class_risk_before=class_before.cpu().tolist(),
            class_risk_after=class_risk(full[:,source].min(1).values)[:,0].cpu().tolist())

    @torch.no_grad()
    def finish(self,source):
        ref=self.reference_meta;pool=self.pending_pool
        d=pairwise(pool,pool[source]);d.masked_fill_((ref['owner_task'][:,None]!=ref['owner_task'][source][None,:]) | (ref['owner_class'][:,None]!=ref['owner_class'][source][None,:]),float('inf'))
        distance,owner=d.min(1)
        if not bool(torch.isfinite(distance).all()):raise RuntimeError('Historical owner lost all representatives')
        size=len(source);bound=torch.zeros(size,device=pool.device)
        bound.scatter_reduce_(0,owner,ref['bound']+distance,reduce='amax')
        mass=torch.zeros(size,device=pool.device,dtype=torch.int64).scatter_add_(0,owner,ref['support_mass'].long())
        weighted=torch.zeros(size,device=pool.device).scatter_add_(0,owner,ref['importance']*ref['support_mass'])
        self.retired_modes+=len(set(ref['mode'].cpu().tolist())-set(ref['mode'][source].cpu().tolist()))
        self.meta={k:v[source].clone() for k,v in ref.items()}
        self.meta.update(bound=bound,support_mass=mass.to(torch.int32),importance=weighted/mass.clamp_min(1))
        if self.variant=='Rimage':self.register_bags(self.current_raw,self.current_raw_ids,pool[source])
        self.pending=None;self.pending_pool=None;self.reference_bound=None;self.reference_meta=None
        self.current_raw=None;self.current_raw_ids=None;self.raw_classes=None;self.class_ids=None;self.bag_previous=None

    @torch.no_grad()
    def register_bags(self,raw,ids,bank):
        # Inherited16 persistent query features/task, organised4 images x4patches.
        byimage=[]
        for image in ids.unique(sorted=True):
            rows=torch.where(ids==image)[0]
            values=[]
            for q in raw[rows].split(256):values.append(pairwise(q,bank).min(1).values.square())
            v=torch.cat(values);count=min(4,len(rows))
            picks=v.topk(count).indices
            if count<4:continue
            byimage.append((float(v.max()),int(image),raw[rows[picks]].clone()))
        byimage.sort(key=lambda x:(-x[0],x[1]))
        if len(byimage)<4:raise RuntimeError('Need at least4 normal training images')
        bag=torch.stack([x[2] for x in byimage[:4]])
        birth=pairwise(bag.flatten(0,1),bank).min(1).values.square().reshape(4,4).max(1).values
        self.bags.append(bag);self.bag_birth.append(birth)
        self.bag_scale.append(birth.median().clamp_min(.001).expand(4).clone())
        self.bag_tasks.append(torch.full((4,),self.task,device=bank.device,dtype=torch.int32))

    def audit(self,bank,ids,labels):
        extra=sum(v.numel()*v.element_size() for v in self.meta.values()) if self.meta else 0
        extra+=sum(t.numel()*t.element_size() for seq in (self.bags,self.bag_birth,self.bag_scale,self.bag_tasks) for t in seq)+128
        extra+=len(bank)*8  # sampler.optimization_class_labels (int64)
        if getattr(self,'bag_previous',None) is not None:extra+=self.bag_previous.numel()*self.bag_previous.element_size()
        core=sum(v.numel()*v.element_size() for v in (bank,ids,labels))
        limit=5000*(bank.shape[1]*4+ids.element_size()+labels.element_size())
        if core+extra>limit:raise RuntimeError(f'Responsibility memory exceeds budget: {core+extra}>{limit}')
        classes={}
        for c in self.meta['owner_class'].unique(sorted=True).tolist():
            mask=self.meta['owner_class']==c
            classes[str(c)]=dict(count=int(mask.sum()),modes=int(self.meta['mode'][mask].unique().numel()),mean_bound=float(self.meta['bound'][mask].mean()),max_bound=float(self.meta['bound'][mask].max()),support_mass=int(self.meta['support_mass'][mask].sum()))
        return dict(per_class=classes,persistent_total_bytes=core+extra,baseline_limit_bytes=limit,auxiliary_bytes=extra,
            prototype_count=len(bank),surviving_modes=int(self.meta['mode'].unique().numel()),
            birth_modes=self.total_birth_modes,retired_modes=self.retired_modes,
            image_bags=sum(len(t) for t in self.bags),support_mass=int(self.meta['support_mass'].sum()))


def continuous_improves(new,old):
    for a,b in zip(new,old):
        tolerance=1e-6*(1+abs(b))
        if a<b-tolerance:return True
        if a>b+tolerance:return False
    return False


def make_memory(variant):
    import os
    if os.environ.get('G_VARIANT') in ('G1','G3'):
        from corelib.r2next import ReservedMemory
        return ReservedMemory()
    if os.environ.get('FULL15_VARIANT'):
        from corelib.full15_memory import make_full15
        return make_full15(os.environ['FULL15_VARIANT'])
    from corelib.responsibility_original import ResponsibilityMemory
    level=int(os.environ.get('RB_VARIANT','R0')[1:])
    if level not in range(6):raise ValueError('RB_VARIANT must be R0..R5')
    memory=ResponsibilityMemory(variant) if level<2 else BalancedMemory(variant)
    memory.level=level
    return memory
