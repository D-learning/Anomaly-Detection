"""Query-aware backward compression with exact candidate loss checks.

GPU computes FP32 distances once; CPU maintains sorted neighbour pointers.
No test labels or persistent query features are introduced by this module.
"""
import numpy as np
import torch
from corelib.refinement import choose_anchors


def risk(drift, groups, alpha=.9, beta=.5):
    output=[]
    for g in np.unique(groups):
        x=drift[groups==g]
        k=max(1,int(np.ceil((1-alpha)*len(x))))
        output.append((1-beta)*x.mean(axis=0)+beta*np.sort(x,axis=0)[-k:].mean(axis=0))
    return np.mean(output,axis=0)


def backward_select(distances, groups, reference, scales, budget, novelty=None):
    """distances QxN; exact masked LOO distances, not an NxN feature matrix."""
    distances=np.asarray(distances,dtype=np.float64)
    groups=np.asarray(groups);reference=np.asarray(reference);scales=np.maximum(scales,1e-6)
    q,n=distances.shape
    if not 1<=budget<=n or not np.isfinite(distances).any(axis=1).all():
        raise ValueError('Invalid compression budget or queries without finite neighbours')
    if budget<2:raise ValueError('Leave-one-out compression requires at least two prototypes')
    if np.isnan(distances).any() or np.any(distances<0):raise ValueError('Invalid distances')
    sorted_indices=np.argsort(distances,axis=1,kind='stable')
    sorted_dist=np.take_along_axis(distances,sorted_indices,axis=1)
    active=np.ones(n,dtype=bool);first=np.zeros(q,dtype=np.int64);second=np.ones(q,dtype=np.int64)
    row=np.arange(q);novelty=np.zeros(n) if novelty is None else np.asarray(novelty)
    groups_unique=np.unique(groups)
    def drift(x):return np.maximum((x-reference[:,None])/scales[:,None],0)
    initial=risk(drift(sorted_dist[:,0,None]),groups).item()
    changes=[]
    for _ in range(n-budget):
        # Advance only pointers whose prototype was removed; no stale nearest2 cache.
        while True:
            invalid=~active[sorted_indices[row,first]]
            if not invalid.any():break
            first[invalid]+=1
        second=np.maximum(second,first+1)
        while True:
            invalid=~active[sorted_indices[row,second]]
            if not invalid.any():break
            second[invalid]+=1
        owner=sorted_indices[row,first];d1=sorted_dist[row,first];d2=sorted_dist[row,second]
        base=drift(d1[:,None])[:,0];removed=drift(d2[:,None])[:,0]
        # First-order mean/tail sensitivity nominates 16 candidates; full risk decides.
        weights=np.zeros(q)
        for group in groups_unique:
            indices=np.where(groups==group)[0];k=max(1,int(np.ceil(.1*len(indices))))
            weights[indices]+=.5/len(indices)/len(groups_unique)
            tail=indices[np.argsort(base[indices],kind='stable')[-k:]]
            weights[tail]+=.5/k/len(groups_unique)
        cost=np.bincount(owner,weights=weights*(removed-base),minlength=n)
        candidates=np.where(active)[0]
        ranked=np.lexsort((candidates,novelty[candidates],cost[candidates]))
        candidates=candidates[ranked[:16]]
        trial=np.where(owner[:,None]==candidates[None,:],d2[:,None],d1[:,None])
        objectives=risk(drift(trial),groups)
        # tie follows nominated redundancy order; retain stable deterministic output.
        chosen=int(candidates[int(np.argmin(objectives))])
        active[chosen]=False
        changes.append(float(objectives.min()))
    selected=np.where(active)[0]
    final=distances[:,selected].min(1)
    final_risk=float(risk(drift(final[:,None]),groups))
    return selected,dict(initial_coverage_risk=initial,final_coverage_risk=final_risk,
        deletions=n-budget,query_count=q,candidate_deletions=16,
        selection='backward approximate candidate nomination, exact macro Mean/CVaR trial acceptance',
        deletion_risk_monotone=all(b>=a-1e-7 for a,b in zip([initial]+changes,changes)))


@torch.no_grad()
def compress(pool, projected, labels, old_count, raw, raw_ids, budget, task, independent=None):
    generator=torch.Generator(device=pool.device).manual_seed(17000+task)
    if independent is None:
        anchors=choose_anchors(labels[:old_count],256,generator,True)
        historical=pool[anchors];history_groups=labels[anchors]
        history_sources=anchors
    else:
        historical=independent['queries'];history_groups=independent['groups']
        history_sources=None
    # Select independent current queries across images, never test labels.
    image_ids=torch.unique(raw_ids)
    image_order=image_ids[torch.randperm(len(image_ids),device=pool.device,generator=generator)[:64]]
    rows=[]
    for image_id in image_order:
        choices=torch.where(raw_ids==image_id)[0]
        rows.append(choices[torch.randint(len(choices),(1,),device=pool.device,generator=generator)[0]])
    current=raw[torch.stack(rows)]
    queries=torch.cat((historical,current));groups=torch.cat((history_groups,torch.full((len(current),),task,device=pool.device,dtype=torch.long)))
    distances=torch.cdist(queries,pool)
    if history_sources is not None:
        distances[torch.arange(len(historical),device=pool.device),history_sources]=float('inf')
    reference=distances.min(1).values
    scales=torch.empty_like(reference)
    for group in groups.unique():
        mask=groups==group;scales[mask]=reference[mask].median().clamp_min(.001)
    # Query-free slots tie-break by redundancy in the same fixed projected space.
    novelty=[]
    for start in range(0,len(pool),256):
        d=torch.cdist(projected[start:start+256],projected)
        local=torch.arange(len(d),device=pool.device)
        d[local,start+local]=float('inf');novelty.append(d.min(1).values)
    selected,diagnostics=backward_select(distances.cpu().numpy(),groups.cpu().numpy(),
        reference.cpu().numpy(),scales.cpu().numpy(),budget,torch.cat(novelty).cpu().numpy())
    diagnostics.update(historical_queries=len(historical),current_queries=len(current),
        independent_history=independent is not None,
        group_queries={str(int(g)):int((groups==g).sum()) for g in groups.unique()})
    return selected.tolist(),diagnostics
