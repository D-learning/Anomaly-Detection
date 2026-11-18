"""Fixed-budget refinement ablations; no test labels or extra persistent queries.

A1: dynamic, leave-one-out-consistent deletion candidates.
A2: task-balanced queries and per-task distance scaling, fixed legacy deletions.
A3: A1+A2; mean-drift acceptance retained.
A4: A3 with task-macro Mean/CVaR acceptance and current-task/worst-task guards.
"""
import math
import torch


def task_risk(drift, groups, alpha=.9, beta=.5, tail=True):
    """Return task-macro risk and per-task means; supports trailing trial axes."""
    risks, means = [], []
    for group in torch.unique(groups, sorted=True):
        values = drift[groups == group]
        mean = values.mean(dim=0)
        count = max(1, math.ceil((1-alpha)*len(values)))
        risk = (1-beta)*mean + beta*values.topk(count, dim=0).values.mean(dim=0) if tail else mean
        risks.append(risk)
        means.append(mean)
    return torch.stack(risks).mean(dim=0), torch.stack(means)


def choose_anchors(labels, count, generator, balanced):
    count = min(count, len(labels))
    if not balanced:
        return torch.randperm(len(labels), device=labels.device, generator=generator)[:count]
    groups = torch.unique(labels, sorted=True)
    pools = [torch.where(labels == g)[0] for g in groups]
    shuffled = [p[torch.randperm(len(p), device=p.device, generator=generator)] for p in pools]
    # Round robin fills the exact same global budget even for tiny task groups.
    selected = []
    for row in range(max(map(len, shuffled))):
        for pool in shuffled:
            if row < len(pool):selected.append(pool[row])
            if len(selected) == count:return torch.stack(selected)
    return torch.stack(selected)


@torch.no_grad()
def refine(old_memory, old_ids, task_memory, task_ids, initial_source_indices,
           old_task_labels, current_task, current_features=None, variant='A1',
           num_anchors=256, max_swaps=32, deletion_candidates=16,
           addition_candidates=8, current_queries=128, alpha=.9, beta=.5,
           current_weight=.5, current_tolerance=.01, worst_tolerance=.01, protection=None):
    if variant not in ('A1','A2','A3','A4','B1','B2','B12'):
        raise ValueError(f'Unknown ablation: {variant}')
    device = task_memory.device
    n_old = len(old_memory) if old_memory is not None else 0
    pool = task_memory if n_old == 0 else torch.cat((old_memory, task_memory))
    ids_pool = task_ids if n_old == 0 else torch.cat((old_ids, task_ids))
    labels_pool = torch.cat((old_task_labels.to(device), torch.full((len(task_memory),), current_task,
                                                                  device=device, dtype=torch.long)))
    source = torch.as_tensor(initial_source_indices, device=device, dtype=torch.long).clone()
    if source.ndim != 1 or not len(source) or source.min()<0 or source.max()>=len(pool):
        raise ValueError('Invalid source indices')
    if len(source.unique()) != len(source):
        raise ValueError('Initial prototype source rows must be unique')
    balanced = variant in ('A2','A3','A4','B1','B2','B12')
    dynamic = variant in ('A1','A3','A4','B2','B12')
    diagnostics = dict(variant=variant, task=current_task+1, swaps=0, reason='no_history',
                       dynamic_deletions=dynamic, balanced_queries=balanced,
                       current_queries=0, query_feature_cache_bytes=0,
                       anchors=0, history=[], budget=len(source))
    floors={}
    if variant in ('B1','B12'):
        from corelib.quota import task_floors
        floors=task_floors(labels_pool,len(source),current_task+1,.15)
    diagnostics['minimum_task_counts']=floors
    def finish():
        for group,minimum in floors.items():
            assert int((labels_pool[source]==group).sum())>=minimum, 'Prototype floor violated'

        diagnostics['task_counts_after'] = {str(int(g)):int((labels_pool[source]==g).sum()) for g in labels_pool.unique()}
        return pool[source].clone(), ids_pool[source].clone(), source, labels_pool[source].clone(), diagnostics
    diagnostics['task_counts_before']={str(int(g)):int((labels_pool[source]==g).sum()) for g in labels_pool.unique()}
    if n_old < 2 or len(source)<2:return finish()
    # A local generator copies the original stream state without consuming it.
    # A1 uses the same anchor rows as legacy H1 for an identical initial bank.
    generator=torch.Generator(device=device)
    if device.type=='cuda':generator.set_state(torch.cuda.get_rng_state(device))
    else:generator.set_state(torch.get_rng_state())
    anchors=choose_anchors(old_task_labels.to(device),num_anchors,generator,balanced)
    H=old_memory[anchors]
    groups=old_task_labels[anchors].to(device) if balanced else torch.zeros(len(H),device=device,dtype=torch.long)
    full_dist=torch.cdist(H,pool)
    full_dist.masked_fill_(anchors[:,None]==torch.arange(len(pool),device=device)[None,:],float('inf'))
    reference=full_dist[:,:n_old].min(1).values
    scale=torch.ones_like(reference)
    if balanced:
        for g in groups.unique():
            mask=groups==g
            scale[mask]=reference[mask].median().clamp_min(1e-6)
    distances=full_dist[:,source]
    diagnostics['anchors']=len(H)
    diagnostics['anchor_task_counts']={str(int(g)):int((old_task_labels[anchors].to(device)==g).sum()) for g in old_task_labels.unique()}
    def drift(values):
        dims=(len(reference),)+(1,)*(values.ndim-1)
        return ((values-reference.reshape(dims))/scale.reshape(dims)).clamp_min(0)
    initial_drift=drift(distances.min(1).values)
    base_risk,base_task_mean=task_risk(initial_drift,groups,alpha,beta)
    diagnostics.update(mean_before=float(task_risk(initial_drift,groups,tail=False)[0]),risk_before=float(base_risk))
    # Preserve legacy unmasked deletion choice only in the A2 isolation.
    legacy_dist=torch.cdist(H,pool[source])
    v,i=legacy_dist.topk(2,dim=1,largest=False)
    cost=torch.zeros(len(source),device=device).scatter_add_(0,i[:,0],v[:,1]-v[:,0])
    def eligible_removals():
        eligible=torch.ones(len(source),device=device,dtype=torch.bool)
        for group,minimum in floors.items():
            mask=labels_pool[source]==group
            if int(mask.sum())<=minimum:eligible[mask]=False
        return eligible
    initial_eligible=eligible_removals()
    fixed_removals=cost.masked_fill(~initial_eligible,float('inf')).topk(min(8,int(initial_eligible.sum())),largest=False).indices
    current_dist=None
    if variant=='A4':
        raw=current_features if current_features is not None else task_memory
        qidx=torch.randperm(len(raw),device=device,generator=generator)[:min(current_queries,len(raw))]
        queries=raw[qidx]
        current_dist=torch.cdist(queries,pool)
        current_reference=current_dist[:,source].min(1).values
        current_scale=current_reference.median().clamp_min(1e-6)
        diagnostics['current_queries']=len(queries)
    def current_loss(values):
        dims=(len(current_reference),)+(1,)*(values.ndim-1)
        return ((values-current_reference.reshape(dims))/current_scale).clamp_min(0).mean(0)
    external=None
    if protection is not None:
        queries=protection['queries']
        external=torch.cdist(queries,pool)
        previous=torch.cdist(queries,old_memory).min(1).values
        birth=protection['birth'];extscale=protection['scales'].clamp_min(.001)
        extgroups=protection['groups']
        def stable(values):
            shape=(len(previous),)+(1,)*(values.ndim-1)
            def huber(x):
                x=(x.abs()-.05).clamp_min(0)
                return torch.where(x<1,.5*x*x,x-.5)
            short=huber((values-previous.reshape(shape))/extscale.reshape(shape))
            long=huber((values-birth.reshape(shape))/extscale.reshape(shape))
            return task_risk(short+.5*long,extgroups,tail=False)[0]
        external_initial=stable(external[:,source].min(1).values)
        diagnostics['independent_queries']=len(queries)
        diagnostics['stability_before']=float(external_initial)
    accepted_slots=[]
    diagnostics['reason']='max_swaps'
    for step in range(max_swaps):
        d1=distances.min(1).values
        before_drift=drift(d1)
        before_mean,_=task_risk(before_drift,groups,tail=False)
        before_risk,_=task_risk(before_drift,groups,alpha,beta)
        if float(before_mean)<1e-6:
            diagnostics['reason']='no_positive_drift';break
        retained=source[source<n_old]
        missing_mask=torch.ones(n_old,device=device,dtype=torch.bool);missing_mask[retained]=False
        missing=torch.where(missing_mask)[0]
        if not len(missing):diagnostics['reason']='no_discarded_history';break
        # Existing H1 add-only risk ranking, task-balanced/scaled in A2-A4.
        add_only=torch.minimum(d1[:,None],full_dist[:,missing])
        add_risk,_=task_risk(drift(add_only),groups,alpha,beta)
        add_uids=missing[add_risk.topk(min(addition_candidates,len(missing)),largest=False).indices]
        vals,idx=distances.topk(2,dim=1,largest=False)
        if dynamic:
            affected=torch.unique(idx[:,0])
            # Include unowned slots with zero history cost, but rank them jointly
            # with current-task cost for A4 before selecting bounded candidates.
            ownership=torch.zeros(len(source),device=device,dtype=torch.bool);ownership[affected]=True
            candidates=torch.arange(len(source),device=device)
            removal_d=torch.where(idx[:,0,None]==candidates[None,:],vals[:,1,None],vals[:,0,None])
            removal_risk,_=task_risk(drift(removal_d),groups,alpha,beta,tail=(variant=='A4'))
            if current_dist is not None:
                cv,ci=current_dist[:,source].topk(2,dim=1,largest=False)
                rem_curr=torch.where(ci[:,0,None]==candidates[None,:],cv[:,1,None],cv[:,0,None])
                removal_risk=removal_risk+current_weight*current_loss(rem_curr)
            eligible=eligible_removals()
            removal_risk=removal_risk.masked_fill(~eligible,float('inf'))
            removals=removal_risk.topk(min(deletion_candidates,int(eligible.sum())),largest=False).indices
        else:removals=fixed_removals[eligible_removals()[fixed_removals]]
        if not len(removals):
            diagnostics['reason']='no_quota_safe_removal';break
        without=torch.where(idx[:,0,None]==removals[None,:],vals[:,1,None],vals[:,0,None])
        trial=torch.minimum(without[:,None,:],full_dist[:,add_uids,None])
        trial_drift=drift(trial)
        trial_mean,_=task_risk(trial_drift,groups,tail=False)
        trial_risk,trial_task_mean=task_risk(trial_drift,groups,alpha,beta)
        if variant=='A4':
            cv,ci=current_dist[:,source].topk(2,dim=1,largest=False)
            c_without=torch.where(ci[:,0,None]==removals[None,:],cv[:,1,None],cv[:,0,None])
            c_trial=torch.minimum(c_without[:,None,:],current_dist[:,add_uids,None])
            curr_cost=current_loss(c_trial)
            current_before=current_loss(cv[:,0])
            objective=trial_risk+current_weight*curr_cost
            feasible=(curr_cost<=current_tolerance)&((trial_task_mean-base_task_mean[:,None,None])<=worst_tolerance).all(0)
            objective=objective.masked_fill(~feasible,float('inf'))
            threshold=before_risk+current_weight*current_before
        else:
            objective=trial_mean;threshold=before_mean
        if external is not None:
            ev,ei=external[:,source].topk(2,dim=1,largest=False)
            e_without=torch.where(ei[:,0,None]==removals[None,:],ev[:,1,None],ev[:,0,None])
            e_trial=torch.minimum(e_without[:,None,:],external[:,add_uids,None])
            trial_stability=stable(e_trial)
            before_stability=stable(ev[:,0])
            objective=trial_mean+.5*trial_stability
            threshold=before_mean+.5*before_stability
            # Do not trade worse historical mean for a lower stability loss.
            objective=objective.masked_fill(trial_mean>before_mean+1e-7,float('inf'))
        best=int(objective.reshape(-1).argmin())
        ai,ri=divmod(best,len(removals))
        if not bool(objective[ai,ri]<threshold-1e-7):
            diagnostics['reason']='no_feasible_improving_swap';break
        slot=removals[ri];uid=add_uids[ai]
        source[slot]=uid;distances[:,slot]=full_dist[:,uid]
        accepted_slots.append(int(slot))
        diagnostics['history'].append(dict(mean=float(trial_mean[ai,ri]),risk=float(trial_risk[ai,ri]),
                                           removed_slot=int(slot),added_source=int(uid),
                                           current_loss=float(curr_cost[ai,ri]) if variant=='A4' else None))
        diagnostics['swaps']+=1
    final_drift=drift(distances.min(1).values)
    final_risk,final_task_means=task_risk(final_drift,groups,alpha,beta)
    diagnostics.update(mean_after=float(task_risk(final_drift,groups,tail=False)[0]),risk_after=float(final_risk),
                       changed_slots=len(set(accepted_slots)),
                       max_task_mean_increase=float((final_task_means-base_task_mean).max()))
    if variant=='A4':diagnostics['current_loss_after']=float(current_loss(current_dist[:,source].min(1).values))
    if external is not None:
        diagnostics['stability_after']=float(stable(external[:,source].min(1).values))
        diagnostics['joint_before']=diagnostics['mean_before']+.5*diagnostics['stability_before']
        diagnostics['joint_after']=diagnostics['mean_after']+.5*diagnostics['stability_after']
    assert len(source.unique())==len(source),'Refinement introduced duplicate source rows' 
    return finish()
