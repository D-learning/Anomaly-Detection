import torch
import os

def rsc_refine(
    old_memory,
    old_ids,
    initial_memory,
    initial_ids,
    current_task_features,
    current_task_ids,
    num_anchors=256,
    max_swaps=32,
    initial_source_indices=None,
):
    S = initial_memory.clone()
    S_ids = initial_ids.clone()
    
    if old_memory is None or old_memory.shape[0] == 0:
        return S, S_ids

    if old_memory.shape[0] < 2 or S.shape[0] < 2 or num_anchors < 2:
        return S, S_ids
    device = initial_memory.device
    old_memory = old_memory.to(device)
    old_ids = old_ids.to(device)
    if initial_source_indices is None:
        raise ValueError("RSC requires prototype source indices, not image IDs")
    S_uids = torch.as_tensor(initial_source_indices, device=device, dtype=torch.long).clone()
    if S_uids.ndim != 1 or len(S_uids) != len(S):
        raise ValueError("RSC source indices must match initial memory rows")
    n_old = old_memory.shape[0]
    old_uids = torch.arange(n_old, device=device)
    k = min(num_anchors, n_old)
    anchor_idx = torch.randperm(n_old, device=device)[:k]
    H = old_memory[anchor_idx]
    d_old_matrix = torch.cdist(H, old_memory)
    d_old_matrix.masked_fill_(anchor_idx[:, None] == old_uids[None, :], float("inf"))
    d_old = d_old_matrix.min(dim=1).values

    # P0-04 修复：基于 H 计算真实删除代价
    Q_all = H 
    dists_all = torch.cdist(Q_all, S)
    vals, idxs = torch.topk(dists_all, 2, dim=1, largest=False)
    d1 = vals[:, 0]
    d2 = vals[:, 1]
    
    removal_cost = torch.zeros(S.shape[0], device=device)
    removal_cost.scatter_add_(0, idxs[:, 0], d2 - d1)
    
    n_rm = min(8, S.shape[0])
    remove_candidates = torch.topk(removal_cost, n_rm, largest=False).indices

    for _ in range(max_swaps):
        # Recompute membership after each accepted swap; IDs remain image IDs.
        retained_old = S_uids[(S_uids >= 0) & (S_uids < n_old)]
        discarded_mask = torch.ones(n_old, dtype=torch.bool, device=device)
        discarded_mask[retained_old] = False
        discarded_uids = old_uids[discarded_mask]
        if not len(discarded_uids):
            break
        discarded = old_memory[discarded_uids]
        discarded_ids = old_ids[discarded_uids]
        d_new_mat = torch.cdist(H, S)
        d_new_mat.masked_fill_(anchor_idx[:, None] == S_uids[None, :], float("inf"))
        d_new = d_new_mat.min(dim=1).values

        drift = torch.clamp(d_new - d_old, min=0)
        loss_before = drift.mean().item()
        if loss_before < 1e-6:
            break

        bad_k = min(32, k)
        bad_idx = torch.topk(drift, bad_k).indices
        bad_anchors = H[bad_idx]

        dist_disc = torch.cdist(discarded, bad_anchors).min(dim=1).values
        n_add = min(8, discarded.shape[0])
        add_indices = torch.topk(dist_disc, n_add, largest=False).indices
        add_candidates = discarded[add_indices]
        add_ids_candidates = discarded_ids[add_indices]

        best_loss = loss_before
        best_S = None
        best_ids = None
        best_uids = None
        
        for add_idx, add_vec in enumerate(add_candidates):
            add_id = add_ids_candidates[add_idx]
            for rem_idx in remove_candidates:
                S_try = S.clone()
                S_ids_try = S_ids.clone()
                
                S_try[rem_idx] = add_vec
                S_ids_try[rem_idx] = add_id
                uids_try = S_uids.clone()
                uids_try[rem_idx] = discarded_uids[add_indices[add_idx]]
                
                d_try_mat = torch.cdist(H, S_try)
                d_try_mat.masked_fill_(anchor_idx[:, None] == uids_try[None, :], float("inf"))
                d_try = d_try_mat.min(dim=1).values
                loss_try = torch.clamp(d_try - d_old, min=0).mean().item()
                
                if loss_try < best_loss:
                    best_loss = loss_try
                    best_S = S_try
                    best_ids = S_ids_try
                    best_uids = uids_try

        if best_S is None:
            break
        S = best_S
        S_ids = best_ids
        S_uids = best_uids

    return S, S_ids


def _h1_mean_cvar(drift, alpha=0.90, beta=0.50):
    """Mean + upper-tail CVaR along the anchor dimension (dimension 0)."""
    import math

    tail_k = max(1, int(math.ceil((1.0 - alpha) * drift.shape[0])))
    tail = torch.topk(drift, tail_k, dim=0, largest=True).values.mean(dim=0)
    return (1.0 - beta) * drift.mean(dim=0) + beta * tail


@torch.no_grad()
def h1_refine(
    old_memory,
    old_ids,
    initial_memory,
    initial_ids,
    current_task_features,
    current_task_ids,
    initial_source_indices,
    num_anchors=256,
    max_swaps=32,
    alpha=0.90,
    beta=0.50,
):
    """B5 + H1 only: use Mean+CVaR to select historical recovery candidates.

    Replaces the legacy top-32-anchor / nearest-discarded candidate selection.
    For each discarded prototype, measure the historical risk reduction if it
    were added, then keep the best 8 candidates. This changes the candidate set,
    rather than merely logging CVaR or reordering the same 8 candidates.

    The legacy deletion candidates (computed once, 8 entries) and strict
    mean-drift improvement acceptance remain unchanged. In particular this is
    not H2 or H3: no new deletion cost, current-query penalty or joint gain.

    initial_source_indices indexes [old_memory; task_coreset] BEFORE sampling.
    These source rows are unique patch UIDs for this update only; image-level
    sample_ids are returned untouched. Persistent UIDs are unnecessary because
    every historical anchor comes from this update's old_memory. Distinct rows
    with identical feature values remain valid neighbours of each other.

    current_task_features/current_task_ids retain the legacy call interface;
    H1 does not use current-task queries or test data.
    """
    if not 0.0 <= alpha < 1.0 or not 0.0 <= beta <= 1.0:
        raise ValueError("H1 requires 0 <= alpha < 1 and 0 <= beta <= 1.")

    S = initial_memory.clone()
    S_ids = initial_ids.clone()
    diagnostics = {
        "method": "B5+H1", "alpha": alpha, "beta": beta,
        "anchors": 0, "swaps": 0, "mean_before": 0.0, "mean_after": 0.0,
        "risk_before": 0.0, "risk_after": 0.0, "reason": "no_history",
    }
    if old_memory is None or old_memory.shape[0] == 0:
        return S, S_ids, diagnostics
    if old_memory.shape[0] < 2 or S.shape[0] < 2 or num_anchors < 2:
        diagnostics["reason"] = "too_few_prototypes_or_anchors"
        return S, S_ids, diagnostics

    device = S.device
    old_memory = old_memory.to(device)
    old_ids = old_ids.to(device)
    S_uids = torch.as_tensor(initial_source_indices, device=device, dtype=torch.long).clone()
    if S_uids.ndim != 1 or len(S_uids) != len(S):
        raise ValueError("H1 source indices must match the initial memory rows.")

    n_old = old_memory.shape[0]
    old_uids = torch.arange(n_old, device=device)
    k = min(num_anchors, n_old)
    # Keep the existing anchor sampling behaviour; no unrelated RNG changes.
    anchor_idx = torch.randperm(n_old, device=device)[:k]
    H = old_memory[anchor_idx]
    H_uids = old_uids[anchor_idx]
    diagnostics["anchors"] = k

    # Same UID-based leave-one-out definition for old, initial and trial banks.
    d_H_old = torch.cdist(H, old_memory)
    d_H_old.masked_fill_(H_uids[:, None] == old_uids[None, :], float("inf"))
    d_old = d_H_old.min(dim=1).values
    d_H_S = torch.cdist(H, S)

    # Preserve the ORIGINAL, fixed deletion candidates, including its unmasked
    # H-to-S nearest/second-nearest distances. Do not introduce H2 here.
    vals, idxs = torch.topk(d_H_S, 2, dim=1, largest=False)
    removal_cost = torch.zeros(S.shape[0], device=device)
    removal_cost.scatter_add_(0, idxs[:, 0], vals[:, 1] - vals[:, 0])
    remove_candidates = torch.topk(removal_cost, min(8, len(S)), largest=False).indices

    d_H_S.masked_fill_(H_uids[:, None] == S_uids[None, :], float("inf"))
    d_new = d_H_S.min(dim=1).values
    drift = torch.clamp(d_new - d_old, min=0)
    diagnostics["mean_before"] = drift.mean().item()
    diagnostics["risk_before"] = _h1_mean_cvar(drift, alpha, beta).item()
    diagnostics["reason"] = "max_swaps"

    for _ in range(max_swaps):
        d_new = d_H_S.min(dim=1).values
        drift = torch.clamp(d_new - d_old, min=0)
        loss_before = drift.mean().item()
        if loss_before < 1e-6:
            diagnostics["reason"] = "no_positive_drift"
            break

        # Recover only genuinely absent source rows. Image IDs need not be
        # unique. Remove accepted additions from the pool on the next round.
        retained_old = S_uids[(S_uids >= 0) & (S_uids < n_old)]
        discarded_mask = torch.ones(n_old, dtype=torch.bool, device=device)
        discarded_mask[retained_old] = False
        discarded_uids = old_uids[discarded_mask]
        if len(discarded_uids) == 0:
            diagnostics["reason"] = "no_discarded_history"
            break

        # H1: rank ALL absent historical prototypes by add-only Mean+CVaR
        # risk reduction. CVaR uses the worst ceil((1-alpha)*k) anchor drifts.
        # d_H_old already excludes an anchor's own UID, including when absent.
        add_distances = d_H_old[:, discarded_uids]
        drift_after_add = torch.clamp(
            torch.minimum(d_new[:, None], add_distances) - d_old[:, None], min=0
        )
        risk_after_add = _h1_mean_cvar(drift_after_add, alpha, beta)
        add_indices = torch.topk(risk_after_add, min(8, len(discarded_uids)), largest=False).indices
        add_uids = discarded_uids[add_indices]

        # Keep the ORIGINAL acceptance objective: strictly lower mean drift.
        # H1 risk selects the recovery pool; it is not an H3 acceptance gate.
        best_loss = loss_before
        best_swap = None
        best_distances = None
        for add_uid in add_uids:
            add_distance = d_H_old[:, add_uid]
            for rem_idx in remove_candidates:
                # Only one distance column changes in a one-for-one swap.
                # Reusing exact cached pairwise distances avoids recomputing
                # the full H-to-S matrix for every proposal.
                d_try_mat = d_H_S.clone()
                d_try_mat[:, rem_idx] = add_distance
                d_try = d_try_mat.min(dim=1).values
                loss_try = torch.clamp(d_try - d_old, min=0).mean().item()
                if loss_try < best_loss:
                    best_loss = loss_try
                    best_swap = (rem_idx, add_uid)
                    best_distances = d_try_mat

        if best_swap is None:
            diagnostics["reason"] = "no_mean_improving_swap"
            break
        rem_idx, add_uid = best_swap
        S[rem_idx] = old_memory[add_uid]
        S_ids[rem_idx] = old_ids[add_uid]
        S_uids[rem_idx] = add_uid
        d_H_S = best_distances
        diagnostics["swaps"] += 1

    final_drift = torch.clamp(d_H_S.min(dim=1).values - d_old, min=0)
    diagnostics["mean_after"] = final_drift.mean().item()
    diagnostics["risk_after"] = _h1_mean_cvar(final_drift, alpha, beta).item()
    return S, S_ids, diagnostics
