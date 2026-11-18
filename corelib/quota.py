"""Deterministic task coverage seeds, counted within the prototype budget."""
import math
import torch


def task_floors(labels, budget, tasks, fraction=.15):
    if not 0 <= fraction <= 1 or tasks < 1:
        raise ValueError('Invalid coverage fraction or task count')
    minimum=math.floor(fraction*budget/tasks)
    return {int(g):min(minimum,int((labels==g).sum())) for g in labels.unique()}


@torch.no_grad()
def coverage_seeds(features, labels, floors):
    """Within each task, start at a central row and cover remaining modes by FPS."""
    seeds=[]
    for group,count in sorted(floors.items()):
        indices=torch.where(labels==group)[0]
        if count==0:continue
        if count>len(indices):raise ValueError('Coverage quota exceeds available rows')
        values=features[indices]
        first=((values-values.mean(0))**2).sum(1).argmin()
        distances=torch.full((len(values),),float('inf'),device=values.device)
        chosen=torch.zeros(len(values),dtype=torch.bool,device=values.device)
        selected=first
        for _ in range(count):
            seeds.append(indices[selected])
            chosen[selected]=True
            distances=torch.minimum(distances,((values-values[selected])**2).sum(1))
            distances.masked_fill_(chosen,-1)
            selected=distances.argmax()
    return torch.stack(seeds).long() if seeds else torch.empty(0,device=features.device,dtype=torch.long)
