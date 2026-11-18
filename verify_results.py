"""Recompute published class and task-group forgetting from all stage CSVs."""
import csv
import json
import pathlib
import statistics

root = pathlib.Path(__file__).resolve().parent / 'results'
summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
order = ['bottle','cable','capsule','carpet','grid','hazelnut','leather',
         'metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']
for row in summary:
    a,b,n = map(int,row['schedule'].split('-'))
    n += 1
    stages = {}
    for t in range(1,n+1):
        with (root/row['variant']/row['schedule']/f'results_s{t}.csv').open(encoding='utf-8') as f:
            stages[t] = {r['Row Names']:r for r in csv.DictReader(f)}
    assert len(stages[n]) == 16
    old = [c for c in order if any(c in stages[t] for t in range(1,n))]
    assert len(old) == row['historical_classes']
    for m in row['final']:
        value = statistics.mean(float(stages[n][c][m]) for c in order)
        assert abs(value-row['final'][m]) < 1e-8
    groups=[];offset=0
    for size in [a]+[b]*(n-1):
        groups.append(order[offset:offset+size]);offset+=size
    for m in row['fm']:
        perclass=[max(float(stages[t][c][m]) for t in range(1,n) if c in stages[t])-float(stages[n][c][m]) for c in old]
        assert abs(statistics.mean(perclass)-row['fm'][m]) < 1e-8
        taskfm=[]
        for k,group in enumerate(groups[:-1]):
            values=[statistics.mean(float(stages[t][c][m]) for c in group) for t in range(k+1,n+1)]
            taskfm.append(max(values[:-1])-values[-1])
        assert abs(statistics.mean(taskfm)-row['task_macro_fm'][m]) < 1e-8
print(f'Verified final metrics, class FM and task-group FM for {len(summary)} runs.')
