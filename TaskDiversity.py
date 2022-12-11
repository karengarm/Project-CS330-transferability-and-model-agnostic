"""  Script to get source and target diversity.
Authors: Phillip Yao-Lakaschus, Karen Garcia"""

import numpy as np
from scipy import spatial

LAYER = 'cls_output.npy'

SOURCE_TASKS = ["squad_v1",
    "squad_v2",
    "duorc_p",
    "duorc_s",
    "drop"]

def cosine_distance(source, target):
  task_emb_path = f"TaskEmb/{{}}/{LAYER}".format
  source_task_emb = np.load(task_emb_path(source)).reshape(-1)
  target_task_emb = np.load(task_emb_path(target)).reshape(-1)
  cosine_distance = spatial.distance.cosine(source_task_emb, target_task_emb)
  return cosine_distance

def averaged_source(source1, source2, source3, target='sst2'):
  task_emb_path = f"TaskEmb/{{}}/{LAYER}".format
  source_task_emb = (np.load(task_emb_path(source1)).reshape(-1) +
                     np.load(task_emb_path(source2)).reshape(-1) +
                     np.load(task_emb_path(source3)).reshape(-1))/3
  target_task_emb = np.load(task_emb_path(target)).reshape(-1)
  cosine_distance = spatial.distance.cosine(source_task_emb, target_task_emb)
  return cosine_distance

def diversity(task1, task2, task3):
  dist12 = cosine_distance(task1, task2)
  dist13 = cosine_distance(task1, task3)
  dist23 = cosine_distance(task2, task3)
  return {'tasks': (task1, task2, task3), 'diversity': (dist12 + dist13 + dist23)/3}

def total_diversity(tasks):
  distances = []
  for t in range(len(tasks) - 1):
    sub_list = tasks[t + 1:]
    for next_task in sub_list:
      distances.append(cosine_distance(tasks[t], next_task))
  return {'tasks': tasks, 'diversity': sum(distances)/len(distances)}

def all_diversities(tasks):
  diversities = []
  for t1, task1 in enumerate(tasks):
    for t2, task2 in enumerate(tasks):
      for t3, task3 in enumerate(tasks):
        if t1 < t2 < t3:
          diversities.append(total_diversity([task1, task2, task3]))
  return diversities

# source diversity

diversities = all_diversities(SOURCE_TASKS)
diversities = sorted(diversities, key=lambda d: d['diversity'])
print(diversities)


# target diversity
target_diversities_source = []
for div in diversities:
    sub_tasks = div['tasks']
    print('total diversity: ', total_diversity(sub_tasks + ['sst2']))
    target_diversities_source.append(total_diversity(sub_tasks + ['sst2']))

diversities_source = []
for task_el in diversities:
    for domain_el in target_diversities_source:
        if task_el['tasks'] == domain_el['tasks']:
            temp = {}
            temp['tasks'] = task_el['tasks']
            temp['domain_diversity'] = domain_el['diversity']
            temp['task_diversity'] = task_el['diversity']
            diversities_source.append(temp)
