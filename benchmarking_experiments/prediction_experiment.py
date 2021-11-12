import sys
sys.path.append('..')
from init_procedure import subset_benchmark
from utils import download_training_data
from benchmarking_utils import real_task, FLOPS_benchmark
from matplotlib import pyplot as plt
import torch
import time
import json

# number of shards in the real task
task_size = 120
FLOPS_ratio = 3.029233303987113e-15

range_fn = lambda : range(5, 15, 5)

# print('running benchmarks')
# scores = [subset_benchmark(num_shards=n) for n in range_fn()]

# print('calculating predictions')
# predictions = [(task_size/score[0], task_size/score[1]) for score in scores]

# print('running the real task')
real_network, real_training = real_task(num_shards=task_size)

# print('calculating error')
# network_errors = [abs(real_network - p[0])/real_network for p in predictions]
# training_errors = [abs(real_training - p[1])/real_training for p in predictions]

# print('recording data')
# f = open('prediction_accuracies.json', 'a')
# f.write('\n'+json.dumps([network_errors, training_errors]))
# f.close()

FLOPS = FLOPS_benchmark()
FLOPS_prediction = task_size*FLOPS_ratio*FLOPS
FLOPS_error = abs(FLOPS_prediction - real_training)/real_training
print(FLOPS_error)

# print('plotting')
# x = list(range_fn())
# plt.plot(x, network_errors, label='network error')
# plt.plot(x, training_errors, label='training error')
# plt.legend()
# plt.show()