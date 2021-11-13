import sys
sys.path.append('..')
from init_procedure import subset_benchmark
from utils import download_training_data
from benchmarking_utils import real_task, FLOPS_benchmark
from config import config_object
import torch
import time
import json

# number of shards in the real task
task_size = 120
FLOPS_ratio = 3.029233303987113e-15

SB_sizes = [1, 2, 4, 8, 32]

# initializing cuda
subset_benchmark(num_shards=1)

num_trials = 15
for i in range(num_trials):
	print('running benchmarks')
	scores = [subset_benchmark(num_shards=n) for n in SB_sizes]

	print('calculating predictions')
	predictions = [(task_size/score[0], task_size/score[1]) for score in scores]

	# print('running the real task')
	real_network, real_training = real_task(num_shards=task_size)

	print('calculating error')
	network_errors = [abs(real_network - p[0])/real_network for p in predictions]
	training_errors = [abs(real_training - p[1])/real_training for p in predictions]

	FLOPS = FLOPS_benchmark()
	FLOPS_prediction = task_size*FLOPS_ratio*FLOPS
	FLOPS_error = abs(FLOPS_prediction - real_training)/real_training

	labels = ['FLOPS']+SB_sizes
	training_errors = [FLOPS_error] + training_errors
	network_errors = [-1] + network_errors

	print('recording data')
	f = open('prediction_accuracies.json', 'a')
	f.write('\n'+json.dumps([network_errors, training_errors]))
	f.close()

	for i in range(len(training_errors)):
		print('method:', labels[i], end=' | ')
		print('training error:', training_errors[i], end=' | ')
		print('network error:', network_errors[i])