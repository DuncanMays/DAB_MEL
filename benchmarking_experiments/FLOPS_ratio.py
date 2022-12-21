# the goal of this file is to determine the ratio between the training time for a shard of data on a real neural net, and the max FLOPS the GPU is capable of, as measure by matrix multiplication

import torch
import sys
sys.path.append('..')
import networks
from benchmarking_utils import FLOPS_benchmark, real_task

def get_ratio():
	score = FLOPS_benchmark()

	n_shards = 60
	real_network, real_training = real_task(num_shards=n_shards)
	time_for_shard = real_training/n_shards

	ratio = time_for_shard/score

	return ratio

if (__name__ == "__main__"):

	FLOPS = FLOPS_benchmark()
	FLOPS = FLOPS_benchmark()
	FLOPS = FLOPS_benchmark()
	
	ratio = get_ratio()

	print('this computer\'s FLOPS ratio is: ', ratio)
