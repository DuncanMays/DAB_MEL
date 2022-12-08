# the goal of this file is to determine the ratio between the training time for a shard of data on a real neural net, and the max FLOPS the GPU is capable of, as measure by matrix multiplication

import torch
import sys
sys.path.append('..')
import networks
from benchmarking_utils import FLOPS_benchmark, real_task

score = FLOPS_benchmark()

n_shards = 60
real_network, real_training = real_task(num_shards=n_shards)
time_for_shard = real_training/n_shards

ratio = time_for_shard/score

print('this computer\'s FLOPS ratio is: ', ratio)