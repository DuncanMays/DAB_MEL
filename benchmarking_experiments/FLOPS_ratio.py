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