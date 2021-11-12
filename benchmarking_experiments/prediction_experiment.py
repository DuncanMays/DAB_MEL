import sys
sys.path.append('..')
from init_procedure import subset_benchmark
from utils import download_training_data
from config import config_object
from matplotlib import pyplot as plt
import torch
import time
import json

# number of shards in the real task
task_size = 120

ModelClass = config_object.model_class
# device = config_object.training_device
device = 'cpu'

criterion = torch.nn.CrossEntropyLoss()
BATCH_SIZE = config_object.client_batch_size

def real_task(num_shards = 1):

	model = ModelClass().to(device)
	optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=config_object.client_learning_rate) 

	print('downloading real data')
	download_start_time = time.time()
	x, y = download_training_data(num_shards)
	download_end_time = time.time()

	download_time = download_end_time - download_start_time

	NUM_BATCHES = x.shape[0]//BATCH_SIZE

	training_start_time = time.time()

	for j in range(NUM_BATCHES):

		x_batch = x[BATCH_SIZE*j: BATCH_SIZE*(j+1)].to(device)
		y_batch = y[BATCH_SIZE*j: BATCH_SIZE*(j+1)].to(device)

		y_hat = model(x_batch)

		loss = criterion(y_hat, y_batch)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	training_end_time = time.time()
	training_time = training_end_time - training_start_time

	return download_time, training_time

range_fn = lambda : range(5, 10, 5)

print('running benchmarks')
scores = [subset_benchmark(num_shards=n) for n in range_fn()]

print('calculating predictions')
predictions = [(task_size/score[0], task_size/score[1]) for score in scores]

print('running the real task')
real_network, real_training = real_task(num_shards=task_size)

print('calculating error')
network_errors = [abs(real_network - p[0])/real_network for p in predictions]
training_errors = [abs(real_training - p[1])/real_training for p in predictions]

print('recording data')
f = open('prediction_accuracies.json', 'a')
f.write('\n'+json.dumps([network_errors, training_errors]))
f.close()

print('plotting')
x = list(range_fn())
plt.plot(x, network_errors, label='network error')
plt.plot(x, training_errors, label='training error')
plt.legend()
plt.show()