import time
import torch
from utils import download_training_data
from networks import TwoNN
from config import config_object
ModelClass = TwoNN

criterion = torch.nn.CrossEntropyLoss()
BATCH_SIZE = config_object.client_batch_size

def subset_benchmark(num_download_shards = 1):

	dummy_model = ModelClass()
	dummy_optimizer = torch.optim.SGD([{'params': dummy_model.parameters()}], lr=config_object.client_learning_rate) 

	print('downloading data for benchmark')
	download_start_time = time.time()
	x, y = download_training_data(num_download_shards)
	download_end_time = time.time()

	download_time = download_end_time - download_start_time

	NUM_BATCHES = x.shape[0]//BATCH_SIZE

	training_start_time = time.time()

	for j in range(NUM_BATCHES):

		x_batch = x[BATCH_SIZE*j: BATCH_SIZE*(j+1)]
		y_batch = y[BATCH_SIZE*j: BATCH_SIZE*(j+1)]

		y_hat = dummy_model(x_batch)

		loss = criterion(y_hat, y_batch)

		dummy_optimizer.zero_grad()
		loss.backward()
		dummy_optimizer.step()

	training_end_time = time.time()
	training_time = training_end_time - training_start_time

	print('init function done')
	return num_download_shards/download_time, num_download_shards/training_time