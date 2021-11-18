import torch
import time
import sys
sys.path.append('..')
from config import config_object
from utils import download_training_data

def FLOPS_benchmark():

	n = 512
	i = 100

	M1 = torch.randn([n, n], dtype=torch.float32).to(device)
	M2 = torch.randn([n, n], dtype=torch.float32).to(device)

	start = time.time()

	for j in range(i):
		M = M1*M2

	end = time.time()
	ops = n**3 + (n-1)*n**2

	return i*ops/(end - start)

ModelClass = config_object.model_class
device = config_object.training_device
BATCH_SIZE = 32
criterion = torch.nn.CrossEntropyLoss()

def real_task(num_shards = 1):

	model = ModelClass().to(device)
	optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=config_object.client_learning_rate) 

	print('downloading real data')
	download_start_time = time.time()
	x, y = download_training_data(num_shards)
	download_end_time = time.time()

	download_time = download_end_time - download_start_time

	print('training on '+device)
	training_start_time = time.time()
	NUM_BATCHES = x.shape[0]//BATCH_SIZE

	for i in range(1):
		
		loss = torch.tensor(0, dtype=torch.float32).to(device)

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