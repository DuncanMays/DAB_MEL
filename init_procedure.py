import time
import torch
import requests
from utils import download_training_data
from config import config_object

ModelClass = config_object.model_class
device = config_object.training_device

criterion = torch.nn.CrossEntropyLoss()
BATCH_SIZE = 32

def subset_benchmark(num_shards = 1):

	dummy_model = ModelClass().to(device)
	dummy_optimizer = torch.optim.SGD([{'params': dummy_model.parameters()}], lr=config_object.client_learning_rate) 

	print('downloading data for benchmark')
	download_start_time = time.time()
	x, y = download_training_data(num_shards)
	download_end_time = time.time()

	download_time = download_end_time - download_start_time

	NUM_BATCHES = x.shape[0]//BATCH_SIZE
	loss = torch.tensor(0, dtype=torch.float32).to(device)
	training_start_time = time.time()

	dummy_optimizer.zero_grad()

	for j in range(NUM_BATCHES):

		x_batch = x[BATCH_SIZE*j: BATCH_SIZE*(j+1)].to(device)
		y_batch = y[BATCH_SIZE*j: BATCH_SIZE*(j+1)].to(device)

		y_hat = dummy_model(x_batch)

		loss += criterion(y_hat, y_batch)

	loss.backward()
	dummy_optimizer.step()

	training_end_time = time.time()
	training_time = training_end_time - training_start_time

	print('benchmark complete')
	return num_shards/download_time, num_shards/training_time

data_url = 'http://'+config_object.data_server_ip+':'+str(config_object.data_server_port)+'/get_training_data'
# this will break if the data server is ever not the same computer as the parameter server
params_url = 'http://'+config_object.parameter_server_ip+':'+str(config_object.parameter_server_port)+'/get_parameters'

def get_model_size():
	print('reporting model size')

	# downloads one shard and records the length of the string
	data_resp = requests.post(url=data_url, data={'num_shards': 1})
	training_data_size = len(data_resp.content)

	# downloads parameters and records the lenght of the string
	params_resp = requests.get(params_url)
	params_size = len(params_resp.content)

	return params_size/training_data_size

training_time_url = 'http://'+config_object.parameter_server_ip+':'+str(config_object.parameter_server_port)+'/get_training_time_limit'
def get_training_time_limit():
	return requests.get(training_time_url).content