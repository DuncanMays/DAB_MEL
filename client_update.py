import time
import requests
import torch
import json
from config import config_object
from networks import TwoNN
from utils import set_parameters, serialize_params, deserialize_params
from tqdm import tqdm

training_data_url = 'http://'+config_object.data_server_ip+':'+str(config_object.data_server_port)+'/get_training_data'
parameters_url = 'http://'+config_object.orchestrator_ip+':'+str(config_object.orchestrator_port)+'/get_parameters'

net = TwoNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([{'params': net.parameters()}], lr=config_object.client_learning_rate)

BATCH_SIZE = config_object.client_batch_size
def train_network(x, y):

	NUM_BATCHES = x.shape[0]//BATCH_SIZE

	for i in range(config_object.client_num_epochs):
		for j in tqdm(range(NUM_BATCHES)):

			x_batch = x[BATCH_SIZE*j: BATCH_SIZE*(j+1)]
			y_batch = y[BATCH_SIZE*j: BATCH_SIZE*(j+1)]

			y_hat = net(x_batch)

			loss = criterion(y_hat, y_batch)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

def client_update():
	print('starting training process')

	# TODO run benchmarks and calculate this from them
	num_shards = 20

	print('downloading data')
	data_resp = requests.post(url=training_data_url, data={'num_shards': num_shards})

	training_data = json.loads(data_resp.content)

	x_train = torch.tensor(training_data['x_data'])/255.0
	y_train = torch.tensor(training_data['y_data'])

	print('downloading parameters')
	params_resp = requests.get(parameters_url)
	set_parameters(net, deserialize_params(params_resp.content))

	print('training')
	train_network(x_train, y_train)

	print('submitting result')
	payload = {
		'num_shards': num_shards,
		'params': serialize_params(net.parameters())
	}

	return payload
