import time
import requests
import torch
import json
from config import config_object
from utils import set_parameters, serialize_params, deserialize_params, download_training_data, val_evaluation
from tqdm import tqdm

ModelClass = config_object.model_class
device = config_object.training_device

net = ModelClass().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([{'params': net.parameters()}], lr=config_object.client_learning_rate)

def train_network(x, y):

	# BATCH_SIZE = max(1, x.shape[0]//config_object.client_batches_per_epoch)

	BATCH_SIZE = config_object.client_batch_size
	NUM_BATCHES = max(1, x.shape[0]//BATCH_SIZE)

	for i in range(config_object.client_num_epochs):
		for j in tqdm(range(NUM_BATCHES)):

			x_batch = x[BATCH_SIZE*j: BATCH_SIZE*(j+1)].to(device)
			y_batch = y[BATCH_SIZE*j: BATCH_SIZE*(j+1)].to(device)

			y_hat = net(x_batch)

			loss = criterion(y_hat, y_batch)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

def client_update(orchestrator_ip):
	print('starting training process')
	global net

	# the URL to return trained parameters to
	parameters_url = 'http://'+config_object.parameter_server_ip+':'+str(config_object.parameter_server_port)+'/get_parameters'

	# TODO read this from a file set by init
	f = open(config_object.init_config_file, 'r')
	# num_shards = json.loads(f.read())['num_shards']
	num_shards = 120
	f.close()

	print('downloading data for training')
	x_train, y_train = download_training_data(num_shards)

	print('downloading parameters')
	params_resp = requests.get(parameters_url)

	set_parameters(net, deserialize_params(params_resp.content))
	net = net.to(device)

	# print('testing parameters')
	# loss, acc = val_evaluation(net, x_train, y_train)
	# print(acc)

	print('training')
	train_network(x_train, y_train)

	# print('testing parameters')
	# loss, acc = val_evaluation(net, x_train, y_train)
	# print(acc)

	print('submitting result')
	payload = {
		'num_shards': num_shards,
		'params': serialize_params(net.parameters())
	}

	return payload
