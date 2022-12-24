import time
import requests
import torch
import json
from config import config_object
from utils import set_parameters, serialize_params, deserialize_params, download_training_data, val_evaluation
from tqdm import tqdm
import time

torch.autograd.set_detect_anomaly(True)

ModelClass = config_object.model_class
device = config_object.training_device

net = ModelClass()
criterion = torch.nn.CrossEntropyLoss()

def train_network(tau, x, y):
	global net
	net = net.to(device)
	optimizer = torch.optim.SGD([{'params': net.parameters()}], lr=config_object.client_learning_rate)

	BATCH_SIZE = 32
	NUM_BATCHES = max(1, x.shape[0]//BATCH_SIZE)

	for i in range(tau):
		
		for j in tqdm(range(NUM_BATCHES)):
			# loss = torch.tensor(0, dtype=torch.float32).to(device)

			x_batch = x[BATCH_SIZE*j: BATCH_SIZE*(j+1)].to(device)
			y_batch = y[BATCH_SIZE*j: BATCH_SIZE*(j+1)].to(device)

			y_hat = net(x_batch)

			loss = criterion(y_hat, y_batch)
		
			optimizer.zero_grad()

			loss.backward()
			optimizer.step()

def client_update(orchestrator_ip):
	print('starting training process')
	global net, optimizer

	start_time = time.time()

	# the URL to return trained parameters to
	parameters_url = 'http://'+config_object.parameter_server_ip+':'+str(config_object.parameter_server_port)+'/get_parameters'

	# TODO read this from a file set by init
	f = open(config_object.init_config_file, 'r')
	info_obj = json.loads(f.read())
	num_shards = int(info_obj['num_shards'])
	num_iterations = int(info_obj['num_iterations'])
	f.close()

	print('downloading data for training')
	x_train, y_train = download_training_data(num_shards)

	print('downloading parameters')
	params_resp = requests.get(parameters_url)

	set_parameters(net, deserialize_params(params_resp.content))
	net.to(device)

	# print('testing parameters')
	# loss, acc = val_evaluation(net, x_train, y_train)
	# print(acc)

	print('training')
	train_network(num_iterations, x_train, y_train)

	# print('testing parameters')
	# loss, acc = val_evaluation(net, x_train, y_train)
	# print(acc)

	print('training complete')
	
	payload = {
		'num_shards': num_shards,
		'params': serialize_params(net.parameters()),
		'time': time.time() - start_time,
	}

	return payload
