import time
import requests
import torch
import json
from config import config_object
from networks import TwoNN
from utils import set_parameters, serialize_params, deserialize_params

training_data_url = 'http://'+config_object.data_server_ip+':'+str(config_object.data_server_port)+'/get_training_data'
parameters_url = 'http://'+config_object.orchestrator_ip+':'+str(config_object.orchestrator_port)+'/get_parameters'

net = TwoNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([{'params': net.parameters()}], lr=config_object.client_learning_rate)

def client_update():
	print('starting training process')

	# TODO run benchmarks and calculate this from them
	num_shards = 5

	print('downloading data')
	data_resp = requests.post(url=training_data_url, data={'num_shards': num_shards})

	training_data = json.loads(data_resp.content)

	x_train = torch.tensor(training_data['x_data'])/255.0
	y_train = torch.tensor(training_data['y_data'])

	print('downloading parameters')
	params_resp = requests.get(parameters_url)
	set_parameters(net, deserialize_params(params_resp.content))

	print('process complete')
	return {'payload': 'hello there!'}
