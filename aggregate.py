import time
import torch
import requests
import json
from config import config_object
from utils import set_parameters, val_evaluation

device = config_object.training_device
net = config_object.model_class()
net.to(device)

testing_data_url = 'http://'+config_object.data_server_ip+':'+str(config_object.data_server_port)+'/get_testing_data'
data_resp = requests.post(url=testing_data_url, data={'num_shards': 20})
testing_data = json.loads(data_resp.content)

x_test = torch.tensor(testing_data['x_data'])/255.0
y_test = torch.tensor(testing_data['y_data'])

def aggregate_parameters(param_list, weights):

	num_clients = len(param_list)
	avg_params = []

	for i, params in enumerate(param_list):

		if (i == 0):
			for p in params:
				avg_params.append(p.clone()*weights[i])

		else:
			for j, p in enumerate(params):
				avg_params[j].data += p.data*weights[i]

	return avg_params

def assess_parameters(net):
	loss, acc = val_evaluation(net, x_test, y_test)

	return loss, acc