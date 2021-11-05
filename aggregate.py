import time
import torch
import requests
import json
from networks import TwoNN
from config import config_object
from utils import set_parameters, val_evaluation

net = TwoNN()

testing_data_url = 'http://'+config_object.data_server_ip+':'+str(config_object.data_server_port)+'/get_testing_data'
data_resp = requests.post(url=testing_data_url, data={'num_shards': 20})
testing_data = json.loads(data_resp.content)

x_test = torch.tensor(testing_data['x_data'])/255.0
y_test = torch.tensor(testing_data['y_data'])

def weighted_average_parameters(param_list, weights):

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

def aggregate_parameters(client_params, weights):
	print('aggregate_parameters has been called')

	avg_params = weighted_average_parameters(client_params, weights)

	set_parameters(net, avg_params)

	loss, acc = val_evaluation(net, x_test, y_test)

	print('aggregation process complete, network loss and accuracy is:')
	print(loss, acc)
	return None