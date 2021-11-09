from config import config_object
from flask import Flask
import requests
from flask import request as route_req
from os import fork
from aggregate import aggregate_parameters, assess_parameters
import time
import json
import torch
from utils import set_parameters, serialize_params, deserialize_params, start_global_cycle, get_active_worker_ips

app = Flask(__name__)
worker_ips = []

device = config_object.training_device

central_model = config_object.model_class()
central_model.to(device)

num_results_submitted = 0
params = []
weights = []
@app.route("/result_submit", methods=['POST'])
def result_submit():
	print('results posted at /result_submit')

	global num_results_submitted, params, weights

	num_results_submitted += 1

	# storing results
	result = route_req.form
	params.append(deserialize_params(result['params']))
	weights.append(int(result['num_shards']))

	if (num_results_submitted < len(worker_ips)):

		return json.dumps({'payload': 'storing result'})

	else:
		# we now aggregate the results

		# normalizing all the weights
		weights = [w/sum(weights) for w in weights]

		# resetting trackers
		num_results_submitted = 0
		params = []
		weights = []

		# we will run this function in a separate process
		def wrapper_fn(a, b):
			temp_net = config_object.model_class()

			new_params = aggregate_parameters(a, b)

			set_parameters(temp_net, new_params)

			loss, acc = assess_parameters(temp_net)

			print('aggregation process complete, network loss and accuracy is:')
			print(loss, acc)

			# we now submit the aggregated parameters to the main process

			payload = {
				'params': serialize_params(temp_net.parameters())
			}

			set_url = 'http://localhost:'+str(config_object.orchestrator_port)+'/set_central_parameters'

			requests.post(url=set_url, data=payload)

			start_global_cycle(worker_ips)

		if fork():
			wrapper_fn(params, weights)

			return json.dumps({'payload': 'this is the fork'})

		else:
			return json.dumps({'payload': 'aggregating results'})

@app.route("/get_parameters", methods=['GET'])
def get_parameters():
	global central_model
	print('serving parameters to:', route_req.remote_addr)
	return serialize_params(central_model.parameters())

@app.route('/set_central_parameters', methods=['POST'])
def set_central_parameters():
	global central_model
	new_params = deserialize_params(route_req.form['params'])
	set_parameters(central_model, new_params)
	return json.dumps({'payload': 'central network updated'})

print('discovering workers')
notice_board_resp = requests.get('http://'+str(config_object.notice_board_ip)+':'+str(config_object.notice_board_port)+'/notice_board')
worker_ips = get_active_worker_ips()

print('starting '+str(len(worker_ips))+' workers')
start_global_cycle(worker_ips)

print('starting orchestration app')
app.run(host='0.0.0.0', port=config_object.orchestrator_port)