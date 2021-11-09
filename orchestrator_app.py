from config import config_object
from flask import Flask
import requests
from flask import request as route_req
from multiprocessing import Process
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

		# the function that we will run in a separate process
		def wrapper_fn(a, b):
			temp_net = config_object.model_class()

			new_params = aggregate_parameters(a, b)

			set_parameters(temp_net, new_params)

			loss, acc = assess_parameters(temp_net)

			print('aggregation process complete, network loss and accuracy is:')
			print(loss, acc)

			payload = {
				'params': serialize_params(temp_net.parameters())
			}

			set_url = 'http://localhost:'+str(config_object.orchestrator_port)+'/set_central_parameters'

			requests.post(url=set_url, data=payload)

			start_global_cycle(worker_ips)

		# start a separate process that runs aggregates and assesses the parameters returned from workers
		process = Process(target=wrapper_fn, args=(params, weights))
		process.start()

		num_results_submitted = 0
		params = []
		weights = []

		return json.dumps({'payload': 'aggregating results'})

@app.route("/get_parameters", methods=['GET'])
def get_parameters():
	print('serving parameters to:', route_req.remote_addr)
	return serialize_params(central_model.parameters())

@app.route('/set_central_parameters', methods=['POST'])
def set_central_paameters():
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