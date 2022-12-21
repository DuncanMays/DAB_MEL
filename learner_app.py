from config import config_object
from flask import Flask
from flask import request as route_req
from multiprocessing import set_start_method
from client_update import client_update
from init_procedure import subset_benchmark, get_model_size, get_training_time_limit
from os import fork
import signal
import requests
import time
import json

# see: https://pythonspeed.com/articles/python-multiprocessing/
# TLDR: forking usually doesn't copy threads to the child processes, pytorch uses a thread to monitor async GPU operations, and so this line lets forks inherit threads
set_start_method('spawn')

notice_board_url = 'http://'+config_object.notice_board_ip+':'+str(config_object.notice_board_port)+'/notice_board'

def notify_board():
	requests.post(url=notice_board_url, data={'task': 'add'})

# TODO on signint, inform the notice board
def shutdown_handler(a, b):
	requests.post(url=notice_board_url, data={'task': 'remove'})
	exit()

signal.signal(signal.SIGINT, shutdown_handler)

app = Flask(__name__)

# this route starts the training routine and is usaully called by orchestrator
@app.route("/start_learning", methods=['GET'])
def start_learning():
	print('get received')

	orchestrator_ip = route_req.remote_addr

	def wrapper_fn(orch_ip):
		result = client_update(orchestrator_ip)

		print('training completed, submitting results')
		result_url = 'http://'+config_object.parameter_server_ip+':'+str(config_object.parameter_server_port)+'/result_submit'
		x = requests.post(url=result_url, data=result)
		print(x.content)

	if fork():
		wrapper_fn(orchestrator_ip)
		return json.dumps({'payload': 'this is the fork'})
	else:	
		return json.dumps({'payload': 'training now'})

# this route runs the subset benchmark and local DAB routine
@app.route("/init_procedure", methods=['GET'])
def init_procedure():

	if fork():
		download_rate, training_rate = subset_benchmark(num_shards=8)
		model_size = get_model_size()

		deadline = get_training_time_limit()

		print('training time limit is:', deadline)

		num_shards = round((deadline - 2*model_size/download_rate)/(config_object.client_num_updates/training_rate + 1/download_rate))
		num_shards = max(1, num_shards)

		f = open(config_object.init_config_file, 'w')
		f.write(json.dumps({
			'num_shards': num_shards,
			'model_size': model_size,
			'num_iterations': config_object.client_num_updates,
			'download_rate': download_rate,
			'training_rate': training_rate,
			'deadline': config_object.client_training_time
		}))
		f.close()

		print('init complete, settings written to file')
		return json.dumps({'payload': 'this is the fork'})

	else:
		return json.dumps({'payload': 'running benchmarks'})

# this route runs the subset benchmark then the baseline data allocation routine. It sends a POST req containing its benchmark scores to the orchestrator, which runs the optimizer and sends a POST req back with training settings
@app.route("/baseline_init_start", methods=['GET'])
def baseline_init_start():

	if fork():
		download_rate, training_rate = subset_benchmark(num_shards=8)
		model_size = get_model_size()

		info = {
			'download_rate': download_rate,
			'training_rate': training_rate,
			'model_size': model_size
		}

		DA_url = 'http://'+config_object.parameter_server_ip+':'+str(config_object.parameter_server_port)+'/submit_characteristics'
		requests.post(url=DA_url, data=info)

		return json.dumps({'payload': 'this is the fork'})

	else:
		return json.dumps({'payload': 'running benchmarks'})

# this is the second route involved in the baseline init procedure. The first one above sends a POST req to the orchestrator, which does some processing, and then responds with a POST req of its own which triggers this route which writes training settings to file.
@app.route("/baseline_init_end", methods=['POST'])
def baseline_init_end():

	print(route_req.form)

	info = route_req.form

	num_shards = info['num_shards']
	num_iterations = info['num_iterations']
	model_size = info['model_size']

	f = open(config_object.init_config_file, 'w')
	f.write(json.dumps({
		'num_shards': num_shards,
		'model_size': model_size,
		'num_iterations': num_iterations,
		'download_rate': 0,
		'training_rate': 0,
		'deadline': config_object.client_training_time
	}))
	f.close()

	print('baseline init complete, settings written to file')
	return json.dumps({'payload': 'recorded info'})

print('notifying notice board')
notify_board()

print('starting app')
app.run(host='0.0.0.0', port=config_object.learner_port)