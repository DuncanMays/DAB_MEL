from config import config_object
from flask import Flask
from flask import request as route_req
from multiprocessing import set_start_method
from client_update import client_update
from init_procedure import subset_benchmark, get_model_size
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

@app.route("/init_procedure", methods=['GET'])
def init_procedure():

	if fork():
		download_rate, training_rate = subset_benchmark(num_shards=8)
		model_size = get_model_size()

		# the training deadline, in seconds
		D = 10;

		num_shards = round((D - 2*model_size/download_rate)/(config_object.client_num_updates/training_rate + 1/download_rate))

		f = open(config_object.init_config_file, 'w')
		f.write(json.dumps({
			'num_shards': num_shards,
			'model_size': model_size,
			'download_rate': download_rate,
			'training_rate': training_rate
		}))
		f.close()

		return json.dumps({'payload': 'this is the fork'})

	else:
		return json.dumps({'payload': 'running benchmarks'})

print('notifying notice board')
notify_board()

print('starting app')
app.run(host='0.0.0.0', port=config_object.learner_port)