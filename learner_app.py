from config import config_object
from flask import Flask
from flask import request as route_req
from multiprocessing import Process
from client_update import client_update
from init_procedure import subset_benchmark
import signal
import requests
import time
import json

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
		result_url = 'http://'+orch_ip+':'+str(config_object.orchestrator_port)+'/result_submit'
		x=requests.post(url=result_url, data=result)

	# start process that runs separate from this thread
	process = Process(target=wrapper_fn, args=(orchestrator_ip, ))
	process.start()

	return json.dumps({'payload': 'training now'})

@app.route("/init_procedure", methods=['GET'])
def init_procedure():

	download_rate, training_rate = subset_benchmark(num_download_shards=1)

	# the training deadline, in seconds
	D = 60;
	# P_d is the amount of data in each data shard, which is one since a data shard was the measurement unit
	P_d = 1;
	# P_m is the amount of data in the model's parameters, which must be expressed as a ratio wrt the size of a data shard
	P_m = 0.354;
	# mu is the number of training iterations, we should probably set this automatically, but for the time being this is fine
	mu = 1;

	num_shards = (D - 2*P_m/download_rate)/(mu/training_rate + P_d/download_rate)

	f = open(config_object.init_config_file, 'w')
	f.write(json.dumps({'num_shards': num_shards}))
	f.close()

	return json.dumps({'payload': 'running benchmarks'})

print('notifying notice board')
notify_board()

print('starting app')
app.run(host='0.0.0.0', port=config_object.learner_port)