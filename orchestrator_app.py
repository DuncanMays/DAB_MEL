from config import config_object
from flask import Flask
import requests
from flask import request as route_req
from multiprocessing import Process
from aggregate import aggregate_parameters
import time
import json

app = Flask(__name__)
worker_ips = []

def start_global_cycle(worker_ips):
	for w_ip in worker_ips:
		url = 'http://'+w_ip+':'+str(config_object.learner_port)+'/start_learning'
		requests.get(url)

num_results_submitted = 0
results = []
@app.route("/result_submit", methods=['POST'])
def result_submit():
	print('post received')

	global num_results_submitted, results

	num_results_submitted += 1

	if (num_results_submitted < len(worker_ips)):

		results.append(route_req.form)

		return json.dumps({'payload': 'storing result'})

	else:
		num_results_submitted = 0

		results.append(route_req.form)

		# the function that we will run in a separate process
		def wrapper_fn(a, b):
			aggregate_parameters(a, b)
			start_global_cycle(worker_ips)

		# start process that runs separate from this thread
		process = Process(target=wrapper_fn, args=(results, 'weights'))
		process.start()

		results = []

		return json.dumps({'payload': 'aggregating results'})

print('discovering workers')
notice_board_resp = requests.get('http://'+config_object.notice_board_ip+':'+str(config_object.notice_board_port)+'/notice_board')
worker_ips = json.loads(notice_board_resp.content)

print('starting '+str(len(worker_ips))+' workers')
print(worker_ips)
start_global_cycle(worker_ips)

print('starting orchestration app')
app.run(host='0.0.0.0', port=config_object.orchestrator_port)