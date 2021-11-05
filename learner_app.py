from config import config_object
from flask import Flask
from flask import request as route_req
from multiprocessing import Process
from client_update import client_update
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

print('notifying notice board')
notify_board()

print('starting app')
app.run(host='0.0.0.0', port=config_object.learner_port)