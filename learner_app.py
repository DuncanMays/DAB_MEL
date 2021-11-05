from config import config_object
from flask import Flask
from multiprocessing import Process
from client_update import client_update
import signal
import requests
import time
import json

notice_board_url = 'http://'+config_object.notice_board_ip+':'+str(config_object.notice_board_port)+'/notice_board'

def notify_board():
	print(notice_board_url)
	requests.post(url=notice_board_url, data={'task': 'add'})

# TODO on signint, inform the notice board
def shutdown_handler():
	requests.post(url=notice_board_url, data={'task': 'remove'})

signal.signal(signal.SIGINT, shutdown_handler)

app = Flask(__name__)

@app.route("/start_learning", methods=['GET'])
def start_learning():
	print('get received')

	# start process that runs separate from this thread
	process = Process(target=client_update, args=())
	process.start()

	return json.dumps({'payload': 'training now'})

print('notifying notice board')
notify_board()

print('starting app')
app.run(host='0.0.0.0', port=config_object.learner_port)