from config import config_object
from flask import Flask
from flask import request as route_req
import json

app = Flask(__name__)

worker_IPs = set()

@app.route("/notice_board", methods=['GET'])
def get_IPs():
	print('request received at get_IPs')
	return json.dumps(list(worker_IPs))

@app.route("/notice_board", methods=['POST'])
def change_IPs():
	print('request received at change_IPs from ', end='')

	remote_addr = route_req.remote_addr
	print(remote_addr)

	if (route_req.form['task'] == 'add'):
		print('adding IP')
		# TODO change this to look if the IP is already in the list and not add it in that case
		worker_IPs.add(remote_addr)

	if (route_req.form['task'] == 'remove'):
		print('removing IP')
		worker_IPs.remove(remote_addr)

	return json.dumps(list(worker_IPs))

print('starting app')
app.run(host='0.0.0.0', port=config_object.notice_board_port)