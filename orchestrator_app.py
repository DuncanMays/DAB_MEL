from config import config_object
from flask import Flask
import requests
from flask import request as route_req
from os import fork
from threading import Thread, Event
from aggregate import aggregate_parameters, assess_parameters
import time
import json
from utils import set_parameters, serialize_params, deserialize_params, get_active_worker_ips, start_global_cycle

app = Flask(__name__)
worker_ips = []

device = config_object.training_device

central_model = config_object.model_class()
central_model.to(device)

max_global_cycles = 5
num_global_cycles = 0
trial_complete_event = Event()
start_time = time.time()
num_results_submitted = 0
params = []
weights = []

training_stats = []

loss, acc = assess_parameters(central_model)

stat_dict = {
	'loss': loss,
	'acc': acc,
	'time': -1
}

training_stats.append(stat_dict)

worker_ips = get_active_worker_ips()

@app.route("/reset", methods=['GET'])
def reset_orchestrator():
	num_global_cycles = 0
	num_results_submitted = 0
	params = []
	weights = []

@app.route("/result_submit", methods=['POST'])
def result_submit():
	global num_results_submitted, params, weights, num_global_cycles, max_global_cycles, start_time

	print(f'results posted at /result_submit, {num_results_submitted}')

	# storing results
	result = route_req.form
	params.append(deserialize_params(result['params']))
	weights.append(int(result['num_shards']))

	num_results_submitted += 1
	if (num_results_submitted < len(worker_ips)):
		return json.dumps({'payload': 'storing result'})

	else:
		cycle_time = time.time() - start_time

		# we now aggregate the results
		num_global_cycles += 1

		# normalizing all the weights
		weights = [w/sum(weights) for w in weights]

		temp_net = config_object.model_class()

		# print('assessing accuracy of '+str(len(params_list))+' returned parameters')
		# for p in params_list:
		# 	new_params = p
		# 	set_parameters(temp_net, new_params)
		# 	loss, acc = assess_parameters(temp_net)
		# 	print('accuracy of returned network is: ', end='')
		# 	print(acc)

		new_params = aggregate_parameters(params, weights)

		set_parameters(temp_net, new_params)

		loss, acc = assess_parameters(temp_net)

		print('aggregation process complete, network loss and accuracy is: ', end='')
		print(loss, acc)

		stat_dict = {
			'loss': loss,
			'acc': acc,
			'time': cycle_time
		}

		training_stats.append(stat_dict)

		# we now submit the aggregated parameters to the main process

		payload = {
			'params': serialize_params(temp_net.parameters())
		}

		# this req is sent on loopback to this app and a route defined below
		set_url = 'http://localhost:'+str(config_object.orchestrator_port)+'/set_central_parameters'
		requests.post(url=set_url, data=payload)

		if (num_global_cycles < max_global_cycles):
			print(f'starting another cycle on {len(worker_ips)} workers')

			# resetting
			params = []
			weights = []

			start_cycle()

		else:
			print('done training, saving results')

			f = open('training_results.json', 'a')
			f.write(json.dumps(training_stats)+'\n')
			f.close()

			trial_complete_event.set()

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

@app.route('/get_training_time_limit', methods=['GET'])
def get_training_time_limit():
	return str(config_object.client_training_time)
	
@app.route('/shutdown', methods=['GET'])
def shutdown():

	func = route_req.environ.get('werkzeug.server.shutdown')
	if func is None:
		raise RuntimeError('Not running with the Werkzeug Server')
	func()

	return 'shutting down'

@app.route('/start_cycle', methods=['GET'])
def start_cycle():
	print(f'starting global cycle on {len(worker_ips)} workers')
	num_results_submitted = 0
	start_time = time.time()
	start_global_cycle(worker_ips)
	return 'starting global cycle'

total_shards = 120
def allocate(worker_characteristics):
	l = []

	total_training_rate = sum([float(w['training_rate']) for w in worker_characteristics])
	total_training_rate += sum([float(w['download_rate']) for w in worker_characteristics])

	for w in worker_characteristics:
		c_k = float(w['training_rate'])
		p_m = float(w['model_size'])
		T = config_object.client_training_time
		b_k = float(w['download_rate'])

		weight = (c_k+b_k)/total_training_rate
		num_shards = round(weight*total_shards)

		print(num_shards)

		tau = max(1, round((c_k/num_shards)*(T-(2*p_m + num_shards)/(b_k))))

		info_obj = {
			'ip_addr': w['ip_addr'],
			'model_size': p_m,
			'num_iterations': tau,
			'num_shards': num_shards
		}

		l.append(info_obj)

	return l

worker_characteristics = []
@app.route('/submit_characteristics', methods=['POST'])
def submit_characteristics():
	global worker_ips, worker_characteristics

	info_obj = {
		'ip_addr': route_req.remote_addr,
		'download_rate': route_req.form['download_rate'],
		'training_rate': route_req.form['training_rate'],
		'model_size': route_req.form['model_size']
	}

	worker_characteristics.append(info_obj)

	if (len(worker_characteristics) < len(worker_ips)):
		return json.dumps({'payload': 'storing characteristcs'})

	else:
		def wrapper_fn(worker_characteristics):
			allocation = allocate(worker_characteristics)

			for a in allocation:
				url = 'http://'+a['ip_addr']+':'+str(config_object.learner_port)+'/baseline_init_end'
				form = {
					'num_iterations': a['num_iterations'],
					'num_shards': a['num_shards'],
					'model_size': a['model_size']
				}

				requests.post(url=url, data=form)

			return

		if fork():
			wrapper_fn(worker_characteristics)

			worker_characteristics = []

			return json.dumps({'payload': 'this is the fork'})

		else:
			return json.dumps({'payload': 'processing characteristics'})			

def init():


def app_fn():
	print('starting orchestration app for '+str(len(worker_ips))+' workers')
	app.run(host='0.0.0.0', port=config_object.orchestrator_port)

def main():
	num_trials = 10

	app_thread = Thread(target=app_fn, daemon=True)
	app_thread.start()

	for t in range(num_trials):
		
		# these lines allocate data to each worker, using either DAB or the baseline CSA
		init()
		# baseline_init()

		# this line starts the learning process
		start_cycle()

		# waits for the trial to complete, then resets everything and loops
		trial_complete_event.wait()
		trial_complete_event = Event()
		reset_orchestrator()

	exit()

if (__name__ == '__main__'):
	main()