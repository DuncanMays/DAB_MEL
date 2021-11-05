from config import config_object
import time
import requests

def client_update(orchestrator_ip):
	print('client_update has been called')

	# waits two seconds
	time.sleep(2)

	print('training completed, submitting results')
	requests.post(url='http://'+orchestrator_ip+':'+str(config_object.orchestrator_port)+'/result_submit', data={'payload': 'hello there!'})

	print('process complete')
	return None
