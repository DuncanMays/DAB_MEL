import time
import torch
from networks import TwoNN

def aggregate_parameters(client_params, weights):
	print('aggregate_parameters has been called')

	# waits one seconds
	time.sleep(1)

	print('aggregation process complete')
	return None