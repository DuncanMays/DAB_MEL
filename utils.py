import torch
import json
import requests
from config import config_object

def start_global_cycle(worker_ips):
	for w_ip in worker_ips:
		url = 'http://'+w_ip+':'+str(config_object.learner_port)+'/start_learning'
		requests.get(url)

def get_active_worker_ips():
	notice_board_resp = requests.get('http://'+config_object.notice_board_ip+':'+str(config_object.notice_board_port)+'/notice_board')
	return json.loads(notice_board_resp.content)

# calculates the accuracy score of a prediction y_hat and the ground truth y
I = torch.eye(10,10)
def get_accuracy(y_hat, y):
	y_vec = torch.tensor([I[int(i)].tolist() for i in y])
	dot = torch.dot(y_hat.flatten(), y_vec.flatten())
	return dot/torch.sum(y_hat)

# gets the accuracy and loss of net on testing data
TEST_BATCH_SIZE = 32
criterion = torch.nn.CrossEntropyLoss()
def val_evaluation(net, x_test, y_test):

	NUM_TEST_BATCHES = x_test.shape[0]//TEST_BATCH_SIZE

	loss = 0
	acc = 0

	for i in range(NUM_TEST_BATCHES):
		x_batch = x_test[TEST_BATCH_SIZE*i : TEST_BATCH_SIZE*(i+1)]
		y_batch = y_test[TEST_BATCH_SIZE*i : TEST_BATCH_SIZE*(i+1)]

		y_hat = net.forward(x_batch)

		loss += criterion(y_hat, y_batch).item()
		acc += get_accuracy(y_hat, y_batch).item()
	
	# normalizing the loss and accuracy
	loss = loss/NUM_TEST_BATCHES
	acc = acc/NUM_TEST_BATCHES

	return loss, acc

# sets the parameters of a neural net
def set_parameters(net, params):
	current_params = list(net.parameters())
	for i, p in enumerate(params): 
		current_params[i].data = p.data.clone()

def serialize_params(params):
	l = []
	for p in params:
		l.append(p.tolist())

	return json.dumps(l)

def deserialize_params(str_params):
	params_lists = json.loads(str_params)
	l = []
	for p in params_lists:
		l.append(torch.tensor(p))

	return l