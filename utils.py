import torch
import json
import requests
from config import config_object

device = config_object.training_device

def start_global_cycle(worker_ips):
	for w_ip in worker_ips:
		url = 'http://'+w_ip+':'+str(config_object.learner_port)+'/start_learning'
		requests.get(url)

def init_workers(worker_ips):
	for w_ip in worker_ips:
		url = 'http://'+w_ip+':'+str(config_object.learner_port)+'/init_procedure'
		requests.get(url)

def get_active_worker_ips():
	notice_board_resp = requests.get('http://'+config_object.notice_board_ip+':'+str(config_object.notice_board_port)+'/notice_board')
	return json.loads(notice_board_resp.content)

# calculates the accuracy score of a prediction y_hat and the ground truth y
I = torch.eye(10,10)
def get_accuracy(y_hat, y):
	y_vec = torch.tensor([I[int(i)].tolist() for i in y]).to(device)
	dot = torch.dot(y_hat.flatten(), y_vec.flatten())
	return dot/torch.sum(y_hat)

def download_training_data(num_shards):
	training_data_url = 'http://'+config_object.data_server_ip+':'+str(config_object.data_server_port)+'/get_training_data'
	data_resp = requests.post(url=training_data_url, data={'num_shards': num_shards})
	training_data = json.loads(data_resp.content)

	x = torch.tensor(training_data['x_data'])/255.0
	y = torch.tensor(training_data['y_data'])

	del training_data

	return x, y

# gets the accuracy and loss of net on testing data
TEST_BATCH_SIZE = 32
criterion = torch.nn.CrossEntropyLoss()
def val_evaluation(net, x_test, y_test):

	NUM_TEST_BATCHES = x_test.shape[0]//TEST_BATCH_SIZE

	loss = 0
	acc = 0

	net = net.to(device)

	for i in range(NUM_TEST_BATCHES):
		x_batch = x_test[TEST_BATCH_SIZE*i : TEST_BATCH_SIZE*(i+1)].to(device)
		y_batch = y_test[TEST_BATCH_SIZE*i : TEST_BATCH_SIZE*(i+1)].to(device)

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