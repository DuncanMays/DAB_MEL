from config import config_object
from flask import Flask
from flask import request as route_req
from keras.datasets import mnist
import torch
import json

print('importing data')

raw_data = mnist.load_data()

print('preprocessing')

shard_size = 500

x_train_raw = raw_data[0][0]
y_train_raw = raw_data[0][1]
x_test_raw = raw_data[1][0]
y_test_raw = raw_data[1][1]

# formatting into sample shape

sample_shape = config_object.sample_shape

x_train = torch.tensor(x_train_raw, dtype=torch.float32).reshape([-1]+sample_shape)
x_test = torch.tensor(x_test_raw, dtype=torch.float32).reshape([-1]+sample_shape)

y_train = torch.tensor(y_train_raw, dtype=torch.long)
y_test = torch.tensor(y_test_raw, dtype=torch.long)

# splitting into shards

num_train_shards = x_train.shape[0]//shard_size
num_test_shards = x_test.shape[0]//shard_size

# cutting off the samples that won't be in any shard
x_train = x_train[0: num_train_shards*shard_size]
y_train = y_train[0: num_train_shards*shard_size]
x_test = x_test[0: num_test_shards*shard_size]
y_test = y_test[0: num_test_shards*shard_size]

# reshaping
x_train = x_train.unsqueeze(dim=0).reshape([num_train_shards, -1]+sample_shape)
y_train = y_train.unsqueeze(dim=0).reshape([num_train_shards, -1])
x_test = x_test.unsqueeze(dim=0).reshape([num_test_shards, -1]+sample_shape)
y_test = y_test.unsqueeze(dim=0).reshape([num_test_shards, -1])

app = Flask(__name__)

@app.route("/get_training_data", methods=['POST'])
def get_training_data():
	print('request received at get_training_data')

	# the number of shards being requested
	num_shards = min(120, int(route_req.form['num_shards']))

	indices = torch.randperm(x_train.shape[0])[0: num_shards]

	x_data = x_train[indices].reshape([-1]+sample_shape)
	y_data = y_train[indices].reshape([-1])

	payload = {'x_data': x_data.tolist(), 'y_data': y_data.tolist()}

	return json.dumps(payload)

@app.route("/get_testing_data", methods=['POST'])
def get_testing_data():
	print('request received at get_testing_data')

	# the number of shards being requested
	num_shards = min(20, int(route_req.form['num_shards']))

	indices = torch.randperm(x_test.shape[0])[0: num_shards]

	x_data = x_test[indices].reshape([-1]+sample_shape)
	y_data = y_test[indices].reshape([-1])

	payload = {'x_data': x_data.tolist(), 'y_data': y_data.tolist()}

	return json.dumps(payload)

print('now serving at port ', config_object.data_server_port)
app.run(host='0.0.0.0', port=config_object.data_server_port)