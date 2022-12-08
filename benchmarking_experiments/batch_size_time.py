import sys
sys.path.append('..')
from init_procedure import subset_benchmark
from client_update import train_network
from utils import download_training_data
from networks import TwoNN, ThreeNN, ConvNet
from config import config_object
from matplotlib import pyplot as plt
from keras.datasets import mnist
from tqdm import tqdm
import time
import torch

device = 'cuda:0'
# device = 'cpu'

net = ThreeNN().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([{'params': net.parameters()}], lr=config_object.client_learning_rate)

print('importing data')
# x_global, y_global = download_training_data(120)
raw_data = mnist.load_data()

print('preprocessing')

shard_size = 500

x_train_raw = raw_data[0][0]
y_train_raw = raw_data[0][1]

# formatting into sample shape

# sample_shape = config_object.sample_shape
sample_shape = [784]


x_global = torch.tensor(x_train_raw, dtype=torch.float32).reshape([-1]+sample_shape)
y_global = torch.tensor(y_train_raw, dtype=torch.long)

def gradient_update(num_samples=32):

	x_batch = x_global[0: num_samples].to(device)
	y_batch = y_global[0: num_samples].to(device)

	y_hat = net(x_batch)

	optimizer.zero_grad()
	loss = criterion(y_hat, y_batch)
	optimizer.step()

def time_gradient_update(num_shards):
	start = time.time()
	gradient_update(num_samples=500*num_shards)
	end = time.time()

	return end - start

def train_on_epoch(batch_size):
	NUM_BATCHES = x_global.shape[0] // batch_size

	for i in tqdm(range(NUM_BATCHES)):
		x_batch = x_global[i*batch_size: (i+1)*batch_size].to(device)
		y_batch = y_global[i*batch_size: (i+1)*batch_size].to(device)

		y_hat = net(x_batch)

		optimizer.zero_grad()
		loss = criterion(y_hat, y_batch)
		optimizer.step()

def time_train_on_epoch(batch_size):
	start = time.time()
	train_on_epoch(batch_size)
	end = time.time()

	return end - start

def train_on_dataset(num_batches=10):

	BATCH_SIZE = x_global.shape[0] // num_batches

	for i in tqdm(range(num_batches)):
		x_batch = x_global[i*BATCH_SIZE: (i+1)*BATCH_SIZE].to(device)
		y_batch = y_global[i*BATCH_SIZE: (i+1)*BATCH_SIZE].to(device)

		y_hat = net(x_batch)

		optimizer.zero_grad()
		loss = criterion(y_hat, y_batch)
		optimizer.step()

def time_train_on_dataset(batch_size):
	start = time.time()
	train_on_dataset(batch_size)
	end = time.time()

	return end - start

def real_task():
	num_shards = 120

	x_train, y_train = download_training_data(num_shards)

	train_network(x_train, y_train)

gradient_update(5)

x = [i+1 for i in range(12)]
y = [time_gradient_update(10*i) for i in x]

plt.plot(x, y)
# plt.plot(x, x)

print(y)

z = [y[i+1] - y[i] for i in range(0, len(y)-1)]

print(z)

plt.show()