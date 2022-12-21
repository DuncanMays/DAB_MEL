from matplotlib import pyplot as plt
import numpy as np
import json

# FLOPS = [17, 6, 10, 11, 13 ]
# SB1 = [2.5, 0.1, 0.2, 2.4, 3.9]
# SB2 = [2.6, 0.7, 0.1, 1.7, 4.3]
# SB4 = [2.1, 1.3, 0.3, 0.1, 4.3]
# SB8 = [0.1, 0.4, 0.3, 0.5, 2.1]

target_file = './benchmarking_experiments/nano_prediction_errors_2.json'

SB_sizes = [2, 4, 6, 8]
data = None

with open(target_file, 'r') as f:
	lines = f.readlines()[1:-1]	
	data = [json.loads(l[0:-1]) for l in lines]

network_data = [d[0] for d in data]
training_data = [d[1] for d in data]

def list_transpose(l1):
	l2 = [[] for i in l1[0]]
	for l in l1:
		for i, e in enumerate(l):
			l2[i].append(e)

	return l2

training_errors = list_transpose(training_data)
training_errors = [100*sum(e)/len(e) for e in training_errors]

network_errors = list_transpose(network_data)
# the first element of network_errors is -1 for the FLOPS method, which doesn not predict network delay and so can be ignored
network_errors = [100*sum(e)/len(e) for e in network_errors][1:]

# print(training_errors)
# print(network_errors)

# exit()

font_size=15

# flops = sum(FLOPS)/len(FLOPS)
# SB_2 = sum(SB1)/len(SB1)
# SB_4 = sum(SB2)/len(SB2)
# SB_6 = sum(SB4)/len(SB4)
# SB_8 = sum(SB8)/len(SB8)

# data = [SB_2, SB_4, SB_6, SB_8]
# flops_data = [flops for d in data]

data = training_errors[1:]
print(data)
flops_data = [training_errors[0] for d in data]

x = [2, 4, 6, 8, 10]
y_ticks = [i for i in range(0, 65, 5)]

plt.plot(x, data, color='red', label='SB')
plt.plot(x, flops_data, color='green', label='FLOPS')

plt.xticks(ticks=x, size=font_size)
plt.yticks(ticks=y_ticks, size=font_size)
plt.ylabel('Average Runtime Benchmarking Error (%)', size=font_size)
plt.xlabel('Benchmark Size (data shards)', size=font_size)

plt.legend(prop={'size': font_size})
plt.show()

y_ticks = [i for i in range(0, 22, 2)]

plt.plot(x, network_errors, color='red', label='SB')

plt.xticks(ticks=x, size=font_size)
plt.yticks(ticks=y_ticks, size=font_size)
plt.ylabel('Average Network Benchmarking Error (%)', size=font_size)
plt.xlabel('Benchmark Size (data shards)', size=font_size)

# plt.legend(prop={'size': font_size})
plt.show()