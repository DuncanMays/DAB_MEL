from matplotlib import pyplot as plt
from math import ceil
import json

nano_data_file = './benchmarking_experiments/nano_prediction_errors_2.json'
laptop_data_file = './benchmarking_experiments/laptop_results/prediction_accuracies_6.json'

def get_avg_errors(target_file):
	data = None

	with open(target_file, 'r') as f:
		lines = f.readlines()[1:-1]	
		data = [json.loads(l[0:-1]) for l in lines]

	network_data = [d[0] for d in data]

	def list_transpose(l1):
		l2 = [[] for i in l1[0]]
		for l in l1:
			for i, e in enumerate(l):
				l2[i].append(e)

		return l2

	network_errors = list_transpose(network_data)
	# the first element of network_errors is -1 for the FLOPS method, which doesn't not predict network delay and so can be ignored
	network_errors = [100*sum(e)/len(e) for e in network_errors][1:]

	return network_errors

laptop_data = get_avg_errors(laptop_data_file)
nano_data = get_avg_errors(nano_data_file)

x = [2, 4, 6, 8, 10]
y_ticks = [i for i in range(0, ceil(max(laptop_data + nano_data)) + 5, 5)]

font_size=15

plt.plot(x, laptop_data, color='yellow', label='WiFi')
print('laptop_data:', laptop_data)
plt.plot(x, nano_data, color='blue', label='Ethernet')
print('nano_data:', nano_data)

plt.xticks(ticks=x, size=font_size)
plt.yticks(ticks=y_ticks, size=font_size)
plt.ylabel('Average Network Benchmarking Error (%)', size=font_size)
plt.xlabel('Benchmark Size (data shards)', size=font_size)

plt.legend(prop={'size': font_size})
plt.title('Network Time Prediction Error', size=25)
# plt.show()