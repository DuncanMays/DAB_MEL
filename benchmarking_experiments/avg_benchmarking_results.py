import json

f = open('./prediction_accuracies.json')

num_methods = 6
num_trials = 10

avg_network_errors = [0 for i in range(num_methods)]
avg_training_errors = [0 for i in range(num_methods)]

for i in range(num_trials):
	[network_errors, training_errors] = json.loads(f.readline()[0:-1])

	for j in range(num_methods):
		avg_network_errors[j] += network_errors[j]
		avg_training_errors[j] += training_errors[j]

for j in range(num_methods):
		avg_network_errors[j] = network_errors[j]/num_trials
		avg_training_errors[j] = training_errors[j]/num_trials

labels = ['FLOPS']+[1, 2, 4, 8, 32]
for i in range(len(avg_training_errors)):
	print('method:', labels[i], end=' | ')
	print('training error:', avg_training_errors[i], end=' | ')
	print('network error:', avg_network_errors[i])