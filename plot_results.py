from matplotlib import pyplot as plt
import json

storage_folder = './results/'

main_files = ['main_k4_T15.json', 'main_k4_T30.json']
baseline_files = ['baseline_k4_T15.json', 'baseline_k4_T30.json']

def read_file(path):
	f = open(path, 'r')
	data = json.loads(f.read())
	f.close()

	return data

def transpose(in_list):
	new_list_one = []
	new_list_two = []

	for e in in_list:
		new_list_one.append(e[0])
		new_list_two.append(e[1])

	return[new_list_one, new_list_two]

[main_loss, main_acc] = transpose(read_file(storage_folder+main_files[0]))
[baseline_loss, baseline_acc] = transpose(read_file(storage_folder+baseline_files[0]))

print(main_acc)

num_iters = len(main_loss)
x = [i for i in range(num_iters)]

plt.title('big title')

plt.subplot(2, 2, 1)
plt.title('loss, D=15s')
plt.plot(x, main_loss[0: num_iters], label='DAB_MEL')
plt.plot(x, baseline_loss[0: num_iters], label='baseline')
plt.legend()

plt.subplot(2, 2, 3)
plt.title('accuracy, D=15s')
plt.plot(x, main_acc[0: num_iters], label='DAB_MEL')
plt.plot(x, baseline_acc[0: num_iters], label='baseline')
plt.legend()

[main_loss, main_acc] = transpose(read_file(storage_folder+main_files[1]))
[baseline_loss, baseline_acc] = transpose(read_file(storage_folder+baseline_files[1]))

print(main_acc)

plt.subplot(2, 2, 2)
plt.title('loss, D=30s')
plt.plot(x, main_loss[0: num_iters], label='DAB_MEL')
plt.plot(x, baseline_loss[0: num_iters], label='baseline')
plt.legend()

plt.subplot(2, 2, 4)
plt.title('accuracy, D=30s')
plt.plot(x, main_acc[0: num_iters], label='DAB_MEL')
plt.plot(x, baseline_acc[0: num_iters], label='baseline')
plt.legend()

plt.tight_layout()
plt.show()