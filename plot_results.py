from matplotlib import pyplot as plt
import json

storage_folder = './TwoNN_results/'
# storage_folder = './ConvNet_results/'

main_files = ['main_k2_T15.json', 'main_k4_T15.json']
baseline_files = ['baseline_k2_T15.json', 'baseline_k4_T15.json']

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

num_iters = len(main_loss)
acc_yticks = [i/10 for i in range(11)]
acc_ylabels = [str(10*i) for i in range(11)]
xticks = [i for i in range(1, num_iters)]
x = [i for i in range(1, num_iters)]
folder = './results/'
font_size=20

plt.title('big title')

# plt.subplot(2, 2, 1)
# plt.title('loss, D=15s')
plt.figure()
plt.plot(x, main_loss[1: num_iters], label='DAB')
plt.plot(x, baseline_loss[1: num_iters], label='CSA')
plt.xlabel('Global Update Index', size=font_size)
plt.ylabel('Loss', size=font_size)
plt.legend(prop={'size': font_size})
plt.xticks(xticks)
plt.savefig(folder+'loss_k2.png')

# plt.subplot(2, 2, 3)
# plt.title('accuracy, D=15s')
plt.figure()
plt.plot(x, main_acc[1: num_iters], label='DAB')
plt.plot(x, baseline_acc[1: num_iters], label='CSA')
plt.xlabel('Global Update Index', size=font_size)
plt.ylabel('Accuracy (%)', size=font_size)
plt.legend(prop={'size': font_size})
plt.yticks(acc_yticks, acc_ylabels)
plt.xticks(xticks)
plt.savefig(folder+'acc_k2.png')

[main_loss, main_acc] = transpose(read_file(storage_folder+main_files[1]))
[baseline_loss, baseline_acc] = transpose(read_file(storage_folder+baseline_files[1]))

# plt.subplot(2, 2, 2)
# plt.title('loss, D=30s')
plt.figure()
plt.plot(x, main_loss[1: num_iters], label='DAB')
plt.plot(x, baseline_loss[1: num_iters], label='CSA')
plt.xlabel('Global Update Index', size=font_size)
plt.ylabel('Loss', size=font_size)
plt.legend(prop={'size': font_size})
plt.xticks(xticks)
plt.savefig(folder+'loss_k4.png')

# plt.subplot(2, 2, 4)
# plt.title('accuracy, D=30s')
plt.figure()
plt.plot(x, main_acc[1: num_iters], label='DAB')
plt.plot(x, baseline_acc[1: num_iters], label='CSA')
plt.xlabel('Global Update Index', size=font_size)
plt.ylabel('Accuracy (%)', size=font_size)
plt.legend(prop={'size': font_size})
plt.yticks(acc_yticks, acc_ylabels)
plt.xticks(xticks)
plt.savefig(folder+'acc_k4.png')

# plt.tight_layout()
# plt.show()