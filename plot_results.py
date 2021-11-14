from matplotlib import pyplot as plt
import json

main_file = 'main_15s_5iter.json'
baseline_file = 'baseline_15s.json'

f = open(main_file, 'r')
main_data = json.loads(f.readline())
f.close()

f = open(baseline_file, 'r')
baseline_data = json.loads(f.readline())
f.close()

main_acc = []
main_loss = []
for d in main_data:
	main_acc.append(d[0])
	main_loss.append(d[1])

baseline_acc = []
baseline_loss = []
for d in main_data:
	baseline_acc.append(d[0])
	baseline_loss.append(d[1])

x = [i for i in range(len(main_data))]

plt.plot(x, main_acc, label='main_acc')
plt.plot(x, main_loss, label='main_loss')

plt.plot(x, baseline_acc, label='baseline_acc')
plt.plot(x, baseline_loss, label='baseline_loss')

plt.legend()
plt.show()