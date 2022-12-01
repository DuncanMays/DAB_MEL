from matplotlib import pyplot as plt
import numpy as np

# Device = ['2070 super', 'Jetson Nano', 'Pi1', 'Pi2', 'Pi3']
# # task size = [120, 120, 15, 15, 15 ]
FLOPS = [17, 6, 10, 11, 13 ]
SB1 = [2.5, 0.1, 0.2, 2.4, 3.9]
SB2 = [2.6, 0.7, 0.1, 1.7, 4.3]
SB4 = [2.1, 1.3, 0.3, 0.1, 4.3]
SB8 = [0.1, 0.4, 0.3, 0.5, 2.1]

# SB2 = [0.2, 1.4, 1.0, 0.6, 0.2]
# SB4 = [0.6, 2.2, 0.5, 0.0, 0.3]
# SB6 = [0.3, 1.9, 0.2, 0.3, 0.0]
# SB8 = [0.3, 2.6, 0.1, 0.3, 0.1]

# x = np.arange(len(Device))

# plt.bar(x-0.3, FLOPS, width=0.15, label='FLOPS')
# plt.bar(x-0.15, SB1, width=0.15, label='SB1')
# plt.bar(x, SB2, width=0.15, label='SB2')
# plt.bar(x+0.15, SB4, width=0.15, label='SB4')
# plt.bar(x+0.3, SB8, width=0.15, label='SB8')

# plt.legend()
# plt.xticks(ticks=x, labels=Device)

# plt.ylabel('error percentage')
# plt.xlabel('Device')

# plt.show()

font_size=15

flops = sum(FLOPS)/len(FLOPS)
SB_2 = sum(SB1)/len(SB1)
SB_4 = sum(SB2)/len(SB2)
SB_6 = sum(SB4)/len(SB4)
SB_8 = sum(SB8)/len(SB8)

data = [SB_2, SB_4, SB_6, SB_8]
flops_data = [flops for d in data]

x = [2, 4, 6, 8]
y_ticks = [i for i in range(0, 15, 2)]

plt.plot(x, data, color='red', label='SB')
plt.plot(x, flops_data, color='green', label='FLOPS')

plt.xticks(ticks=x, size=font_size)
plt.yticks(ticks=y_ticks, size=font_size)
plt.ylabel('Average Runtime Benchmarking Error (%)', size=font_size)
plt.xlabel('Benchmark Size (data shards)', size=font_size)

plt.legend(prop={'size': font_size})
plt.show()