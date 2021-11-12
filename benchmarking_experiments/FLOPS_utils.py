import torch
import time

device = 'cpu'

def FLOPS_benchmark():

	n = 512
	i = 50

	M1 = torch.randn([n, n], dtype=torch.float32).to(device)
	M2 = torch.randn([n, n], dtype=torch.float32).to(device)

	start = time.time()

	for j in range(i):
		M = M1*M2

	end = time.time()
	ops = n**3 + (n-1)*n**2

	return ops/(end - start)