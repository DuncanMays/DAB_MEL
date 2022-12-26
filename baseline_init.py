import asyncio
import aiohttp
from config import config_object
from utils import get_active_worker_ips

async def async_GET(url):
	async with aiohttp.ClientSession() as session:
		async with session.get(url) as resp:
			return resp.status, await resp.text()

worker_ips = get_active_worker_ips()

async def init_worker(w_ip):
	url = 'http://'+w_ip+':'+str(config_object.learner_port)+'/basaeline_init_procedure'
	return await async_GET(url)

async def baseline_init():
	init_promises = []

	for w_ip in worker_ips:
		init_promises.append(init_worker(w_ip))

	await asyncio.gather(*init_promises)

if (__name__ == '__main__'):
	asyncio.run(baseline_init())
