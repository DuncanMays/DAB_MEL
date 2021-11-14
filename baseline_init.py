from config import config_object
from utils import start_global_cycle, get_active_worker_ips
import requests

print('discovering workers')
notice_board_resp = requests.get('http://'+str(config_object.notice_board_ip)+':'+str(config_object.notice_board_port)+'/notice_board')
worker_ips = get_active_worker_ips()

print('starting '+str(len(worker_ips))+' baseline init process')
for ip in worker_ips:
	url = 'http://'+ip+':'+str(config_object.learner_port)+'/baseline_init_start'
	requests.get(url)
