from config import config_object
from utils import start_global_cycle, get_active_worker_ips
import requests

print('discovering workers')
notice_board_resp = requests.get('http://'+str(config_object.notice_board_ip)+':'+str(config_object.notice_board_port)+'/notice_board')
worker_ips = get_active_worker_ips()

print('starting '+str(len(worker_ips))+' workers')
# sends a req to each learner to inform them to start learning
start_global_cycle(worker_ips)
