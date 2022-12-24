from config import config_object
import requests

print('killing orchestrator')
url = 'http://'+str(config_object.parameter_server_ip)+':'+str(config_object.orchestrator_port)+'/shutdown'
notice_board_resp = requests.get(url)
