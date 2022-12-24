from config import config_object
import requests

url = 'http://'+str(config_object.parameter_server_ip)+':'+str(config_object.orchestrator_port)+'/start_cycle'
x = requests.get(url)
print(x.content)