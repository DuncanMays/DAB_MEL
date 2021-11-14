from types import SimpleNamespace
from networks import TwoNN, ThreeNN, ConvNet
import torch

def set_training_device():
	if torch.cuda.is_available():
		return 'cuda:0'
	else:
		return 'cpu'

primative = {
	'client_learning_rate': 0.1,
	'client_num_updates': 5,
	'client_training_time': 30,
	'data_server_ip': '192.168.2.19',
	'data_server_port': 5003,
	'init_config_file': './init_config.json',
	'learner_port': 5001,
	'model_class': TwoNN,
	'notice_board_ip': '192.168.2.19',
	'notice_board_port': 5002,
	'orchestrator_port': 5000,
	'parameter_server_ip': '192.168.2.19',
	'parameter_server_port': 5000,
	'sample_shape': [784],
	'training_device': set_training_device()
}

config_object = SimpleNamespace(**primative)