from types import SimpleNamespace

primative = {
	'client_batch_size': 10,
	'client_learning_rate': 0.1,
	'client_num_epochs': 1,
	'data_server_ip': '192.168.2.19',
	'data_server_port': 5003,
	'learner_port': 5001,
	'notice_board_ip': '192.168.2.19',
	'notice_board_port': 5002,
	'orchestrator_port': 5000
}

config_object = SimpleNamespace(**primative)
