from types import SimpleNamespace

primative = {
	'data_server_port': 5003,
	'learner_port': 5001,
	'notice_board_port': 5002,
	'notice_board_ip': '192.168.2.19',
	'orchestrator_port': 5000
}

config_object = SimpleNamespace(**primative)
