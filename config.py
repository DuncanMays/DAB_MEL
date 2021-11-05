from types import SimpleNamespace

primative = {
	'orchestrator_port': 5000,
	'learner_port': 5001,
	'notice_board_port': 5002,
	'notice_board_ip': '127.0.0.1'
}

config_object = SimpleNamespace(**primative)
