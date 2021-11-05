import time

def client_update():
	print('client_update has been called')

	# waits two seconds
	time.sleep(2)

	print('process complete')
	return {'payload': 'hello there!'}
