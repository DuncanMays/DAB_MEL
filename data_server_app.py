from config import config_object
from flask import Flask
import json

app = Flask(__name__)

# TODO load real data
data_shard = [i for i in range(10_000)]

# TODO change to POST that serves a variable number of shards
@app.route("/get_data", methods=['GET'])
def get_data():
	print('request received at get_data')
	return json.dumps(data_shard)

print('starting app')
app.run(host='0.0.0.0', port=config_object.data_server_port)