import json
import random

import requests

resp = requests.post('http://127.0.0.1:8000', json.dumps({'input': [random.uniform(0, 1)] * 784}))
print(resp.content.decode('utf-8'))
