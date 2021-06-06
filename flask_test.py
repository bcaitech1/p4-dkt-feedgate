import json
import requests
import numpy as np


headers = {'Content-Type':'application/json'}
address = "http://127.0.0.1:2431/inference"
data = {'images':'test'}

result = requests.post(address, data=json.dumps(data), headers=headers)

json_result = result.json()
print(json_result[0])
# print(str(result.content, encoding='utf-8'))
