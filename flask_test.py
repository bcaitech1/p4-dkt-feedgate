import json
import requests
import numpy as np


headers = {'Content-Type':'application/json'}
address = "http://127.0.0.1:2431/get_score"
data = [{'q_content': '사람', 'tag': 7597, 'test_id': 'A010000060', 'assess_id': 'A010060001', 'answer': '1'}, {'q_content': '동물', 'tag': 397, 'test_id': 'A050000094', 'assess_id': 'A050094005', 'answer': '0'}, {'q_content': '자연', 'tag': 451, 'test_id': 'A050000155', 'assess_id': 'A050155004', 'answer': '0'}, {'q_content': '물건', 'tag': 587, 'test_id': 'A060000017', 'assess_id': 'A060017006', 'answer': '0'}, {'q_content': '실제', 'tag': 4803, 'test_id': 'A080000031', 'assess_id': 'A080031001', 'answer': '1'}]
[['A010060001', 'A010000060', 7597, '1'], ['A050094005', 'A050000094', 397, '0'], ['A050155004', 'A050000155', 451, '0'], ['A060017006', 'A060000017', 587, '0'], ['A080031001', 'A080000031', 4803, '1']]

result = requests.post(address, data=json.dumps(data), headers=headers)

json_result = result.json()
print(json_result) # type : int
# print(str(result.content, encoding='utf-8'))
