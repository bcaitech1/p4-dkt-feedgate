# import requests

# response = requests.post("http://127.0.0.1:5000/inference", json=[[5.1, 3.5, 1.4, 0.2]])
# print(response.text)


import requests

response = requests.post("http://127.0.0.1:5000/inference", json=[[5.1, 3.5, 1.4, 0.2]])

print(response.text)
