import requests

url = "http://localhost:30000/generate"
data = {"text": "What is the capital of France?"}

response = requests.post(url, json=data)
print(response.json())
