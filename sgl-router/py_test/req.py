import requests

base_url = "http://127.0.0.1:2157"

response = requests.post(f"{base_url}/generate", json={"text": "Kanye west is, ", "temperature": 0}, headers={"Authorization": "Bearer correct_api_key"})
print(f"status code: {response.status_code}, response: {response.text}")