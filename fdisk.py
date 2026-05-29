# successful update with same architecture and size
import requests 
port=6990
url = f"http://localhost:{port}/update_weights_from_disk"
# data = {"model_path": "/home/weights/Qwen3-30B-A3B"}
data = {"model_path": "/home/wzy/qwen3-30b-a3b"}

response = requests.post(url, json=data)
print(response.text)
assert response.json()["success"] is True
assert response.json()["message"] == "Succeeded to update model weights."
