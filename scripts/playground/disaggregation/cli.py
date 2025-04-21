prompt = "Hello " * 16000

import json

import requests

response = requests.post(
    "http://0.0.0.0:8000/generate",
    json={"text": prompt, "sampling_params": {"temperature": 0}},
)


print("Response content (raw):", response.content)
