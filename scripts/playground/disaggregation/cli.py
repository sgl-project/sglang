prompt = [0] * 431

import json

import requests

response = requests.post(
    "http://0.0.0.0:8000/generate",
    json={"input_ids": [prompt] * 32, "sampling_params": {"temperature": 0}},
)


# print("Response content (raw):", response.content)
