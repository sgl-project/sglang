# launch server
# python -m sglang.launch_server --model mistralai/Mistral-7B-Instruct-v0.3 --lora-paths /home/ying/test_lora lora1=/home/ying/test_lora_1 lora2=/home/ying/test_lora_2 --disable-radix --disable-cuda-graph --max-loras-per-batch 4

# send requests
# lora_path[i] specifies the LoRA used for text[i], so make sure they have the same length
# use None to specify base-only prompt, e.x. "lora_path": [None, "/home/ying/test_lora"]
import json

import requests

url = "http://127.0.0.1:30000"
json_data = {
    "text": [
        "prompt 1",
        "prompt 2",
        "prompt 3",
        "prompt 4",
        "prompt 5",
        "prompt 6",
        "prompt 7",
    ],
    "sampling_params": {"max_new_tokens": 32},
    "lora_path": [
        "/home/ying/test_lora",
        "lora1",
        "lora2",
        "lora1",
        "lora2",
        None,
        None,
    ],
}
response = requests.post(
    url + "/generate",
    json=json_data,
)
print(json.dumps(response.json()))
