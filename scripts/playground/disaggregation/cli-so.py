import json

import requests

port = 8000

json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

# JSON
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": "Here is the information of the capital of France in the JSON format.\n",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "json_schema": json_schema,
        },
    },
)

print(response.json())


# python3 -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --trust-remote-code --disaggregation-mode prefill --tp 2 --disaggregation-ib-device mlx5_roce0,mlx5_roce1 --speculative-algorithm EAGLE --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B --speculative-num-steps 3 --speculative-eagle-topk 4 --speculative-num-draft-tokens 16 --cuda-graph-max-bs 8 --host 127.0.0.1 --port 8100
