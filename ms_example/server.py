import os

import requests

from sglang.utils import launch_server_cmd, print_highlight, wait_for_server

current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["MF_MODEL_CONFIG"] = os.path.join(
    current_dir, "predict_qwen2_5_7b_instruct_800l_A2.yaml"
)

server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server \
        --model-path /home/ckpt/Qwen3-8B \
        --host 0.0.0.0 \
        --device npu \
        --model-impl mindspore \
        --max-total-tokens=20000 \
        --attention-backend ascend \
        --mem-fraction-static 0.8 \
        --tp-size 1 \
        --dp-size 1",
    port=37654,
)

wait_for_server(f"http://localhost:{port}")

url = f"http://localhost:{port}/generate"
data = {"text": "What is the capital of France?"}

response = requests.post(url, json=data)
print_highlight(response.json())
