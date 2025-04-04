from sglang.test.test_utils import is_in_ci
from sglang.utils import print_highlight, wait_for_server

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

# This is equivalent to running the following command in your terminal

# python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0

server_process, port = launch_server_cmd(
    """
python -m sglang.launch_server --model-path nvidia/Llama-3_3-Nemotron-Super-49B-v1  --tp 2 \
 --host 0.0.0.0 --trust-remote-code
"""
)

wait_for_server(f"http://localhost:{port}")
import json
import subprocess

curl_command = f"""
curl -s http://localhost:{port}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{{"model": "nvidia/Llama-3_3-Nemotron-Super-49B-v1", "messages": [{{"role": "user", "content": "What is the capital of France?"}}]}}'
"""

response = json.loads(subprocess.check_output(curl_command, shell=True))
print_highlight(response)
