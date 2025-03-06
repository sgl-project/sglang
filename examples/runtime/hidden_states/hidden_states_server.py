"""
Usage:

python hidden_states_server.py

Note that each time you change the `return_hidden_states` parameter,
the cuda graph will be recaptured, which might lead to a performance hit.
So avoid getting hidden states and completions alternately.
"""

import requests

from sglang.test.test_utils import is_in_ci
from sglang.utils import print_highlight, terminate_process, wait_for_server

if is_in_ci():
    from docs.backend.patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd


def main():
    # Launch the server
    server_process, port = launch_server_cmd(
        "python -m sglang.launch_server --model-path Alibaba-NLP/gte-Qwen2-1.5B-instruct --host 0.0.0.0"
    )
    wait_for_server(f"http://localhost:{port}")

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 10,
    }

    json_data = {
        "text": prompts,
        "sampling_params": sampling_params,
        "return_hidden_states": True,
    }

    response = requests.post(
        f"http://localhost:{port}/generate",
        json=json_data,
    )

    outputs = response.json()
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(
            f"Prompt: {prompt}\n"
            f"Generated text: {output['text']}\n"
            f"Prompt_Tokens: {output['meta_info']['prompt_tokens']}\t"
            f"Completion_tokens: {output['meta_info']['completion_tokens']}\n"
            f"Hidden states: {output['meta_info']['hidden_states']}"
        )
        print()

    terminate_process(server_process)


if __name__ == "__main__":
    main()
