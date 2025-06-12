"""
Usage:

python hidden_states_server.py

Note that each time you change the `return_hidden_states` parameter,
the cuda graph will be recaptured, which might lead to a performance hit.
So avoid getting hidden states and completions alternately.
"""

import requests
import torch

from sglang.test.test_utils import is_in_ci
from sglang.utils import terminate_process, wait_for_server

if is_in_ci():
    from docs.backend.patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd


def main():
    # Launch the server
    server_process, port = launch_server_cmd(
        "python -m sglang.launch_server --model-path Alibaba-NLP/gte-Qwen2-1.5B-instruct --enable-return-hidden-states --host 0.0.0.0"
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

    terminate_process(server_process)

    outputs = response.json()
    for prompt, output in zip(prompts, outputs):
        for i in range(len(output["meta_info"]["hidden_states"])):
            output["meta_info"]["hidden_states"][i] = torch.tensor(
                output["meta_info"]["hidden_states"][i], dtype=torch.bfloat16
            )
        print("===============================")
        print(
            f"Prompt: {prompt}\n"
            f"Generated text: {output['text']}\n"
            f"Prompt_Tokens: {output['meta_info']['prompt_tokens']}\t"
            f"Completion_tokens: {output['meta_info']['completion_tokens']}"
        )
        print("Hidden states: ")
        hidden_states = torch.cat(
            [
                i.unsqueeze(0) if len(i.shape) == 1 else i
                for i in output["meta_info"]["hidden_states"]
            ]
        )
        print(hidden_states)
        print()


if __name__ == "__main__":
    main()
