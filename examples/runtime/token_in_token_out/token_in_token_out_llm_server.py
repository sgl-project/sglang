"""
Usage:

python token_in_token_out_llm_server.py

"""

import requests

from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import is_in_ci
from sglang.utils import terminate_process, wait_for_server

if is_in_ci():
    from docs.backend.patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd


MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"


def main():
    # Launch the server
    server_process, port = launch_server_cmd(
        f"python -m sglang.launch_server --model-path {MODEL_PATH} --skip-tokenizer-init --host 0.0.0.0"
    )
    wait_for_server(f"http://localhost:{port}")

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params object.
    sampling_params = {"temperature": 0.8, "top_p": 0.95}

    # Tokenize inputs
    tokenizer = get_tokenizer(MODEL_PATH)
    token_ids_list = [tokenizer.encode(prompt) for prompt in prompts]

    json_data = {
        "input_ids": token_ids_list,
        "sampling_params": sampling_params,
    }

    response = requests.post(
        f"http://localhost:{port}/generate",
        json=json_data,
    )

    outputs = response.json()
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        decode_output = tokenizer.decode(output["output_ids"])
        print(
            f"Prompt: {prompt}\nGenerated token ids: {output['output_ids']}\nGenerated text: {decode_output}"
        )
        print()

    terminate_process(server_process)


if __name__ == "__main__":
    main()
