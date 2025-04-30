"""
Usage:

python token_in_token_out_vlm_server.py

"""

from typing import Tuple

import requests
from transformers import AutoProcessor

from sglang.lang.chat_template import get_chat_template_by_model_path
from sglang.test.test_utils import DEFAULT_IMAGE_URL, is_in_ci
from sglang.utils import terminate_process, wait_for_server

if is_in_ci():
    from docs.backend.patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd


MODEL_PATH = "Qwen/Qwen2-VL-2B"


def get_input_ids() -> Tuple[list[int], list]:
    chat_template = get_chat_template_by_model_path(MODEL_PATH)
    text = f"{chat_template.image_token}What is in this picture?"
    image_data = [DEFAULT_IMAGE_URL]

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    input_ids = (
        processor.tokenizer(
            text=[text],
            return_tensors="pt",
        )
        .input_ids[0]
        .tolist()
    )

    return input_ids, image_data


def main():
    # Launch the server
    server_process, port = launch_server_cmd(
        f"python -m sglang.launch_server --model-path {MODEL_PATH} --skip-tokenizer-init --host 0.0.0.0"
    )
    wait_for_server(f"http://localhost:{port}")

    input_ids, image_data = get_input_ids()

    sampling_params = {
        "temperature": 0.8,
        "max_new_tokens": 32,
    }

    json_data = {
        "input_ids": input_ids,
        "image_data": image_data,
        "sampling_params": sampling_params,
    }

    response = requests.post(
        f"http://localhost:{port}/generate",
        json=json_data,
    )

    output = response.json()
    print("===============================")
    print(f"Output token ids: ", output["output_ids"])

    terminate_process(server_process)


if __name__ == "__main__":
    main()
