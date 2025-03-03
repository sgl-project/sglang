import argparse
import dataclasses
from io import BytesIO
from typing import Tuple

import requests
from PIL import Image
from transformers import AutoProcessor

from sglang import Engine
from sglang.lang.chat_template import get_chat_template_by_model_path
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import DEFAULT_IMAGE_URL


def get_input_ids(
    server_args: ServerArgs, model_config: ModelConfig
) -> Tuple[list[int], list]:
    chat_template = get_chat_template_by_model_path(model_config.model_path)
    text = f"{chat_template.image_token}What is in this picture?"
    images = [Image.open(BytesIO(requests.get(DEFAULT_IMAGE_URL).content))]
    image_data = [DEFAULT_IMAGE_URL]

    processor = AutoProcessor.from_pretrained(
        model_config.model_path, trust_remote_code=server_args.trust_remote_code
    )

    inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt",
    )

    return inputs.input_ids[0].tolist(), image_data


def token_in_out_example(
    server_args: ServerArgs,
):
    input_ids, image_data = get_input_ids(
        server_args,
        ModelConfig(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            model_override_args=server_args.json_model_override_args,
        ),
    )
    backend = Engine(**dataclasses.asdict(server_args))

    output = backend.generate(
        input_ids=input_ids,
        image_data=image_data,
        sampling_params={
            "temperature": 0.8,
            "max_new_tokens": 32,
        },
    )

    print("===============================")
    print(f"Output token ids: ", output["output_ids"])

    backend.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = [
        "--model-path=Qwen/Qwen2-VL-2B",
    ]
    args = parser.parse_args(args=args)
    server_args = ServerArgs.from_cli_args(args)
    server_args.skip_tokenizer_init = True
    token_in_out_example(server_args)
