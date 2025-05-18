"""
Usage:
python offline_batch_inference_vlm.py --model-path Qwen/Qwen2-VL-7B-Instruct
"""

import argparse
import dataclasses

import sglang as sgl
from sglang.srt.conversation import chat_templates
from sglang.srt.server_args import ServerArgs


def main(
    server_args: ServerArgs,
):
    vlm = sgl.Engine(**dataclasses.asdict(server_args))

    conv = chat_templates[server_args.chat_template].copy()
    image_token = conv.image_token

    image_url = "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"

    prompt = f"What's in this image?\n{image_token}"

    sampling_params = {
        "temperature": 0.001,
        "max_new_tokens": 30,
    }

    output = vlm.generate(
        prompt=prompt,
        image_data=image_url,
        sampling_params=sampling_params,
    )

    print("===============================")
    print(f"Prompt: {prompt}")
    print(f"Generated text: {output['text']}")

    vlm.shutdown()


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()

    server_args = ServerArgs.from_cli_args(args)
    main(server_args)
