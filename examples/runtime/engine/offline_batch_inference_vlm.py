"""
Usage:
python offline_batch_inference_vlm.py --model-path Qwen/Qwen2-VL-7B-Instruct --chat-template=qwen2-vl
"""

import argparse
import dataclasses

from transformers import AutoProcessor

import sglang as sgl
from sglang.srt.openai_api.adapter import v1_chat_generate_request
from sglang.srt.openai_api.protocol import ChatCompletionRequest
from sglang.srt.server_args import ServerArgs


def main(
    server_args: ServerArgs,
):
    # Create an LLM.
    vlm = sgl.Engine(**dataclasses.asdict(server_args))

    # prepare prompts.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true",
                    },
                },
            ],
        }
    ]
    chat_request = ChatCompletionRequest(
        messages=messages,
        model=server_args.model_path,
        temperature=0.8,
        top_p=0.95,
    )
    gen_request, _ = v1_chat_generate_request(
        [chat_request],
        vlm.tokenizer_manager,
    )

    outputs = vlm.generate(
        input_ids=gen_request.input_ids,
        image_data=gen_request.image_data,
        sampling_params=gen_request.sampling_params,
    )

    print("===============================")
    print(f"Prompt: {messages[0]['content'][0]['text']}")
    print(f"Generated text: {outputs['text']}")


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    main(server_args)
