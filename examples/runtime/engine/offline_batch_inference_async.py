"""
Usage:
python offline_batch_inference_async.py --model-path Qwen/Qwen2-VL-7B-Instruct

Note:
This demo shows the usage of async generation,
which is useful to implement an online-like generation with batched inference.
"""

import argparse
import asyncio
import dataclasses
import time

import sglang as sgl
from sglang.srt.server_args import ServerArgs


class InferenceEngine:
    def __init__(self, **kwargs):
        self.engine = sgl.Engine(**kwargs)

    async def generate(self, prompt, sampling_params):
        result = await self.engine.async_generate(prompt, sampling_params)
        return result


async def run_server(server_args):
    inference = InferenceEngine(**dataclasses.asdict(server_args))

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ] * 100

    # Create a sampling params object.
    sampling_params = {"temperature": 0.8, "top_p": 0.95}

    # Run the generation tasks concurrently in async mode.
    tasks = []
    for prompt in prompts:
        task = asyncio.create_task(inference.generate(prompt, sampling_params))
        tasks.append(task)

    # Get and print the result
    for task in tasks:
        await task
        while True:
            if not task.done():
                time.sleep(1)
            else:
                result = task.result()
                print(f"Generated text: {result['text']}")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    asyncio.run(run_server(server_args))
