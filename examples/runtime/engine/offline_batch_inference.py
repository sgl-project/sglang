"""
Usage:
python3 offline_batch_inference.py  --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import dataclasses

import sglang as sgl
from sglang.srt.server_args import ServerArgs


def main(
    server_args: ServerArgs,
):
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = {"temperature": 0.8, "top_p": 0.95}

    # Create an LLM.
    llm = sgl.Engine(**dataclasses.asdict(server_args))

    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    main(server_args)
