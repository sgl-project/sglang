"""
Usage:
python3 offline_batch_inference.py  --model /mnt/cephfs/kavioyu/models/Qwen3-8B
"""

import argparse
import dataclasses

import sglang as sgl
from sglang.srt.server_args import ServerArgs


def main(
    server_args: ServerArgs,
):
    # Sample prompts.
    text = open('prompt.txt', 'r').read()
    
    prompts = [
        text + '请直接回答：张无忌会什么武功？', 
        text + '请直接回答：张无忌真正钟情的是谁？'
    ]
    # Create a sampling params object.
    #sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 2048}
    sampling_params = {"top_k": 1, "max_new_tokens": 1280}
    server_args.disable_cuda_graph = False
    server_args.is_sparse_attn = False
    server_args.cuda_graph_bs = [1, 2]
    server_args.page_size = 64

    # Create an LLM.
    llm = sgl.Engine(**dataclasses.asdict(server_args))

    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        #print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
        print(output['text'])


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    main(server_args)
