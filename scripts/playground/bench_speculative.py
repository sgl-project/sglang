"""
Usage:
# single GPU
python3 bench_speculative.py --model-path meta-llama/Llama-2-7b-chat-hf --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B

# multiple GPU
python3 bench_speculative.py --model-path deepseek-ai/DeepSeek-V3 --speculative-draft-model-path lmsys/DeepSeek-V3-NextN --tp-size 8 --trust-remote-code --batch-size 1 4 8 16 32 --steps 0 1 2 --topk 0 1 2 4 --num_draft_tokens 0 2 4 8
"""

import argparse
import asyncio
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import requests
from transformers import AutoTokenizer

from sglang.bench_serving import (
    DatasetRow,
    benchmark,
    sample_mmmu_requests,
    set_global_args,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    kill_process_tree,
    popen_launch_server,
)


def node0_print(msg):
    if server_args.node_rank == 0:
        print(msg)


prompts = [
    "Human: Give me a fully functional FastAPI server. Show the full, long python code without stop.\n\nAssistant:",
    "Human: Imagine you are an experienced Ethereum developer tasked with creating a smart contract for a blockchain messenger. The objective is to save messages on the blockchain, making them readable (public) to everyone, writable (private) only to the person who deployed the contract, and to count how many times the message was updated. Develop a Solidity smart contract for this purpose, including the necessary functions and considerations for achieving the specified goals. Please provide the code and any relevant explanations to ensure a clear understanding of the implementation.\n\nAssistant:",
    "Human: Write a travel blog post to Hawaii.\n\nAssistant:",
    "Human: I want you to act as an English translator, spelling corrector and improver. I will speak to you in any language and you will detect the language, translate it and answer in the corrected and improved version of my text, in English. I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level English words and sentences. Keep the meaning same, but make them more literary. My first sentence is 'istanbulu cok seviyom burada olmak cok guzel'. Answer in more than 5000 words.\n\nAssistant:",
    "Human: I want you to act as a storyteller. You will come up with entertaining stories that are engaging, imaginative and captivating for the audience. It can be fairy tales, educational stories or any other type of stories which has the potential to capture people's attention and imagination. Depending on the target audience, you may choose specific themes or topics for your storytelling session e.g., if it’s children then you can talk about animals; If it’s adults then history-based tales might engage them better etc. Answer in more than 5000 words. My first request is 'I need an interesting story on perseverance.'\n\nAssistant:",
    "Human: Solve x^2 = -1. Think step-by-step. Give me a long detailed explanation. \n\nAssistant:",
    "Human: Tell me about the president of the USA in wikipedia style.\n\nAssistant:",
    "Human: Hello? Who are you? Write code, math, and poem to explanin yourself.\n\nAssistant:",
]


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return []


def send_one_batch(base_url, num_prompts, batch_size, tokenizer, is_multimodal):
    # format: (prompt, input_len, output len). We set input_len as a dummy value 0.
    if is_multimodal:
        backend = "sglang-oai-chat"
        api_url = f"{base_url}/v1/chat/completions"
        input_requests = sample_mmmu_requests(
            num_prompts,
            tokenizer,
            backend=backend,
            fixed_output_len=512,
        )
    else:
        padded_prompts = (prompts * ((num_prompts + len(prompts) - 1) // len(prompts)))[
            :num_prompts
        ]
        input_requests: List[DatasetRow] = [
            DatasetRow(p, 0, 512) for p in padded_prompts
        ]
        backend = "sglang"
        api_url = f"{base_url}/generate"

    # We need to set some dummy values in order to call `benchmark` below.
    args = SimpleNamespace(
        disable_ignore_eos=False,
        disable_stream=False,
        return_logprob=False,
        backend=backend,
        dataset_name="custom",
        num_prompts=None,
        sharegpt_output_len=None,
        random_input_len=None,
        random_output_len=None,
        random_range_ratio=None,
        output_file=None,
        warmup_requests=1,
        output_details=False,
    )
    set_global_args(args)

    # Run benchmark
    results = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id="default",
            tokenizer=tokenizer,
            input_requests=input_requests,
            request_rate=float("inf"),
            max_concurrency=batch_size,
            disable_tqdm=False,
            lora_names=None,
            extra_request_body={},
            profile=None,
        )
    )

    assert results["completed"] == len(input_requests)
    acc_length = results["accept_length"] or 1.0
    avg_output_token = results["total_output_tokens"] / results["completed"]

    server_info = requests.get(base_url + "/get_server_info").json()
    # We use 20% percentile instead of median on purpose
    step_time = np.percentile(
        server_info["internal_states"][0]["step_time_dict"][str(batch_size)], 20
    )
    speed = 1 / step_time * acc_length

    return (
        round(acc_length, 3),
        round(step_time, 5),
        round(speed, 3),
        avg_output_token,
    )


def main(args, server_args):
    base_url = "http://127.0.0.1:20000"

    configs = []
    for batch_size in args.batch_size:
        for steps in args.steps:
            for topk in args.topk:
                for num_draft_tokens in args.num_draft_tokens:
                    if steps * topk + 1 < num_draft_tokens:
                        continue

                    if (steps == 0 or topk == 0 or num_draft_tokens == 0) and (
                        steps + topk + num_draft_tokens != 0
                    ):
                        # steps == 0 and topk == 0 and num_draft_tokens == 0 is a special case for non-speculative decoding.
                        continue

                    configs.append((batch_size, steps, topk, num_draft_tokens))

    for i in range(args.start, args.end or len(configs)):
        batch_size, steps, topk, num_draft_tokens = configs[i]

        node0_print(
            f"Start {i=}: {batch_size=}, {steps=}, {topk=}, {num_draft_tokens=}"
        )

        # Create an LLM.
        if steps == 0:
            other_args = []
        else:
            other_args = [
                "--speculative-num-steps",
                steps,
                "--speculative-eagle-topk",
                topk,
                "--speculative-num-draft-tokens",
                num_draft_tokens,
            ]
            if server_args.speculative_draft_model_path is not None:
                other_args.extend(
                    [
                        "--speculative-draft-model-path",
                        server_args.speculative_draft_model_path,
                        "--speculative-algorithm",
                        server_args.speculative_algorithm,
                    ]
                )

        other_args.extend(
            [
                "--cuda-graph-max-bs",
                batch_size,
                "--mem-fraction-static",
                server_args.mem_fraction_static,
                "--tp-size",
                server_args.tp_size,
                "--max-running-requests",
                batch_size,
            ]
        )

        if server_args.trust_remote_code:
            other_args.extend(
                [
                    "--trust-remote-code",
                ]
            )

        if server_args.attention_backend:
            other_args.extend(
                [
                    "--attention-backend",
                    server_args.attention_backend,
                ]
            )

        if server_args.quantization:
            other_args.extend(
                [
                    "--quantization",
                    server_args.quantization,
                ]
            )

        process = popen_launch_server(
            args.model_path,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env={
                "SGLANG_RECORD_STEP_TIME": "1",
                **os.environ,
            },
        )

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=server_args.trust_remote_code
        )

        try:
            # Warmup
            send_one_batch(
                base_url, batch_size, batch_size, tokenizer, args.is_multimodal
            )

            # Benchmark
            acc_length, step_time, speed, completion_tokens = send_one_batch(
                base_url,
                max(args.num_prompts, batch_size),
                batch_size,
                tokenizer,
                args.is_multimodal,
            )
        finally:
            kill_process_tree(process.pid)

        node0_print(
            f"Finish {i=}: {batch_size=}, {steps=}, {topk=}, {num_draft_tokens=}, {speed=:.2f} token/s, step_time={step_time * 1000:.2f} ms"
        )

        record = {
            "batch_size": batch_size,
            "steps": steps,
            "topk": topk,
            "num_draft_tokens": num_draft_tokens,
            "acc_length": acc_length,
            "step_time": step_time,
            "speed": speed,
            "completion_tokens": completion_tokens,
        }

        with open(args.output, "a") as fout:
            fout.write(json.dumps(record) + "\n")

        # Wait for the server to shutdown
        time.sleep(5)


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8, 16),
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=(0, 1, 3, 5, 7),  # use (0, 1, 2, 3, 4) for large batch size
    )
    parser.add_argument(
        "--topk",
        type=int,
        nargs="+",
        default=(0, 1, 2, 4, 8),
    )
    parser.add_argument(
        "--num_draft_tokens",
        type=int,
        nargs="+",
        default=(0, 2, 4, 8, 16, 32),  # use (0, 2, 4, 8) for large batch size
    )
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--output", type=str, default="output.jsonl")
    parser.add_argument("--is-multimodal", action="store_true", default=False)
    args = parser.parse_args()
    server_args: ServerArgs = ServerArgs.from_cli_args(args)

    main(args, server_args)
