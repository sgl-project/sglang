"""
Usage:
python3 bench_one.py --input-len 2048 --batch-size 1 2 4 8 16 32 64 128 256 512
"""

import argparse
import json
import time

import numpy as np
import requests


def run_one_batch_size(bs):
    url = f"{args.host}:{args.port}"
    max_new_tokens = args.max_tokens

    if args.input_len:
        input_ids = [
            [int(x) for x in np.random.randint(0, high=16384, size=(args.input_len,))]
            for _ in range(bs)
        ]
    else:
        text = [f"{i, }" for i in range(bs)]

    tic = time.time()
    if args.backend == "srt":
        if args.input_len:
            inputs = {"input_ids": input_ids}
        else:
            inputs = {"text": text}

        response = requests.post(
            url + "/generate",
            json={
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
                **inputs,
            },
        )
    elif args.backend == "lightllm":
        response = requests.post(
            url + "/generate",
            json={
                "inputs": text[0],
                "parameters": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
        )
    elif args.backend == "vllm":
        if args.input_len:
            inputs = {"prompt": input_ids}
        else:
            inputs = {"prompt": text}

        response = requests.post(
            url + "/v1/completions",
            json={
                "model": args.vllm_model_name,
                "temperature": 0,
                "max_tokens": max_new_tokens,
                "ignore_eos": True,
                **inputs,
            },
        )
    elif args.backend == "ginfer":
        import grpc
        from ginfer import sampler_pb2, sampler_pb2_grpc

        sampler_channel = grpc.insecure_channel(url.replace("http://", ""))
        sampler = sampler_pb2_grpc.SamplerStub(sampler_channel)

        tic = time.time()
        sample_request = sampler_pb2.SampleTextRequest(
            prompt=text[0],
            settings=sampler_pb2.SampleSettings(
                max_len=max_new_tokens,
                rng_seed=0,
                temperature=0,
                nucleus_p=1,
            ),
        )
        stream = sampler.SampleText(sample_request)
        response = "".join([x.text for x in stream])
    latency = time.time() - tic

    if isinstance(response, str):
        ret = response
    else:
        ret = response.json()
    print(ret)

    input_len = args.input_len if args.input_len else 1
    output_len = max_new_tokens

    output_throughput = bs * max_new_tokens / latency
    overall_throughput = bs * (input_len + output_len) / latency
    print(f"latency: {latency:.2f} s")
    print(f"output throughput: {output_throughput:.2f} token/s")
    print(f"(input + output) throughput: {overall_throughput:.2f} token/s")

    with open("results.jsonl", "a") as fout:
        res = {
            "backend": args.backend,
            "input_len": args.input_len,
            "output_len": args.max_tokens,
            "batch_size": bs,
            "latency": latency,
            "output_throughput": output_throughput,
            "overall_throughput": overall_throughput,
        }
        fout.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--backend", type=str, default="srt")
    parser.add_argument("--input-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, nargs="*", default=[1])
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--vllm-model-name", type=str, default="meta-llama/Meta-Llama-3-70B"
    )
    args = parser.parse_args()

    if args.port is None:
        if args.backend == "srt":
            args.port = 30000
        elif args.backend == "vllm":
            args.port = 21000
        elif args.backend == "lightllm":
            args.port = 22000
        elif args.backend == "ginfer":
            args.port = 9988
        else:
            raise ValueError(f"Invalid backend: {args.backend}")

    for bs in args.batch_size:
        run_one_batch_size(bs)
