import argparse
import random
import time

import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--backend", type=str, default="srt")
    args = parser.parse_args()

    if args.port is None:
        if args.backend == "srt":
            args.port = 30000
        elif args.backend == "vllm":
            args.port = 21000
        elif args.backend == "lightllm":
            args.port = 22000
        elif args.backend == "xinfer":
            args.port = 9988
        else:
            raise ValueError(f"Invalid backend: {args.backend}")

    url = f"{args.host}:{args.port}"
    a = random.randint(0, 1 << 20)
    max_new_tokens = 256
    prompt = f"{a, }"

    tic = time.time()
    if args.backend == "srt":
        response = requests.post(
            url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
        )
    elif args.backend == "lightllm":
        response = requests.post(
            url + "/generate",
            json={
                "inputs": prompt,
                "parameters": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
        )
    elif args.backend == "vllm":
        response = requests.post(
            url + "/generate",
            json={
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": max_new_tokens,
            },
        )
    elif args.backend == "xinfer":
        import grpc
        from xlm.proto import sampler_pb2, sampler_pb2_grpc

        sampler_channel = grpc.insecure_channel(url.replace("http://", ""))
        sampler = sampler_pb2_grpc.SamplerStub(sampler_channel)

        tic = time.time()
        sample_request = sampler_pb2.SampleTextRequest(
            prompt=prompt,
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

    speed = max_new_tokens / latency
    print(f"latency: {latency:.2f} s, speed: {speed:.2f} token/s")
