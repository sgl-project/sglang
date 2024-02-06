"""
Usage:
python3 -m sglang.launch_server --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 --port 30000
python3 test_httpserver_decode_stream.py

Output:
The capital of France is Paris.\nThe capital of the United States is Washington, D.C.\nThe capital of Canada is Ottawa.\nThe capital of Japan is Tokyo
"""

import argparse
import json

import requests


def test_decode_stream(url, return_logprob):
    response = requests.post(
        url + "/generate",
        json={
            "text": "The capital of France is",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 128,
            },
            "stream": True,
            "return_logprob": return_logprob,
        },
        stream=True,
    )

    prev = 0
    for chunk in response.iter_lines(decode_unicode=False):
        chunk = chunk.decode("utf-8")
        if chunk and chunk.startswith("data:"):
            if chunk == "data: [DONE]":
                break
            data = json.loads(chunk[5:].strip("\n"))

            if return_logprob:
                assert data["meta_info"]["prompt_logprob"] is not None
                assert data["meta_info"]["token_logprob"] is not None
                assert data["meta_info"]["normalized_prompt_logprob"] is not None
                if prev == 0:  # Skip prompt logprobs
                    prev = data["meta_info"]["prompt_tokens"]
                for token_txt, _, logprob in data["meta_info"]["token_logprob"][prev:]:
                    print(f"{token_txt}\t{logprob}", flush=True)
                prev = len(data["meta_info"]["token_logprob"])
            else:
                output = data["text"].strip()
                print(output[prev:], end="", flush=True)
                prev = len(output)
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    url = f"{args.host}:{args.port}"

    test_decode_stream(url, False)
    test_decode_stream(url, True)
