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


def test_decode_stream(url, return_logprob, top_logprobs_num):
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
            "top_logprobs_num": top_logprobs_num,
            "return_text_in_logprobs": True,
            "logprob_start_len": 0,
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
                assert data["meta_info"]["prefill_token_logprobs"] is not None
                assert data["meta_info"]["decode_token_logprobs"] is not None
                assert data["meta_info"]["normalized_prompt_logprob"] is not None
                for logprob, token_id, token_text in data["meta_info"][
                    "decode_token_logprobs"
                ][prev:]:
                    print(f"{token_text:12s}\t{logprob}\t{token_id}", flush=True)
                prev = len(data["meta_info"]["decode_token_logprobs"])
            else:
                output = data["text"].strip()
                print(output[prev:], end="", flush=True)
                prev = len(output)

    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    url = f"{args.host}:{args.port}"

    test_decode_stream(url, False, 0)
    test_decode_stream(url, True, 0)
    test_decode_stream(url, True, 3)
