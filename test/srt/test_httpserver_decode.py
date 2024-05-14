"""
Usage:
python3 -m sglang.launch_server --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 --port 30000
python3 test_httpserver_decode.py

Output:
The capital of France is Paris.\nThe capital of the United States is Washington, D.C.\nThe capital of Canada is Ottawa.\nThe capital of Japan is Tokyo
"""

import argparse
import json

import requests


def test_decode(url, return_logprob, top_logprobs_num, return_text):
    response = requests.post(
        url + "/generate",
        json={
            "text": "The capital of France is",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 32,
            },
            "stream": False,
            "return_logprob": return_logprob,
            "top_logprobs_num": top_logprobs_num,
            "return_text_in_logprobs": return_text,
            "logprob_start_len": 0,
        },
    )
    print(json.dumps(response.json()))
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    url = f"{args.host}:{args.port}"

    test_decode(url, False, 0, False)
    test_decode(url, True, 0, False)
    test_decode(url, True, 0, True)
    test_decode(url, True, 3, False)
    test_decode(url, True, 3, True)
