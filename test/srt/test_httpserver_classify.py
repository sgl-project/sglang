"""
Usage:
python3 test_httpserver_classify.py
"""

import argparse
import requests


def test_classify(url):
    response = requests.post(
        url + "/generate",
        json={
            "text": [
                "The capital of France is<|eot_id|>",
                "How about you?<|eot_id|>",
            ],

            "sampling_params": {
                "max_new_tokens": 0,
            },
            "return_logprob": True,
        },
    )
    print(response.json())

    logits = response.json()[0]["meta_info"]["normalized_prompt_logprob"]
    print("logits", logits)
    logits = response.json()[1]["meta_info"]["normalized_prompt_logprob"]
    print("logits", logits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    url = f"{args.host}:{args.port}"

    test_classify(url)