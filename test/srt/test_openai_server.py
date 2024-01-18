"""
python3 -m sglang.launch_server --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 --port 30000

Output:
The capital of France is Paris.\nThe capital of the United States is Washington, D.C.\nThe capital of Canada is Ottawa.\nThe capital of Japan is Tokyo
"""

import argparse

import openai


def test_completion(args):
    client = openai.Client(api_key="EMPTY", base_url=args.base_url)
    response = client.completions.create(
        model="default",
        prompt="The capital of France is",
        temperature=0,
        max_tokens=32,
    )
    print(response.choices[0].text)
    assert response.id
    assert response.created
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens > 0


def test_completion_stream(args):
    client = openai.Client(api_key="EMPTY", base_url=args.base_url)
    response = client.completions.create(
        model="default",
        prompt="The capital of France is",
        temperature=0,
        max_tokens=32,
        stream=True,
    )
    for r in response:
        print(r.choices[0].text, end="", flush=True)
        assert r.id
        assert r.created
        assert r.usage.prompt_tokens > 0
        assert r.usage.completion_tokens > 0
        assert r.usage.total_tokens > 0
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:30000/v1")
    args = parser.parse_args()

    test_completion(args)
    test_completion_stream(args)
