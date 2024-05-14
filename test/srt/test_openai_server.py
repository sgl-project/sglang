"""
First run the following command to launch the server.
Note that TinyLlama adopts different chat templates in different versions.
For v0.4, the chat template is chatml.

python3 -m sglang.launch_server --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 \
--port 30000 --chat-template chatml

Output example:
The capital of France is Paris.
The capital of the United States is Washington, D.C.
The capital of Canada is Ottawa.
The capital of Japan is Tokyo
"""

import argparse
import json

import openai


def test_completion(args, echo, logprobs):
    client = openai.Client(api_key="EMPTY", base_url=args.base_url)
    response = client.completions.create(
        model="default",
        prompt="The capital of France is",
        temperature=0,
        max_tokens=32,
        echo=echo,
        logprobs=logprobs,
    )
    text = response.choices[0].text
    print(response.choices[0].text)
    if echo:
        assert text.startswith("The capital of France is")
    if logprobs:
        print(response.choices[0].logprobs.top_logprobs)
        assert response.choices[0].logprobs
        if echo:
            assert response.choices[0].logprobs.token_logprobs[0] == None
        else:
            assert response.choices[0].logprobs.token_logprobs[0] != None
    assert response.id
    assert response.created
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens > 0
    print("=" * 100)


def test_completion_stream(args, echo, logprobs):
    client = openai.Client(api_key="EMPTY", base_url=args.base_url)
    response = client.completions.create(
        model="default",
        prompt="The capital of France is",
        temperature=0,
        max_tokens=32,
        stream=True,
        echo=echo,
        logprobs=logprobs,
    )
    first = True
    for r in response:
        if first:
            if echo:
                assert r.choices[0].text.startswith("The capital of France is")
            first = False
        if logprobs:
            print(
                f"{r.choices[0].text:12s}\t" f"{r.choices[0].logprobs.token_logprobs}",
                flush=True,
            )
            print(r.choices[0].logprobs.top_logprobs)
        else:
            print(r.choices[0].text, end="", flush=True)
        assert r.id
        assert r.usage.prompt_tokens > 0
        assert r.usage.completion_tokens > 0
        assert r.usage.total_tokens > 0
    print("=" * 100)


def test_chat_completion(args):
    client = openai.Client(api_key="EMPTY", base_url=args.base_url)
    response = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant"},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        temperature=0,
        max_tokens=32,
    )
    print(response.choices[0].message.content)
    assert response.id
    assert response.created
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens > 0
    print("=" * 100)


def test_chat_completion_image(args):
    client = openai.Client(api_key="EMPTY", base_url=args.base_url)
    response = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/mixtral_8x7b.jpg"
                        },
                    },
                ],
            },
        ],
        temperature=0,
        max_tokens=32,
    )
    print(response.choices[0].message.content)
    assert response.id
    assert response.created
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens > 0
    print("=" * 100)


def test_chat_completion_stream(args):
    client = openai.Client(api_key="EMPTY", base_url=args.base_url)
    response = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant"},
            {"role": "user", "content": "List 3 countries and their capitals."},
        ],
        temperature=0,
        max_tokens=64,
        stream=True,
    )
    is_first = True
    for chunk in response:
        if is_first:
            is_first = False
            assert chunk.choices[0].delta.role == "assistant"
            continue

        data = chunk.choices[0].delta
        if not data.content:
            continue
        print(data.content, end="", flush=True)
    print("=" * 100)


def test_regex(args):
    client = openai.Client(api_key="EMPTY", base_url=args.base_url)

    regex = (
        r"""\{\n"""
        + r"""   "name": "[\w]+",\n"""
        + r"""   "population": [\d]+\n"""
        + r"""\}"""
    )

    response = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant"},
            {"role": "user", "content": "Introduce the capital of France."},
        ],
        temperature=0,
        max_tokens=128,
        extra_body={"regex": regex},
    )
    text = response.choices[0].message.content
    print(json.loads(text))
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:30000/v1")
    parser.add_argument(
        "--test-image", action="store_true", help="Enables testing image inputs"
    )
    args = parser.parse_args()

    test_completion(args, echo=False, logprobs=False)
    test_completion(args, echo=True, logprobs=False)
    test_completion(args, echo=False, logprobs=True)
    test_completion(args, echo=True, logprobs=True)
    test_completion(args, echo=False, logprobs=3)
    test_completion(args, echo=True, logprobs=3)
    test_completion_stream(args, echo=False, logprobs=False)
    test_completion_stream(args, echo=True, logprobs=False)
    test_completion_stream(args, echo=False, logprobs=True)
    test_completion_stream(args, echo=True, logprobs=True)
    test_completion_stream(args, echo=False, logprobs=3)
    test_completion_stream(args, echo=True, logprobs=3)
    test_chat_completion(args)
    test_chat_completion_stream(args)
    test_regex(args)
    if args.test_image:
        test_chat_completion_image(args)
