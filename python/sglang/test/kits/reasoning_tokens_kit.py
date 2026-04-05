import json

import requests

from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.utils.hf_transformers_utils import get_tokenizer


def run_reasoning_tokens_test(base_url, model, reasoning_parser_name, api_key=None):
    """Run reasoning_tokens usage tests against a server with --reasoning-parser.

    Covers chat API (streaming + non-streaming) × (thinking + non-thinking),
    plus /generate API exact count verification.
    """
    tokenizer = get_tokenizer(model)
    parser = ReasoningParser(reasoning_parser_name)
    think_end_token_id = tokenizer.convert_tokens_to_ids(
        parser.detector.think_end_token
    )
    assert think_end_token_id is not None

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def chat_request(enable_thinking, stream=False):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "What is 1+3?"}],
            "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return requests.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=stream,
        )

    def extract_streaming_usage(response):
        usage = None
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data:") or decoded.startswith("data: [DONE]"):
                continue
            data = json.loads(decoded[len("data:") :].strip())
            if data.get("usage"):
                usage = data["usage"]
        return usage

    # Chat API non-streaming
    resp = chat_request(enable_thinking=True)
    assert resp.status_code == 200, resp.text
    usage = resp.json()["usage"]
    assert usage["reasoning_tokens"] > 0
    assert usage["reasoning_tokens"] < usage["completion_tokens"]

    resp = chat_request(enable_thinking=False)
    assert resp.status_code == 200, resp.text
    assert resp.json()["usage"]["reasoning_tokens"] == 0

    # Chat API streaming
    with chat_request(enable_thinking=True, stream=True) as resp:
        assert resp.status_code == 200
        usage = extract_streaming_usage(resp)
        assert usage is not None
        assert usage["reasoning_tokens"] > 0
        assert usage["reasoning_tokens"] < usage["completion_tokens"]

    with chat_request(enable_thinking=False, stream=True) as resp:
        assert resp.status_code == 200
        usage = extract_streaming_usage(resp)
        assert usage is not None
        assert usage["reasoning_tokens"] == 0

    # Generate API exact count
    messages = [{"role": "user", "content": "What is 1+3?"}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    resp = requests.post(
        f"{base_url}/generate",
        headers=headers,
        json={
            "text": prompt,
            "sampling_params": {"max_new_tokens": 1024},
            "require_reasoning": True,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    reported = data["meta_info"]["reasoning_tokens"]
    actual = data["output_ids"].index(think_end_token_id) + 1
    assert reported == actual, f"reasoning_tokens mismatch: {reported} != {actual}"
