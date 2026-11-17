"""Send Qwen3-8B requests and check real HiCache D2H/H2D activity."""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


def get_json(url, timeout):
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def metric_total(base_url, metric_name, timeout):
    response = requests.get(f"{base_url}/metrics", timeout=timeout)
    response.raise_for_status()
    pattern = re.compile(
        "^" + re.escape(metric_name) + r"(?:\{([^}]*)\})?\s+([\d.eE+-]+)$"
    )
    values = []
    for line in response.text.splitlines():
        match = pattern.match(line)
        if match and re.search(r'\bpool="kv"', match.group(1) or ""):
            values.append(float(match.group(2)))
    return sum(values)


def wait_for_increase(base_url, name, before, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = metric_total(base_url, name, timeout)
        if current > before:
            return current
        time.sleep(1)
    raise AssertionError(f"{name} did not increase; before={before}")


def make_prompt(tokenizer, case_id, length):
    prefix = tokenizer.encode(
        f"Case {case_id}: Explain this sequence in one sentence. ",
        add_special_tokens=False,
    )
    filler = tokenizer.encode(
        f"A unique sequence for case {case_id} continues here. ",
        add_special_tokens=False,
    )
    return (prefix + filler * ((length - len(prefix)) // len(filler) + 1))[:length]


def generate(base_url, input_ids, timeout, output_tokens):
    response = requests.post(
        f"{base_url}/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": output_tokens},
            "return_logprob": True,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data.get("text"), str):
        raise AssertionError("/generate did not return text")
    return data


def token_ids(result):
    logprobs = result.get("meta_info", {}).get("output_token_logprobs")
    if not logprobs:
        raise AssertionError("/generate did not return output_token_logprobs")
    return [item[1] for item in logprobs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=("layer_first", "page_first"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_path = Path(os.environ["MODEL_PATH"])
    port = os.environ["PORT"]
    timeout = int(os.environ["REQUEST_TIMEOUT"])
    prompt_tokens = int(os.environ["PROMPT_TOKENS"])
    output_tokens = int(os.environ["OUTPUT_TOKENS"])
    filler_requests = int(os.environ["FILLER_REQUESTS"])
    base_url = f"http://127.0.0.1:{port}"
    result = {"layout": args.layout, "model": str(model_path), "passed": False}
    try:
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "qwen3" or "Qwen3ForCausalLM" not in config.get(
            "architectures", []
        ):
            raise AssertionError("MODEL_PATH is not a Qwen3ForCausalLM model")
        qwen3_8b_geometry = {
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
        }
        if any(config.get(key) != value for key, value in qwen3_8b_geometry.items()):
            raise AssertionError("MODEL_PATH config does not match Qwen3-8B geometry")
        info = get_json(f"{base_url}/server_info", timeout)
        model_info = get_json(f"{base_url}/model_info", timeout)
        if Path(info.get("model_path", "")).resolve() != model_path.resolve():
            raise AssertionError(f"server loaded {info.get('model_path')!r}, not {model_path}")
        expected = {
            "hicache_io_backend": "kernel_ascend",
            "hicache_mem_layout": args.layout,
            "hicache_write_policy": "write_through",
            "enable_hierarchical_cache": True,
        }
        for field, value in expected.items():
            if info.get(field) != value:
                raise AssertionError(f"resolved {field}={info.get(field)!r}, expected {value!r}")
        if model_info.get("model_type") != "qwen3" or "Qwen3ForCausalLM" not in model_info.get(
            "architectures", []
        ):
            raise AssertionError(f"server did not load Qwen3ForCausalLM: {model_info}")

        backup_name = "sglang:hicache_backup_tokens_total"
        load_name = "sglang:load_back_tokens_total"
        before_backup = metric_total(base_url, backup_name, timeout)
        before_load = metric_total(base_url, load_name, timeout)
        result["resolved"] = {key: info.get(key) for key in expected}
        result["model_info"] = model_info
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        prompt = make_prompt(tokenizer, 0, prompt_tokens)
        result["prompt_tokens"] = len(prompt)
        result["filler_requests"] = filler_requests
        first = generate(base_url, prompt, timeout, output_tokens)
        first_ids = token_ids(first)
        result["first_token_ids"] = first_ids
        result["first_text"] = first["text"]
        for case_id in range(1, filler_requests + 1):
            generate(
                base_url,
                make_prompt(tokenizer, case_id, prompt_tokens),
                timeout,
                output_tokens,
            )
        after_backup = wait_for_increase(base_url, backup_name, before_backup, timeout)
        result["backup_tokens_delta"] = after_backup - before_backup
        replay = generate(base_url, prompt, timeout, output_tokens)
        replay_ids = token_ids(replay)
        result["replay_token_ids"] = replay_ids
        result["replay_text"] = replay["text"]
        cache_details = replay.get("meta_info", {}).get("cached_tokens_details") or {}
        result["cached_tokens_details"] = cache_details
        after_load = wait_for_increase(base_url, load_name, before_load, timeout)
        result["load_back_tokens_delta"] = after_load - before_load
        if cache_details.get("host", 0) <= 0:
            raise AssertionError(f"replay did not hit host cache: {cache_details}")
        if first_ids != replay_ids or first["text"] != replay["text"]:
            raise AssertionError("replayed output differs from the first request")
        result["passed"] = True
        print(f"{args.layout}: D2H/H2D metrics increased; host hit and output matched")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"{args.layout}: {result['error']}", file=sys.stderr)
    finally:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
