"""Three-round GSM8K accuracy and L2 KV replay case for Qwen3-8B on NPU."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

from sglang.test.simple_eval_mixed_prefix_gsm8k import (
    get_answer_value,
    get_few_shot_examples,
    get_one_example,
)


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def get_json(url, timeout):
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def metric_total(base_url, name, timeout, pool=None):
    response = requests.get(f"{base_url}/metrics", timeout=timeout)
    response.raise_for_status()
    pattern = re.compile(r"^" + re.escape(name) + r"(?:\{([^}]*)\})?\s+([\d.eE+-]+)$")
    values = []
    for line in response.text.splitlines():
        match = pattern.match(line)
        if match and (pool is None or re.search(rf'\bpool="{re.escape(pool)}"', match[1] or "")):
            values.append(float(match[2]))
    return sum(values)


def wait_for_metric(base_url, name, before, timeout, pool=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = metric_total(base_url, name, timeout, pool)
        if current > before:
            return current
        time.sleep(1)
    raise AssertionError(f"{name} did not increase from {before}")


def generate(base_url, input_ids, max_new_tokens, timeout, logprobs=False):
    response = requests.post(
        f"{base_url}/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "stop": ["Question:", "Assistant:", "<|separator|>"],
            },
            "return_logprob": logprobs,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data.get("text"), str):
        raise AssertionError("/generate did not return text")
    return data


def output_ids(data):
    logprobs = data.get("meta_info", {}).get("output_token_logprobs")
    if logprobs is None:
        raise AssertionError("/generate did not return output_token_logprobs")
    return [item[1] for item in logprobs]


def make_filler(tokenizer, train, start, target_tokens, max_questions):
    # Round two uses train questions only. The evaluated test prompts never appear here.
    chunks = [f"Independent cache eviction request {start}.\n"]
    for offset in range(max_questions):
        row = train[start + offset]
        chunks.append(f"Question {start + offset}: {row['question']}\n")
        ids = tokenizer.encode("".join(chunks), add_special_tokens=False)
        if len(ids) >= target_tokens:
            return ids[:target_tokens], start + offset + 1
    raise AssertionError("GSM8K train set is too small for the requested filler size")


def score_rows(rows):
    return sum(row["correct"] for row in rows) / len(rows)


def shared_prefix_length(prompts):
    if len(prompts) == 1:
        return 0
    for i in range(min(map(len, prompts))):
        if any(prompt[i] != prompts[0][i] for prompt in prompts[1:]):
            return i
    return min(map(len, prompts))


def git_commit(path):
    proc = subprocess.run(
        ["git", "-C", path, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=("layer_first", "page_first"), required=True)
    parser.add_argument("--mode", choices=("l2", "device"), default="l2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_path = Path(os.environ["MODEL_PATH"])
    timeout = int(os.environ["REQUEST_TIMEOUT"])
    n = int(os.environ["GSM8K_NUM_QUESTIONS"])
    shots = int(os.environ["GSM8K_NUM_SHOTS"])
    max_new_tokens = int(os.environ["GSM8K_MAX_NEW_TOKENS"])
    max_total_tokens = int(os.environ["MAX_TOTAL_TOKENS"])
    page_size = int(os.environ["PAGE_SIZE"])
    filler_tokens = int(os.environ["FILLER_PROMPT_TOKENS"])
    max_fillers = int(os.environ["MAX_FILLER_REQUESTS"])
    threshold = os.environ.get("GSM8K_ACCURACY", "")
    threshold = float(threshold) if threshold else None
    base_url = f"http://127.0.0.1:{os.environ['PORT']}"
    result = {
        "layout": args.layout,
        "mode": args.mode,
        "model": str(model_path),
        "num_questions": n,
        "num_shots": shots,
        "accuracy_threshold": threshold,
        "sglang_commit": git_commit(os.environ["SGLANG_ROOT"]),
        "kernel_commit": git_commit(os.environ["KERNEL_ROOT"]),
        "npu_devices": os.environ["NPU_DEVICES"],
        "tp_size": int(os.environ["TP_SIZE"]),
        "rounds": {"first": [], "filler": [], "replay": []},
        "passed": False,
    }
    try:
        if n < 1 or shots < 0 or page_size < 1 or filler_tokens < 1 or max_fillers < 1:
            raise ValueError("question, filler and shot counts must be valid")
        if filler_tokens + 1 >= max_total_tokens:
            raise ValueError("FILLER_PROMPT_TOKENS must fit MAX_TOTAL_TOKENS")
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        geometry = {
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
        }
        if config.get("model_type") != "qwen3" or "Qwen3ForCausalLM" not in config.get("architectures", []):
            raise AssertionError("MODEL_PATH is not Qwen3ForCausalLM")
        if any(config.get(key) != value for key, value in geometry.items()):
            raise AssertionError("MODEL_PATH does not match Qwen3-8B geometry")

        info = get_json(f"{base_url}/server_info", timeout)
        model_info = get_json(f"{base_url}/model_info", timeout)
        if Path(info.get("model_path", "")).resolve() != model_path.resolve():
            raise AssertionError(f"server loaded {info.get('model_path')!r}")
        expected = {"enable_hierarchical_cache": args.mode == "l2"}
        if args.mode == "l2":
            expected.update({
                "hicache_io_backend": "kernel_ascend",
                "hicache_mem_layout": args.layout,
                "hicache_write_policy": "write_through",
            })
        for field, value in expected.items():
            if info.get(field) != value:
                raise AssertionError(f"resolved {field}={info.get(field)!r}; expected {value!r}")
        if model_info.get("model_type") != "qwen3" or "Qwen3ForCausalLM" not in model_info.get("architectures", []):
            raise AssertionError(f"server loaded unexpected model: {model_info}")
        result["resolved"] = {key: info[key] for key in expected}

        train = read_jsonl(os.environ["GSM8K_TRAIN_PATH"])
        test = read_jsonl(os.environ["GSM8K_TEST_PATH"])
        if len(test) < n or len(train) <= shots:
            raise ValueError("GSM8K train/test set is smaller than the requested sample")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        few_shot = get_few_shot_examples(train, shots)
        prompts = []
        for index in range(n):
            content = few_shot + get_one_example(test, index, include_answer=False)
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(tokenizer.encode(rendered, add_special_tokens=False))
        if max(len(prompt) for prompt in prompts) + max_new_tokens >= max_total_tokens:
            raise ValueError("a GSM8K prompt plus its output does not fit MAX_TOTAL_TOKENS")
        common_pages = shared_prefix_length(prompts) // page_size
        unique_page_indices = [
            i for i, prompt in enumerate(prompts)
            if len(prompt) // page_size > common_pages
        ]
        result["questions_with_unique_prompt_pages"] = len(unique_page_indices)
        if not unique_page_indices:
            raise ValueError("no question has a unique full prompt page; lower PAGE_SIZE")

        backup_name = "sglang:hicache_backup_tokens_total"
        load_name = "sglang:load_back_tokens_total"
        eviction_name = "sglang:evicted_tokens_total"
        backup_before = metric_total(base_url, backup_name, timeout, "kv")
        load_before = metric_total(base_url, load_name, timeout, "kv")
        eviction_before = metric_total(base_url, eviction_name, timeout)

        # Round 1: evaluate the test questions normally.
        for index, prompt in enumerate(prompts):
            data = generate(base_url, prompt, max_new_tokens, timeout, logprobs=True)
            answer = get_answer_value(data["text"])
            expected_answer = get_answer_value(test[index]["answer"])
            row = {
                "index": index,
                "expected": expected_answer,
                "answer": answer,
                "correct": answer == expected_answer,
                "text": data["text"],
                "output_token_ids": output_ids(data),
            }
            result["rounds"]["first"].append(row)
            print(f"first {index + 1}/{n}: correct={row['correct']}", flush=True)
        result["first_accuracy"] = score_rows(result["rounds"]["first"])

        if args.mode == "l2":
            # Round 2: fill more than one device-cache capacity and observe eviction.
            next_train = shots
            filler_input_tokens = 0
            required_filler_tokens = max_total_tokens + filler_tokens
            for filler_index in range(max_fillers):
                prompt, next_train = make_filler(tokenizer, train, next_train, filler_tokens, 100)
                data = generate(base_url, prompt, 1, timeout)
                filler_input_tokens += len(prompt)
                result["rounds"]["filler"].append({
                    "index": filler_index,
                    "prompt_tokens": len(prompt),
                    "output": data["text"],
                })
                evicted = metric_total(base_url, eviction_name, timeout) - eviction_before
                print(f"filler {filler_index + 1}: input_tokens={filler_input_tokens}, evicted={evicted}", flush=True)
                if filler_input_tokens >= required_filler_tokens and evicted > 0:
                    break
            else:
                raise AssertionError("filler requests did not fill L1 and trigger eviction")
            result["filler_input_tokens"] = filler_input_tokens
            result["evicted_tokens_delta"] = evicted
            backup_after_filler = wait_for_metric(base_url, backup_name, backup_before, timeout, "kv")
            result["backup_tokens_delta"] = backup_after_filler - backup_before

        # Round 3: replay the *identical token IDs* for every test question.
        for index, prompt in enumerate(prompts):
            data = generate(base_url, prompt, max_new_tokens, timeout, logprobs=True)
            details = data.get("meta_info", {}).get("cached_tokens_details") or {}
            answer = get_answer_value(data["text"])
            expected_answer = get_answer_value(test[index]["answer"])
            row = {
                "index": index,
                "expected": expected_answer,
                "answer": answer,
                "correct": answer == expected_answer,
                "text": data["text"],
                "output_token_ids": output_ids(data),
                "cached_tokens_details": details,
            }
            result["rounds"]["replay"].append(row)
            print(f"replay {index + 1}/{n}: correct={row['correct']}, host={details.get('host', 0)}", flush=True)
        result["replay_accuracy"] = score_rows(result["rounds"]["replay"])
        result["accuracy_delta"] = result["replay_accuracy"] - result["first_accuracy"]
        result["host_hit_questions"] = sum(
            row["cached_tokens_details"].get("host", 0) > 0
            for row in result["rounds"]["replay"]
        )
        result["device_hit_questions"] = sum(
            row["cached_tokens_details"].get("device", 0) > 0
            for row in result["rounds"]["replay"]
        )
        result["same_output_questions"] = sum(
            first["output_token_ids"] == replay["output_token_ids"]
            for first, replay in zip(result["rounds"]["first"], result["rounds"]["replay"])
        )
        load_after_replay = (
            wait_for_metric(base_url, load_name, load_before, timeout, "kv")
            if args.mode == "l2" else metric_total(base_url, load_name, timeout, "kv")
        )
        result["load_back_tokens_delta"] = load_after_replay - load_before
        if args.mode == "l2":
            missing_unique_hits = [
                i for i in unique_page_indices
                if result["rounds"]["replay"][i]["cached_tokens_details"].get("host", 0) <= 0
            ]
            result["unique_page_questions_without_host_hit"] = missing_unique_hits
            if missing_unique_hits:
                raise AssertionError(
                    f"{len(missing_unique_hits)} questions with full unique prompt pages did not hit L2"
                )
            if result["replay_accuracy"] < result["first_accuracy"]:
                raise AssertionError("GSM8K accuracy dropped after L2 replay")
        else:
            result["evicted_tokens_delta"] = metric_total(base_url, eviction_name, timeout) - eviction_before
            if result["evicted_tokens_delta"] != 0:
                raise AssertionError("device-only control evicted KV; increase MAX_TOTAL_TOKENS")
            if result["host_hit_questions"] or result["load_back_tokens_delta"]:
                raise AssertionError("device-only control unexpectedly loaded KV from Host")
            if result["device_hit_questions"] != n:
                raise AssertionError("device-only control did not hit L1 for every question")
        if threshold is not None and result["replay_accuracy"] < threshold:
            raise AssertionError(f"replay accuracy {result['replay_accuracy']:.4f} < threshold {threshold:.4f}")
        result["passed"] = True
        print(
            f"{args.layout}/{args.mode}: first={result['first_accuracy']:.4f}, "
            f"replay={result['replay_accuracy']:.4f}, "
            f"host_hits={result['host_hit_questions']}/{n}, "
            f"device_hits={result['device_hit_questions']}/{n}",
            flush=True,
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"{args.layout}: {result['error']}", file=sys.stderr)
    finally:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
