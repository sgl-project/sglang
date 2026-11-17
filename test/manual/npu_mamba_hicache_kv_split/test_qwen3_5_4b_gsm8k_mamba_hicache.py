"""GSM8K accuracy with a Host-L2 reload before every replay on NPU.

Round 1 answers every question once with standard few-shot prompts (no
synthetic padding) and records answers, accuracy and output token IDs.
After round 1, filler requests (train questions only) evict the device cache
before each replay. Round 3 replays the identical token IDs in the same order;
every replay must hit Host L2 and load Mamba state back to the device.

Pass conditions: replay accuracy >= first-round accuracy, replay accuracy
>= GSM8K_ACCURACY threshold (if set), every replay hits Host L2 and keeps its
first-round parsed answer, Mamba H2D grows for every replay, and no device
token is dropped without backup. Output token consistency is diagnostic only.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

from sglang.test.simple_eval_mixed_prefix_gsm8k import (
    INVALID,
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


def make_filler(tokenizer, train, start, ordinal, target_tokens):
    # Fillers use train questions only. The evaluated test prompts never
    # appear here. The start index wraps around the train set and the
    # ordinal in the header keeps every filler's token sequence unique even
    # when rows repeat, so filler count is bounded by runtime, not dataset
    # size.
    start = start % len(train)
    chunks = [f"Independent cache eviction request {ordinal}.\n"]
    for offset in range(len(train)):
        row = train[(start + offset) % len(train)]
        chunks.append(f"Question {start + offset}: {row['question']}\n")
        ids = tokenizer.encode("".join(chunks), add_special_tokens=False)
        if len(ids) >= target_tokens:
            return ids[:target_tokens], (start + offset + 1) % len(train)
    raise AssertionError("GSM8K train set is too small for the requested filler size")


def score_rows(rows):
    return sum(row["correct"] for row in rows) / len(rows)


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_path = Path(os.environ["MODEL_PATH"])
    timeout = int(os.environ["REQUEST_TIMEOUT"])
    n = int(os.environ["GSM8K_NUM_QUESTIONS"])
    shots = int(os.environ["GSM8K_NUM_SHOTS"])
    max_new_tokens = int(os.environ["GSM8K_MAX_NEW_TOKENS"])
    max_total_tokens = int(os.environ["MAX_TOTAL_TOKENS"])
    filler_tokens = int(os.environ["FILLER_PROMPT_TOKENS"])
    threshold = os.environ.get("GSM8K_ACCURACY", "")
    threshold = float(threshold) if threshold else None
    base_url = f"http://127.0.0.1:{os.environ['PORT']}"
    result = {
        "mode": "mamba_l2",
        "flow": "first_all_then_evict_replay_each",
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
        if n < 1 or shots < 0 or filler_tokens < 1:
            raise ValueError("question and filler counts must be valid")
        if filler_tokens + 1 >= max_total_tokens:
            raise ValueError("FILLER_PROMPT_TOKENS must fit MAX_TOTAL_TOKENS")
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "qwen3_5" or (
            "Qwen3_5ForConditionalGeneration" not in config.get("architectures", [])
        ):
            raise AssertionError("MODEL_PATH is not Qwen3.5-4B")
        text_config = config.get("text_config", config)
        if text_config.get("model_type") != "qwen3_5_text":
            raise AssertionError("MODEL_PATH is not a Qwen3.5 text model")
        layer_types = text_config.get("layer_types", [])
        if (
            text_config.get("num_hidden_layers") != 32
            or text_config.get("hidden_size") != 2560
            or layer_types.count("linear_attention") != 24
            or layer_types.count("full_attention") != 8
        ):
            raise AssertionError("MODEL_PATH does not match Qwen3.5-4B hybrid geometry")

        info = get_json(f"{base_url}/server_info", timeout)
        if Path(info.get("model_path", "")).resolve() != model_path.resolve():
            raise AssertionError(f"server loaded {info.get('model_path')!r}")
        expected = {
            "enable_hierarchical_cache": True,
            "hicache_io_backend": "kernel_ascend",
            "hicache_write_policy": "write_through",
        }
        for field, value in expected.items():
            if info.get(field) != value:
                raise AssertionError(f"resolved {field}={info.get(field)!r}; expected {value!r}")
        result["resolved"] = {key: info[key] for key in expected}

        train = read_jsonl(os.environ["GSM8K_TRAIN_PATH"])
        test = read_jsonl(os.environ["GSM8K_TEST_PATH"])
        if len(test) < n or len(train) <= shots:
            raise ValueError("GSM8K train/test set is smaller than the requested sample")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        few_shot = get_few_shot_examples(train, shots)
        prompts = []
        for index in range(n):
            # Standard GSM8K prompt: shared few-shot prefix + question.
            # No synthetic padding: prompts stay comparable with the standard
            # GSM8K baseline and historical accuracy thresholds.
            content = few_shot + get_one_example(test, index, include_answer=False)
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(tokenizer.encode(rendered, add_special_tokens=False))

        backup_name = "sglang:hicache_backup_tokens_total"
        load_name = "sglang:load_back_tokens_total"
        eviction_name = "sglang:evicted_tokens_total"
        dropped_name = "sglang:hicache_dropped_tokens_total"
        next_train = shots
        filler_budget = int(os.environ["MAX_FILLER_REQUESTS"])
        filler_input_tokens = 0

        # -- Round 1: answer every question once --
        for index, prompt in enumerate(prompts):
            data = generate(base_url, prompt, max_new_tokens, timeout, logprobs=True)
            answer = get_answer_value(data["text"])
            expected_answer = get_answer_value(test[index]["answer"])
            result["rounds"]["first"].append({
                "index": index,
                "prompt_tokens": len(prompt),
                "expected": expected_answer,
                "answer": answer,
                "correct": answer == expected_answer,
                "text": data["text"],
                "output_token_ids": output_ids(data),
            })
            print(f"first {index + 1}/{n}: correct={answer == expected_answer}", flush=True)

        fillers_needed = math.ceil(max_total_tokens / filler_tokens)
        filler_ordinal = 0

        def evict_before_replay(replay_index):
            nonlocal next_train, filler_input_tokens, filler_ordinal
            evicted_start = metric_total(base_url, eviction_name, timeout)
            backup_start = metric_total(base_url, backup_name, timeout, "mamba")
            kv_backup_start = metric_total(base_url, backup_name, timeout, "kv")
            dropped_start = metric_total(base_url, dropped_name, timeout)
            if filler_budget < fillers_needed:
                raise ValueError(
                    f"MAX_FILLER_REQUESTS={filler_budget} is below the "
                    f"{fillers_needed} fillers needed per replay"
                )
            for _ in range(fillers_needed):
                filler_prompt, next_train = make_filler(
                    tokenizer, train, next_train, filler_ordinal, filler_tokens
                )
                generate(base_url, filler_prompt, 1, timeout)
                filler_ordinal += 1
                filler_input_tokens += len(filler_prompt)
                result["rounds"]["filler"].append({
                    "replay_index": replay_index,
                    "prompt_tokens": len(filler_prompt),
                })
            evicted = metric_total(base_url, eviction_name, timeout) - evicted_start
            if evicted <= 0:
                raise AssertionError(f"replay {replay_index}: fillers caused no eviction")
            # A write-through KV copy can finish after the filler response.
            # Repeated loads can retain a Host copy, so later rounds need not
            # increment backup counters again.
            kv_backup = (
                metric_total(base_url, backup_name, timeout, "kv") - kv_backup_start
            )
            if replay_index == 0 or kv_backup > 0:
                settle_deadline = time.monotonic() + timeout
                last_kv_backup = -1.0
                while time.monotonic() < settle_deadline:
                    kv_backup = (
                        metric_total(base_url, backup_name, timeout, "kv") - kv_backup_start
                    )
                    if kv_backup == last_kv_backup and kv_backup > 0:
                        break
                    last_kv_backup = kv_backup
                    time.sleep(2)
                else:
                    raise AssertionError(f"replay {replay_index}: KV backup did not settle")
            mamba_backup = (
                metric_total(base_url, backup_name, timeout, "mamba") - backup_start
            )
            if replay_index == 0:
                if kv_backup <= 0:
                    raise AssertionError("initial fillers did not back up KV to Host")
                wait_for_metric(base_url, backup_name, backup_start, timeout, "mamba")
                mamba_backup = (
                    metric_total(base_url, backup_name, timeout, "mamba") - backup_start
                )
            dropped = metric_total(base_url, dropped_name, timeout) - dropped_start
            if dropped > 0:
                raise AssertionError(
                    f"replay {replay_index}: {dropped} device tokens were dropped"
                )
            print(
                f"fillers before replay {replay_index + 1}/{n}: "
                f"sent={fillers_needed}, evicted={evicted}, "
                f"mamba_backup={mamba_backup}, kv_backup={kv_backup}",
                flush=True,
            )

        # -- Evict, then replay each original prompt from Host L2 --
        result["mamba_load_tokens_delta"] = 0
        for index, prompt in enumerate(prompts):
            evict_before_replay(index)
            load_start = metric_total(base_url, load_name, timeout, "mamba")
            data = generate(base_url, prompt, max_new_tokens, timeout, logprobs=True)
            details = data.get("meta_info", {}).get("cached_tokens_details") or {}
            answer = get_answer_value(data["text"])
            expected_answer = get_answer_value(test[index]["answer"])
            first_answer = result["rounds"]["first"][index]["answer"]
            host = details.get("host", 0)
            device = details.get("device", 0)
            same_answer = answer != INVALID and answer == first_answer
            result["rounds"]["replay"].append({
                "index": index,
                "expected": expected_answer,
                "answer": answer,
                "correct": answer == expected_answer,
                "matches_first_answer": same_answer,
                "text": data["text"],
                "output_token_ids": output_ids(data),
                "cached_tokens_details": details,
            })
            if host <= 0:
                raise AssertionError(f"replay {index}: no Host L2 hit (device={device})")
            load_after = wait_for_metric(
                base_url, load_name, load_start, min(timeout, 30), "mamba"
            )
            mamba_load_delta = load_after - load_start
            result["mamba_load_tokens_delta"] += mamba_load_delta
            result["rounds"]["replay"][-1]["mamba_load_tokens"] = mamba_load_delta
            print(
                f"replay {index + 1}/{n}: correct={answer == expected_answer}, "
                f"host={host}, device={device}, mamba_load={mamba_load_delta}, "
                f"same_answer={same_answer}",
                flush=True,
            )
            if not same_answer:
                raise AssertionError(
                    f"replay {index}: GSM8K answer changed from "
                    f"{first_answer!r} to {answer!r} after Host reload"
                )

        result["first_accuracy"] = score_rows(result["rounds"]["first"])
        result["replay_accuracy"] = score_rows(result["rounds"]["replay"])
        result["accuracy_delta"] = result["replay_accuracy"] - result["first_accuracy"]
        result["filler_input_tokens"] = filler_input_tokens
        result["host_hit_questions"] = sum(
            row["cached_tokens_details"].get("host", 0) > 0
            for row in result["rounds"]["replay"]
        )
        result["host_answer_mismatch_questions"] = [
            row["index"]
            for row in result["rounds"]["replay"]
            if row["cached_tokens_details"].get("host", 0) > 0
            and not row["matches_first_answer"]
        ]
        result["device_hit_questions"] = sum(
            row["cached_tokens_details"].get("device", 0) > 0
            for row in result["rounds"]["replay"]
        )
        result["same_output_questions"] = sum(
            first["output_token_ids"] == replay["output_token_ids"]
            for first, replay in zip(result["rounds"]["first"], result["rounds"]["replay"])
        )
        # Output divergence on Host-reloaded questions is diagnostic (bf16
        # checkpoint-tail recompute micro-differences can flip greedy
        # argmax); answers may still be correct.
        for first, replay in zip(result["rounds"]["first"], result["rounds"]["replay"]):
            if first["output_token_ids"] != replay["output_token_ids"]:
                print(
                    f"mamba_l2: warning: question {replay['index']} greedy output diverged "
                    f"({len(first['output_token_ids'])} -> {len(replay['output_token_ids'])} tokens)",
                    flush=True,
                )
        if result["replay_accuracy"] < result["first_accuracy"]:
            raise AssertionError("GSM8K accuracy dropped after Mamba L2 replay")
        if result["host_hit_questions"] < n:
            raise AssertionError(
                f"only {result['host_hit_questions']}/{n} replays hit Host L2"
            )
        if result["host_answer_mismatch_questions"]:
            raise AssertionError(
                "Host-reloaded questions changed their GSM8K answers: "
                f"{result['host_answer_mismatch_questions']}"
            )
        if result["mamba_load_tokens_delta"] <= 0:
            raise AssertionError("Mamba H2D load-back tokens did not grow")
        if threshold is not None and result["replay_accuracy"] < threshold:
            raise AssertionError(f"replay accuracy {result['replay_accuracy']:.4f} < threshold {threshold:.4f}")
        result["passed"] = True
        print(
            f"mamba_l2: first={result['first_accuracy']:.4f}, "
            f"replay={result['replay_accuracy']:.4f}, "
            f"host_hits={result['host_hit_questions']}/{n}, "
            f"device_hits={result['device_hit_questions']}/{n}",
            flush=True,
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"mamba_l2: {result['error']}", file=sys.stderr)
    finally:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
