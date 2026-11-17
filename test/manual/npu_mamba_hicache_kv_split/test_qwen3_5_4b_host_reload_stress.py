"""Targeted Host-L2 reload stress test for Mamba state transfer on NPU.

Per-question A -> E -> R (Answer -> Evict -> Replay), with per-question
unique padding and a checkpoint-grid seed request so every question's
replay hits Host L2 deterministically. This is a reload-path stress test,
NOT a standard GSM8K accuracy case: the unique numeric padding changes the
prompt distribution, so its accuracy is not comparable with the standard
GSM8K baseline. Use test_qwen3_5_4b_gsm8k_mamba_hicache.py for accuracy.

Mechanics (see README for the full rationale):

- A0 seed: the answer request's radix leaf spans prompt+output, so the
  replay's page-aligned match splits that leaf, and the split parent holds
  no mamba state (mamba is leaf-only data); the match then validates no
  node and the Host hit is lost. A leaf ending exactly on a
  checkpoint-grid depth inside the prompt keeps a restorable state, so the
  replay's match stops at a node that passes validation.
- S scrub (q1+): the previous question's replay load-back reloads its
  whole chain onto the device and consumes the host backups, leaving
  unbacked device residue; inserting the next seed under that pressure
  loses the seeded chain before A (observed: q1 A matched only the shared
  prefix). A scrub filler round first evicts that residue and re-backs it
  to Host L2, so every seed inserts into a device holding only
  host-backed filler pages -- the same conditions as question 0 on a
  fresh server.
- E deterministic eviction: fill the whole device capacity with unique
  filler content (filler total tokens >= device capacity), because
  evicted_tokens_total is rank-inflated (x TP rank) and cannot gate
  per-question eviction, and shared-prefix pages must also leave the
  device or the replay hits Device instead of Host.
- Filler budget: each eviction needs
  ceil(MAX_TOTAL_TOKENS / FILLER_PROMPT_TOKENS) fillers; the budget is
  granted per question, with a separate global cap across the whole run.
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
    # size (a 50-question run needs ~100 fillers).
    start = start % len(train)
    chunks = [f"Independent cache eviction request {ordinal}.\n"]
    for offset in range(len(train)):
        row = train[(start + offset) % len(train)]
        chunks.append(f"Question {start + offset}: {row['question']}\n")
        ids = tokenizer.encode("".join(chunks), add_special_tokens=False)
        if len(ids) >= target_tokens:
            return ids[:target_tokens], (start + offset + 1) % len(train)
    raise AssertionError("GSM8K train set is too small for the requested filler size")


def unique_pad(index, target_tokens):
    # Deterministic per-question text, roughly 1 token per 5-digit number plus
    # separators, so token counts land near target_tokens and never collide
    # between questions.
    count = int(target_tokens * 1.1) + 16
    numbers = " ".join(str((index * 7919 + j * 104729) % 100000) for j in range(count))
    return f"\n\n[Reference context {index}: {numbers}]\n\n"


def shared_prefix_length(prompts):
    if len(prompts) == 1:
        return 0
    for i in range(min(map(len, prompts))):
        if any(prompt[i] != prompts[0][i] for prompt in prompts[1:]):
            return i
    return min(map(len, prompts))


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
    page_size = int(os.environ["PAGE_SIZE"])
    filler_tokens = int(os.environ["FILLER_PROMPT_TOKENS"])
    max_fillers_per_question = int(os.environ["MAX_FILLER_REQUESTS"])
    global_filler_cap = int(os.environ.get("GLOBAL_FILLER_CAP", "0"))
    threshold = os.environ.get("GSM8K_ACCURACY", "")
    threshold = float(threshold) if threshold else None
    base_url = f"http://127.0.0.1:{os.environ['PORT']}"
    result = {
        "mode": "host_reload_stress",
        "flow": "per_question_flush_A0_A_E_R",
        "model": str(model_path),
        "num_questions": n,
        "num_shots": shots,
        "accuracy_threshold": threshold,
        "sglang_commit": git_commit(os.environ["SGLANG_ROOT"]),
        "kernel_commit": git_commit(os.environ["KERNEL_ROOT"]),
        "npu_devices": os.environ["NPU_DEVICES"],
        "tp_size": int(os.environ["TP_SIZE"]),
        "rounds": {"first": [], "filler": [], "replay": []},
        "cache_flushes": 0,
        "passed": False,
    }
    try:
        if n < 1 or shots < 0 or page_size < 1 or filler_tokens < 1 or max_fillers_per_question < 1:
            raise ValueError("question, filler and shot counts must be valid")
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
        unique_pad_tokens = int(os.environ["UNIQUE_PAD_TOKENS"])
        prompts = []
        for index in range(n):
            # A per-question unique pad between the shared few-shot prefix and
            # the question. Without it the few-shot prefix dominates: a replayed
            # question reloads the shared prefix back into the device radix, so
            # host hits degenerate into device hits and never exercise the
            # Host reload path. The pad gives each question full unique pages
            # (with Mamba checkpoints on the track grid).
            pad = unique_pad(index, unique_pad_tokens)
            content = few_shot + pad + get_one_example(test, index, include_answer=False)
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(tokenizer.encode(rendered, add_special_tokens=False))
        if max(len(prompt) for prompt in prompts) + max_new_tokens >= max_total_tokens:
            raise ValueError("a GSM8K prompt plus its output does not fit MAX_TOTAL_TOKENS")
        checkpoint_grid = math.lcm(page_size, int(info["mamba_track_interval"]))
        result["mamba_checkpoint_grid"] = checkpoint_grid
        # Host hit depths are page-aligned match lengths, so the shared
        # few-shot prefix boundary must be page-aligned too. A replay whose
        # host hit stops at this boundary only reloaded the shared prefix,
        # not this question's own pages -- which is exactly the v8 q1 case
        # (host=1280 == shared prefix) the reviewer called out.
        shared_tokens = shared_prefix_length(prompts)
        prefix_boundary = shared_tokens // page_size * page_size
        result["shared_prefix_tokens"] = shared_tokens
        result["shared_prefix_boundary"] = prefix_boundary

        backup_name = "sglang:hicache_backup_tokens_total"
        load_name = "sglang:load_back_tokens_total"
        eviction_name = "sglang:evicted_tokens_total"
        dropped_name = "sglang:hicache_dropped_tokens_total"
        next_train = shots
        fillers_needed = math.ceil(max_total_tokens / filler_tokens)
        global_fillers_sent = 0
        filler_input_tokens = 0
        mamba_load_total = 0
        # Per-question A0 -> A -> E -> R. Clear the previous question after
        # its replay so each seed starts from an empty radix tree. Flushing
        # between questions does not affect this question's E -> R Host load.
        for index, prompt in enumerate(prompts):
            # Previous R consumes its Host backup and leaves device residue.
            # Extra filler requests cannot reliably preserve the next seed
            # leaf under that tree. Reset only between independent questions.
            if index > 0:
                response = requests.post(
                    f"{base_url}/flush_cache",
                    params={"timeout": 30},
                    timeout=timeout,
                )
                response.raise_for_status()
                result["cache_flushes"] += 1
                print(f"q{index} cache flushed before seed", flush=True)

            # -- A0: seed a mamba checkpoint leaf on the track grid --
            match_len = (len(prompt) - 1) // page_size * page_size
            seed_len = match_len // checkpoint_grid * checkpoint_grid
            if seed_len <= prefix_boundary:
                raise AssertionError(
                    f"question {index}: seed checkpoint {seed_len} does not "
                    f"extend past shared prefix boundary {prefix_boundary}"
                )
            generate(base_url, prompt[:seed_len], 1, timeout)

            # -- A: answer the question once --
            data = generate(base_url, prompt, max_new_tokens, timeout, logprobs=True)
            answer = get_answer_value(data["text"])
            expected_answer = get_answer_value(test[index]["answer"])
            first_ids = output_ids(data)
            result["rounds"]["first"].append({
                "index": index,
                "seed_len": seed_len,
                "prompt_tokens": len(prompt),
                "expected": expected_answer,
                "answer": answer,
                "correct": answer == expected_answer,
                "text": data["text"],
                "output_token_ids": first_ids,
            })
            print(f"first {index + 1}/{n}: correct={answer == expected_answer}", flush=True)

            # -- E: fillers until this question is fully evicted from device --
            # Fill the whole device capacity with unique filler content, which
            # evicts every page of this question (including shared-prefix pages)
            # deterministically; the backup counters gate the D2H traffic.
            evicted_start = metric_total(base_url, eviction_name, timeout)
            backup_start = metric_total(base_url, backup_name, timeout, "mamba")
            kv_backup_start = metric_total(base_url, backup_name, timeout, "kv")
            dropped_start = metric_total(base_url, dropped_name, timeout)
            fillers_sent = 0
            mamba_backup = 0
            kv_backup = 0
            while (
                fillers_sent < fillers_needed
                or mamba_backup <= 0
                or kv_backup <= 0
            ) and fillers_sent < max_fillers_per_question:
                if global_filler_cap and global_fillers_sent >= global_filler_cap:
                    raise AssertionError(
                        f"global filler cap {global_filler_cap} reached at question {index}"
                    )
                filler_prompt, next_train = make_filler(
                    tokenizer, train, next_train, global_fillers_sent, filler_tokens
                )
                generate(base_url, filler_prompt, 1, timeout)
                fillers_sent += 1
                global_fillers_sent += 1
                filler_input_tokens += len(filler_prompt)
                result["rounds"]["filler"].append({
                    "question": index,
                    "prompt_tokens": len(filler_prompt),
                })
                mamba_backup = (
                    metric_total(base_url, backup_name, timeout, "mamba") - backup_start
                )
                kv_backup = (
                    metric_total(base_url, backup_name, timeout, "kv") - kv_backup_start
                )
                print(
                    f"q{index} filler {fillers_sent}/{fillers_needed}: "
                    f"mamba_backup={mamba_backup}, kv_backup={kv_backup}",
                    flush=True,
                )
            if fillers_sent < fillers_needed or mamba_backup <= 0 or kv_backup <= 0:
                raise AssertionError(
                    f"question {index}: fillers did not evict it from L1 "
                    f"(fillers={fillers_sent}/{fillers_needed}, "
                    f"mamba_backup={mamba_backup}, kv_backup={kv_backup})"
                )
            # The write-through D2H for KV pages runs asynchronously; poll
            # until the KV backup counter stops growing.
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
                raise AssertionError(f"question {index}: KV backup did not settle")
            dropped = metric_total(base_url, dropped_name, timeout) - dropped_start
            if dropped > 0:
                raise AssertionError(
                    f"question {index}: {dropped} device tokens were destroyed "
                    f"instead of backed up to Host L2"
                )
            wait_for_metric(base_url, backup_name, backup_start, timeout, "mamba")

            # -- R: replay the identical token IDs immediately --
            load_start = metric_total(base_url, load_name, timeout, "mamba")
            data = generate(base_url, prompt, max_new_tokens, timeout, logprobs=True)
            details = data.get("meta_info", {}).get("cached_tokens_details") or {}
            answer = get_answer_value(data["text"])
            same_answer = (
                answer != INVALID
                and answer == result["rounds"]["first"][index]["answer"]
            )
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
            host = details.get("host", 0)
            device = details.get("device", 0)
            unique_host = host - prefix_boundary
            result["rounds"]["replay"][-1]["unique_host_tokens"] = unique_host
            print(
                f"replay {index + 1}/{n}: correct={answer == expected_answer}, "
                f"host={host} (unique={unique_host}, boundary={prefix_boundary}), "
                f"device={device}",
                flush=True,
            )
            load_after = wait_for_metric(base_url, load_name, load_start, timeout, "mamba")
            mamba_load_delta = load_after - load_start
            mamba_load_total += mamba_load_delta
            result["rounds"]["replay"][-1]["mamba_load_tokens"] = mamba_load_delta
            if host <= 0:
                raise AssertionError(
                    f"question {index} replay missed Host L2 "
                    f"(host=0, device={device}, mamba_load={mamba_load_delta})"
                )
            # A Host hit just beyond the shared prefix could contain only KV
            # pages. Require it to reach the seeded Mamba checkpoint itself.
            if host < seed_len:
                raise AssertionError(
                    f"question {index} Host hit {host} stopped before its own "
                    f"Mamba seed checkpoint {seed_len} (shared boundary "
                    f"{prefix_boundary}, mamba_load={mamba_load_delta})"
                )
            if device != 0:
                raise AssertionError(
                    f"question {index} replay also used {device} Device tokens"
                )
            if not same_answer:
                raise AssertionError(
                    f"question {index} GSM8K answer changed from "
                    f"{result['rounds']['first'][index]['answer']!r} to {answer!r} "
                    "after Host reload"
                )

        result["first_accuracy"] = score_rows(result["rounds"]["first"])
        result["replay_accuracy"] = score_rows(result["rounds"]["replay"])
        result["accuracy_delta"] = result["replay_accuracy"] - result["first_accuracy"]
        result["filler_input_tokens"] = filler_input_tokens
        result["global_fillers_sent"] = global_fillers_sent
        result["mamba_load_tokens_delta"] = mamba_load_total
        result["host_hit_questions"] = sum(
            row["cached_tokens_details"].get("host", 0) > 0
            for row in result["rounds"]["replay"]
        )
        result["unique_host_questions"] = sum(
            row.get("unique_host_tokens", 0) > 0
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
        for first, replay in zip(result["rounds"]["first"], result["rounds"]["replay"]):
            if first["output_token_ids"] != replay["output_token_ids"]:
                print(
                    f"stress: warning: question {replay['index']} hit Host L2 "
                    f"but greedy output diverged "
                    f"({len(first['output_token_ids'])} -> {len(replay['output_token_ids'])} tokens)",
                    flush=True,
                )
        if result["replay_accuracy"] < result["first_accuracy"]:
            raise AssertionError("accuracy dropped after Host reload (stress case)")
        if result["unique_host_questions"] < n:
            raise AssertionError(
                f"only {result['unique_host_questions']}/{n} questions reloaded "
                f"their own pages from Host L2 (beyond shared prefix "
                f"boundary {prefix_boundary})"
            )
        if threshold is not None and result["replay_accuracy"] < threshold:
            raise AssertionError(f"replay accuracy {result['replay_accuracy']:.4f} < threshold {threshold:.4f}")
        result["passed"] = True
        print(
            f"stress: first={result['first_accuracy']:.4f}, "
            f"replay={result['replay_accuracy']:.4f}, "
            f"unique_host_hits={result['unique_host_questions']}/{n} "
            f"(boundary={prefix_boundary}), "
            f"device_hits={result['device_hit_questions']}/{n}, "
            f"fillers={global_fillers_sent}",
            flush=True,
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"stress: {result['error']}", file=sys.stderr)
    finally:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
