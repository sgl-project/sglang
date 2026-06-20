#!/usr/bin/env python3
import argparse
import ast
import json
import os
import random
import re
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import tiktoken
from transformers import AutoTokenizer
from sglang.test.simple_eval_common import (
    ANSWER_PATTERN_MULTICHOICE,
    format_multichoice_question,
)
from sglang.utils import download_and_cache_file, read_jsonl


MMLU_CHOICES = ["A", "B", "C", "D"]
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
INVALID = -9999999


def load_dsv4_encoder(model_path: str):
    encoding_dir = Path(model_path) / "encoding"
    if not (encoding_dir / "encoding_dsv4.py").exists():
        raise FileNotFoundError(f"DeepSeek-V4 encoder not found under {encoding_dir}")
    import sys

    sys.path.insert(0, str(encoding_dir))
    import encoding_dsv4  # type: ignore

    return encoding_dsv4


def encode_user_prompt(encoder, prompt: str, thinking_mode: str) -> str:
    if encoder is None:
        raise RuntimeError("DSV4 encoder is required for chat/thinking prompt mode")
    return encoder.encode_messages(
        [{"role": "user", "content": prompt}],
        thinking_mode=thinking_mode,
        drop_thinking=True,
    )


def post_generate(
    session: requests.Session,
    url: str,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str],
    timeout: int,
    regex: str | None = None,
) -> list[dict[str, Any]]:
    sampling_params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "stop": stop,
    }
    if regex is not None:
        sampling_params["regex"] = regex
    payload = {
        "text": prompts,
        "sampling_params": sampling_params,
    }
    last_error = None
    for attempt in range(6):
        try:
            resp = session.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return [data]
            return data
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"/generate failed after retries: {last_error}")


def post_generate_logprob(
    session: requests.Session,
    url: str,
    prompts: list[str],
    token_ids: list[int],
    timeout: int,
) -> list[dict[str, Any]]:
    payload = {
        "text": prompts,
        "sampling_params": {
            "temperature": 0,
            "top_p": 1.0,
            "max_new_tokens": 1,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
        "token_ids_logprob": token_ids,
    }
    last_error = None
    for attempt in range(6):
        try:
            resp = session.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return [data]
            return data
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"/generate logprob failed after retries: {last_error}")


def run_batches(
    prompts: list[str],
    args: argparse.Namespace,
    encoder,
    max_new_tokens: int,
) -> tuple[list[str], list[dict[str, Any]], float]:
    if encoder is None:
        encoded = prompts
        stop: list[str] = []
    else:
        encoded = [encode_user_prompt(encoder, p, args.thinking_mode) for p in prompts]
        stop = [encoder.eos_token]
    url = f"http://{args.host}:{args.port}/generate"
    outputs: list[str] = []
    metas: list[dict[str, Any]] = []
    tic = time.perf_counter()
    with requests.Session() as session:
        for start in range(0, len(encoded), args.batch_size):
            batch = encoded[start : start + args.batch_size]
            data = post_generate(
                session,
                url,
                batch,
                max_new_tokens=max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stop=stop,
                timeout=args.timeout,
            )
            if len(data) != len(batch):
                raise RuntimeError(
                    f"Expected {len(batch)} responses, got {len(data)} at offset {start}"
                )
            outputs.extend((x.get("text") or "") for x in data)
            metas.extend((x.get("meta_info") or {}) for x in data)
            print(
                f"batch {start + len(batch)}/{len(encoded)} done",
                flush=True,
            )
    return outputs, metas, time.perf_counter() - tic


def run_logprob_batches(
    prompts: list[str],
    args: argparse.Namespace,
    token_ids: list[int],
) -> tuple[list[str], list[dict[str, Any]], list[list[float]], float]:
    url = f"http://{args.host}:{args.port}/generate"
    outputs: list[str] = []
    metas: list[dict[str, Any]] = []
    all_scores: list[list[float]] = []
    tic = time.perf_counter()
    with requests.Session() as session:
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            data = post_generate_logprob(
                session,
                url,
                batch,
                token_ids=token_ids,
                timeout=args.timeout,
            )
            if len(data) != len(batch):
                raise RuntimeError(
                    f"Expected {len(batch)} responses, got {len(data)} at offset {start}"
                )
            for item in data:
                meta = item.get("meta_info") or {}
                scores = extract_requested_output_logprobs(meta, len(token_ids))
                outputs.append(item.get("text") or "")
                metas.append(meta)
                all_scores.append(scores)
            print(f"batch {start + len(batch)}/{len(prompts)} done", flush=True)
    return outputs, metas, all_scores, time.perf_counter() - tic


def extract_requested_output_logprobs(
    meta: dict[str, Any], expected_len: int
) -> list[float]:
    rows = meta.get("output_token_ids_logprobs")
    if not rows:
        raise RuntimeError(f"Missing output_token_ids_logprobs in meta_info: {meta}")
    first_pos = rows[0]
    if len(first_pos) < expected_len:
        raise RuntimeError(
            f"Expected {expected_len} token-id logprobs, got {len(first_pos)}: {first_pos}"
        )

    scores: list[float] = []
    for entry in first_pos[:expected_len]:
        if isinstance(entry, (list, tuple)):
            scores.append(float(entry[0]))
        else:
            scores.append(float(entry))
    return scores


def choice_token_ids(
    model_path: str, choices: list[str], prefix: str
) -> tuple[list[str], list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if prefix == "space":
        candidate_groups = [[" " + x for x in choices]]
    elif prefix == "none":
        candidate_groups = [choices]
    else:
        candidate_groups = [[" " + x for x in choices], choices]
    for candidate_choices in candidate_groups:
        ids: list[int] = []
        ok = True
        for choice in candidate_choices:
            tokenized = tokenizer.encode(choice, add_special_tokens=False)
            if len(tokenized) != 1:
                ok = False
                break
            ids.append(int(tokenized[0]))
        if ok:
            return candidate_choices, ids
    raise RuntimeError(
        "Could not find single-token A/B/C/D choices with or without leading spaces"
    )


def mmlu_format_subject(subject: str) -> str:
    return " " + " ".join(subject.split("_"))


def mmlu_format_example(df: pd.DataFrame, idx: int, include_answer: bool) -> str:
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += f"\n{MMLU_CHOICES[j]}. {df.iloc[idx, j + 1]}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {df.iloc[idx, k + 1]}\n\n"
    return prompt


def mmlu_gen_prompt(train_df: pd.DataFrame, subject: str, k: int) -> str:
    prompt = (
        "The following are multiple choice questions (with answers) about"
        f"{mmlu_format_subject(subject)}.\n\n"
    )
    for i in range(k):
        prompt += mmlu_format_example(train_df, i, include_answer=True)
    return prompt


def download_mmlu_data(data_dir: str) -> None:
    if os.path.isdir(os.path.join(data_dir, "test")):
        return
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "data.tar")
    subprocess.check_call(
        ["wget", "-O", tar_path, "https://people.eecs.berkeley.edu/~hendrycks/data.tar"]
    )
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=data_dir, filter="data")
    nested = os.path.join(data_dir, "data")
    if os.path.isdir(nested):
        for item in os.listdir(nested):
            os.rename(os.path.join(nested, item), os.path.join(data_dir, item))
        os.rmdir(nested)
    os.remove(tar_path)


def extract_choice(text: str) -> str | None:
    match = re.search(ANSWER_PATTERN_MULTICHOICE, text)
    if match:
        return match.group(1).upper()
    match = re.search(r"(?i)\b([ABCD])\b", text.strip())
    return match.group(1).upper() if match else None


def run_mmlu(args: argparse.Namespace, encoder) -> dict[str, Any]:
    download_mmlu_data(args.mmlu_data_dir)
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    subjects = sorted(
        f.split("_test.csv")[0]
        for f in os.listdir(os.path.join(args.mmlu_data_dir, "test"))
        if f.endswith("_test.csv")
    )
    if args.nsub is not None:
        subjects = subjects[: args.nsub]

    prompts: list[str] = []
    labels: list[str] = []
    subject_ranges: list[tuple[str, int, int]] = []
    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.mmlu_data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.mmlu_ntrain]
        test_df = pd.read_csv(
            os.path.join(args.mmlu_data_dir, "test", subject + "_test.csv"), header=None
        )
        k = args.mmlu_ntrain
        few_shot = mmlu_gen_prompt(dev_df, subject, k)
        while k > 0 and len(tokenizer.encode(few_shot)) > 1536:
            k -= 1
            few_shot = mmlu_gen_prompt(dev_df, subject, k)

        begin = len(prompts)
        for i in range(test_df.shape[0]):
            prompts.append(few_shot + mmlu_format_example(test_df, i, False))
            labels.append(test_df.iloc[i, test_df.shape[1] - 1])
        subject_ranges.append((subject, begin, len(prompts)))

    if args.num_examples is not None:
        prompts = prompts[: args.num_examples]
        labels = labels[: args.num_examples]

    if args.mmlu_prompt_mode == "dsv4-chat":
        prompts_for_request = [
            encode_user_prompt(encoder, p, args.thinking_mode) for p in prompts
        ]
    else:
        prompts_for_request = prompts

    if args.mmlu_scoring == "logprob":
        choice_strings, token_ids = choice_token_ids(
            args.model_path, MMLU_CHOICES, args.mmlu_choice_prefix
        )
        print(
            "MMLU_LOGPROB_CHOICES",
            json.dumps(dict(zip(choice_strings, token_ids)), sort_keys=True),
            flush=True,
        )
        outputs, metas, logprob_scores, latency = run_logprob_batches(
            prompts_for_request, args, token_ids
        )
        preds = [MMLU_CHOICES[int(np.argmax(x))] for x in logprob_scores]
        for meta, scores_by_choice in zip(metas, logprob_scores):
            meta["mmlu_choice_logprobs"] = dict(zip(MMLU_CHOICES, scores_by_choice))
    else:
        url = f"http://{args.host}:{args.port}/generate"
        outputs = []
        metas = []
        tic = time.perf_counter()
        with requests.Session() as session:
            for start in range(0, len(prompts_for_request), args.batch_size):
                batch = prompts_for_request[start : start + args.batch_size]
                data = post_generate(
                    session,
                    url,
                    batch,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop=[encoder.eos_token] if args.mmlu_prompt_mode == "dsv4-chat" else [],
                    timeout=args.timeout,
                    regex=r" ?[ABCD]" if args.mmlu_scoring == "regex" else None,
                )
                outputs.extend((x.get("text") or "") for x in data)
                metas.extend((x.get("meta_info") or {}) for x in data)
                print(
                    f"batch {start + len(batch)}/{len(prompts_for_request)} done",
                    flush=True,
                )
        latency = time.perf_counter() - tic
        preds = [extract_choice(x) for x in outputs]
    scores = [float(p == y) for p, y in zip(preds, labels)]
    for subject, begin, end in subject_ranges:
        if begin >= len(scores):
            continue
        end = min(end, len(scores))
        print(f"subject: {subject}, #q:{end - begin}, acc: {np.mean(scores[begin:end]):.3f}")

    return finish_result(args, "mmlu", prompts, outputs, metas, preds, labels, scores, latency)


def gsm8k_get_one_example(lines: list[dict[str, Any]], i: int, include_answer: bool) -> str:
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def gsm8k_get_answer_value(answer_str: str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", answer_str)
    if not numbers:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID


def run_gsm8k(args: argparse.Namespace, encoder) -> dict[str, Any]:
    data_path = args.gsm8k_data_path or download_and_cache_file(GSM8K_URL)
    lines = list(read_jsonl(data_path))
    few_shot = "".join(
        gsm8k_get_one_example(lines, i, True) + "\n\n"
        for i in range(args.num_shots)
    )
    eval_lines = lines[args.num_shots :]
    if args.num_examples is not None:
        eval_lines = eval_lines[: args.num_examples]
    prompts = [few_shot + "Question: " + x["question"] + "\nAnswer:" for x in eval_lines]
    labels = [gsm8k_get_answer_value(x["answer"]) for x in eval_lines]

    outputs, metas, latency = run_batches(prompts, args, encoder, args.max_new_tokens)
    preds = [gsm8k_get_answer_value(x) for x in outputs]
    scores = [float(p == y) for p, y in zip(preds, labels)]
    invalid = float(np.mean([p == INVALID for p in preds])) if preds else 0.0
    return finish_result(
        args, "gsm8k", prompts, outputs, metas, preds, labels, scores, latency, invalid
    )


def run_gpqa(args: argparse.Namespace, encoder) -> dict[str, Any]:
    df = pd.read_csv(
        "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv",
        storage_options={"timeout": 30},
    )
    examples = [row.to_dict() for _, row in df.iterrows()]
    rng = random.Random(0)
    if args.num_examples is not None:
        examples = rng.sample(examples, min(args.num_examples, len(examples)))
    examples = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]

    prompts: list[str] = []
    labels: list[str] = []
    for row in examples:
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        choices = [choices[i] for i in row["permutation"]]
        correct_index = choices.index(row["Correct Answer"])
        labels.append("ABCD"[correct_index])
        prompts.append(
            format_multichoice_question(
                {
                    "A": choices[0],
                    "B": choices[1],
                    "C": choices[2],
                    "D": choices[3],
                    "Question": row["Question"],
                }
            )
        )

    outputs, metas, latency = run_batches(prompts, args, encoder, args.max_new_tokens)
    preds = [extract_choice(x) for x in outputs]
    scores = [float(p == y) for p, y in zip(preds, labels)]
    invalid = float(np.mean([p is None for p in preds])) if preds else 0.0
    return finish_result(
        args, "gpqa", prompts, outputs, metas, preds, labels, scores, latency, invalid
    )


def finish_result(
    args: argparse.Namespace,
    task: str,
    prompts: list[str],
    outputs: list[str],
    metas: list[dict[str, Any]],
    preds: list[Any],
    labels: list[Any],
    scores: list[float],
    latency: float,
    invalid: float | None = None,
) -> dict[str, Any]:
    output_tokens = sum(int(m.get("completion_tokens") or 0) for m in metas)
    result = {
        "task": task,
        "backend": "srt-native-dsv4-encoder",
        "thinking_mode": args.thinking_mode,
        "latency": round(latency, 3),
        "accuracy": round(float(np.mean(scores)) if scores else 0.0, 3),
        "score": round(float(np.mean(scores)) if scores else 0.0, 3),
        "num_requests": len(prompts),
        "output_throughput": round(output_tokens / latency, 3) if latency else 0.0,
        "other": {
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "mmlu_prompt_mode": getattr(args, "mmlu_prompt_mode", None),
            "mmlu_scoring": getattr(args, "mmlu_scoring", None),
            "mmlu_choice_prefix": getattr(args, "mmlu_choice_prefix", None),
            "prompt_mode": getattr(args, "prompt_mode", None),
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    }
    if invalid is not None:
        result["invalid"] = round(invalid, 3)

    raw_path = Path(args.raw_result_file)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w") as fout:
        for prompt, output, pred, label, meta in zip(prompts, outputs, preds, labels, metas):
            fout.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "output": output,
                        "pred": pred,
                        "label": label,
                        "meta_info": meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    result_path = Path(args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.suffix == ".jsonl":
        with result_path.open("a") as fout:
            fout.write(json.dumps(result) + "\n")
    else:
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    print(f"Accuracy: {result['accuracy']:.3f}")
    if invalid is not None:
        print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {result['output_throughput']:.3f} token/s")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["mmlu", "gsm8k", "gpqa"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--thinking-mode", choices=["chat", "thinking"], default="chat")
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--raw-result-file", required=True)
    parser.add_argument("--mmlu-data-dir", default="benchmark/mmlu/data")
    parser.add_argument("--mmlu-ntrain", type=int, default=5)
    parser.add_argument("--mmlu-prompt-mode", choices=["raw", "dsv4-chat"], default="raw")
    parser.add_argument(
        "--mmlu-scoring",
        choices=["logprob", "regex", "generate"],
        default="regex",
    )
    parser.add_argument("--mmlu-choice-prefix", choices=["auto", "space", "none"], default="auto")
    parser.add_argument("--prompt-mode", choices=["raw", "dsv4-chat"], default="dsv4-chat")
    parser.add_argument("--nsub", type=int)
    parser.add_argument("--num-shots", type=int, default=8)
    parser.add_argument("--gsm8k-data-path")
    args = parser.parse_args()

    needs_encoder = not (
        (args.task == "mmlu" and args.mmlu_prompt_mode == "raw")
        or (args.task in {"gsm8k", "gpqa"} and args.prompt_mode == "raw")
    )
    encoder = None
    if needs_encoder:
        encoder = load_dsv4_encoder(args.model_path)
    if args.task == "mmlu":
        run_mmlu(args, encoder)
    elif args.task == "gsm8k":
        run_gsm8k(args, encoder)
    else:
        run_gpqa(args, encoder)


if __name__ == "__main__":
    main()
