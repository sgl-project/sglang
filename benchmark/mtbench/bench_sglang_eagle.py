"""Benchmark EAGLE/EAGLE3 on the full two-turn MT-Bench workload.

The OpenAI chat endpoint makes the server apply the target model's Hugging
Face chat template. ``return_meta_info`` retains speculative-decoding metrics.
"""

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
import uuid
from collections import Counter

import aiohttp
import requests

from sglang.test.test_utils import add_common_sglang_args_and_parse
from sglang.utils import download_and_cache_file

MT_BENCH_COMMIT = "587d5cfa1609a43d192cedb8441cac3c17db105d"
MT_BENCH_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/"
    f"{MT_BENCH_COMMIT}/fastchat/llm_judge/data/mt_bench/question.jsonl"
)
TEMPERATURES = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}


def load_questions(filename, count):
    if count <= 0:
        raise ValueError("--num-questions must be positive")
    with open(filename, "r") as fin:
        questions = [json.loads(line) for line in fin if line.strip()][:count]
    if len(questions) != count:
        raise ValueError(f"Expected {count} questions, found {len(questions)}")

    ids = [question["question_id"] for question in questions]
    if len(ids) != len(set(ids)):
        duplicates = [key for key, value in Counter(ids).items() if value > 1]
        raise ValueError(f"Duplicate MT-Bench question IDs: {duplicates}")
    if any(
        question.get("category") not in TEMPERATURES
        or len(question.get("turns", ())) != 2
        for question in questions
    ):
        raise ValueError("Every selected MT-Bench question must have two valid turns")
    if count == 80 and Counter(q["category"] for q in questions) != Counter(
        {category: 10 for category in TEMPERATURES}
    ):
        raise ValueError("Full MT-Bench must contain 10 questions per category")
    return questions


def question_set_sha256(questions):
    value = "\n".join(
        json.dumps(question, sort_keys=True, separators=(",", ":"))
        for question in questions
    )
    return hashlib.sha256(value.encode()).hexdigest()


def load_answers(filename):
    if not filename:
        return {}
    answers = {}
    with open(os.path.expanduser(filename), "r") as fin:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            choices = obj["choices"]
            choice = choices[0] if isinstance(choices, list) else choices
            turns = choice["turns"]
            if len(turns) != 2:
                raise ValueError(
                    f"Answer for question {obj['question_id']} must have two turns"
                )
            answers[obj["question_id"]] = turns
    if not answers:
        raise ValueError(f"Answer file is empty: {filename}")
    return answers


def write_answers(filename, model, questions, conversations):
    with open(os.path.expanduser(filename), "w") as fout:
        for question, conversation in zip(questions, conversations):
            turns = [turn.get("text", "") for turn in conversation["turns"]]
            value = {
                "question_id": question["question_id"],
                "answer_id": uuid.uuid4().hex,
                "model_id": model,
                "choices": [{"index": 0, "turns": turns}],
                "tstamp": time.time(),
            }
            fout.write(json.dumps(value, ensure_ascii=False) + "\n")


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * fraction
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def aggregate(records):
    successful = [record for record in records if "error" not in record]
    metas = [record.get("meta_info") or {} for record in successful]
    output_tokens = sum(int(meta.get("completion_tokens", 0)) for meta in metas)
    verify_count = sum(int(meta.get("spec_verify_ct", 0)) for meta in metas)
    correct = sum(int(meta.get("spec_num_correct_drafts", 0)) for meta in metas)
    proposed = sum(int(meta.get("spec_num_proposed_drafts", 0)) for meta in metas)
    cap_histogram = Counter()
    for meta in metas:
        for cap_len, count in enumerate(meta.get("spec_cap_lens_histogram") or []):
            cap_histogram[cap_len] += int(count)
    cap_count = sum(cap_histogram.values())
    cap_tokens = sum(cap_len * count for cap_len, count in cap_histogram.items())
    latencies = [record["latency"] for record in successful]
    return {
        "requests": len(records),
        "completed": len(successful),
        "errors": len(records) - len(successful),
        "length_truncated": sum(
            record.get("finish_reason") == "length" for record in successful
        ),
        "prompt_tokens": sum(int(meta.get("prompt_tokens", 0)) for meta in metas),
        "cached_tokens": sum(int(meta.get("cached_tokens", 0)) for meta in metas),
        "completion_tokens": output_tokens,
        "spec_verify_count": verify_count,
        "accept_length": output_tokens / verify_count if verify_count else None,
        "accept_rate": correct / proposed if proposed else None,
        "verify_cap_length": cap_tokens / cap_count if cap_count else None,
        "verify_cap_histogram": dict(sorted(cap_histogram.items())),
        "latency_s": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.5),
            "p90": percentile(latencies, 0.9),
            "p99": percentile(latencies, 0.99),
        },
    }


def summarize(conversations, wall_latency):
    records = [turn for conversation in conversations for turn in conversation["turns"]]
    overall = aggregate(records)
    overall.update(
        wall_latency_s=wall_latency,
        output_throughput=overall["completion_tokens"] / wall_latency,
        conversations_per_second=len(conversations) / wall_latency,
    )
    conversation_latencies = [
        conversation["latency"]
        for conversation in conversations
        if "error" not in conversation
    ]
    overall["conversation_latency_s"] = {
        "mean": (
            statistics.fmean(conversation_latencies) if conversation_latencies else None
        ),
        "p50": percentile(conversation_latencies, 0.5),
        "p90": percentile(conversation_latencies, 0.9),
        "p99": percentile(conversation_latencies, 0.99),
    }
    return {
        "overall": overall,
        "by_turn": {
            str(turn): aggregate(
                [record for record in records if record["turn"] == turn]
            )
            for turn in (1, 2)
        },
        "by_category": {
            category: aggregate(
                [record for record in records if record["category"] == category]
            )
            for category in TEMPERATURES
        },
    }


def compare_answers(conversations, reference):
    if not reference:
        return None
    exact = 0
    mismatches = []
    for conversation in conversations:
        question_id = conversation["question_id"]
        expected = reference[question_id]
        for turn in (1, 2):
            actual = (
                conversation["turns"][turn - 1].get("text")
                if len(conversation["turns"]) >= turn
                else None
            )
            if actual == expected[turn - 1]:
                exact += 1
            elif len(mismatches) < 20:
                mismatches.append({"question_id": question_id, "turn": turn})
    total = len(conversations) * 2
    return {
        "exact_turns": exact,
        "total_turns": total,
        "exact_match_rate": exact / total,
        "mismatches": mismatches,
    }


async def generate_turn(
    session, semaphore, base_url, model, messages, question, turn, args, max_tokens
):
    temperature = (
        TEMPERATURES[question["category"]]
        if args.temperature_policy == "official"
        else 0.0
    )
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 1.0,
        "seed": args.seed,
        "max_completion_tokens": max_tokens,
        "stream": False,
        "ignore_eos": False,
        "return_meta_info": True,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
    }
    async with semaphore:
        tic = time.perf_counter()
        try:
            async with session.post(
                base_url + "/v1/chat/completions", json=body
            ) as response:
                payload = await response.json(content_type=None)
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {payload}")
        except Exception as exc:
            return {
                "question_id": question["question_id"],
                "category": question["category"],
                "turn": turn,
                "latency": time.perf_counter() - tic,
                "error": repr(exc),
            }

    choice = payload["choices"][0]
    return {
        "question_id": question["question_id"],
        "category": question["category"],
        "turn": turn,
        "latency": time.perf_counter() - tic,
        "text": choice["message"].get("content") or "",
        "token_ids": choice.get("token_ids"),
        "finish_reason": choice.get("finish_reason"),
        "meta_info": choice.get("meta_info") or {},
    }


async def run_conversation(
    session,
    semaphore,
    base_url,
    model,
    question,
    frozen_turn_one,
    args,
    max_tokens,
):
    tic = time.perf_counter()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question["turns"][0]},
    ]
    first = await generate_turn(
        session,
        semaphore,
        base_url,
        model,
        messages,
        question,
        1,
        args,
        max_tokens,
    )
    if "error" in first:
        return {
            "question_id": question["question_id"],
            "category": question["category"],
            "latency": time.perf_counter() - tic,
            "turns": [first],
            "error": first["error"],
        }

    assistant = frozen_turn_one.get(question["question_id"], [first["text"]])[0]
    messages.extend(
        [
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": question["turns"][1]},
        ]
    )
    second = await generate_turn(
        session,
        semaphore,
        base_url,
        model,
        messages,
        question,
        2,
        args,
        max_tokens,
    )
    result = {
        "question_id": question["question_id"],
        "category": question["category"],
        "latency": time.perf_counter() - tic,
        "turns": [first, second],
    }
    if "error" in second:
        result["error"] = second["error"]
    return result


async def run_workload(questions, base_url, model, frozen_turn_one, args, max_tokens):
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    semaphore = asyncio.Semaphore(args.parallel)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tic = time.perf_counter()
        conversations = await asyncio.gather(
            *[
                run_conversation(
                    session,
                    semaphore,
                    base_url,
                    model,
                    question,
                    frozen_turn_one,
                    args,
                    max_tokens,
                )
                for question in questions
            ]
        )
        return conversations, time.perf_counter() - tic


def main(args):
    if args.require_exact_match and not args.reference_answer_file:
        raise ValueError("--require-exact-match needs --reference-answer-file")
    question_file = args.question_file or download_and_cache_file(MT_BENCH_URL)
    questions = load_questions(question_file, args.num_questions)
    frozen_turn_one = load_answers(args.frozen_turn_one_file)
    reference = load_answers(args.reference_answer_file)
    selected_ids = {question["question_id"] for question in questions}
    for name, filename, answers in (
        ("frozen turn-one", args.frozen_turn_one_file, frozen_turn_one),
        ("reference", args.reference_answer_file, reference),
    ):
        missing = sorted(selected_ids - answers.keys()) if filename else []
        if missing:
            raise ValueError(f"{name} file is missing question IDs: {missing}")

    base_url = f"http://{args.host}:{args.port}"
    response = requests.get(base_url + "/model_info", timeout=60)
    response.raise_for_status()
    model_info = response.json()
    model = model_info["model_path"]

    if args.warmup_questions:
        warmup, _ = asyncio.run(
            run_workload(
                questions[: args.warmup_questions],
                base_url,
                model,
                frozen_turn_one,
                args,
                min(args.max_new_tokens, 32),
            )
        )
        if any("error" in conversation for conversation in warmup):
            raise RuntimeError("MT-Bench warmup failed")
        requests.post(base_url + "/flush_cache", timeout=60).raise_for_status()

    conversations, wall_latency = asyncio.run(
        run_workload(
            questions,
            base_url,
            model,
            frozen_turn_one,
            args,
            args.max_new_tokens,
        )
    )
    metrics = summarize(conversations, wall_latency)
    parity = compare_answers(conversations, reference)
    result = {
        "task": "mtbench",
        "run_label": args.run_label,
        "backend": args.backend,
        "model": model,
        "tokenizer": model_info.get("tokenizer_path"),
        "num_gpus": args.num_gpus,
        "dataset_commit": MT_BENCH_COMMIT,
        "question_set_sha256": question_set_sha256(questions),
        "num_questions": len(questions),
        "parallel": args.parallel,
        "max_new_tokens": args.max_new_tokens,
        "temperature_policy": args.temperature_policy,
        "enable_thinking": args.enable_thinking,
        "seed": args.seed,
        "system_prompt": "You are a helpful assistant.",
        "frozen_turn_one": bool(args.frozen_turn_one_file),
        "metrics": metrics,
        "parity": parity,
    }
    with open(os.path.expanduser(args.result_file), "a") as fout:
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    answer_file = args.answer_file or f"tmp_output_{args.backend}.jsonl"
    write_answers(answer_file, model, questions, conversations)
    if args.raw_result_file:
        with open(os.path.expanduser(args.raw_result_file), "w") as fout:
            for conversation in conversations:
                fout.write(json.dumps(conversation, ensure_ascii=False) + "\n")

    overall = metrics["overall"]
    accept = overall["accept_length"]
    accept_text = f"{accept:.3f}" if accept is not None else "n/a"
    print(
        f"{len(questions)} questions, {overall['output_throughput']:.2f} token/s, "
        f"accept length {accept_text}, {overall['errors']} errors"
    )
    if overall["errors"]:
        raise RuntimeError(f"{overall['errors']} MT-Bench turns failed")
    if (
        args.require_exact_match
        and parity
        and parity["exact_turns"] != parity["total_turns"]
    ):
        raise RuntimeError(
            f"Exact parity failed: {parity['exact_turns']}/"
            f"{parity['total_turns']} turns"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--answer-file", type=str)
    parser.add_argument("--reference-answer-file", type=str)
    parser.add_argument("--frozen-turn-one-file", type=str)
    parser.add_argument("--require-exact-match", action="store_true")
    parser.add_argument("--num-questions", type=int, default=80)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--warmup-questions", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", type=str)
    parser.add_argument(
        "--temperature-policy",
        choices=("greedy", "official"),
        default="greedy",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    args = add_common_sglang_args_and_parse(parser)
    main(args)
