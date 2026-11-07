"""Checkpoint-backed smoke, repeatability and serving performance client.

Generate identical token-ID fixtures once, then point the same client at each
server mode. Use a mode/configuration-specific --tag to retain raw results.
The generated needle checks are smoke coverage, not a general quality metric.
Performance includes serving/scheduling/prefill; profiling runs stay separate."""

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import statistics
import time
import urllib.request

ROOT = pathlib.Path.cwd()
MODEL_PATH = None
BASE = "http://127.0.0.1:30000"


def prepare():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    rows = []
    for length in (512, 8192, 32768):
        for variant in range(4):
            needle = f"仓库 ZX-{variant + 17} 的校验码为 LIM{7319 + variant * 113}。"
            lines = "".join(
                f"记录{i}: 普通物料批次，数量{(i * 73 + variant * 19) % 997}，日期2026年10月。\n"
                for i in range(5000)
            )
            start = tok.encode(
                "以下是仓库清单。请阅读全部记录。\n", add_special_tokens=False
            )
            middle = tok.encode(lines, add_special_tokens=False)
            fact = tok.encode(
                "\n唯一关键记录：" + needle + "\n", add_special_tokens=False
            )
            suffix = f"\n根据上文，仓库 ZX-{variant + 17} 的校验码是什么？只输出校验码，不要解释。"
            # Keep the exact target token count and the original chat template.
            base_messages = [{"role": "user", "content": "PLACEHOLDER" + suffix}]
            template = tok.apply_chat_template(
                base_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            before, after = template.split("PLACEHOLDER", 1)
            a, z = (
                tok.encode(before, add_special_tokens=False),
                tok.encode(after, add_special_tokens=False),
            )
            room = length - len(a) - len(z) - len(start) - len(fact)
            assert room > 0
            split = room // 2
            body = start + middle[:split] + fact + middle[split:room]
            ids = a + body + z
            assert len(ids) == length, (len(ids), length)
            rows.append(
                {
                    "name": f"L{length}-v{variant}",
                    "input_ids": ids,
                    "expected": f"LIM{7319 + variant * 113}",
                    "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                }
            )
    (ROOT / "results/prompts.json").write_text(json.dumps(rows, ensure_ascii=False))
    print(
        json.dumps(
            [{k: v for k, v in row.items() if k != "input_ids"} for row in rows],
            ensure_ascii=False,
        ),
        flush=True,
    )


def request(row, output, logprobs=False, ignore_eos=False, stop=None):
    started_unix = time.time()
    body = {
        "input_ids": row["input_ids"],
        "stream": True,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output,
            "ignore_eos": ignore_eos,
        },
        "return_logprob": logprobs,
        "logprob_start_len": -1,
    }
    if logprobs:
        body["top_logprobs_num"] = 5
    if stop:
        body["sampling_params"]["stop"] = stop
    req = urllib.request.Request(
        BASE + "/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    times, counts, final = [], [], None
    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            value = raw[6:].strip()
            if value == b"[DONE]":
                break
            data = json.loads(value)
            if "error" in data:
                raise RuntimeError(data)
            final = data
            count = data.get("meta_info", {}).get("completion_tokens", 0)
            if not counts or count > counts[-1]:
                times.append(time.perf_counter() - t0)
                counts.append(count)
    elapsed = time.perf_counter() - t0
    if final is None:
        raise RuntimeError("Server returned no generation payload")
    n = final.get("meta_info", {}).get("completion_tokens", 0)
    if ignore_eos and n != output:
        raise RuntimeError(f"Expected {output} completion tokens, got {n}.")
    return {
        "case": row["name"],
        "input_length": len(row["input_ids"]),
        "prompt_sha256": row["sha256"],
        "requested_output": output,
        "started_unix": started_unix,
        "finished_unix": time.time(),
        "actual_output": n,
        "elapsed_s": elapsed,
        "ttft_s": times[0] if times else None,
        "tpot_s": (times[-1] - times[0]) / (counts[-1] - counts[0])
        if len(times) > 1 and counts[-1] > counts[0]
        else None,
        "arrival_s": times,
        "completion_counts": counts,
        "expected": row["expected"],
        "answer_contains_expected": row["expected"] in final.get("text", ""),
        "response": final,
    }


def run(args):
    rows = json.loads((ROOT / "results/prompts.json").read_text())
    out = ROOT / "results" / f"{args.tag}-{args.phase}.jsonl"
    selected = [
        r for r in rows if (args.length == 0 or len(r["input_ids"]) == args.length)
    ]
    if not selected:
        raise ValueError("No prepared prompts match --length.")
    if args.phase == "smoke":
        selected = [r for r in selected if r["name"] in ("L512-v0", "L8192-v0")]
        for row in selected:
            result = request(row, 64, logprobs=True)
            with out.open("a") as file:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in result.items()
                        if k not in ("response", "arrival_s", "completion_counts")
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            print(result["response"].get("text", "")[:800], flush=True)
    elif args.phase == "correctness":
        for repeat in range(2):
            for row in selected:
                result = request(row, args.output, logprobs=True)
                result["repeat"] = repeat
                with out.open("a") as file:
                    file.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(
                    row["name"],
                    repeat,
                    result["actual_output"],
                    result["answer_contains_expected"],
                    flush=True,
                )
    elif args.phase == "perf":
        group = selected[: args.concurrency]
        assert len(group) == args.concurrency
        for repeat in range(args.rounds + 1):
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency
            ) as pool:
                result = list(
                    pool.map(
                        lambda row: request(row, args.output, ignore_eos=True), group
                    )
                )
            elapsed = time.perf_counter() - t0
            record = {
                "repeat": repeat,
                "warmup": repeat == 0,
                "concurrency": args.concurrency,
                "wall_s": elapsed,
                "output_tokens_per_s": sum(r["actual_output"] for r in result)
                / elapsed,
                "median_tpot_s": statistics.median(r["tpot_s"] for r in result),
                "median_ttft_s": statistics.median(r["ttft_s"] for r in result),
                "requests": result,
            }
            with out.open("a") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                json.dumps({k: v for k, v in record.items() if k != "requests"}),
                flush=True,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase", choices=["prepare", "smoke", "correctness", "perf"], required=True
    )
    p.add_argument("--tag", default="lim-c4096")
    p.add_argument("--length", type=int, default=8192)
    p.add_argument("--output", type=int, default=128)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--workdir", type=pathlib.Path, default=pathlib.Path.cwd())
    p.add_argument("--model-path", help="Local tokenizer path, required for prepare.")
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    args = p.parse_args()
    if args.rounds < 1 or args.output < 2 or not 1 <= args.concurrency <= 4:
        p.error("Require rounds >= 1, output >= 2, and concurrency in [1,4].")
    ROOT, MODEL_PATH, BASE = args.workdir, args.model_path, args.base_url.rstrip("/")
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    if args.phase == "prepare" and MODEL_PATH is None:
        p.error("--model-path is required for --phase prepare")
    prepare() if args.phase == "prepare" else run(args)
