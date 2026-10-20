"""Custom long few-shot GSM8K check; not a standard GSM8K score or KL test."""

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import re
import time

import bench_lim_model as model_client

ROOT = pathlib.Path.cwd() / "results"
MODEL_PATH = None
DATA_PATH = None


def prepare():
    from transformers import AutoTokenizer

    path = DATA_PATH
    data = json.loads(path.read_text())
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if len(data) < 72:
        raise ValueError(
            "Need at least 72 rows to keep the 40 demonstrations disjoint."
        )
    demos = data[-40:]
    prefix = "".join(
        "Question: "
        + r["conversations"][0]["value"]
        + "\nAnswer: "
        + r["conversations"][1]["value"]
        + "\n\n"
        for r in demos
    )
    rows = []
    for i, example in enumerate(data[:32]):
        text = (
            prefix + "Question: " + example["conversations"][0]["value"] + "\nAnswer:"
        )
        ids = tok.encode(text, add_special_tokens=False)
        assert 4096 < len(ids) < 32000
        answer = (
            example["conversations"][1]["value"]
            .rsplit("####", 1)[1]
            .strip()
            .replace(",", "")
        )
        rows.append(
            {
                "name": f"gsm8k-{i}",
                "input_ids": ids,
                "expected": answer,
                "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            }
        )
    (ROOT / "quality-prompts.json").write_text(json.dumps(rows))
    manifest = {
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "evaluation_indices": [0, 31],
        "demonstration_indices": [len(data) - 40, len(data) - 1],
        "num_shots": 40,
        "scope": "Custom 40-shot GSM8K subset with disjoint test-set demonstrations; not a standard GSM8K benchmark or a KL consistency test",
        "prompt_lengths": [len(r["input_ids"]) for r in rows],
        "output_limit": 512,
    }
    (ROOT / "quality-plan.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


def evaluate(tag):
    rows = json.loads((ROOT / "quality-prompts.json").read_text())
    started = time.perf_counter()
    out = ROOT / f"{tag}-quality.jsonl"

    def one(row):
        result = model_client.request(
            row, 512, logprobs=True, stop=["Question", "Assistant:", "<|separator|>"]
        )
        numbers = re.findall(
            r"-?\d+(?:\.\d+)?", result["response"].get("text", "").replace(",", "")
        )
        predicted = numbers[-1] if numbers else None
        result.update(
            predicted_answer=predicted, numeric_correct=predicted == row["expected"]
        )
        return result

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(one, rows):
            results.append(result)
            with out.open("a") as file:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                result["case"],
                result["numeric_correct"],
                result["predicted_answer"],
                result["expected"],
                result["actual_output"],
                flush=True,
            )
    summary = {
        "correct": sum(r["numeric_correct"] for r in results),
        "total": len(results),
        "elapsed_s": time.perf_counter() - started,
        "truncated_at_512": sum(r["actual_output"] == 512 for r in results),
        "scope": "Custom 40-shot long-prompt GSM8K subset, not standard benchmark score",
    }
    (ROOT / f"{tag}-quality-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--tag", default="baseline-c4096-ep16")
    parser.add_argument("--workdir", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--model-path", help="Local tokenizer path for preparation.")
    parser.add_argument(
        "--data-path", type=pathlib.Path, help="Local GSM8K conversations JSON."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    args = parser.parse_args()
    ROOT = args.workdir / "results"
    MODEL_PATH, DATA_PATH = args.model_path, args.data_path
    ROOT.mkdir(parents=True, exist_ok=True)
    model_client.BASE = args.base_url.rstrip("/")
    if args.prepare and (MODEL_PATH is None or DATA_PATH is None):
        parser.error("--prepare requires --model-path and --data-path")
    prepare() if args.prepare else evaluate(args.tag)
