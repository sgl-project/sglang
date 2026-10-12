"""Reproduce PR #41913's raw-completion GSM8K protocol (20 training shots)."""

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

import requests

_DATA_SHA256 = {
    "train": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
    "test": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
}


def load_data(split, directory):
    path = directory / f"gsm8k_{split}.jsonl"
    if not path.exists():
        url = (
            "https://raw.githubusercontent.com/openai/grade-school-math/"
            f"master/grade_school_math/data/{split}.jsonl"
        )
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != _DATA_SHA256[split]:
            raise ValueError(
                f"Unexpected GSM8K {split} data; historical checksum differs"
            )
        path.write_bytes(data)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != _DATA_SHA256[split]:
        raise ValueError(f"Unexpected GSM8K {split} data; historical checksum differs")
    return [json.loads(line) for line in data.splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--questions", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test = load_data("test", args.output_dir)[: args.questions]
    shots = load_data("train", args.output_dir)[:20]
    prefix = (
        "\n\n".join(
            "Question: " + s["question"] + "\nAnswer: " + s["answer"] for s in shots
        )
        + "\n\n"
    )
    url = f"http://127.0.0.1:{args.port}/generate"

    def answer(text):
        numbers = re.findall(r"\d+", text.replace(",", ""))
        return int(numbers[-1]) if numbers else None

    def work(item):
        index, sample = item
        payload = {
            "text": prefix + "Question: " + sample["question"] + "\nAnswer:",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 512,
                "stop": ["Question", "Assistant:", "<|separator|>"],
            },
        }
        response = requests.post(url, json=payload, timeout=1800)
        response.raise_for_status()
        output = response.json()
        expected = answer(sample["answer"])
        predicted = answer(output["text"])
        return {
            "index": index,
            "expected": expected,
            "predicted": predicted,
            "correct": expected == predicted,
            **output,
        }

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        outputs = list(executor.map(work, enumerate(test)))
    seconds = time.perf_counter() - start
    summary = {
        "tag": args.tag,
        "questions": len(test),
        "shots": len(shots),
        "shots_source": "train",
        "accuracy": sum(o["correct"] for o in outputs) / len(outputs),
        "invalid": sum(o["predicted"] is None for o in outputs) / len(outputs),
        "seconds": seconds,
        "output_tokens_per_second": sum(
            o["meta_info"]["completion_tokens"] for o in outputs
        )
        / seconds,
    }
    (args.output_dir / f"{args.tag}_gsm8k_raw.json").write_text(
        json.dumps(outputs, indent=2)
    )
    (args.output_dir / f"{args.tag}_gsm8k.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("GSM8K_RESULT", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
