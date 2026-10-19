"""Replay three TweetEval failures against a running DiffusionGemma server.

Exit 0 only when all three requests return valid decisions. The current decoder
returns HTTP 400 for all three and exits 1. See README_diffusiongemma_prefix.md.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import httpx

MODEL = "google/diffusiongemma-26B-A4B-it"
FILE_HASHES = {
    "manifest.json": "06ba9d34680703e0fb8b17841b7c313e40dcde18ddeff9685898a0dbba158e83",
    "test.jsonl": "013a8bee08cd7661b7215a07f9832b5d070dc9e1a5e2a1144cfe3fb66a679591",
}
CASE_IDS = (
    "6e64a2415fdf017af9dc16f1efda36ea560ab62799c6218c696fea5171d58dde",
    "2e1ab45c412a4f46cca6adcecdf8eb8bac2a558623e54c3917f8d6f8baa276b3",
    "f950a583343b1b72d1aa9787a0a982f4e093c70d01767acab53cbb42cef3bb2c",
)


def load_cases(data_dir):
    contents = {}
    for name, expected in FILE_HASHES.items():
        raw = (data_dir / name).read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise ValueError(f"{name}: expected SHA256 {expected}, got {actual}")
        contents[name] = raw.decode("utf-8")
    manifest = json.loads(contents["manifest.json"])
    rows = {
        row["id"]: row
        for line in contents["test.jsonl"].splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    return manifest, [rows[case_id] for case_id in CASE_IDS]


def validate_decision(data, labels):
    answer = data["answers"]["decision"]
    probabilities = answer["probabilities"]
    if answer["type"] != "choice" or set(probabilities) != set(labels):
        raise ValueError("Unexpected answer type or labels")
    values = list(probabilities.values())
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in values):
        raise ValueError("Invalid probabilities")
    if not math.isclose(sum(values), 1, abs_tol=1e-5):
        raise ValueError("Probabilities do not sum to 1")
    if answer["choice"] != max(probabilities, key=probabilities.get):
        raise ValueError("Choice does not match the highest probability")
    if data["usage"]["output_tokens"] != 0:
        raise ValueError("Decision scoring must report zero output tokens")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest, rows = load_cases(args.data_dir)
    results = []
    with httpx.Client(base_url=args.base_url, timeout=300) as client:
        for row in rows:
            body = {
                "model": args.model,
                "state": row["text"],
                "questions": {"decision": manifest["question"]},
            }
            result = {"id": row["id"], "request": body, "passed": False}
            try:
                response = client.post("/v1/systemone", json=body)
                result.update(status=response.status_code, response=response.text)
                response.raise_for_status()
                validate_decision(response.json(), manifest["labels"])
                result["passed"] = True
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                result["error"] = str(exc)
            results.append(result)
            print(f"{row['id']}: {'PASS' if result['passed'] else 'FAIL'}")
    report = {
        "model": args.model,
        "base_url": args.base_url,
        "dataset_sha256": FILE_HASHES,
        "passed": all(result["passed"] for result in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
