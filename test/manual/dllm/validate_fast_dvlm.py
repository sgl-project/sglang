"""Capture and compare Fast-dVLM GPU runs; see README.md in this directory."""

import argparse
import importlib.metadata
import json
import re
import subprocess
import time
from pathlib import Path

MODEL = "Efficient-Large-Model/Fast_dVLM_3B"
BLOCK = 32
SCHEMA = 1
PREFIX = (
    "You are a careful assistant. Answer in plain sentences. "
    "Context: a warehouse ships small, medium and large parcels. "
    "Small parcels cost 4 dollars, medium cost 9 dollars and large cost 15 dollars. "
    "On Monday it shipped 12 small, 7 medium and 3 large parcels. "
    "On Tuesday it shipped 5 small, 11 medium and 6 large parcels. "
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def token_ids(value):
    return isinstance(value, list) and all(type(x) is int and x >= 0 for x in value)


def records_by_id(data):
    if data.get("schema") != SCHEMA:
        raise ValueError("Unsupported artifact schema")
    rows = data.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty records list")
    indexed = {}
    for row in rows:
        rid = row.get("id")
        if not isinstance(rid, str) or not rid or rid in indexed:
            raise ValueError("Record IDs must be nonempty and unique")
        if not token_ids(row.get("input_ids")) or not row["input_ids"]:
            raise ValueError(f"{rid}: missing or invalid input_ids")
        indexed[rid] = row
    return indexed


def prepare(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    records = []
    questions = [
        "How much revenue did Monday bring in?",
        "How much revenue did Tuesday bring in?",
        "Which day shipped more parcels?",
        "Write a short poem about parcels.",
    ]
    targets = [(f"aligned-{i}", question, 0) for i, question in enumerate(questions)]
    targets += [(f"remainder-{r:02d}", questions[0], r) for r in range(1, BLOCK)]
    for rid, question, remainder in targets:
        for padding in range(4 * BLOCK):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": PREFIX + question + " ." * padding}],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) % BLOCK == remainder:
                records.append({"id": rid, "input_ids": list(ids)})
                break
        else:
            raise ValueError(f"Could not construct prompt remainder {remainder}")
    write_json(
        args.out,
        {
            "schema": SCHEMA,
            "model": MODEL,
            "revision": args.revision,
            "records": records,
        },
    )
    print(f"Wrote {len(records)} prompts covering all {BLOCK} block remainders")


def run(args):
    import torch

    import sglang as sgl

    manifest = read_json(args.prompts)
    records = list(records_by_id(manifest).values())
    if args.aligned_only:
        records = [r for r in records if len(r["input_ids"]) % BLOCK == 0]
    if not records:
        raise ValueError("No selected prompts")
    if args.fork and (args.radix != "off" or args.max_running != 1 or args.batched):
        raise ValueError(
            "The reference fork requires radix off and one running request"
        )
    if manifest.get("model") != MODEL or not re.fullmatch(
        r"[0-9a-f]{40}", manifest.get("revision", "")
    ):
        raise ValueError(
            "Use a manifest with the Fast-dVLM model and a pinned revision"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("GPU capture requires a CUDA environment")

    kwargs = {
        "model_path": MODEL,
        "revision": manifest["revision"],
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "attention_backend": "flashinfer",
        "mem_fraction_static": args.mem_fraction_static,
        "max_running_requests": args.max_running,
        "dllm_algorithm": "HierarchyBlock",
    }
    if args.fork:
        kwargs["chunked_prefill_size"] = 16384
        if args.eager:
            kwargs["disable_cuda_graph"] = True
    else:
        kwargs["disable_radix_cache"] = args.radix == "off"
        kwargs["disable_decode_cuda_graph"] = args.eager

    sampling = {"temperature": 0, "max_new_tokens": args.max_new_tokens}
    engine = sgl.Engine(**kwargs)
    try:
        # Warm kernels/graphs and prime the shared prefix for cache-enabled runs.
        engine.generate(input_ids=records[0]["input_ids"], sampling_params=sampling)
        start = time.perf_counter()
        if args.batched:
            outputs = engine.generate(
                input_ids=[r["input_ids"] for r in records], sampling_params=sampling
            )
        else:
            outputs = [
                engine.generate(input_ids=r["input_ids"], sampling_params=sampling)
                for r in records
            ]
        elapsed = time.perf_counter() - start
    finally:
        engine.shutdown()
    if not isinstance(outputs, list) or len(outputs) != len(records):
        raise ValueError("Engine did not return one result per prompt")
    captured = []
    for record, output in zip(records, outputs):
        ids = output.get("output_ids")
        if not token_ids(ids) or not ids:
            raise ValueError(f"{record['id']}: missing or empty output_ids")
        meta = output.get("meta_info", {})
        if meta.get("finish_reason", {}).get("type") not in ("length", "stop"):
            raise ValueError(f"{record['id']}: unexpected finish_reason")
        captured.append(
            {
                **record,
                "output_ids": ids,
                "cached_tokens": meta.get("cached_tokens"),
                "finish_reason": meta["finish_reason"],
            }
        )
    packages = {}
    for name in ("sglang", "torch", "flashinfer-python", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    data = {
        "schema": SCHEMA,
        "model": MODEL,
        "revision": manifest["revision"],
        "sampling": sampling,
        "engine": kwargs,
        "batched": args.batched,
        "fork": args.fork,
        "environment": {
            "packages": packages,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "cwd_git_head": git.stdout.strip() if git.returncode == 0 else None,
        },
        "elapsed_s": elapsed,
        "output_tok_per_s": sum(len(r["output_ids"]) for r in captured) / elapsed,
        "records": captured,
    }
    write_json(args.out, data)
    print(f"Wrote {len(captured)} results to {args.out}")


def compare(left, right, require_cache_hit=False):
    a, b = records_by_id(left), records_by_id(right)
    for key in ("model", "revision", "sampling"):
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError(f"Missing or different {key}")
    if set(a) != set(b):
        raise ValueError("Result sets contain different prompt IDs")
    differences = []
    for rid, ra in a.items():
        rb = b[rid]
        if ra["input_ids"] != rb["input_ids"]:
            raise ValueError(f"{rid}: input token IDs differ")
        for row in (ra, rb):
            if not token_ids(row.get("output_ids")) or not row["output_ids"]:
                raise ValueError(f"{rid}: missing or empty output token IDs")
            reason = row.get("finish_reason")
            if not isinstance(reason, dict) or reason.get("type") not in (
                "length",
                "stop",
            ):
                raise ValueError(f"{rid}: missing or unsuccessful finish_reason")
        if require_cache_hit and (
            type(rb.get("cached_tokens")) is not int or rb["cached_tokens"] <= 0
        ):
            raise ValueError(f"{rid}: second run has no positive cache hit")
        x, y = ra["output_ids"], rb["output_ids"]
        if x != y or ra["finish_reason"] != rb["finish_reason"]:
            first = next(
                (i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y))
            )
            differences.append(
                {
                    "id": rid,
                    "prompt_remainder": len(ra["input_ids"]) % BLOCK,
                    "first_difference": first,
                    "output_lengths": [len(x), len(y)],
                }
            )
    return {
        "total": len(a),
        "exact": len(a) - len(differences),
        "differences": differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument(
        "--revision", required=True, help="40-character checkpoint commit"
    )
    prep.add_argument("--local-files-only", action="store_true")
    prep.add_argument("--out", required=True)
    capture = sub.add_parser("run")
    capture.add_argument("--prompts", required=True)
    capture.add_argument("--out", required=True)
    capture.add_argument("--fork", action="store_true")
    capture.add_argument("--aligned-only", action="store_true")
    capture.add_argument("--radix", choices=("on", "off"), default="off")
    capture.add_argument("--max-running", type=int, default=1)
    capture.add_argument("--batched", action="store_true")
    capture.add_argument("--eager", action="store_true")
    capture.add_argument("--max-new-tokens", type=int, default=128)
    capture.add_argument("--mem-fraction-static", type=float, default=0.42)
    check = sub.add_parser("compare")
    check.add_argument("left")
    check.add_argument("right")
    check.add_argument("--require-cache-hit", action="store_true")
    check.add_argument(
        "--allow-differences",
        action="store_true",
        help="Report numerical/output differences without failing",
    )
    args = parser.parse_args()
    if args.command == "prepare":
        if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
            parser.error("--revision must be a 40-character checkpoint commit")
        prepare(args)
    elif args.command == "run":
        if args.max_running <= 0 or args.max_new_tokens <= 0:
            parser.error("Request and token counts must be positive")
        run(args)
    else:
        result = compare(
            read_json(args.left), read_json(args.right), args.require_cache_hit
        )
        print(json.dumps(result, indent=2))
        if result["differences"] and not args.allow_differences:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
