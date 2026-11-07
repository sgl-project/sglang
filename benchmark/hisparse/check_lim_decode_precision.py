"""Diagnostic-only fixed-history decode logits; never use these requests for timing.

The custom processor saves untouched model logits before forcing the reference
token. API logprobs after forcing are deliberately ignored. Full-vocabulary KL
is computed in float64 on CPU; identical token histories are checked explicitly.

Run this client in the server's filesystem namespace with a shared absolute
--workdir and --enable-custom-logit-processor on the server. Each capture tag
stores full logits, verified HTTP results and frozen token-ID fixtures. At
vocab=154880, 32 samples x 512 positions require about 10 GB per capture tag.
Use a fresh tag for every capture; keep these diagnostics outside timed runs.
For fixed chunked prefill, pass --prefill-chunk-size and use concurrency one.
The helper validates that radix reuse and dynamic chunking are disabled, skips
discarded intermediate-chunk sampling calls, and verifies the final token history.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path


def processor_string():
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

    class CaptureAndForce(CustomLogitProcessor):
        def __call__(self, logits, custom_param_list):
            import json
            from pathlib import Path

            import torch

            for row, params in enumerate(custom_param_list):
                # Processor instances can be rebuilt at batch merges and at the
                # prefill/decode boundary. Request custom_params outlive them.
                call = params.get("_lim_precision_call", 0)
                params["_lim_precision_call"] = call + 1
                skip = params.get("prefill_sampling_calls_to_skip", 0)
                if call < skip:
                    continue
                step = call - skip
                forced = params.get("force_ids")
                if forced is not None and step >= len(forced):
                    raise RuntimeError("Teacher continuation exhausted")
                if torch.distributed.get_rank() == 0:
                    target = (
                        Path(params["capture_root"]) / params["tag"] / params["case"]
                    )
                    target.mkdir(parents=True, exist_ok=True)
                    raw = logits[row].detach().float().cpu()
                    if not torch.isfinite(raw).all():
                        raise RuntimeError("Nonfinite untouched model logits")
                    token = int(raw.argmax()) if forced is None else forced[step]
                    torch.save(raw, target / f"{step:04d}.pt")
                    with (target / "steps.jsonl").open("a", encoding="utf-8") as file:
                        file.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "raw_sampling_call": call,
                                    "prefill_sampling_calls_skipped": skip,
                                    "forced_token": token,
                                    "raw_argmax": int(raw.argmax()),
                                    "active_processor_rows": logits.shape[0],
                                    "vocab_size": raw.numel(),
                                }
                            )
                            + "\n"
                        )
                if forced is not None:
                    logits[row].fill_(-float("inf"))
                    logits[row, forced[step]] = 0
            return logits

    return CaptureAndForce.to_str()


def request(args, processor, row):
    forced = row.get("force_ids")
    payload = {
        "input_ids": row["input_ids"],
        "stream": False,
        "custom_logit_processor": processor,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.tokens,
            "ignore_eos": True,
            "custom_params": {
                "request_key": args.tag + ":" + row["name"],
                "tag": args.tag,
                "case": row["name"],
                "capture_root": str(args.workdir / "results/teacher-logits"),
                "force_ids": forced,
                "prefill_sampling_calls_to_skip": partial_prefill_sample_count(
                    len(row["input_ids"]), args.prefill_chunk_size
                ),
            },
        },
        "return_logprob": True,
        "logprob_start_len": -1,
        "return_text_in_logprobs": False,
    }
    started = time.time()
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as response:
        result = json.load(response)
    response_root = args.workdir / "results/teacher-http" / args.tag
    response_root.mkdir(parents=True, exist_ok=True)
    (response_root / f"{row['name']}.json").write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8"
    )
    if "error" in result:
        raise RuntimeError(result)
    observed = [int(item[1]) for item in result["meta_info"]["output_token_logprobs"]]
    if len(observed) != args.tokens:
        raise RuntimeError(f"Incomplete continuation: {row['name']} {len(observed)}")
    if forced is not None and observed != forced:
        raise RuntimeError(f"Teacher history differs: {row['name']}")
    directory = args.workdir / "results/teacher-logits" / args.tag / row["name"]
    steps = [
        json.loads(line)
        for line in (directory / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if [r["step"] for r in steps] != list(range(args.tokens)):
        raise RuntimeError(f"Invalid processor step sequence: {row['name']}")
    expected_skip = partial_prefill_sample_count(
        len(row["input_ids"]), args.prefill_chunk_size
    )
    if [r["raw_sampling_call"] for r in steps] != list(
        range(expected_skip, expected_skip + args.tokens)
    ):
        raise RuntimeError(f"Chunked sampling calls differ: {row['name']}")
    if [r["forced_token"] for r in steps] != observed:
        raise RuntimeError(f"Capture/sampler token mismatch: {row['name']}")
    return {
        "name": row["name"],
        "input_ids": row["input_ids"],
        "force_ids": observed,
        "input_sha256": hashlib.sha256(
            json.dumps(row["input_ids"]).encode()
        ).hexdigest(),
        "started_unix": started,
        "finished_unix": time.time(),
        "response": result,
    }


def partial_prefill_sample_count(input_length, chunk_size):
    return (
        max(0, (input_length + chunk_size - 1) // chunk_size - 1)
        if chunk_size > 0
        else 0
    )


def validate_prefill_settings(base_url, chunk_size, concurrency):
    with urllib.request.urlopen(
        base_url.rstrip("/") + "/server_info", timeout=120
    ) as response:
        info = json.load(response)
    settings = info.get("server_args", info)
    configured = settings.get("chunked_prefill_size")
    expected = chunk_size if chunk_size > 0 else -1
    if configured != expected and not (expected == -1 and configured is None):
        raise RuntimeError(
            f"Pass the server's fixed prefill chunk size explicitly: server={configured}, requested={expected}"
        )
    if chunk_size > 0 and (
        concurrency != 1
        or not settings.get("disable_radix_cache")
        or settings.get("enable_dynamic_chunking")
        or settings.get("enable_mixed_chunk")
    ):
        raise RuntimeError(
            "Chunked logits capture requires real concurrency 1, disabled radix cache, fixed chunks and no mixed chunks"
        )
    return settings


def capture(args):
    validate_prefill_settings(args.base_url, args.prefill_chunk_size, args.concurrency)
    source = json.loads(args.fixtures.read_text(encoding="utf-8"))
    if len(source) < args.samples:
        raise RuntimeError("Insufficient precision fixtures")
    rows = source[: args.samples]
    if args.force and any(len(r.get("force_ids", [])) != args.tokens for r in rows):
        raise RuntimeError(
            "Every teacher fixture must have the exact continuation length"
        )
    if not args.force:
        rows = [{k: v for k, v in row.items() if k != "force_ids"} for row in rows]
    root = args.workdir / "results"
    if (root / "teacher-logits" / args.tag).exists():
        raise RuntimeError("Capture tag already exists; choose a fresh tag")
    processor = processor_string()
    output = root / f"{args.tag}-teacher-requests.jsonl"
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for result in pool.map(lambda row: request(args, processor, row), rows):
            results.append(result)
            with output.open("a", encoding="utf-8") as file:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    {
                        "case": result["name"],
                        "tokens": len(result["force_ids"]),
                        "history_verified": True,
                    }
                ),
                flush=True,
            )
    fixtures = [
        {
            k: v
            for k, v in r.items()
            if k in ("name", "input_ids", "force_ids", "input_sha256")
        }
        for r in results
    ]
    (root / f"{args.tag}-teacher-fixtures.json").write_text(
        json.dumps(fixtures), encoding="utf-8"
    )
    (root / f"{args.tag}-capture-status.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "samples": len(results),
                "tokens_per_sample": args.tokens,
                "prefill_chunk_size": args.prefill_chunk_size,
                "fixed_prefill_configuration_verified": True,
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "fixed_histories_sha256": hashlib.sha256(
                    json.dumps(fixtures, sort_keys=True).encode()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def compare(args):
    import torch

    fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))[: args.samples]
    root = args.workdir / "results/teacher-logits"
    for tag in (args.reference, args.tag):
        status = json.loads(
            (args.workdir / "results" / f"{tag}-capture-status.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            status["status"] != "verified"
            or status["samples"] < args.samples
            or status["tokens_per_sample"] != args.tokens
        ):
            raise RuntimeError("Comparison requires a complete, verified capture")
        observed = json.loads(
            (args.workdir / "results" / f"{tag}-teacher-fixtures.json").read_text(
                encoding="utf-8"
            )
        )
        by_name = {row["name"]: row for row in observed}
        for row in fixtures:
            recorded = by_name[row["name"]]
            if (
                recorded["input_ids"] != row["input_ids"]
                or recorded["force_ids"] != row["force_ids"]
            ):
                raise RuntimeError("Cannot compare logits on different token histories")
    records = []
    for row in fixtures:
        for step in range(args.tokens):
            reference = torch.load(
                root / args.reference / row["name"] / f"{step:04d}.pt",
                weights_only=True,
            ).double()
            candidate = torch.load(
                root / args.tag / row["name"] / f"{step:04d}.pt", weights_only=True
            ).double()
            if (
                reference.shape != candidate.shape
                or not torch.isfinite(candidate).all()
            ):
                raise RuntimeError("Invalid candidate logits")
            p_log, q_log = reference.log_softmax(-1), candidate.log_softmax(-1)
            kl = float((p_log.exp() * (p_log - q_log)).sum())
            token = row["force_ids"][step]
            records.append(
                {
                    "case": row["name"],
                    "step": step,
                    "kl_reference_to_candidate": kl,
                    "reference_token_logprob_abs_diff": float(
                        (p_log[token] - q_log[token]).abs()
                    ),
                    "argmax_agrees": int(reference.argmax()) == int(candidate.argmax()),
                    "logits_bitwise_equal": torch.equal(reference, candidate),
                }
            )
    kls = sorted(r["kl_reference_to_candidate"] for r in records)
    if not records or any(not math.isfinite(x) for x in kls):
        raise RuntimeError("Empty or nonfinite precision comparison")
    summary = {
        "reference": args.reference,
        "candidate": args.tag,
        "samples": len(fixtures),
        "tokens_per_sample": args.tokens,
        "total_aligned_tokens": len(records),
        "metric": "full-vocabulary KL(p_reference || p_candidate), float64 CPU",
        "mean_kl": sum(kls) / len(kls),
        "max_kl": max(kls),
        "p99_kl": kls[min(len(kls) - 1, math.ceil(len(kls) * 0.99) - 1)],
        "fraction_kl_above_1e_minus_3": sum(x > 1e-3 for x in kls) / len(kls),
        "fraction_kl_above_1e_minus_2": sum(x > 1e-2 for x in kls) / len(kls),
        "argmax_agreement": sum(r["argmax_agrees"] for r in records) / len(records),
        "bitwise_equal_fraction": sum(r["logits_bitwise_equal"] for r in records)
        / len(records),
        "max_reference_token_logprob_abs_diff": max(
            r["reference_token_logprob_abs_diff"] for r in records
        ),
        "limitations": "Diagnostic custom processor adds CPU synchronization and IO; not performance data. Cross-policy KL includes the policy change. Repeat each reference to measure same-mode variability.",
    }
    stem = args.workdir / "results" / f"{args.reference}-vs-{args.tag}-teacher"
    stem.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    stem.with_suffix(".jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("capture", "compare"))
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=0,
        help="Fixed server chunk size; 0 requires unchunked prefill",
    )
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    options = parser.parse_args()
    if options.phase == "compare" and not options.reference:
        parser.error("compare requires --reference")
    capture(options) if options.phase == "capture" else compare(options)
