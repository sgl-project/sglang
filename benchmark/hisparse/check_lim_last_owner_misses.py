"""Diagnostic only: sample the last IndexShare owner's live LIM counters.

The adapter reuses output buffers across owners, so these are last-owner data,
not a model-wide hit rate. CPU reads deliberately synchronize; never include
this request in timing. Sampling step zero is prefill and is excluded.
"""

import argparse
import json
import urllib.request
from pathlib import Path


def processor_string():
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

    class SampleLastOwner(CustomLogitProcessor):
        def __call__(self, logits, custom_param_list):
            import json
            from pathlib import Path

            import torch

            if logits.shape[0] != 1:
                raise RuntimeError("Last-owner diagnostic requires actual batch one")
            params = custom_param_list[0]
            call = params.get("_lim_miss_sampling_call", 0)
            params["_lim_miss_sampling_call"] = call + 1
            skip = params.get("prefill_sampling_calls_to_skip", 0)
            if call < skip:
                return logits
            step = call - skip
            if torch.distributed.get_rank() != 0 or step == 0:
                return logits
            from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload import (
                host_callback,
            )
            from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager import (
                get_sparse_kv_manager,
            )

            manager = get_sparse_kv_manager()
            adapter = manager.lim_adapter if manager is not None else None
            if adapter is None or not adapter.is_native:
                raise RuntimeError("Native LIM manager is required")
            prefix = adapter.last_prefix.cpu().tolist()
            owners = [
                manager.start_layer + i
                for i, rows in enumerate(prefix)
                if any(n >= 0 for n in rows[: manager.size])
            ]
            record = {
                "sampling_step": step,
                "raw_sampling_call": call,
                "last_owner_layer_id": max(owners),
                "active_owner_count": len(owners),
                "K": adapter.top_k,
                "C": adapter.capacity,
                "prefix_topk_misses": int(adapter.topk_misses[0].cpu()),
                "copy_count": int(adapter.miss_counts[0].cpu()),
                "valid_topk_sources": int((adapter.source_ids[0] >= 0).sum().cpu()),
                "index_k_buffer_shape": list(
                    manager.paged_kv_cache.index_k_buffer.shape
                ),
                "index_k_buffer_bytes": manager.paged_kv_cache.index_k_buffer.numel()
                * manager.paged_kv_cache.index_k_buffer.element_size(),
                "managed_hbm_budget_bytes": manager.budget.managed_hbm_bytes,
                "host_kv_budget_bytes": manager.budget.host_kv_bytes,
                "framework_acl_reporter_count": len(host_callback._REPORTERS),
            }
            with Path(params["miss_log"]).open("a") as file:
                file.write(json.dumps(record) + "\n")
            return logits

    return SampleLastOwner.to_str()


def main():
    from check_lim_decode_precision import (
        partial_prefill_sample_count,
        validate_prefill_settings,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:18732")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--prefill-chunk-size", type=int, default=0)
    args = parser.parse_args()
    if args.tokens < 3:
        parser.error(
            "At least three output tokens are required for cold and steady samples"
        )
    if args.output.exists() or args.output.with_suffix(".jsonl").exists():
        raise RuntimeError("Use a fresh diagnostic output path")
    row = next(
        r for r in json.loads(args.fixtures.read_text()) if r["name"] == args.case
    )
    validate_prefill_settings(args.url, args.prefill_chunk_size, 1)
    skip = partial_prefill_sample_count(len(row["input_ids"]), args.prefill_chunk_size)
    log = args.output.with_suffix(".jsonl")
    payload = {
        "input_ids": row["input_ids"],
        "custom_logit_processor": processor_string(),
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.tokens,
            "ignore_eos": True,
            "custom_params": {
                "miss_log": str(log),
                "prefill_sampling_calls_to_skip": skip,
            },
        },
    }
    request = urllib.request.Request(
        args.url + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    if result.get("meta_info", {}).get("completion_tokens") != args.tokens:
        raise RuntimeError("Incomplete diagnostic response")
    records = [json.loads(line) for line in log.read_text().splitlines()]
    if [r["sampling_step"] for r in records] != list(range(1, args.tokens)):
        raise RuntimeError("Missing decode samples or unexpected processor lifecycle")
    if [r["raw_sampling_call"] for r in records] != list(
        range(skip + 1, skip + args.tokens)
    ):
        raise RuntimeError("Unexpected partial-prefill sampling calls")
    first = records[0]
    if first["copy_count"] != first["C"] or first["prefix_topk_misses"] != first["K"]:
        raise RuntimeError("A fresh request did not publish the expected cold fill")
    steady = records[1:]
    if any(
        r["copy_count"] != r["prefix_topk_misses"]
        or not 0 <= r["copy_count"] <= r["K"]
        or r["valid_topk_sources"] != r["K"]
        for r in steady
    ):
        raise RuntimeError("Q1 steady copy/miss bookkeeping differs")
    summary = {
        "diagnostic_only": True,
        "scope": "Last IndexShare owner only, rank 0, actual batch 1, Q1",
        "case": args.case,
        "input_tokens": len(row["input_ids"]),
        "last_owner_layer_ids": sorted({r["last_owner_layer_id"] for r in records}),
        "first_fill": first,
        "steady_samples": len(steady),
        "steady_mean_prefix_topk_misses": sum(r["prefix_topk_misses"] for r in steady)
        / len(steady),
        "steady_prefix_topk_hit_fraction": 1
        - sum(r["prefix_topk_misses"] for r in steady) / sum(r["K"] for r in steady),
        "limits": "Excludes full-tail tokens and padding; cannot estimate 21-owner average or SWE-bench cache locality.",
        "response": result,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "response"}, indent=2))


if __name__ == "__main__":
    main()
