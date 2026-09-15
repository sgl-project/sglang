"""Compare complete global outputs, including every TP replica and EP rank."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from workload import gather_outputs, make_input

REL_TOL = 2e-2


def compare_artifacts(a, b):
    """Reject mismatched workloads, missing outputs and numerical corruption."""
    for key in (
        "global_tokens",
        "input_sha256",
        "weights_sha256",
        "phase",
        "layer_id",
        "skew_experts",
        "world_size",
    ):
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(f"workload mismatch: {key}")
    if {a["backend"], b["backend"]} != {"none", "deepep"}:
        raise ValueError("compare one TP artifact against one EP artifact")
    tp, ep = (a, b) if a["backend"] == "none" else (b, a)
    if len(tp["replicas"]) != tp["world_size"]:
        raise ValueError("missing TP replica")
    reference = tp["output"].float()
    if reference.shape[0] != tp["global_tokens"]:
        raise ValueError("incomplete global output")
    checks = []
    for candidate in [ep["output"], *tp["replicas"]]:
        candidate = candidate.float()
        if candidate.shape != reference.shape:
            raise ValueError("output shape mismatch")
        if not torch.isfinite(candidate).all() or not torch.isfinite(reference).all():
            raise ValueError("non-finite output")
        max_abs = (candidate - reference).abs().max().item()
        scale = reference.abs().max().item()
        rel = max_abs / (scale + 1e-9)
        checks.append(dict(max_abs=max_abs, relative=rel, passed=rel <= REL_TOL))
    return dict(
        passed=all(x["passed"] for x in checks), tolerance=REL_TOL, checks=checks
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", nargs=2, metavar=("TP", "EP"))
    parser.add_argument("--model-path")
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--backend", default="none", choices=["none", "deepep"])
    parser.add_argument("--global-tokens", type=int, default=2048)
    parser.add_argument("--skew-experts", type=int, default=0)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--input-seed", type=int, default=20260922)
    parser.add_argument("--phase", default="prefill", choices=["prefill", "decode"])
    parser.add_argument("--save")
    parser.add_argument("--out", help="comparison JSON")
    args = parser.parse_args()
    if args.compare:
        try:
            result = compare_artifacts(
                *(
                    torch.load(p, map_location="cpu", weights_only=True)
                    for p in args.compare
                )
            )
        except (ValueError, KeyError) as exc:
            result = dict(passed=False, error=str(exc))
        print(json.dumps(result), flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        sys.exit(0 if result["passed"] else 1)
    if not args.model_path or not args.save:
        parser.error("--model-path and --save are required unless --compare is used")
    if args.global_tokens < 1:
        parser.error("--global-tokens must be positive")

    import torch_npu  # noqa: F401
    from bench_moe_tp_ep import (
        build_block,
        environment_metadata,
        finalize_weights,
        load_real_weights,
        make_forward_batch,
    )

    block, config, sa = build_block(args.model_path, args.ep_size, args.backend, "auto")
    load_real_weights(block, config, args.model_path, args.layer_id, args.skew_experts)
    finalize_weights(block)
    hidden, counts, input_hash = make_input(
        args.global_tokens,
        config.hidden_size,
        args.backend,
        args.input_seed,
        torch.device("npu"),
    )
    fb = make_forward_batch(hidden.shape[0], is_extend=args.phase == "prefill")
    with torch.inference_mode():
        out = block(hidden, fb)
        torch.npu.synchronize()
        full, replicas = gather_outputs(out, counts, args.backend)
    if dist.get_rank() == 0:
        artifact = dict(
            schema_version=2,
            output=full,
            replicas=replicas,
            backend=args.backend,
            global_tokens=args.global_tokens,
            tokens_per_rank=counts,
            input_sha256=input_hash,
            input_seed=args.input_seed,
            weights_sha256=block.benchmark_weight_sha256,
            phase=args.phase,
            layer_id=args.layer_id,
            skew_experts=args.skew_experts,
            world_size=dist.get_world_size(),
            metadata=environment_metadata(),
        )
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, args.save)
        print(
            f"saved {args.save}: global={args.global_tokens}, per_rank={counts}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
