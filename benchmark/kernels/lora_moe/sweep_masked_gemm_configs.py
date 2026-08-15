"""Sweep masked base-GEMM launch configs and emit the M-bucketed JSON store.

Runs the MoE LoRA providers' S2/S4 masked GEMMs over a grid of ``expected_m``
buckets on the current device and writes the store files consumed by
``gemm_config_store.load_config_table``.  Only buckets that beat the built-in
heuristic by at least ``--min-gain`` get entries — sparse tables are fine,
nearest-M covers the gaps.

Run on the target device (e.g. the GB300 pod) with the layer's real
TP-sharded geometry, e.g. qwen3.5 TP4:

    python benchmark/kernels/lora_moe/sweep_masked_gemm_configs.py \
        --provider both --num-local-experts <E_local> --hidden-size <H> \
        --intermediate-size <I> --gate-up-slices 2 --top-k <K> \
        --output-dir python/sglang/srt/lora/moe/configs/base_gemm
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch

from sglang.srt.lora.moe.base_gemm_provider.gemm_config_store import (
    config_file_name,
    cutedsl_version,
)
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo
from sglang.srt.utils import get_device_name

DEFAULT_BUCKETS = "4,8,16,32,48,64,96,128,192,256,384,512"


def _time_ms(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def _workload(bucket_m: int, args, device) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = max(1, bucket_m * args.num_local_experts // args.top_k)
    # Balanced routing with distinct experts per token (requires top_k <= E).
    topk_ids = (
        torch.arange(num_tokens * args.top_k, device=device)
        .view(num_tokens, args.top_k)
        .remainder(args.num_local_experts)
        .to(torch.int32)
    )
    hidden = torch.randn(
        (num_tokens, args.hidden_size), device=device, dtype=torch.bfloat16
    )
    return hidden, topk_ids


def _quant_info(args, device) -> MoeLoraBf16QuantInfo:
    shape13 = (
        args.num_local_experts,
        args.gate_up_slices * args.intermediate_size,
        args.hidden_size,
    )
    shape2 = (args.num_local_experts, args.hidden_size, args.intermediate_size)
    return MoeLoraBf16QuantInfo(
        w13_weight=torch.randn(shape13, device=device, dtype=torch.bfloat16) * 0.05,
        w2_weight=torch.randn(shape2, device=device, dtype=torch.bfloat16) * 0.05,
        num_local_experts=args.num_local_experts,
        intermediate_size=args.intermediate_size,
        hidden_size=args.hidden_size,
    )


def _stage_times(provider, ws) -> float:
    gateup_out = torch.empty(
        provider.gateup_out_shape(ws), device="cuda", dtype=torch.bfloat16
    )
    act = torch.randn(provider.act_out_shape(ws), device="cuda", dtype=torch.bfloat16)
    down_out = torch.empty(
        provider.down_out_shape(ws), device="cuda", dtype=torch.bfloat16
    )
    t1 = _time_ms(lambda: provider.gateup(ws, gateup_out), warmup=10, iters=50)
    t2 = _time_ms(lambda: provider.down(ws, act, down_out), warmup=10, iters=50)
    return t1 + t2


def sweep_cutedsl(args, buckets, device) -> dict[int, dict]:
    from sglang.srt.lora.moe.base_gemm_provider.cutedsl_bf16 import (
        CuteDslBf16Provider,
    )
    from sglang.srt.lora.moe.base_gemm_provider.cutedsl_masked.api import (
        MaskedGroupedGemmConfig,
    )

    provider = CuteDslBf16Provider(_quant_info(args, device))
    provider._config_table = None  # sweep against the pristine heuristic
    for spec in args.tiles.split(",") if args.tiles else ():
        token_width, clusters = (int(part) for part in spec.split(":"))
        if token_width in provider._compiled:
            continue
        provider._tile_configs[token_width] = MaskedGroupedGemmConfig(
            mma_tiler_mn=(provider.OUTPUT_WIDTH, token_width),
            cluster_shape_mn=provider.CLUSTER_SHAPE_MN,
            use_2cta_instrs=provider.USE_2CTA_INSTRS,
            occupancy=1,
            mma_inst_tile_k=4,
            persistent_clusters=clusters,
            swap_ab=True,
            direct_schedule=True,
        )
        provider._compiled[token_width] = {}
        provider._compile_stage(token_width, "gemm1", produce_pdl=False)
        provider._compile_stage(token_width, "gemm2", produce_pdl=False)
    torch.cuda.synchronize(device)

    clusters_of = {
        width: provider._tile_configs[width].persistent_clusters
        for width in provider._compiled
    }
    results: dict[int, dict] = {}
    for bucket_m in buckets:
        hidden, topk_ids = _workload(bucket_m, args, device)
        m_max = (hidden.shape[0] // 256 + 1) * 256
        heuristic = CuteDslBf16Provider._token_width_for(provider, m_max, bucket_m)
        per_width = {}
        for width in sorted(provider._compiled):
            if m_max > width * provider._max_token_clusters:
                continue
            provider._token_width_for = lambda m_max, expected_m, _w=width: _w
            ws = provider.prepare(hidden, topk_ids, args.top_k)
            per_width[width] = _stage_times(provider, ws)
        del provider._token_width_for
        best = min(per_width, key=per_width.get)
        gain = 1.0 - per_width[best] / per_width[heuristic]
        results[bucket_m] = {
            "heuristic": heuristic,
            "best": best,
            "gain": gain,
            "times": per_width,
            "clusters": clusters_of[best],
        }
    return results


def sweep_deepgemm(args, buckets, device) -> dict[int, dict]:
    from sglang.srt.lora.moe.base_gemm_provider.deep_gemm_bf16 import (
        DeepGemmBf16Provider,
    )

    provider = DeepGemmBf16Provider(_quant_info(args, device))
    provider._config_table = None
    results: dict[int, dict] = {}
    for bucket_m in buckets:
        hidden, topk_ids = _workload(bucket_m, args, device)
        ws = provider.prepare(hidden, topk_ids, args.top_k)
        identity = ws.expected_m
        candidates = sorted(
            {max(1, identity // 2), identity, 2 * identity, (identity + 63) // 64 * 64}
        )
        per_hint = {}
        for hint in candidates:
            ws.expected_m = hint
            per_hint[hint] = _stage_times(provider, ws)
        best = min(per_hint, key=per_hint.get)
        gain = 1.0 - per_hint[best] / per_hint[identity]
        results[bucket_m] = {
            "heuristic": identity,
            "best": best,
            "gain": gain,
            "times": per_hint,
        }
    return results


def _emit(provider_key, results, args, payload_key, version) -> None:
    buckets = {
        str(m): {payload_key: row["best"]}
        for m, row in sorted(results.items())
        if row["best"] != row["heuristic"] and row["gain"] >= args.min_gain
    }
    name = config_file_name(
        provider_key,
        num_local_experts=args.num_local_experts,
        n_gemm1=args.gate_up_slices * args.intermediate_size,
        n_gemm2=args.hidden_size,
        k=args.hidden_size,
        device_name=get_device_name(),
    )
    print(f"\n== {provider_key} ==")
    for m, row in sorted(results.items()):
        marker = "*" if str(m) in buckets else " "
        print(
            f"{marker} M={m:>4} heuristic={row['heuristic']:>5} "
            f"best={row['best']:>5} gain={row['gain'] * 100:+.1f}% "
            f"times={ {k: round(v, 4) for k, v in row['times'].items()} }"
        )
    if not buckets:
        print(
            f"no bucket beat the heuristic by >= {args.min_gain:.0%}; "
            f"not writing {name}"
        )
        return
    payload: dict = {"version": version, "buckets": buckets}
    if payload_key == "token_width":
        payload["tiles"] = sorted(
            (
                {
                    "token_width": row["best"],
                    "persistent_clusters": row["clusters"],
                }
                for m, row in results.items()
                if str(m) in buckets
            ),
            key=lambda tile: tile["token_width"],
        )
        seen = set()
        payload["tiles"] = [
            tile
            for tile in payload["tiles"]
            if tile["token_width"] not in seen and not seen.add(tile["token_width"])
        ]
    path = os.path.join(args.output_dir, name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", choices=("cutedsl", "deepgemm", "both"), default="both"
    )
    parser.add_argument("--num-local-experts", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--gate-up-slices", type=int, default=2)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument(
        "--buckets", default=DEFAULT_BUCKETS, help="comma-separated expected_m grid"
    )
    parser.add_argument(
        "--tiles", default="", help="extra CuTeDSL candidates, e.g. '32:128,128:152'"
    )
    parser.add_argument("--min-gain", type=float, default=0.02)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    device = torch.device("cuda")
    buckets = [int(m) for m in args.buckets.split(",")]
    device_name = get_device_name()
    if args.provider in ("cutedsl", "both"):
        results = sweep_cutedsl(args, buckets, device)
        version = {"generated_on": device_name}
        if cutedsl_version():
            version["cutedsl"] = cutedsl_version()
        _emit("cutedsl_bf16", results, args, "token_width", version)
    if args.provider in ("deepgemm", "both"):
        results = sweep_deepgemm(args, buckets, device)
        _emit(
            "deepgemm_bf16", results, args, "expected_m", {"generated_on": device_name}
        )


if __name__ == "__main__":
    main()
