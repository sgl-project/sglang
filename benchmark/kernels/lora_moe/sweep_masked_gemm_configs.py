"""Sweep BF16/FP8 base-GEMM tiles using the target device and TP-sharded geometry.

Emit nearest-M config overrides only for gains above --min-gain.
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


def _fp8_block_quant(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize 128x128 blocks, including partial TP-shard tail blocks."""
    e, rows, cols = weight.shape
    block = 128
    rb = (rows + block - 1) // block
    cb = (cols + block - 1) // block
    scale = torch.empty((e, rb, cb), dtype=torch.float32, device=weight.device)
    q = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    fmax = torch.finfo(torch.float8_e4m3fn).max
    for i in range(rb):
        for j in range(cb):
            blk = weight[
                :, i * block : (i + 1) * block, j * block : (j + 1) * block
            ].float()
            amax = blk.abs().amax(dim=(1, 2)).clamp(min=1e-6)
            s = amax / fmax
            scale[:, i, j] = s
            q[:, i * block : (i + 1) * block, j * block : (j + 1) * block] = (
                blk / s[:, None, None]
            ).to(torch.float8_e4m3fn)
    return q, scale


def _fp8_quant_info(args, device):
    """Build synthetic FP8 weights with FP32 block scales."""
    from sglang.srt.lora.moe.quant_info import MoeLoraFp8QuantInfo

    bf16 = _quant_info(args, device)
    w13_q, w13_s = _fp8_block_quant(bf16.w13_weight)
    w2_q, w2_s = _fp8_block_quant(bf16.w2_weight)
    return MoeLoraFp8QuantInfo(
        w13_weight=w13_q,
        w13_scale=w13_s,
        w2_weight=w2_q,
        w2_scale=w2_s,
        block_shape=(128, 128),
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


def sweep_cutedsl(
    args, buckets, device, provider_cls=None, quant_info=None
) -> dict[int, dict]:
    from sglang.srt.lora.moe.base_gemm_provider.cutedsl_bf16 import (
        CuteDslBf16MaskedProvider,
    )
    from sglang.srt.lora.moe.kernels.cutedsl.api import (
        GroupedGemmConfig,
    )

    if provider_cls is None:
        provider_cls = CuteDslBf16MaskedProvider
    if quant_info is None:
        quant_info = _quant_info(args, device)
    provider = provider_cls(quant_info)
    # Ignore the installed bucket selection for the heuristic baseline; tile
    # and cluster settings already compiled (installed store or --tiles) remain.
    provider._config_table = None
    for spec in args.tiles.split(",") if args.tiles else ():
        token_width, clusters = (int(part) for part in spec.split(":"))
        if token_width in provider._compiled:
            print(f"tile width {token_width} is already compiled; ignoring :{clusters}")
            continue
        provider._tile_configs[token_width] = GroupedGemmConfig(
            mma_tiler_mn=(provider.OUTPUT_WIDTH, token_width),
            cluster_shape_mn=provider.CLUSTER_SHAPE_MN,
            use_2cta_instrs=provider.USE_2CTA_INSTRS,
            occupancy=1,
            mma_inst_tile_k=4,
            persistent_clusters=clusters,
        )
        provider._compiled[token_width] = {}
        try:
            provider._compile_stage(token_width, "gemm1")
            provider._compile_stage(token_width, "gemm2")
        except Exception as exc:  # kernel rejects unsupported tile geometries
            print(f"skipping tile width {token_width}: {exc}")
            provider._compiled.pop(token_width, None)
            provider._tile_configs.pop(token_width, None)
    torch.cuda.synchronize(device)

    clusters_of = {
        width: provider._tile_configs[width].persistent_clusters
        for width in provider._compiled
    }
    results: dict[int, dict] = {}
    for bucket_m in buckets:
        hidden, topk_ids = _workload(bucket_m, args, device)
        m_max = (hidden.shape[0] // 256 + 1) * 256
        heuristic = provider_cls._token_width_for(provider, m_max, bucket_m)
        per_width = {}
        for width in sorted(provider._compiled):
            max_clusters = getattr(provider, "_max_token_clusters", None)
            if max_clusters is not None and m_max > width * max_clusters:
                continue
            try:
                provider._admit_tile_width(width)
            except ValueError:
                continue
            provider._token_width_for = lambda m_max, expected_m, _w=width: _w
            ws = provider.prepare(hidden, topk_ids, args.top_k)
            per_width[width] = _stage_times(provider, ws)
            # Release masked slabs before the next candidate to bound memory.
            del ws
            torch.cuda.empty_cache()
        del provider._token_width_for
        best = min(per_width, key=per_width.get)
        gain = 1.0 - per_width[best] / per_width[heuristic]
        results[bucket_m] = {
            "heuristic": heuristic,
            "best": best,
            "gain": gain,
            "times": per_width,
            "clusters": clusters_of[best],
            "heuristic_clusters": clusters_of[heuristic],
        }
    return results


def _emit(provider_key, results, args, version) -> None:
    # The store picks the nearest emitted bucket, so every measured bucket is
    # written: a sub-threshold cell keeps its heuristic width instead of
    # inheriting a neighbour's winner.
    def _wins(row) -> bool:
        return row["best"] != row["heuristic"] and row["gain"] >= args.min_gain

    def _chosen(row) -> tuple[int, int]:
        if _wins(row):
            return row["best"], row["clusters"]
        return row["heuristic"], row["heuristic_clusters"]

    buckets = {
        str(m): {"token_width": _chosen(row)[0]} for m, row in sorted(results.items())
    }
    winners = {str(m) for m, row in results.items() if _wins(row)}
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
        marker = "*" if str(m) in winners else " "
        print(
            f"{marker} M={m:>4} heuristic={row['heuristic']:>5} "
            f"best={row['best']:>5} gain={row['gain'] * 100:+.1f}% "
            f"times={ {k: round(v, 4) for k, v in row['times'].items()} }"
        )
    if not winners:
        print(
            f"no bucket beat the heuristic by >= {args.min_gain:.0%}; "
            f"not writing {name}"
        )
        return
    payload: dict = {"version": version, "buckets": buckets}
    payload["tiles"] = sorted(
        (
            {"token_width": width, "persistent_clusters": clusters}
            for width, clusters in (_chosen(row) for row in results.values())
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
        "--family",
        choices=("bf16", "fp8"),
        default="bf16",
        help="weight family to sweep; picks the matching providers",
    )
    parser.add_argument(
        "--domain",
        choices=("masked", "contiguous"),
        default="masked",
        help="base-GEMM row domain: masked (decode rows) or contiguous "
        "(prefill rows)",
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
    cutedsl_ver = {"generated_on": device_name}
    if cutedsl_version():
        cutedsl_ver["cutedsl"] = cutedsl_version()
    if args.family == "bf16" and args.domain == "contiguous":
        from sglang.srt.lora.moe.base_gemm_provider.cutedsl_bf16 import (
            CuteDslBf16ContiguousProvider,
        )

        results = sweep_cutedsl(
            args, buckets, device, provider_cls=CuteDslBf16ContiguousProvider
        )
        _emit("cutedsl_bf16_contiguous", results, args, cutedsl_ver)
    elif args.family == "fp8" and args.domain == "contiguous":
        from sglang.srt.lora.moe.base_gemm_provider.cutedsl_fp8 import (
            CuteDslFp8ContiguousProvider,
        )

        results = sweep_cutedsl(
            args,
            buckets,
            device,
            provider_cls=CuteDslFp8ContiguousProvider,
            quant_info=_fp8_quant_info(args, device),
        )
        _emit("cutedsl_fp8_contiguous", results, args, cutedsl_ver)
    elif args.family == "bf16":
        results = sweep_cutedsl(args, buckets, device)
        _emit("cutedsl_bf16_masked", results, args, cutedsl_ver)
    elif args.family == "fp8":
        from sglang.srt.lora.moe.base_gemm_provider.cutedsl_fp8 import (
            CuteDslFp8MaskedProvider,
        )

        results = sweep_cutedsl(
            args,
            buckets,
            device,
            provider_cls=CuteDslFp8MaskedProvider,
            quant_info=_fp8_quant_info(args, device),
        )
        _emit("cutedsl_fp8_masked", results, args, cutedsl_ver)


if __name__ == "__main__":
    main()
