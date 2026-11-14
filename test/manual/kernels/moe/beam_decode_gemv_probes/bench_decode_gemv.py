"""Single-token latency of the Beam decode GEMV against the Triton fused MoE.

Manual and not CI-registered. Run the correctness test first, then this probe,
from the repository root on one SM103 GPU:

    python test/manual/kernels/moe/test_beam_decode_gemv.py
    python test/manual/kernels/moe/beam_decode_gemv_probes/bench_decode_gemv.py

All three timed paths return the same [1, 5120] BF16 sum over the 8 routed
experts: `fused_experts_impl` with interleaved SwiGLU and its own expert
combine (`fused_moe`), `fused_experts_impl(defer_sum=True)` followed by the
GEMV path's reducer (`fused_moe_defer`), and `beam_decode_gemv` followed by an
explicit FP32 expert sum rounded to BF16 (`decode_gemv`). None includes TP
communication or the fused all-reduce boundary.
"""

import argparse
import contextlib
import json
import os
import statistics

import torch
import triton

from sglang.kernels.ops.moe.beam_decode_gemv import beam_decode_gemv

# Beam's experts, hidden size, TP4 intermediate size and top-k.
E, H, I, TOPK = 256, 5120, 640, 8
ARMS = ("fused_moe", "fused_moe_defer", "decode_gemv")


def _init_runtime():
    # Same single-rank setup as test_fused_swiglu_epilogue.py: fused_experts_impl
    # reads global server args and allocates its output under the TP group.
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from sglang.test.test_utils import publish_build_topology

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    for key, value in (
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "29642"),
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
    ):
        os.environ.setdefault(key, value)
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    publish_build_topology(tp_size=1, ep_size=1, pp_size=1)
    initialize_model_parallel(backend="gloo")


def _shutdown_runtime():
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    destroy_model_parallel()
    destroy_distributed_environment()


def _capture(fn):
    for _ in range(3):  # Compile and autotune before capture.
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, out


def _time_samples(graph, flush, repeats):
    """Microseconds of `repeats` single replays, each after an untimed L2 flush."""
    events = []
    for _ in range(repeats):
        flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    return [s.elapsed_time(e) * 1e3 for s, e in events]


def _spread(values):
    values = list(values)
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--blocks", type=int, default=6, help="palindromic blocks (>= 5)"
    )
    parser.add_argument("--repeats", type=int, default=20, help="replays per run")
    parser.add_argument("--flush-l2-multiple", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fused-config",
        type=json.loads,
        help="Triton config JSON for both fused GEMMs (default: runtime selection)",
    )
    args = parser.parse_args()
    if args.blocks < 5 or args.flush_l2_multiple < 5:
        parser.error("use at least 5 blocks and a flush of at least 5x L2")

    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    if capability != (10, 3):
        raise SystemExit(f"beam_decode_gemv targets SM103; this GPU is {capability}")

    _init_runtime()
    try:
        from sglang.srt.layers.moe.moe_runner.triton_utils import override_config
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts_impl,
        )
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
            try_get_optimal_moe_config,
        )
        from sglang.srt.utils.device import get_nvidia_driver_version_str

        torch.manual_seed(args.seed)
        # Interleaved gate/up rows: row 2j is gate j and row 2j + 1 is up j.
        w13 = (torch.randn(E, 2 * I, H, device="cuda") / H**0.5).bfloat16()
        w2 = (torch.randn(E, H, I, device="cuda") / I**0.5).bfloat16()
        hidden = torch.randn(1, H, device="cuda").bfloat16()
        ids = torch.randperm(E, device="cuda")[:TOPK].view(1, TOPK).int()
        weights = torch.softmax(torch.randn(1, TOPK, device="cuda"), -1)

        def config_scope():
            if args.fused_config is None:
                return contextlib.nullcontext()
            return override_config(args.fused_config)

        def fused(defer_sum):
            with config_scope():
                return fused_experts_impl(
                    hidden,
                    w13,
                    w2,
                    weights,
                    ids,
                    inplace=False,
                    # As fused_experts passes when every expert is local (EP=1).
                    filter_expert=False,
                    routed_scaling_factor=1.0,
                    fuse_swiglu_interleaved=True,
                    defer_sum=defer_sum,
                )

        def reduce(per_expert):
            return per_expert.sum(dim=1, dtype=torch.float32).bfloat16()

        paths = {
            "fused_moe": lambda: fused(defer_sum=False),
            "fused_moe_defer": lambda: reduce(fused(defer_sum=True)),
            "decode_gemv": lambda: reduce(
                beam_decode_gemv(hidden, w13, w2, weights, ids)
            ),
        }
        graphs, outputs = {}, {}
        for name, fn in paths.items():
            graphs[name], outputs[name] = _capture(fn)
        for name in ("fused_moe", "fused_moe_defer"):
            # The correctness test's tolerance; the paths are not bitwise equal.
            torch.testing.assert_close(
                outputs[name], outputs["decode_gemv"], rtol=1.6e-2, atol=1e-3
            )
        # The lookup fused_experts_impl makes for this BF16 shape at M=1.
        with config_scope():
            config, (down_config, _) = try_get_optimal_moe_config(
                w13.shape, w2.shape, TOPK, None, 1, return_down_config=True
            )
        l2_bytes = props.L2_cache_size
        flush = torch.empty(
            args.flush_l2_multiple * l2_bytes // 4, dtype=torch.int32, device="cuda"
        )
        print(
            json.dumps(
                {
                    "gpu": props.name,
                    "capability": capability,
                    "sm_count": props.multi_processor_count,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "driver": get_nvidia_driver_version_str(),
                    "triton": triton.__version__,
                    "requested_fused_config": args.fused_config or "runtime",
                    "fused_moe_config": config,
                    "fused_moe_down_config": down_config,
                    "l2_bytes": l2_bytes,
                    "flush_bytes": flush.numel() * 4,
                    # Only the routed experts' weights are read.
                    "routed_weight_bytes": TOPK * 3 * I * H * 2,
                }
            ),
            flush=True,
        )

        block_means = []
        for block in range(args.blocks):
            runs = {name: [] for name in ARMS}
            for name in ARMS + ARMS[::-1]:
                samples = _time_samples(graphs[name], flush, args.repeats)
                runs[name].append(
                    {
                        "samples_us": samples,
                        "mean_us": statistics.fmean(samples),
                        "median_us": statistics.median(samples),
                    }
                )
            means = {
                name: statistics.fmean(run["mean_us"] for run in runs[name])
                for name in ARMS
            }
            block_means.append(means)
            print(
                json.dumps({"block": block, "mean_us": means, "runs": runs}), flush=True
            )
        print(
            json.dumps(
                {
                    "blocks": args.blocks,
                    "repeats": args.repeats,
                    "block_mean_us": {
                        name: _spread(m[name] for m in block_means) for name in ARMS
                    },
                    "decode_gemv_minus_fused_moe_us": _spread(
                        m["decode_gemv"] - m["fused_moe"] for m in block_means
                    ),
                    "decode_gemv_minus_fused_moe_defer_us": _spread(
                        m["decode_gemv"] - m["fused_moe_defer"] for m in block_means
                    ),
                }
            ),
            flush=True,
        )
    finally:
        _shutdown_runtime()


if __name__ == "__main__":
    main()
