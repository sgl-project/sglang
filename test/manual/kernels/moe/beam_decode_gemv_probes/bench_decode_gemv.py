"""Single-token latency of the Beam decode GEMV against the Triton fused MoE.

Manual and not CI-registered. Run the correctness test first, then this probe,
from the repository root on one SM103 GPU (GB300):

    python test/manual/kernels/moe/test_beam_decode_gemv.py
    python test/manual/kernels/moe/beam_decode_gemv_probes/bench_decode_gemv.py

Both timed paths return the same [1, 5120] BF16 sum over the 8 routed experts:
`fused_experts_impl` with interleaved SwiGLU and its own expert combine, and
`beam_decode_gemv` followed by an explicit FP32 expert sum rounded to BF16.
Neither path includes TP communication or the fused all-reduce boundary.
"""

import argparse
import json
import os
import statistics

import torch
import triton

from sglang.kernels.ops.moe.beam_decode_gemv import beam_decode_gemv

# Beam's experts, hidden size, TP4 intermediate size and top-k.
E, H, I, TOPK = 256, 5120, 640, 8


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


def _time_us(graph, flush, repeats):
    """Median of `repeats` single replays, each after an untimed L2 flush."""
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
    return statistics.median(s.elapsed_time(e) * 1e3 for s, e in events)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--blocks", type=int, default=6, help="ABBA blocks (>= 5)")
    parser.add_argument("--repeats", type=int, default=20, help="replays per run")
    parser.add_argument("--flush-l2-multiple", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.blocks < 5 or args.flush_l2_multiple < 5:
        parser.error("use at least 5 ABBA blocks and a flush of at least 5x L2")

    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    if capability != (10, 3):
        raise SystemExit(f"beam_decode_gemv targets SM103; this GPU is {capability}")

    _init_runtime()
    try:
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts_impl,
        )

        torch.manual_seed(args.seed)
        # Interleaved gate/up rows: row 2j is gate j and row 2j + 1 is up j.
        w13 = (torch.randn(E, 2 * I, H, device="cuda") / H**0.5).bfloat16()
        w2 = (torch.randn(E, H, I, device="cuda") / I**0.5).bfloat16()
        hidden = torch.randn(1, H, device="cuda").bfloat16()
        ids = torch.randperm(E, device="cuda")[:TOPK].view(1, TOPK).int()
        weights = torch.softmax(torch.randn(1, TOPK, device="cuda"), -1)

        def fused():
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
            )

        def gemv():
            out = beam_decode_gemv(hidden, w13, w2, weights, ids)
            return out.sum(dim=1, dtype=torch.float32).bfloat16()

        graphs = {"fused_moe": _capture(fused), "decode_gemv": _capture(gemv)}
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
                    "triton": triton.__version__,
                    "l2_bytes": l2_bytes,
                    "flush_bytes": flush.numel() * 4,
                    # Only the routed experts' weights are read.
                    "routed_weight_bytes": TOPK * 3 * I * H * 2,
                    "max_abs_output_diff": (
                        graphs["fused_moe"][1].float()
                        - graphs["decode_gemv"][1].float()
                    )
                    .abs()
                    .max()
                    .item(),
                }
            ),
            flush=True,
        )

        pairs = []
        for block in range(args.blocks):
            times = {name: [] for name in graphs}
            for name in ("fused_moe", "decode_gemv", "decode_gemv", "fused_moe"):
                times[name].append(_time_us(graphs[name][0], flush, args.repeats))
            pair = {name: statistics.mean(t) for name, t in times.items()}
            pair["delta_us"] = pair["decode_gemv"] - pair["fused_moe"]
            pairs.append(pair)
            print(json.dumps({"block": block, **pair}), flush=True)
        print(
            json.dumps(
                {
                    "blocks": args.blocks,
                    "repeats": args.repeats,
                    "median_fused_moe_us": statistics.median(
                        p["fused_moe"] for p in pairs
                    ),
                    "median_decode_gemv_us": statistics.median(
                        p["decode_gemv"] for p in pairs
                    ),
                    "median_delta_us": statistics.median(p["delta_us"] for p in pairs),
                }
            ),
            flush=True,
        )
    finally:
        _shutdown_runtime()


if __name__ == "__main__":
    main()
