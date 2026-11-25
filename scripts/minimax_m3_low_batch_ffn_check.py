"""Manual GPU check of the MiniMax-M3.1 low-batch FFN on one TP layer, through
the fused finalize + all-reduce + residual + RMSNorm boundary that consumes it.

Each rank builds its own shard of one MoE layer (random NVFP4 routed experts
processed by SGLang's loader, MXFP8 shared expert) and runs, on the same rows:

* low-batch: ``LowBatchFfn._run`` (shared expert on an alt stream, routed
  chain on the current one) -> ``MoeDeferredFinalize`` -> the boundary kernel
  ``moe_finalize_all_reduce_residual_norm`` (as ``CustomAllReduceFusion``
  calls it);
* stock: ``moe_fused_gate`` top-k -> TRT-LLM-gen routed experts with
  ``do_finalize=False`` + an MXFP8 torch shared reference -> the same
  boundary kernel.

It checks the low-batch boundary output against the stock one (per-row
relative L2), every rank bit for bit, the unfused fallback (``finish``-style
``moe_finalize_fuse_shared`` + NCCL all-reduce) against the fused boundary,
and the same in a CUDA graph (captured once, replayed on new rows), which is
where FC2's early PDL trigger meets the boundary kernel. ``--bench`` adds
CUDA-graph replay timings (cold L2) of the routed chains and the boundary.

    torchrun --nproc-per-node 4 scripts/minimax_m3_low_batch_ffn_check.py
    torchrun --nproc-per-node 4 scripts/minimax_m3_low_batch_ffn_check.py --bench
    torchrun --nproc-per-node 8 scripts/minimax_m3_low_batch_ffn_check.py --columns 384

Needs SM100-family GPUs (B200 / B300 / GB200 / GB300). Not CI-registered.
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import statistics
from types import SimpleNamespace

import torch
import torch.distributed as dist

EPS = 1e-6
ROUTED_SCALING_FACTOR = 2.0
TOLERANCE = 0.02


def _init(max_push_size: int = 4 * 1024 * 1024):
    import sglang.srt.distributed.parallel_state as ps
    from sglang.kernels.ops.communication import all_reduce_fusion
    from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )
    from sglang.srt.runtime_context import get_parallel

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    coord = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    ps._WORLD = coord
    get_parallel().override_permanently(world_group=coord)
    atexit.register(dist.destroy_process_group)
    comm = CustomAllReduceV2(
        coord.cpu_group,
        torch.device(f"cuda:{local_rank}"),
        max_pull_size=0,
        max_pull_blocks=0,
        max_push_size=max_push_size,
        max_push_blocks=512,
    )
    assert not comm.disabled, "custom all-reduce v2 is disabled on this group"
    all_reduce_fusion.register_comm(lambda: comm.obj, world_size=comm.world_size)
    register_comm_cleanup(comm)
    return coord


def _log(coord, *args):
    if coord.rank == 0:
        print(*args, flush=True)


def _probe(coord, routed, shared, stock_deferred):
    """Step 0: what this check (and the served path) assumes about the image."""
    import cutlass
    import flashinfer

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    _log(
        coord,
        f"[probe] torch {torch.__version__}, flashinfer {flashinfer.__version__}, "
        f"cutlass-dsl {getattr(cutlass, '__version__', '?')}, {props.name} "
        f"sm_{props.major}{props.minor}, {props.multi_processor_count} SMs, "
        f"world {coord.world_size}",
    )
    for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
        tensor = getattr(routed, name)
        _log(coord, f"[probe] experts.{name}: {tuple(tensor.shape)} {tensor.dtype}")
    for name in ("g1_alphas", "g2_alphas"):
        tensor = getattr(routed, name)
        _log(
            coord,
            f"[probe] experts.{name}: {tuple(tensor.shape)} {tensor.dtype} "
            f"contiguous={tensor.is_contiguous()}",
        )
    _log(
        coord,
        f"[probe] shared gate_up swizzled scale: "
        f"{tuple(shared.gate_up_swizzled.shape)} {shared.gate_up_swizzled.dtype}",
    )
    _log(
        coord,
        "[probe] stock deferred handoff: gemm2_out "
        f"{tuple(stock_deferred.gemm2_out.shape)} {stock_deferred.gemm2_out.dtype}, "
        f"expert_weights {tuple(stock_deferred.expert_weights.shape)} "
        f"{stock_deferred.expert_weights.dtype} (low-batch hands FP32), permuted "
        f"{tuple(stock_deferred.expanded_idx_to_permuted_idx.shape)} "
        f"{stock_deferred.expanded_idx_to_permuted_idx.dtype}",
    )


def _fake_moe(routed, shared, bias, gate):
    """The attributes of a ``MiniMaxM3MoE`` that ``LowBatchFfn`` reads."""

    def linear(weight, swizzled):
        return SimpleNamespace(weight=weight, weight_scale_inv_swizzled=swizzled)

    return SimpleNamespace(
        experts=routed,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        e_score_correction_bias=bias,
        shared_experts=SimpleNamespace(
            gate_up_proj=linear(shared.gate_up, shared.gate_up_swizzled),
            down_proj=linear(shared.down, shared.down_swizzled),
        ),
        alt_stream=torch.cuda.Stream(),
        _compute_router_logits=lambda x: torch.mm(x, gate.t(), out_dtype=torch.float32),
    )


def _boundary(handoff, residual, gamma, world_size):
    from sglang.kernels.ops.communication import all_reduce_fusion

    return all_reduce_fusion.moe_finalize_all_reduce_residual_norm(
        handoff.routed_output,
        handoff.permuted_indices.reshape(-1),
        handoff.expert_weights,
        handoff.expert_weights.shape[1],
        handoff.gated_shared_output,
        residual,
        gamma,
        EPS,
        world_size=world_size,
        hidden_dim=residual.shape[1],
        residual_out=torch.empty_like(residual),
    )


def _stock_handoff(x, moe, routed, shared):
    from sglang.srt.layers.layer_boundary.fusions.moe_finalize import (
        MoeDeferredFinalize,
    )
    from sglang.test.kernels import minimax_m3_low_batch_ffn as ref

    logits = moe._compute_router_logits(x)
    weights, ids = ref.stock_topk(
        logits, moe.e_score_correction_bias, ROUTED_SCALING_FACTOR
    )
    deferred = ref.stock_routed_deferred(x, weights, ids, routed)
    handoff = MoeDeferredFinalize.from_flashinfer(
        deferred,
        gated_shared_output=ref.shared_reference(x, shared),
        m=x.shape[0],
        reduce=lambda out: out,
    )
    return handoff, deferred, logits


def _gather(coord, tensor):
    out = [torch.empty_like(tensor) for _ in range(coord.world_size)]
    dist.all_gather(out, tensor, group=coord.device_group)
    return out


def _compare(coord, label, rows, got, want, skip_rows):
    from sglang.test.kernels import minimax_m3_low_batch_ffn as ref

    errors = ref.relative_errors(got, want)
    errors = errors[~skip_rows] if bool(skip_rows.any()) else errors
    worst = float(errors.max()) if errors.numel() else 0.0
    status = "ok" if worst < TOLERANCE else "FAIL"
    _log(coord, f"  rows {rows:2d} {label}: max row rel err {worst:.4f} {status}")
    return worst < TOLERANCE


def _same_on_all_ranks(coord, label, rows, tensor):
    same = all(torch.equal(other, tensor) for other in _gather(coord, tensor))
    _log(coord, f"  rows {rows:2d} {label}: identical on all ranks {same}")
    return same


def _check_rows(coord, ffn, moe, routed, shared, rows, generator, graph):
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        finalize_flashinfer_trtllm_deferred_output,
    )
    from sglang.test.kernels import minimax_m3_low_batch_ffn as ref

    hidden = routed.hidden_size
    x = (torch.randn(rows, hidden, generator=generator, device="cuda") * 2).bfloat16()
    residual = torch.randn(rows, hidden, generator=generator, device="cuda").bfloat16()
    gamma = (
        1 + 0.1 * torch.randn(hidden, generator=generator, device="cuda")
    ).bfloat16()
    if rows > 1:
        x[1] = 0

    stock, _, logits = _stock_handoff(x, moe, routed, shared)
    stock_normed, stock_residual = _boundary(stock, residual, gamma, coord.world_size)
    ties = ref.near_tie_rows(logits, moe.e_score_correction_bias)

    if not graph:
        handoff = ffn._run(moe, x)
        normed, residual_out = _boundary(handoff, residual, gamma, coord.world_size)
    else:
        static_x = torch.zeros_like(x)
        static_residual = torch.zeros_like(residual)
        ffn._run(moe, static_x)
        torch.cuda.synchronize()
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph):
            handoff = ffn._run(moe, static_x)
            normed, residual_out = _boundary(
                handoff, static_residual, gamma, coord.world_size
            )
        # Replay on other rows first, so stale state from it would show.
        static_x.copy_(x.roll(1, 0))
        static_residual.copy_(residual.roll(1, 0))
        cuda_graph.replay()
        static_x.copy_(x)
        static_residual.copy_(residual)
        cuda_graph.replay()
    torch.cuda.synchronize()

    mode = "graph" if graph else "eager"
    ok = _compare(
        coord, f"{mode} normed vs stock", rows, normed, stock_normed, ties
    ) & _compare(
        coord,
        f"{mode} residual vs stock",
        rows,
        residual_out,
        stock_residual,
        ties,
    )
    ok &= _same_on_all_ranks(coord, f"{mode} normed", rows, normed)

    # The unfused fallback (MoeDeferredFinalize.finish) on the same handoff.
    deferred = SimpleNamespace(
        gemm2_out=handoff.routed_output,
        expanded_idx_to_permuted_idx=handoff.permuted_indices.reshape(-1),
        expert_weights=handoff.expert_weights,
        top_k=handoff.expert_weights.shape[1],
    )
    partial = finalize_flashinfer_trtllm_deferred_output(
        deferred, handoff.gated_shared_output
    ).float()
    dist.all_reduce(partial, group=coord.device_group)
    unfused_residual = partial + residual.float()
    unfused_normed = (
        unfused_residual
        * torch.rsqrt(unfused_residual.pow(2).mean(-1, keepdim=True) + EPS)
        * gamma.float()
    )
    no_skip = torch.zeros_like(ties)
    ok &= _compare(
        coord,
        f"{mode} residual vs unfused finish + NCCL",
        rows,
        residual_out,
        unfused_residual,
        no_skip,
    )
    ok &= _compare(
        coord,
        f"{mode} normed vs unfused finish + NCCL",
        rows,
        normed,
        unfused_normed,
        no_skip,
    )
    return ok


def _bench(coord, ffn, moe, routed, shared, rows_list, replays):
    from sglang.test.kernels import minimax_m3_low_batch_ffn as ref

    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    hidden = routed.hidden_size

    def time_graph(fn):
        fn()
        torch.cuda.synchronize()
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph):
            fn()
        times = []
        for _ in range(replays):
            flush.zero_()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            cuda_graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1e3)
        return statistics.median(times)

    _log(coord, "[bench] median us per CUDA-graph replay, cold L2 (256 MB flush)")
    for rows in rows_list:
        x = torch.randn(rows, hidden, device="cuda").bfloat16()
        logits = moe._compute_router_logits(x)
        bias = moe.e_score_correction_bias

        def stock_routed():
            weights, ids = ref.stock_topk(logits, bias, ROUTED_SCALING_FACTOR)
            ref.stock_routed_deferred(x, weights, ids, routed)

        def low_routed():
            ffn._routed_ffn(
                x,
                logits,
                bias=bias,
                routed_scaling_factor=ROUTED_SCALING_FACTOR,
                w13=routed.w13_weight,
                w13_scale=routed.w13_weight_scale,
                w2=routed.w2_weight,
                w2_scale=routed.w2_weight_scale,
                alpha1=routed.g1_alphas,
                alpha2=routed.g2_alphas,
                columns=routed.intermediate_size_per_partition,
            )

        stock_us = time_graph(stock_routed)
        low_us = time_graph(low_routed)
        shared_us = time_graph(lambda: ffn._shared(x))
        full_us = time_graph(lambda: ffn._run(moe, x))
        _log(
            coord,
            f"  rows {rows:2d}: stock top-k+quant+routed {stock_us:7.1f}  "
            f"low-batch routed {low_us:7.1f}  low-batch shared {shared_us:7.1f}  "
            f"low-batch routed||shared {full_us:7.1f}",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--columns", type=int, default=None, help="768 or 384")
    parser.add_argument("--rows", default="1,3,8,12,16,24,32")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--replays", type=int, default=200)
    args = parser.parse_args()

    coord = _init()
    logging.disable(logging.INFO)
    from sglang.srt.models.minimax_common.nvidia.low_batch_ffn import LowBatchFfn
    from sglang.test.kernels import minimax_m3_low_batch_ffn as ref

    columns = args.columns or {4: 768, 8: 384}[coord.world_size]
    rows_list = [int(r) for r in args.rows.split(",")]
    # Each rank its own shard; router and inputs shared by all ranks.
    routed = ref.build_routed_layer(
        columns, torch.Generator(device="cuda").manual_seed(1000 + coord.rank)
    )
    shared = ref.build_shared(
        columns, torch.Generator(device="cuda").manual_seed(2000 + coord.rank)
    )
    common = torch.Generator(device="cuda").manual_seed(0)
    hidden = routed.hidden_size
    gate = (torch.randn(128, hidden, generator=common, device="cuda") * 0.02).bfloat16()
    bias = torch.rand(128, generator=common, device="cuda") * 2
    moe = _fake_moe(routed, shared, bias, gate)

    probe_x = torch.randn(8, hidden, device="cuda").bfloat16()
    _probe(coord, routed, shared, _stock_handoff(probe_x, moe, routed, shared)[1])

    ffn = LowBatchFfn(moe)  # compiles and warms up, outside any capture
    _log(coord, f"[check] TP{coord.world_size}, {columns} columns per rank")
    ok = True
    for graph in (False, True):
        for rows in rows_list:
            generator = torch.Generator(device="cuda").manual_seed(rows)
            ok &= _check_rows(coord, ffn, moe, routed, shared, rows, generator, graph)
    _log(coord, f"[check] {'PASS' if ok else 'FAIL'}")
    if args.bench:
        _bench(coord, ffn, moe, routed, shared, rows_list, args.replays)
    dist.barrier()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
