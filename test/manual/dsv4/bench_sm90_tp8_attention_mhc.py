"""8 x Hopper native witness for the opt-in V4.1 attention mHC epilogue.

Run on an exclusive host with 8 peer-accessible SM90 GPUs, under the same
SGLang ENV as the server. No model weights, network listener or gateway needed:

    SGLANG_OPT_DSV41_SM90_ATTN_MHC_AR=1 \
      timeout --kill-after=30s 900s python -m torch.distributed.run \
      --standalone --nproc-per-node=8 test/manual/dsv4/bench_sm90_tp8_attention_mhc.py \
      --output mhc-sm90-tp8.json

Correctness uses actual model-boundary dispatch, the installed TP8 V2 lease,
an unfused production-kernel reference and an independent FP32 oracle. The
benchmark times the full attention boundary only, not whole-model throughput.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.srt.distributed import parallel_state as ps
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.distributed.device_communicators.symm_workspace import (
    finalize_symm_workspaces,
    v2_push_slot_bytes,
)
from sglang.srt.layers.layer_boundary.layout import Layout, SumGroup
from sglang.srt.layers.layer_boundary.row_transport import TransportRows
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
    can_fuse_all_reduce,
    register_fused_all_reduce_comm,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models import deepseek_v4
from sglang.srt.models import deepseek_v4_boundary as boundary
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.runtime_context import get_parallel
from sglang.test.test_utils import publish_build_topology

H, EPS, TP = 5120, 1e-6, 8
ROWS = (1, 6, 12, 13)


def init():
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == TP
    assert torch.cuda.device_count() == TP
    torch.cuda.set_device(local_rank)
    assert torch.cuda.get_device_capability() == (9, 0)
    ps.set_custom_all_reduce(True)
    ps.init_distributed_environment(
        world_size=TP, rank=rank, local_rank=local_rank, timeout=120
    )
    publish_build_topology(world_rank=rank, tp_size=TP, ep_size=TP)
    ps.initialize_model_parallel()
    group = get_parallel().tp_group
    assert get_parallel().attn_tp_group is group
    assert get_parallel().moe_ep_size == TP
    assert isinstance(group.ca_comm, CustomAllReduceV2) and not group.ca_comm.disabled
    assert v2_push_slot_bytes(group.cpu_group) == 128 * 1024
    assert mhc.sm90_attn_mhc_enabled()
    # Use the same registration/finalization order as precompile and BaseRunner.
    register_fused_all_reduce_comm()
    finalize_symm_workspaces()
    assert can_fuse_all_reduce(12, H, group=group)
    assert not can_fuse_all_reduce(13, H, group=group)
    return group


def make_inputs(rows, rank, seed):
    common = torch.Generator(device="cuda").manual_seed(seed)
    local = torch.Generator(device="cuda").manual_seed(seed * 31 + rank)
    partial = (
        torch.randn(rows, H, device="cuda", generator=local) / TP**0.5
    ).bfloat16()
    residual = torch.randn(rows, 4, H, device="cuda", generator=common).bfloat16()
    pre = torch.rand(rows, 4, device="cuda", generator=common)
    post = 2 * torch.rand(rows, 4, device="cuda", generator=common)
    comb = torch.rand(rows, 4, 4, device="cuda", generator=common) + 0.1
    for _ in range(10):
        comb = comb / comb.sum(-1, keepdim=True)
        comb = comb / comb.sum(-2, keepdim=True)
    return partial, residual, pre, post, comb


def steps(group, inputs, norm, mode):
    cfg = mhc.HcConfig(4, 20, 1e-5, EPS, H, True, False)
    hc = mhc.HcSubLayer(cfg, None, None, None, norm)
    next_boundary = mhc.make_boundary(norm, accepts_mxfp8=False, hc=hc)
    layer = SimpleNamespace(
        config=SimpleNamespace(model_type="deepseek_v41"),
        self_attn=SimpleNamespace(attn_tp_size=TP),
        attn_hc=hc,
        ffn_hc=hc,
        hc_cfg=cfg,
        hc_stats_stream=None,
        post_attention_layernorm=norm,
        boundary_mhc_fusions=True,
        boundary_pre_from_prev=True,
        _can_fuse_attn_mhc=False,
    )
    layer._can_fuse_sm90_attn_mhc = (
        deepseek_v4.DeepseekV4CudaDecoderLayer._can_fuse_sm90_attn_mhc.func(layer)
    )
    layout = Layout(frozenset())
    rows = TransportRows(layout, layout, layout, SumGroup.ATTN_TP)
    call = boundary.LayerCall()
    batch = SimpleNamespace(forward_mode=mode, can_run_tbo=False)
    reader = boundary.HcRead(layer, attention=False)
    update = boundary.HcUpdate(layer, attention=True)
    fallback = inputs[0].shape[0] > mhc.SM90_ATTN_MHC_MAX_ROWS

    def baseline():
        x, residual, pre, post, comb = inputs
        # The graph-mode 2shot_pull may consume its input in place. A real
        # wo_b produces a new partial at every boundary. Do the same for the
        # unmeasured fallback while keeping the measured push path clone-free.
        if fallback:
            x = x.clone()
        state = mhc.run_attn_post(
            hc,
            group.all_reduce(x),
            residual,
            stats_stream=None,
            next=next_boundary,
            world_size=TP,
            precomputed=(pre, post, comb),
        )
        return state.residual, mhc.combine(hc, state)

    def candidate():
        x, residual, pre, post, comb = inputs
        if fallback:
            x = x.clone()
        transport = boundary.AttentionTransport(layer, x, batch, call, rows)
        # The real boundary must reject 13 rows, not expand the push plane.
        assert transport.defers_sum == (x.shape[0] <= 12)
        owed = transport.prepare_output(mhc.AttnOutput(x))
        if transport.defers_sum:
            result = reader._complete_read(
                owed,
                boundary.ReadState(residual, stats=(pre, post, comb), call=call),
                batch,
                norm,
                update=update,
            )
            assert result is not None
            normalized, state = result
            return state.streams, normalized
        state = mhc.run_attn_post(
            hc,
            owed.complete(),
            residual,
            stats_stream=None,
            next=next_boundary,
            world_size=TP,
            precomputed=(pre, post, comb),
        )
        return state.residual, mhc.combine(hc, state)

    return baseline, candidate


def gather(group, tensor):
    out = [torch.empty_like(tensor) for _ in range(TP)]
    dist.all_gather(out, tensor, group=group.device_group)
    return out


def oracle(group, inputs, gamma):
    x, residual, pre, post, comb = inputs
    reduced = torch.stack([t.float() for t in gather(group, x)]).sum(0).bfloat16()
    updated = (
        post[:, :, None] * reduced[:, None].float()
        + (comb[:, :, :, None] * residual[:, :, None].float()).sum(1)
    ).bfloat16()
    collapsed = (pre[:, :, None] * updated.float()).sum(1).bfloat16().float()
    normalized = collapsed * torch.rsqrt(collapsed.square().mean(-1, True) + EPS)
    return updated, (normalized * gamma.float()).bfloat16()


def check(group, got, ref, phase):
    errors = []
    for name, actual, expected in zip(("residual", "normalized"), got, ref):
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=2e-2,
            atol=2e-2,
            msg=lambda msg: f"{phase}/{name}: {msg}",
        )
        for peer in gather(group, actual):
            assert torch.equal(actual, peer), f"{phase}/{name}: differs across ranks"
        errors.append(float((actual.float() - expected.float()).abs().max()))
    return errors


def capture(group, step, *, repeats=1):
    with group.graph_capture() as context:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph, stream=context.stream, capture_error_mode="thread_local"
        ):
            for _ in range(repeats):
                output = step()
    torch.cuda.synchronize()
    return graph, output


def bench(group, first, second, count, trials, graph=False):
    # Alternate A/B and B/A to temper drift; the slowest rank dominates latency.
    variants = (first, second)
    if graph:
        # Hold the last graph outputs as well as the graph itself until all
        # replays finish; their storage belongs to the graph's memory pool.
        captures = [capture(group, step, repeats=count) for step in variants]
        variants = tuple(g.replay for g, _ in captures)
    samples = [[], []]
    for trial in range(trials):
        for index in (0, 1) if trial % 2 == 0 else (1, 0):
            step = variants[index]
            for _ in range(3):
                step()
            torch.cuda.synchronize()
            dist.barrier(group=group.cpu_group)
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(1 if graph else count):
                step()
            end.record()
            end.synchronize()
            latency = start.elapsed_time(end) * 1000 / count
            slowest = torch.tensor(latency, dtype=torch.float64)
            dist.all_reduce(slowest, op=dist.ReduceOp.MAX, group=group.cpu_group)
            samples[index].append(float(slowest))
    return dict(
        baseline_us=samples[0],
        candidate_us=samples[1],
        baseline_median_us=statistics.median(samples[0]),
        candidate_median_us=statistics.median(samples[1]),
    )


@torch.inference_mode()
def main(args):
    group = init()
    report = dict(
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        tp_size=TP,
        ep_size=get_parallel().moe_ep_size,
        push_slot_bytes=v2_push_slot_bytes(group.cpu_group),
        multicast=group.ca_comm.has_multicast,
        cases=[],
    )
    norm = RMSNorm(H, eps=EPS, weight_dtype=torch.bfloat16).cuda()
    norm.weight.copy_(
        (
            1
            + torch.rand(
                H,
                device="cuda",
                generator=torch.Generator(device="cuda").manual_seed(7),
            )
            * 0.1
        ).bfloat16()
    )
    for rows in args.rows:
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            label = f"{rows}/{mode.name}"
            inputs = make_inputs(rows, group.rank_in_group, 42)
            # Keep a separate reference to the original partial; the oracle
            # must never read a value consumed by a previous collective.
            eager_expected = tuple(t.clone() for t in inputs)
            # Prove the comparator is the actual default AR V2 path, not NCCL fallback.
            assert group.ca_comm.should_custom_ar(inputs[0])
            assert group._resolve_outplace_all_reduce_method(inputs[0]) == "ca"
            assert not group._can_use_flashinfer_allreduce(inputs[0])
            baseline, candidate = steps(group, inputs, norm, mode)
            ref = baseline()
            got = candidate()
            for tensor, original in zip(inputs, eager_expected):
                torch.testing.assert_close(tensor, original, atol=0, rtol=0)
            eager_err = check(group, got, ref, f"{label}/eager")
            oracle_err = check(
                group,
                got,
                oracle(group, eager_expected, norm.weight),
                f"{label}/oracle",
            )

            # Interleave fused/ordinary AR on the same push epochs, then replay
            # with changed data to catch incorrect phase reuse and stale inputs.
            def mixed():
                result = candidate()
                # 13 rows exceed the V2 push slot, so its normal graph AR
                # uses a fresh input (as the model does) for each 2shot pull.
                partial = (
                    inputs[0]
                    if rows <= mhc.SM90_ATTN_MHC_MAX_ROWS
                    else inputs[0].clone()
                )
                return result, group.all_reduce(partial), candidate()

            graph, (first, reduced, last) = capture(group, mixed)
            replay_error = []
            replay_oracle_error = []
            for seed in (43, 97, 128):
                expected = make_inputs(rows, group.rank_in_group, seed)
                for target, source in zip(inputs, expected):
                    target.copy_(source)
                graph.replay()
                torch.cuda.synchronize()
                for tensor, source in zip(inputs, expected):
                    torch.testing.assert_close(
                        tensor,
                        source,
                        atol=0,
                        rtol=0,
                        msg=lambda msg: (
                            f"{label}/graph seed={seed}: source mutated: {msg}"
                        ),
                    )
                ref = baseline()
                replay_error.append(
                    check(group, first, ref, f"{label}/graph first seed={seed}")
                )
                replay_error.append(
                    check(group, last, ref, f"{label}/graph last seed={seed}")
                )
                replay_ref = oracle(group, expected, norm.weight)
                replay_oracle_error.append(
                    check(
                        group,
                        first,
                        replay_ref,
                        f"{label}/graph oracle first seed={seed}",
                    )
                )
                replay_oracle_error.append(
                    check(
                        group,
                        last,
                        replay_ref,
                        f"{label}/graph oracle last seed={seed}",
                    )
                )
                torch.testing.assert_close(
                    reduced,
                    group.all_reduce(expected[0].clone()),
                    rtol=2e-2,
                    atol=2e-2,
                    msg=lambda msg: f"{label}/graph ordinary AR seed={seed}: {msg}",
                )
            del graph

            case = dict(
                rows=rows,
                mode=mode.name,
                fused=rows <= 12,
                eager_max_abs=eager_err,
                oracle_max_abs=oracle_err,
                replay_max_abs=replay_error,
                replay_oracle_max_abs=replay_oracle_error,
            )
            if rows <= mhc.SM90_ATTN_MHC_MAX_ROWS and not args.correctness_only:
                case["eager"] = bench(
                    group, baseline, candidate, args.iterations, args.trials
                )
                case["graph"] = bench(
                    group, baseline, candidate, args.iterations, args.trials, graph=True
                )
            report["cases"].append(case)
            if group.rank_in_group == 0:
                print(json.dumps(case), flush=True)
    dist.barrier(group=group.cpu_group)
    if group.rank_in_group == 0:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--rows", type=int, nargs="+", choices=ROWS, default=ROWS)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert 0 < args.iterations <= 256 and 0 < args.trials <= 20
    main(args)
