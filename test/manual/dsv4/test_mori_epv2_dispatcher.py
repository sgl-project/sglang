"""Eight-GPU identity round trip through the SGLang MORI EPv2 adapter.

TBO=1 runs the FP4-asymmetric two-child TBO path instead; there TEST_CUDA_GRAPH=1
captures and replays REPLAY_ITERS times (default 3), and
SGLANG_MORI_EPV2_AITER_DIRECT_OUTPUT=0/1 exercises staging or direct writes.
The identity expert checks buffer correctness, not AITER performance.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

import sglang.srt.layers.dp_attention as dp_attention
import sglang.srt.layers.moe.token_dispatcher.moriep as adapter
from sglang.srt.batch_overlap.two_batch_overlap import MaybeTboDeepEPDispatcher
from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.moriep import round_logical_recv_rows
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode, MoeA2ABackend
from sglang.srt.runtime_context import get_flags


def _fp4_lookup():
    # e2m1 code points, low nibble first.
    values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    return torch.tensor(
        values + [-v for v in values], dtype=torch.float32, device="cuda"
    )


def _expert_output(dispatched, dispatch: str, fp4_lookup=None):
    if dispatch == "bf16":
        output = dispatched.hidden_states
        return output[: dispatched.recv_cap] if dispatched.recv_cap > 0 else output

    if dispatch == "fp4":
        packed = dispatched.hidden_states.view(torch.uint8)
        nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)
        values = fp4_lookup[nibbles.long()]
    else:
        values = dispatched.hidden_states.float()
    # fp8 uses fp32 scales per 128 columns; fp4 and mxfp8 use e8m0 per 32.
    block = 128 if dispatch == "fp8" else 32
    scales = dispatched.hidden_states_scale.float().repeat_interleave(block, dim=1)
    output = (values * scales[:, : values.shape[1]]).to(torch.bfloat16)
    valid_rows = (
        torch.arange(output.shape[0], device=output.device)
        < dispatched.num_recv_tokens_per_expert.reshape(-1)[0]
    )
    output = torch.where(valid_rows[:, None], output, torch.zeros_like(output))
    return output[: dispatched.recv_cap] if dispatched.recv_cap > 0 else output


class _Group:
    def __init__(self, process_group):
        self.cpu_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.rank_in_group = dist.get_rank(process_group)

    def broadcast_object(self, obj, src=0):
        values = [obj if self.rank_in_group == src else None]
        dist.broadcast_object_list(values, src=src, group=self.cpu_group)
        return values[0]


def _expected_unique_destinations(topk_ids: torch.Tensor, experts_per_rank: int):
    return torch.tensor(
        [
            len(set(row.tolist()))
            for row in (topk_ids.cpu().to(torch.int64) // experts_per_rank)
        ],
        dtype=torch.float32,
    ).view(-1, 1)


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world_size == 8
    torch.cuda.set_device(local_rank)
    if os.environ.get("TBO", "0") == "1":
        _run_tbo(rank, world_size)

    hidden_size = int(os.environ.get("HIDDEN", "7168"))
    topk = int(os.environ.get("TOPK", "6"))
    experts_per_rank = int(os.environ.get("EPR", "48"))
    num_experts = world_size * experts_per_rank
    tokens = int(os.environ.get("TOKENS", "64"))
    # DISPATCH / COMBINE map to SGLANG_MORI_DISPATCH_DTYPE / SGLANG_MORI_COMBINE_DTYPE.
    dispatch = os.environ.get(
        "DISPATCH", "fp4" if os.environ.get("FP4", "0") == "1" else "bf16"
    )
    combine = os.environ.get("COMBINE", "bf16")
    os.environ["SGLANG_MORI_DISPATCH_DTYPE"] = dispatch
    os.environ["SGLANG_MORI_COMBINE_DTYPE"] = combine
    fp4_enabled = dispatch == "fp4"
    fp4_lookup = _fp4_lookup() if fp4_enabled else None
    if os.environ.get("EMPTY_LAST_RANK", "0") == "1" and rank == world_size - 1:
        tokens = 0

    # ATTN_TP > 1 models attention TP: each DP group's TOKENS * (group + 1) rows
    # are tensor_split over its attention-TP ranks, as the model does before MoE.
    attn_tp_size = int(os.environ.get("ATTN_TP", "1"))
    attn_dp_size = world_size // attn_tp_size
    attn_dp_rank, attn_tp_rank = divmod(rank, attn_tp_size)
    if attn_tp_size == 1:
        sender_rows = [None] * world_size
        dist.all_gather_object(sender_rows, tokens)
    else:
        sender_rows = [tokens * (group + 1) for group in range(attn_dp_size)]
        tokens = (
            torch.empty(sender_rows[attn_dp_rank], device="cpu")
            .tensor_split(attn_tp_size)[attn_tp_rank]
            .numel()
        )
    # STALE_ROWS models DSpark draft forwards that keep the target step's metadata.
    stale_rows = os.environ.get("STALE_ROWS")
    metadata_rows = [int(stale_rows)] if stale_rows else sender_rows
    dp_attention.set_dp_buffer_len(sum(sender_rows), tokens, False, metadata_rows)

    adapter.is_tbo_enabled = lambda: False
    adapter.get_parallel = lambda: SimpleNamespace(
        moe_ep_size=world_size,
        moe_ep_rank=rank,
        tp_size=world_size,
        attn_dp_size=attn_dp_size,
        attn_dp_rank=attn_dp_rank,
        attn_tp_size=attn_tp_size,
        attn_tp_rank=attn_tp_rank,
        attn_cp_size=1,
        moe_tp_size=1,
        moe_dp_size=1,
        launch_world_rank=rank,
    )
    group = _Group(dist.group.WORLD)
    envs.SGLANG_MORI_EP_V2.set(True)
    dispatcher = adapter.MoriEPDispatcher(
        group=group,
        router_topk=topk,
        num_experts=num_experts,
        num_local_experts=experts_per_rank,
        hidden_size=hidden_size,
        params_dtype=torch.bfloat16,
        deepep_mode=DeepEPMode.NORMAL,
    )
    dispatcher.set_quant_config(
        {"weight_dtype": (torch.float4_e2m1fn_x2 if fp4_enabled else torch.bfloat16)}
    )

    generator = torch.Generator(device="cpu").manual_seed(20260803 + rank)
    hidden = torch.randn(
        tokens, hidden_size, generator=generator, dtype=torch.bfloat16
    ).cuda()
    if os.environ.get("SKEWED", "0") == "1":
        first_expert = int(os.environ.get("SKEW_RANK", "0")) * experts_per_rank
        topk_ids = (
            torch.arange(first_expert, first_expert + topk, dtype=torch.int32)
            .repeat(tokens, 1)
            .cuda()
        )
    else:
        topk_ids = torch.randint(
            0,
            num_experts,
            (tokens, topk),
            generator=generator,
            dtype=torch.int32,
        ).cuda()
    topk_weights = torch.rand(
        tokens, topk, generator=generator, dtype=torch.float32
    ).cuda()
    topk_output = StandardTopKOutput(topk_weights, topk_ids, None)

    dispatched = dispatcher.dispatch(hidden, topk_output)
    impl = dispatcher._get_impl()
    assert impl.dispatch_dtype.name == dispatch, (impl.dispatch_dtype, dispatch)
    assert impl.combine_dtype.name == combine, (impl.combine_dtype, combine)
    expected_cap = 0
    if impl._manual_recv_cap > 0:
        expected_cap = min(impl._manual_recv_cap, impl.mori_op.cfg.effective_max_recv)
    elif impl._trim_recv:
        cluster_rows = (
            sum(sender_rows)
            if attn_dp_size > 1
            else attn_tp_size * tokens + attn_tp_rank
        )
        if cluster_rows > 0:
            rounded = round_logical_recv_rows(
                cluster_rows, pow2_buckets=impl._recv_cap_pow2_buckets
            )
            # A bound that covers the whole receive buffer keeps the full view.
            if rounded < impl.mori_op.cfg.effective_max_recv:
                expected_cap = rounded
    assert dispatched.recv_cap == expected_cap
    recv_rows = int(dispatched.num_recv_tokens_per_expert.reshape(-1)[0].item())
    assert recv_rows <= sum(sender_rows), (recv_rows, sender_rows)
    assert expected_cap == 0 or recv_rows <= expected_cap, (recv_rows, expected_cap)
    combined = dispatcher.combine(
        (
            _expert_output(dispatched, dispatch, fp4_lookup),
            dispatched.topk_ids,
            dispatched.topk_weights,
        )
    )
    torch.cuda.synchronize()
    dispatcher._get_impl().mori_op.comm.barrier()

    expected = (
        _expected_unique_destinations(topk_ids, experts_per_rank) * hidden.float().cpu()
    ).to(torch.bfloat16)
    # Arbitrary bounds: fp4 keeps ~1 mantissa bit, e4m3 keeps 3 (per quantization hop).
    if fp4_enabled:
        tolerance = 6e-1
    elif dispatch != "bf16" or combine != "bf16":
        tolerance = 1.5e-1
    else:
        tolerance = 2e-2
    error = (combined.float().cpu() - expected.float()).abs()
    ok = torch.allclose(
        combined.float().cpu(), expected.float(), atol=tolerance, rtol=tolerance
    )
    failures = torch.tensor([not ok], dtype=torch.int32)
    dist.all_reduce(failures)
    if rank == 0:
        print(
            "# MORI-EPV2-SGLANG-IDENTITY: "
            f"{'PASS' if failures.item() == 0 else 'FAIL'} "
            f"tokens={tokens} hidden={hidden_size} topk={topk} "
            f"skewed={os.environ.get('SKEWED', '0')} "
            f"empty_last_rank={os.environ.get('EMPTY_LAST_RANK', '0')} "
            f"attn_tp={attn_tp_size} sender_rows={sender_rows} stale_rows={stale_rows} "
            f"recv_cap={dispatched.recv_cap}",
            f"dispatch={dispatch} combine={combine} "
            f"combine_mode={impl.mori_op.cfg.combine_mode}",
            f"mean_abs_error={error.mean().item():.6f}",
            f"max_abs_error={error.max().item() if error.numel() else 0:.6f}",
            flush=True,
        )

    graph_replays = int(os.environ.get("GRAPH_REPLAYS", "0"))
    if graph_replays:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_dispatched = dispatcher.dispatch(hidden, topk_output)
            graph_combined = dispatcher.combine(
                (
                    _expert_output(graph_dispatched, dispatch, fp4_lookup),
                    graph_dispatched.topk_ids,
                    graph_dispatched.topk_weights,
                )
            )
        torch.cuda.synchronize()
        dispatcher._get_impl().mori_op.comm.barrier()
        for _ in range(graph_replays):
            graph.replay()
            torch.cuda.synchronize()
            dispatcher._get_impl().mori_op.comm.barrier()
        graph_ok = torch.allclose(
            graph_combined.float().cpu(),
            expected.float(),
            atol=tolerance,
            rtol=tolerance,
        )
        graph_failures = torch.tensor([not graph_ok], dtype=torch.int32)
        dist.all_reduce(graph_failures)

        eager_dispatched = dispatcher.dispatch(hidden, topk_output)
        eager_after_graph = dispatcher.combine(
            (
                _expert_output(eager_dispatched, dispatch, fp4_lookup),
                eager_dispatched.topk_ids,
                eager_dispatched.topk_weights,
            )
        )
        torch.cuda.synchronize()
        dispatcher._get_impl().mori_op.comm.barrier()
        eager_after_graph_ok = torch.allclose(
            eager_after_graph.float().cpu(),
            expected.float(),
            atol=tolerance,
            rtol=tolerance,
        )
        eager_after_graph_failures = torch.tensor(
            [not eager_after_graph_ok], dtype=torch.int32
        )
        dist.all_reduce(eager_after_graph_failures)
        failures += graph_failures + eager_after_graph_failures
        if rank == 0:
            print(
                "# MORI-EPV2-SGLANG-GRAPH: "
                f"{'PASS' if graph_failures.item() == 0 else 'FAIL'} "
                f"replays={graph_replays}; "
                "post_graph_eager="
                f"{'PASS' if eager_after_graph_failures.item() == 0 else 'FAIL'}",
                flush=True,
            )

    dispatcher._get_impl().mori_op.close()
    dispatcher._get_impl().mori_op.comm.destroy()
    dist.destroy_process_group()
    raise SystemExit(int(failures.item() != 0))


def _run_tbo(rank, world_size):
    hidden_size, topk, experts_per_rank = 7168, 6, 48
    num_experts = world_size * experts_per_rank
    adapter.get_parallel = lambda: SimpleNamespace(
        moe_ep_size=world_size,
        moe_ep_rank=rank,
        tp_size=world_size,
        attn_dp_size=world_size,
        attn_dp_rank=rank,
        attn_tp_size=1,
        attn_tp_rank=0,
        attn_cp_size=1,
        moe_tp_size=1,
        moe_dp_size=1,
        launch_world_rank=rank,
        world_rank=rank,
    )
    group = _Group(dist.group.WORLD)
    kwargs = dict(
        group=group,
        router_topk=topk,
        num_experts=num_experts,
        num_local_experts=experts_per_rank,
        hidden_size=hidden_size,
        params_dtype=torch.bfloat16,
        async_finish=True,
    )
    with (
        envs.SGLANG_MORI_EP_V2.override(True),
        get_flags().moe.override(tbo_enabled=True, a2a_backend=MoeA2ABackend.MORI),
    ):
        dispatcher = MaybeTboDeepEPDispatcher(**kwargs)
    assert len(dispatcher._inners) == 2
    for child in dispatcher._inners:
        child.set_quant_config({"weight_dtype": torch.float4_e2m1fn_x2})
    children = [child._get_impl() for child in dispatcher._inners]
    assert children[0]._comm_stream is children[1]._comm_stream
    assert children[0]._launch_config == adapter._MoriEPv2LaunchConfig(32, 4, 48, 4)
    assert children[0].mori_op is not children[1].mori_op
    probe_trim = os.environ.get("PROBE_TBO_TRIM", "0") == "1"
    if probe_trim:
        # Validation-only probe: keep the real shared TBO stream/dispatchers,
        # but allow each child to consume metadata snapshotted immediately
        # before its dispatch_a. Production remains conservatively gated off.
        for child in children:
            child._tbo_enabled = False

    fp4_lookup = _fp4_lookup()
    failures = torch.zeros(1, dtype=torch.int32)
    inputs = []
    child_token_counts = (
        (12, 0, 7, 2, 9, 0, 4, 1),
        (0, 11, 3, 0, 5, 8, 1, 6),
    )
    for child_id, token_counts in enumerate(child_token_counts):
        tokens = token_counts[rank]
        generator = torch.Generator(device="cpu").manual_seed(
            20260805 + child_id * 100 + rank
        )
        hidden = torch.randn(
            tokens, hidden_size, dtype=torch.bfloat16, generator=generator
        ).cuda()
        ids = torch.randint(
            0,
            num_experts,
            (tokens, topk),
            dtype=torch.int32,
            generator=generator,
        ).cuda()
        weights = torch.rand(
            tokens, topk, dtype=torch.float32, generator=generator
        ).cuda()
        inputs.append((hidden, ids, StandardTopKOutput(weights, ids, None)))

    def run_step():
        for child_id, (hidden, _ids, topk_output) in enumerate(inputs):
            token_counts = child_token_counts[child_id]
            dp_attention.set_dp_buffer_len(
                sum(token_counts), token_counts[rank], False, list(token_counts)
            )
            dispatcher.dispatch_a(
                tbo_subbatch_index=child_id,
                hidden_states=hidden,
                topk_output=topk_output,
            )
        outputs = [
            dispatcher.dispatch_b(tbo_subbatch_index=child_id) for child_id in range(2)
        ]
        direct_views = []
        for child_id, (child, output) in enumerate(zip(children, outputs)):
            if child._manual_recv_cap > 0:
                assert output.recv_cap == min(
                    child._manual_recv_cap, child.mori_op.cfg.effective_max_recv
                )
            elif child._trim_recv and probe_trim:
                expected_cap = round_logical_recv_rows(
                    sum(child_token_counts[child_id]),
                    pow2_buckets=child._recv_cap_pow2_buckets,
                )
                assert output.recv_cap == expected_cap
            else:
                assert output.recv_cap == 0
            expected_direct = child._direct_output and callable(
                getattr(child.mori_op, "combine_in_view", None)
            )
            assert (output.expert_output is not None) == expected_direct
            if output.expert_output is not None:
                assert (
                    output.expert_output.data_ptr()
                    == child.mori_op.combine_in_view().data_ptr()
                )
                direct_views.append(output.expert_output.data_ptr())
        assert len(direct_views) == len(set(direct_views)), (
            "TBO children share an output buffer"
        )
        for child_id, output in enumerate(outputs):
            expert_out = _expert_output(output, "fp4", fp4_lookup)
            if output.expert_output is not None:
                direct_out = output.expert_output[: expert_out.shape[0]]
                direct_out.copy_(expert_out)
                expert_out = direct_out
            dispatcher.combine_a(
                tbo_subbatch_index=child_id,
                combine_input=(expert_out, output.topk_ids, output.topk_weights),
            )
        actual = [
            dispatcher.combine_b(tbo_subbatch_index=child_id)[
                : inputs[child_id][0].shape[0]
            ]
            for child_id in range(2)
        ]
        return actual, outputs

    def check(actual, atol=0.6, rtol=0.6):
        torch.cuda.synchronize()
        for child_id, (hidden, ids, _topk_output) in enumerate(inputs):
            if not torch.allclose(
                actual[child_id].float().cpu(),
                (
                    _expected_unique_destinations(ids, experts_per_rank)
                    * hidden.float().cpu()
                )
                .to(torch.bfloat16)
                .float(),
                atol=atol,
                rtol=rtol,
            ):
                failures.add_(1)

    replay_iters = int(os.environ.get("REPLAY_ITERS", "3"))
    assert replay_iters > 0
    use_graph = os.environ.get("TEST_CUDA_GRAPH", "0") == "1"
    graph = None
    # Exercise random FP4 inputs and warm up both instances before graph capture.
    for _ in range(2 if use_graph else 1):
        actual, outputs = run_step()
        check(actual)
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual, outputs = run_step()
        graph.replay()
        check(actual)
    for iteration in range(replay_iters):
        # Change values and routing while retaining the captured input addresses.
        for child_id, (hidden, ids, _topk_output) in enumerate(inputs):
            # Powers of two are exact in FP4; opposite signs expose child aliasing.
            value = 2.0 ** (iteration % 4 - 3)
            hidden.fill_(value if child_id == 0 else -2 * value)
            ids.copy_((ids + 13) % num_experts)
        if graph is None:
            actual, outputs = run_step()
        else:
            graph.replay()
        check(actual, atol=2e-2, rtol=2e-2)
    graph = None
    torch.cuda.synchronize()
    dist.all_reduce(failures)
    if rank == 0:
        summary = {
            "status": "PASS" if failures.item() == 0 else "FAIL",
            "probe_trim": probe_trim,
            "direct_output": [output.expert_output is not None for output in outputs],
            "cuda_graph": use_graph,
            "replay_iters": replay_iters,
            "recv_caps": [output.recv_cap for output in outputs],
            "child_sender_rows": child_token_counts,
            "failures": failures.item(),
        }
        print(
            "# MORI-EPV2-FP4-TBO: " + json.dumps(summary, sort_keys=True),
            flush=True,
        )
        if result_json := os.environ.get("RESULT_JSON"):
            Path(result_json).write_text(json.dumps(summary, indent=2) + "\n")
    for child in children:
        child.mori_op.close()
        child.mori_op.comm.destroy()
    dist.destroy_process_group()
    raise SystemExit(int(failures.item() != 0))


if __name__ == "__main__":
    main()
