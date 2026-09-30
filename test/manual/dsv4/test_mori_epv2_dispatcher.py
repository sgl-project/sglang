"""Eight-GPU identity round trip through the SGLang MORI EPv2 adapter."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

import sglang.srt.layers.dp_attention as dp_attention
import sglang.srt.layers.moe.token_dispatcher.moriep as adapter
from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.moriep import (
    round_logical_recv_rows,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode


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
    fp4_lookup = (
        torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
            device="cuda",
        )
        if fp4_enabled
        else None
    )
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
    envs.SGLANG_MORI_EP_VERSION.set("epv2")
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
        expected_cap = impl.mori_op.cfg.effective_max_recv
        if 0 < cluster_rows < expected_cap:
            expected_cap = round_logical_recv_rows(
                cluster_rows, pow2_buckets=impl._recv_cap_pow2_buckets
            )
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


if __name__ == "__main__":
    main()
