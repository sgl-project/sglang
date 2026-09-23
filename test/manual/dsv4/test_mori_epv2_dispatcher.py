"""Eight-GPU identity round trip through the SGLang MORI EPv2 adapter."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

import sglang.srt.layers.moe.token_dispatcher.moriep as adapter
from sglang.srt.environ import envs
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode


def _expert_output(dispatched, fp4_enabled: bool, fp4_lookup=None):
    if not fp4_enabled:
        return dispatched.hidden_states

    packed = dispatched.hidden_states.view(torch.uint8)
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)
    values = fp4_lookup[nibbles.long()]
    scales = dispatched.hidden_states_scale.repeat_interleave(32, dim=1).float()
    output = (values * scales[:, : values.shape[1]]).to(torch.bfloat16)
    valid_rows = (
        torch.arange(output.shape[0], device=output.device)
        < dispatched.num_recv_tokens_per_expert.reshape(-1)[0]
    )
    return torch.where(valid_rows[:, None], output, torch.zeros_like(output))


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
    fp4_enabled = os.environ.get("FP4", "0") == "1"
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

    adapter.is_tbo_enabled = lambda: False
    adapter.get_parallel = lambda: SimpleNamespace(
        moe_ep_size=world_size,
        moe_ep_rank=rank,
        world_rank=rank,
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
        topk_ids = torch.arange(topk, dtype=torch.int32).repeat(tokens, 1).cuda()
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
    combined = dispatcher.combine(
        (
            _expert_output(dispatched, fp4_enabled, fp4_lookup),
            dispatched.topk_ids,
            dispatched.topk_weights,
        )
    )
    torch.cuda.synchronize()
    dispatcher._get_impl().mori_op.comm.barrier()

    expected = (
        _expected_unique_destinations(topk_ids, experts_per_rank) * hidden.float().cpu()
    ).to(torch.bfloat16)
    tolerance = 6e-1 if fp4_enabled else 2e-2
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
            f"empty_last_rank={os.environ.get('EMPTY_LAST_RANK', '0')}",
            f"fp4={fp4_enabled}",
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
                    _expert_output(graph_dispatched, fp4_enabled, fp4_lookup),
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
                _expert_output(eager_dispatched, fp4_enabled, fp4_lookup),
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
