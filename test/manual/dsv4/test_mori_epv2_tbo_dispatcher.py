"""Eight-GPU MORI EPv2 FP4-asymmetric two-child TBO coverage."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

import sglang.srt.layers.moe.token_dispatcher.moriepv2 as adapter
from sglang.srt.batch_overlap.two_batch_overlap import MaybeTboDeepEPDispatcher
from sglang.srt.environ import envs
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.runtime_context import get_flags


class _Group:
    def __init__(self, process_group):
        self.cpu_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.rank_in_group = dist.get_rank(process_group)

    def broadcast_object(self, obj, src=0):
        values = [obj if self.rank_in_group == src else None]
        dist.broadcast_object_list(values, src=src, group=self.cpu_group)
        return values[0]


def _dequantize(dispatched):
    from aiter.utility.fp4_utils import mxfp4_to_f32

    values = mxfp4_to_f32(dispatched.hidden_states)
    scales = dispatched.hidden_states_scale.repeat_interleave(32, dim=1).float()
    output = (values * scales[:, : values.shape[1]]).to(torch.bfloat16)
    valid_rows = (
        torch.arange(output.shape[0], device=output.device)
        < dispatched.num_recv_tokens_per_expert.reshape(-1)[0]
    )
    return torch.where(valid_rows[:, None], output, torch.zeros_like(output))


def _expected(hidden, topk_ids, experts_per_rank):
    factors = torch.tensor(
        [len(set(row.tolist())) for row in (topk_ids.cpu().long() // experts_per_rank)],
        dtype=torch.float32,
    ).view(-1, 1)
    return (factors * hidden.float().cpu()).to(torch.bfloat16)


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world_size == 8
    torch.cuda.set_device(local_rank)

    hidden_size, topk, experts_per_rank = 7168, 6, 48
    num_experts = world_size * experts_per_rank
    adapter.get_parallel = lambda: SimpleNamespace(
        moe_ep_size=world_size,
        moe_ep_rank=rank,
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
        envs.SGLANG_MORI_EP_VERSION.override("epv2"),
        get_flags().moe.override(tbo_enabled=True, a2a_backend=MoeA2ABackend.MORI),
    ):
        dispatcher = MaybeTboDeepEPDispatcher(**kwargs)
    children = dispatcher._inners
    assert len(children) == 2
    assert children[0]._comm_stream is children[1]._comm_stream
    assert children[0]._geometry == (32, 4, 48, 4)
    for child in children:
        child.set_quant_config({"weight_dtype": torch.float4_e2m1fn_x2})
    assert children[0].op is not children[1].op

    failures = torch.zeros(1, dtype=torch.int32)
    inputs = []
    for child_id, token_counts in enumerate(
        ((12, 0, 7, 2, 9, 0, 4, 1), (0, 11, 3, 0, 5, 8, 1, 6))
    ):
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

    for child_id, (hidden, _ids, topk_output) in enumerate(inputs):
        dispatcher.dispatch_a(
            tbo_subbatch_index=child_id,
            hidden_states=hidden,
            topk_output=topk_output,
            dynamic_recv_cluster_rows=sum(
                (12, 0, 7, 2, 9, 0, 4, 1)
                if child_id == 0
                else (0, 11, 3, 0, 5, 8, 1, 6)
            ),
        )
    outputs = [
        dispatcher.dispatch_b(tbo_subbatch_index=child_id) for child_id in range(2)
    ]
    for child_id, output in enumerate(outputs):
        dispatcher.combine_a(
            tbo_subbatch_index=child_id,
            combine_input=(_dequantize(output), output.topk_ids, output.topk_weights),
        )
    actual = [
        dispatcher.combine_b(tbo_subbatch_index=child_id)[
            : inputs[child_id][0].shape[0]
        ]
        for child_id in range(2)
    ]
    torch.cuda.synchronize()
    for child_id, (hidden, ids, _topk_output) in enumerate(inputs):
        if not torch.allclose(
            actual[child_id].float().cpu(),
            _expected(hidden, ids, experts_per_rank).float(),
            atol=0.6,
            rtol=0.6,
        ):
            failures += 1
    dist.all_reduce(failures)
    if rank == 0:
        print(
            "# MORI-EPV2-FP4-TBO: "
            f"{'PASS' if failures.item() == 0 else 'FAIL'} failures={failures.item()}",
            flush=True,
        )
    for child in children:
        child.op.close()
        child.op.comm.destroy()
    dist.destroy_process_group()
    raise SystemExit(int(failures.item() != 0))


if __name__ == "__main__":
    main()
