"""Eight-GPU MORI EPv2 FP4-asymmetric two-child TBO coverage.

Set TEST_CUDA_GRAPH=1 to capture and replay REPLAY_ITERS times (default 3).
SGLANG_MORI_EPV2_AITER_DIRECT_OUTPUT=0/1 exercises staging or direct writes.
The identity expert below checks buffer correctness, not AITER performance.
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
from sglang.srt.layers.moe.token_dispatcher.moriep import (
    round_logical_recv_rows,
)
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
    output = torch.where(valid_rows[:, None], output, torch.zeros_like(output))
    return output[: dispatched.recv_cap] if dispatched.recv_cap > 0 else output


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
        envs.SGLANG_MORI_EP_VERSION.override("epv2"),
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
            elif not child._trim_recv:
                assert output.recv_cap == 0
            elif probe_trim:
                expected_cap = round_logical_recv_rows(
                    sum(child_token_counts[child_id]),
                    pow2_buckets=child._recv_cap_pow2_buckets,
                )
                assert output.recv_cap == expected_cap
            else:
                assert output.recv_cap == child.mori_op.cfg.effective_max_recv
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
            expert_out = _dequantize(output)
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
                _expected(hidden, ids, experts_per_rank).float(),
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
