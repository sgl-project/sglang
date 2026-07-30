"""8-GPU FlyDSL dispatcher/TBO integration coverage without model weights.

This is intentionally dispatcher-level, not model-level coverage.  Constructing
``FlyDSLEPDispatcher.flydsl_op`` through ``init_flydsl_op`` requires SGLang's
fully initialized parallel topology and process-group wrapper.  The harness
therefore constructs the production FlyDSL ops directly and injects them into
the production two-child ``MaybeTboDeepEPDispatcher``.  It still exercises the
real child selection, dispatcher stage machines, dedicated-stream event
handoffs, FlyDSL dispatch/combine kernels, and per-child communication state.
It does not claim coverage of model loading, MoE compute, CUDA graphs, or the
serving runtime. It does exercise production eager dynamic recv-cap resolution.
"""

import os
import socket
import time
from types import SimpleNamespace

import mori.shmem as ms
import torch
import torch.distributed as dist

from sglang.kernels.third_party.flydsl_a2a import (
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
)
from sglang.srt.batch_overlap.two_batch_overlap import MaybeTboDeepEPDispatcher
from sglang.srt.layers.moe.token_dispatcher.flydslep import (
    FlyDSLEPDispatcher,
    FlyDSLEPNormalCombineInput,
    _resolve_eager_recv_cap,
    _resolve_tbo_child_cluster_rows,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.runtime_context import get_flags

_TOKEN_COUNTS = (
    ((12, 0, 7, 2, 9, 0, 4, 1), (0, 11, 3, 0, 5, 8, 1, 6)),
    ((0, 6, 0, 10, 2, 4, 1, 7), (9, 0, 12, 1, 0, 2, 8, 3)),
)
_HOT_PES = ((0, 7), (3, 5))
_CHILD_PHYSICAL_CAPS = (128, 128)


def _make_skewed_inputs(rank, world_size, round_id, child_id, hidden, epr, topk):
    cur = _TOKEN_COUNTS[round_id][child_id][rank]
    device = torch.device("cuda", rank)
    torch.manual_seed(1000 + 100 * round_id + 10 * child_id + rank)
    hidden_states = torch.randn(cur, hidden, dtype=torch.bfloat16, device=device)
    topk_weights = torch.rand(cur, topk, dtype=torch.float32, device=device)
    topk_ids = torch.empty(cur, topk, dtype=torch.int32, device=device)
    hot_pe = _HOT_PES[round_id][child_id]
    for token in range(cur):
        for slot in range(topk):
            # Five routes target one PE; the sixth spreads across all PEs.
            pe = hot_pe if slot < topk - 1 else (rank + token + child_id) % world_size
            local_expert = (rank * 7 + token * topk + slot) % epr
            topk_ids[token, slot] = pe * epr + local_expert
    return hidden_states, StandardTopKOutput(topk_weights, topk_ids, None)


def _make_op(rank, world_size, hidden, epr, topk, recv_cap):
    return FlyDSLDispatchCombineIntraNodeOp(
        FlyDSLDispatchCombineConfig(
            rank=rank,
            world_size=world_size,
            hidden_dim=hidden,
            max_num_inp_token_per_rank=64,
            num_experts_per_rank=epr,
            num_experts_per_token=topk,
            data_type=torch.bfloat16,
            max_token_type_size=torch.bfloat16.itemsize,
            dispatch_block_num=16,
            dispatch_warp_num_per_block=4,
            combine_block_num=16,
            combine_warp_num_per_block=4,
            max_total_recv_tokens=recv_cap,
        )
    )


def _combine_input(dispatch_output):
    return FlyDSLEPNormalCombineInput(
        hidden_states=dispatch_output.hidden_states,
        topk_ids=dispatch_output.topk_ids,
        topk_weights=dispatch_output.topk_weights,
    )


def _worker(rank, world_size, port):
    os.environ.update(
        LOCAL_RANK=str(rank),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="localhost",
        MASTER_PORT=str(port),
        SGLANG_FLYDSL_DYNAMIC_RECV_CAP="false",
        SGLANG_FLYDSL_TBO_USE_COMM_STREAM="true",
    )
    os.environ.pop("SGLANG_FLYDSL_DYNAMIC_RECV_CAP_EAGER", None)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group_name = "flydsl_tbo_dispatcher"
    torch._C._distributed_c10d._register_process_group(group_name, dist.group.WORLD)
    ms.shmem_torch_process_group_init(group_name)

    hidden, epr, topk = 7168, 48, 6
    group = SimpleNamespace(cpu_group=dist.group.WORLD)
    dispatcher_kwargs = dict(
        group=group,
        router_topk=topk,
        num_experts=None,
        num_local_experts=epr,
        hidden_size=hidden,
        params_dtype=torch.bfloat16,
    )
    with get_flags().moe.override(tbo_enabled=True, a2a_backend=MoeA2ABackend.FLYDSL):
        tbo_dispatcher = MaybeTboDeepEPDispatcher(
            **dispatcher_kwargs, async_finish=True
        )
        reference = FlyDSLEPDispatcher(**dispatcher_kwargs, async_finish=False)

    children = tbo_dispatcher._inners
    assert len(children) == 2
    assert [child.instance_id for child in children] == [0, 1]
    assert children[0]._comm_stream is not None
    assert children[0]._comm_stream is children[1]._comm_stream

    for child, recv_cap in zip(children, _CHILD_PHYSICAL_CAPS, strict=True):
        child._flydsl_op = _make_op(
            rank, world_size, hidden, epr, topk, recv_cap=recv_cap
        )
    # max_total_recv_tokens=0 is the sequential/full-cap reference (8 * 64).
    reference._flydsl_op = _make_op(rank, world_size, hidden, epr, topk, recv_cap=0)
    assert children[0]._flydsl_op is not children[1]._flydsl_op
    assert tuple(child.flydsl_op.cfg.effective_max_recv for child in children) == (
        _CHILD_PHYSICAL_CAPS
    )
    assert reference.flydsl_op.cfg.effective_max_recv == world_size * 64
    ms.shmem_barrier_all()

    for round_id, child_order in enumerate(((0, 1), (1, 0))):
        child_rows_by_rank = tuple(
            tuple(_TOKEN_COUNTS[round_id][child_id][pe] for child_id in range(2))
            for pe in range(world_size)
        )
        parent_rows = [sum(rows) for rows in child_rows_by_rank]
        cluster_rows = _resolve_tbo_child_cluster_rows(parent_rows, child_rows_by_rank)
        assert cluster_rows is not None
        expected_caps = tuple(
            _resolve_eager_recv_cap(rows, physical)
            for rows, physical in zip(cluster_rows, _CHILD_PHYSICAL_CAPS, strict=True)
        )
        assert all(cap is not None for cap in expected_caps)
        if round_id == 1:
            assert expected_caps == (32, 64)

        inputs = [
            _make_skewed_inputs(rank, world_size, round_id, child_id, hidden, epr, topk)
            for child_id in range(2)
        ]

        expected = []
        for hidden_states, topk_output in inputs:
            dispatched = reference.dispatch(hidden_states, topk_output)
            expected.append(reference.combine(_combine_input(dispatched)).clone())
        torch.cuda.synchronize()

        for child_id in child_order:
            hidden_states, topk_output = inputs[child_id]
            tbo_dispatcher.dispatch_a(
                tbo_subbatch_index=child_id,
                hidden_states=hidden_states,
                topk_output=topk_output,
                dynamic_recv_cluster_rows=cluster_rows[child_id],
            )
        assert all(child._stage.name == "AFTER_DISPATCH_A" for child in children)
        assert all(
            child._dispatch_intermediate_state[0].data_ptr()
            == inputs[child_id][0].data_ptr()
            for child_id, child in enumerate(children)
        )

        dispatched = [None, None]
        for child_id in child_order:
            dispatched[child_id] = tbo_dispatcher.dispatch_b(
                tbo_subbatch_index=child_id
            )
        assert all(child._stage.name == "AFTER_DISPATCH_B" for child in children)
        assert all(
            not hasattr(child, "_dispatch_intermediate_state") for child in children
        )
        assert dispatched[0].topk_ids.data_ptr() != dispatched[1].topk_ids.data_ptr()

        for child_id in child_order:
            tbo_dispatcher.combine_a(
                tbo_subbatch_index=child_id,
                combine_input=_combine_input(dispatched[child_id]),
            )
        assert all(child._stage.name == "AFTER_COMBINE_A" for child in children)

        actual = [None, None]
        for child_id in child_order:
            actual[child_id] = tbo_dispatcher.combine_b(tbo_subbatch_index=child_id)[
                : inputs[child_id][0].shape[0]
            ]
        torch.cuda.synchronize()

        for child_id, child in enumerate(children):
            recv_cap = expected_caps[child_id]
            total_recv = int(dispatched[child_id].num_recv_tokens_per_expert.item())
            assert dispatched[child_id].hidden_states.shape[0] == recv_cap
            assert child._op_recv_cap == recv_cap
            assert total_recv <= recv_cap
            assert next(reversed(child.flydsl_op._disp_jit_cache))[-1] == recv_cap
            assert next(reversed(child.flydsl_op._comb_jit_cache))[-1] == recv_cap
            torch.testing.assert_close(
                actual[child_id], expected[child_id], rtol=0, atol=0
            )
            assert child._stage.name == "INITIAL"
            assert child._num_tokens == inputs[child_id][0].shape[0]
            assert not hasattr(child, "_combine_intermediate_state")

    ms.shmem_barrier_all()
    try:
        ms.shmem_finalize()
    except Exception:
        pass
    dist.destroy_process_group()


def test_flydsl_tbo_dispatcher_two_children():
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]

    process_context = torch.multiprocessing.spawn(
        _worker, args=(8, port), nprocs=8, join=False
    )
    deadline = time.monotonic() + 600
    while not process_context.join(
        timeout=max(0, deadline - time.monotonic()), grace_period=10
    ):
        if time.monotonic() < deadline:
            continue
        for process in process_context.processes:
            if process.is_alive():
                process.terminate()
        for process in process_context.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
        raise AssertionError("FlyDSL two-child TBO integration test deadlocked")


if __name__ == "__main__":
    test_flydsl_tbo_dispatcher_two_children()
