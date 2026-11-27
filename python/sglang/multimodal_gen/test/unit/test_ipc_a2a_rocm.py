# SPDX-License-Identifier: Apache-2.0
"""ROCm copy-engine Ulysses head-group exchange: gating and the slot protocol.

The protocol tests run every rank as a thread of this process on host memory:
a copy lands when it is queued (so a flag lands after its data, as on one SDMA
stream) and a flag wait spins on the flag word. That checks the slot offsets,
the sequence numbers, the counter reset and the pool regrowth without GPUs;
the multi-GPU parity test is single_test_file/test_ipc_a2a_rocm_multi_gpu.py.
"""

import contextlib
import ctypes
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.multimodal_gen.runtime.distributed.device_communicators import (
    ipc_a2a_rocm as ce,
)
from sglang.multimodal_gen.runtime.layers import usp

WORLD = 4
HEADS, HEAD_SIZE = 56, 8


def _attend(q, k, v):
    # head-independent but row-order dependent, like attention
    return torch.cumsum(q.float() * k.flip(0).float(), dim=0).to(q.dtype) + v


def _pack(q, k, v, n):
    """Destination-major ``[n, rows, heads / n, 3 * head_size]``, q|k|v per head."""
    rows, heads, head_size = q.shape
    split = [t.reshape(rows, n, heads // n, head_size) for t in (q, k, v)]
    return torch.cat(split, dim=-1).transpose(0, 1).contiguous()


def _reference(qf, kf, vf, groups):
    """Every head group attended over all ranks' rows, ``[world * rows, ...]``."""
    group_heads = qf.shape[1] // (WORLD * groups)
    out = torch.empty_like(qf)
    for block in range(WORLD * groups):
        h = slice(block * group_heads, (block + 1) * group_heads)
        out[:, h] = _attend(qf[:, h], kf[:, h], vf[:, h])
    return out


class _Stream:
    """A stream whose work ran when it was queued."""

    cuda_stream = 0

    def record_event(self):
        return object()

    def wait_event(self, event):
        pass

    def wait_stream(self, stream):
        pass


class _HostGroup:
    """The ranks of one Ulysses group, as threads sharing host buffers."""

    def __init__(self, world):
        self.world = world
        self.sync = threading.Barrier(world, timeout=60)
        self.slots = [None] * world

    def share(self, t, group, rank, world):
        self.slots[rank] = t
        self.sync.wait()
        members = list(self.slots)
        self.sync.wait()
        return members


class _HostHip:
    def __init__(self, group):
        self.group = group

    def copy(self, stream, dst, src, nbytes):
        ctypes.memmove(dst, src, nbytes)

    def copy2d(self, stream, dst, dpitch, src, spitch, width, height):
        for row in range(height):
            ctypes.memmove(dst + row * dpitch, src + row * spitch, width)

    def wait(self, stream, addr, value):
        deadline = time.monotonic() + 60
        while ctypes.c_int32.from_address(addr).value < value:
            if self.group.sync.broken:
                raise RuntimeError("another rank failed")
            if time.monotonic() > deadline:
                raise TimeoutError(f"flag stayed below {value}")
            time.sleep(1e-4)


def _host_rank(group, rank, seq_table):
    state = ce.IpcA2ARocmState()
    state.group, state.rank, state.world = group, rank, group.world
    state.hip = _HostHip(group)
    state.seqtab = torch.arange(seq_table, dtype=torch.int32)
    state.peers = [(rank + s) % group.world for s in range(1, group.world)]
    state.copy_streams = [_Stream() for _ in state.peers]
    state.inited = True
    return state


@contextlib.contextmanager
def _host_transport(group, seq_table):
    zeros = torch.zeros
    with (
        patch.object(ce, "_share", group.share),
        patch.object(ce, "_SEQ_TABLE", seq_table),
        patch.object(
            ce, "dist", SimpleNamespace(barrier=lambda group: group.sync.wait())
        ),
        patch.object(ce.torch, "zeros", lambda *a, device=None, **kw: zeros(*a, **kw)),
        patch.object(ce.torch.cuda, "Stream", _Stream),
        patch.object(ce.torch.cuda, "current_stream", _Stream),
        patch.object(ce.torch.cuda, "stream", lambda s: contextlib.nullcontext()),
        patch.object(ce.torch.cuda, "synchronize", lambda: None),
        patch.object(ce.torch.cuda, "ipc_collect", lambda: None),
    ):
        yield


def _run_ranks(inputs, groups, seq_table, **call_kwargs):
    """Every rank's outputs of the calls over ``inputs`` (full q/k/v triples)."""
    group = _HostGroup(WORLD)

    def rank_main(rank):
        state = _host_rank(group, rank, seq_table)
        outs = []
        try:
            for qf, kf, vf in inputs:
                rows = qf.shape[0] // WORLD
                q, k, v = (t[rank * rows : (rank + 1) * rows] for t in (qf, kf, vf))
                out = state.pipelined_attention(
                    q, k, v, _attend, groups, _pack, **call_kwargs
                )
                outs.append(out.clone())
        except BaseException:
            group.sync.abort()
            raise
        return state, outs

    with _host_transport(group, seq_table), ThreadPoolExecutor(WORLD) as pool:
        futures = [pool.submit(rank_main, rank) for rank in range(WORLD)]
        return [f.result(timeout=300) for f in futures]


def _inputs(rows_per_call, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return [
        tuple(
            torch.randn(WORLD * rows, HEADS, HEAD_SIZE, generator=gen).bfloat16()
            for _ in range(3)
        )
        for rows in rows_per_call
    ]


@pytest.mark.parametrize("group_pack", [False, True], ids=["pack_kernel", "group_pack"])
@pytest.mark.parametrize("groups", [2, 7])
def test_copy_engine_exchange_gives_the_collective_bytes(groups, group_pack):
    # the 9-row calls outgrow the pool sized for 5 rows; the table is short
    # enough that the flag counters reset twice in between
    rows_per_call = [5, 5, 5, 9, 9, 9, 5]
    inputs = _inputs(rows_per_call)
    with patch.object(ce, "_GROUP_PACK", group_pack):
        ranks = _run_ranks(inputs, groups, seq_table=2 * groups + 2)
    for call, (qf, kf, vf) in enumerate(inputs):
        ref = _reference(qf, kf, vf, groups)
        rows = rows_per_call[call]
        for rank, (_, outs) in enumerate(ranks):
            assert torch.equal(outs[call], ref[rank * rows : (rank + 1) * rows]), (
                f"call {call} rank {rank}"
            )
    for state, _ in ranks:
        pool = state.pools[torch.bfloat16]
        assert pool.sizes[0] >= 9 * HEADS * 3 * HEAD_SIZE
        # four calls since the regrowth: only a reset leaves fewer counted
        assert pool.calls < 4


# ---- gating ----------------------------------------------------------------


@pytest.fixture
def fresh_state():
    ce.IPC_A2A_ROCM.reset()
    yield ce.IPC_A2A_ROCM
    ce.IPC_A2A_ROCM.reset()


def test_copy_engine_stays_off_without_the_env(fresh_state, monkeypatch):
    monkeypatch.delenv("SGLANG_DIFFUSION_ROCM_CE_A2A", raising=False)
    with patch.object(fresh_state, "init") as init:
        assert not ce.rocm_ce_ready(object())
    init.assert_not_called()


def test_copy_engine_is_not_set_up_under_graph_capture(fresh_state, monkeypatch):
    monkeypatch.setenv("SGLANG_DIFFUSION_ROCM_CE_A2A", "1")
    with (
        patch.object(fresh_state, "init") as init,
        patch.object(ce.torch.cuda, "is_current_stream_capturing", return_value=True),
    ):
        assert not ce.rocm_ce_ready(object())
    init.assert_not_called()
    assert not fresh_state.failed


@pytest.mark.parametrize(
    "local_ok, peers_ok", [(True, True), (True, False), (False, True)]
)
def test_copy_engine_runs_only_when_every_rank_set_it_up(
    fresh_state, monkeypatch, local_ok, peers_ok
):
    monkeypatch.setenv("SGLANG_DIFFUSION_ROCM_CE_A2A", "1")
    group = object()

    def init(g):
        if not local_ok:
            raise ce._Unsupported("device 0 cannot reach [1]")
        fresh_state.group, fresh_state.inited = g, True

    def all_reduce(agreed, op, group):
        agreed.fill_(min(int(agreed.item()), int(peers_ok)))

    tensor = torch.tensor
    with (
        patch.object(fresh_state, "init", side_effect=init) as setup,
        patch.object(ce.torch.cuda, "is_current_stream_capturing", return_value=False),
        patch.object(
            ce.torch, "tensor", lambda *a, device=None, **kw: tensor(*a, **kw)
        ),
        patch.object(ce.dist, "all_reduce", side_effect=all_reduce),
    ):
        ready = ce.rocm_ce_ready(group)
        again = ce.rocm_ce_ready(group)
    assert ready == again == (local_ok and peers_ok)
    # set up once; a failure on any rank sticks, every rank keeps RCCL
    setup.assert_called_once_with(group)
    assert fresh_state.failed == (not ready)


def test_copy_engine_sets_up_again_for_another_group(fresh_state, monkeypatch):
    monkeypatch.setenv("SGLANG_DIFFUSION_ROCM_CE_A2A", "1")
    old, new = object(), object()
    fresh_state.group, fresh_state.inited = old, True

    def init(g):
        fresh_state.group, fresh_state.inited = g, True

    tensor = torch.tensor
    with (
        patch.object(fresh_state, "init", side_effect=init) as setup,
        patch.object(ce.torch.cuda, "is_current_stream_capturing", return_value=False),
        patch.object(
            ce.torch, "tensor", lambda *a, device=None, **kw: tensor(*a, **kw)
        ),
        patch.object(ce.dist, "all_reduce"),
    ):
        assert ce.rocm_ce_ready(old)
        setup.assert_not_called()
        assert ce.rocm_ce_ready(new)
    setup.assert_called_once_with(new)


@pytest.mark.parametrize(
    "hip, ready, capturing",
    [
        (None, True, False),
        ("7.0", False, False),
        ("7.0", True, False),
        ("7.0", True, True),
    ],
    ids=["cuda", "rocm_not_ready", "rocm", "rocm_capturing"],
)
def test_usp_pipeline_takes_the_copy_engines_only_when_ready_on_rocm(
    hip, ready, capturing
):
    takes = hip is not None and ready and not capturing
    rows, groups = 3, 2
    q = MagicMock(
        shape=(rows, HEADS, HEAD_SIZE),
        is_cuda=True,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    group = object()

    class Collective(Exception):
        pass

    with (
        patch.object(torch.version, "hip", hip),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=capturing),
        patch.object(
            usp, "get_sp_group", return_value=SimpleNamespace(ulysses_group=group)
        ),
        patch.object(usp, "get_ulysses_parallel_world_size", return_value=WORLD),
        patch.object(ce, "rocm_ce_ready", return_value=ready) as ready_check,
        patch.object(ce.IPC_A2A_ROCM, "pipelined_attention", return_value="ce") as run,
        patch.object(usp, "pack_qkv_destination_major", side_effect=Collective),
    ):
        if takes:
            assert (
                usp.ulysses_head_group_pipelined_attention(q, q, q, _attend, groups)
                == "ce"
            )
        else:
            with pytest.raises(Collective):
                usp.ulysses_head_group_pipelined_attention(q, q, q, _attend, groups)
    if hip is None or capturing:
        ready_check.assert_not_called()
    else:
        ready_check.assert_called_once_with(group)
    assert run.called == takes
    if not run.called:
        return
    # the pack handed over is the collective path's destination-major kernel
    pack = run.call_args.args[5]
    q, k, v = (torch.randn(rows, HEADS, HEAD_SIZE).bfloat16() for _ in range(3))

    def kernel(q, k, v, n, out):
        return out.copy_(_pack(q, k, v, n))

    with patch.object(usp, "pack_qkv_destination_major", side_effect=kernel):
        packed = pack(q, k, v, WORLD * groups)
    assert torch.equal(packed, _pack(q, k, v, WORLD * groups))
