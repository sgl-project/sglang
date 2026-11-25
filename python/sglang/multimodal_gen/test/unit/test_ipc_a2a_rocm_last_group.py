# SPDX-License-Identifier: Apache-2.0
"""SGLANG_DIFFUSION_ROCM_CE_LAST: the last head group of the ROCm copy-engine pipeline.

The CPU tests run one rank of ``IpcA2ARocmState.pipelined_attention`` with host
buffers standing in for the peers' IPC mappings and ``memmove`` for the SDMA
copies, so the bytes every peer receives can be compared between the default
send and ``klast``. The GPU test runs the real pipeline on four ROCm GPUs.
"""

import ctypes
import importlib
import os
import sys
import types
from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.multimodal_gen.runtime.distributed.device_communicators import (
    ipc_a2a_rocm as ce,
)
from sglang.srt.utils.network import get_free_port_below_ephemeral

WORLD, GROUPS, GROUP_HEADS, HEAD_DIM, ROWS = 4, 3, 2, 8, 5
HEADS = WORLD * GROUPS * GROUP_HEADS
_AITER = "sglang.multimodal_gen.runtime.layers.attention.backends.aiter"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("", frozenset()),
        ("klast", {"klast"}),
        ("tlast", {"tlast"}),
        ("klast,tlast", {"klast", "tlast"}),
        ("tlast,,klast,", {"klast", "tlast"}),
    ],
)
def test_last_group_options_parse_from_the_environment(monkeypatch, value, expected):
    monkeypatch.setenv("SGLANG_DIFFUSION_ROCM_CE_LAST", value)
    try:
        assert importlib.reload(ce)._LAST == expected
    finally:
        monkeypatch.delenv("SGLANG_DIFFUSION_ROCM_CE_LAST")
        importlib.reload(ce)
    assert ce._LAST == frozenset()


class _Stream:
    """Stands in for torch.cuda.Stream; only records what was queued on it."""

    def __init__(self, *args, **kwargs):
        self.log = []

    def wait_event(self, event):
        self.log.append(("wait_event", event))

    def record_event(self):
        event = object()
        self.log.append(("record_event", event))
        return event

    def wait_stream(self, stream):
        self.log.append(("wait_stream", stream))


class _Hip:
    """SDMA copies as host memmoves between the CPU buffers; waits are no-ops."""

    def __init__(self):
        self.copies = []

    def copy(self, stream, dst, src, nbytes):
        ctypes.memmove(dst, src, nbytes)
        self.copies.append((stream, nbytes))

    def wait(self, stream, addr, value):
        pass


@pytest.fixture
def host_streams(monkeypatch):
    """torch.cuda streams and the pool's peer mappings replaced by host objects."""
    stack = [_Stream()]

    @contextmanager
    def stream(s):
        stack.append(s)
        try:
            yield
        finally:
            stack.pop()

    zeros = torch.zeros
    monkeypatch.setattr(torch.cuda, "Stream", _Stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: stack[-1])
    monkeypatch.setattr(torch.cuda, "stream", stream)
    monkeypatch.setattr(
        torch,
        "zeros",
        lambda *a, device=None, **k: zeros(
            *a, device=None if device == "cuda" else device, **k
        ),
    )
    # every peer's buffer is a separate host tensor of the same size
    monkeypatch.setattr(
        ce,
        "_share",
        lambda t, group, rank, world: [
            t if m == rank else torch.zeros_like(t) for m in range(world)
        ],
    )
    # tlast's hook into the attention backend, without importing aiter
    backend = types.ModuleType(_AITER)
    backend._TAIL_STREAM = None
    package = importlib.import_module(_AITER.rsplit(".", 1)[0])
    monkeypatch.setattr(package, "aiter", backend, raising=False)
    monkeypatch.setitem(sys.modules, _AITER, backend)
    return stack, backend


def _pack(q, k, v, n):
    # destination-major [n, rows, heads / n, 3 * head_dim], q|k|v per head
    rows, heads, hd = q.shape
    qkv = torch.cat((q, k, v), dim=-1).view(rows, n, heads // n, 3 * hd)
    return qkv.permute(1, 0, 2, 3).contiguous()


def _run(monkeypatch, host_streams, rank, last, fail_last=False):
    """One call of rank `rank`; returns its output, what every peer received
    and how each group attended."""
    stack, backend = host_streams
    monkeypatch.setattr(ce, "_LAST", frozenset(last))
    state = ce.IpcA2ARocmState()
    state.rank, state.world, state.inited = rank, WORLD, True
    state.hip = _Hip()
    state.seqtab = torch.arange(64, dtype=torch.int32)
    state.peers = [(rank + s) % WORLD for s in range(1, WORLD)]
    state.copy_streams = [_Stream() for _ in state.peers]
    g_in = WORLD * ROWS * GROUP_HEADS * 3 * HEAD_DIM
    g_out = WORLD * ROWS * GROUP_HEADS * HEAD_DIM
    pool = state._pool(torch.bfloat16, GROUPS * g_in, GROUPS * g_out, GROUPS)
    gen = torch.Generator().manual_seed(rank)
    # the peers' blocks of every group, as if they had been delivered
    pool.inb[rank].copy_(torch.randn(pool.inb[rank].shape, generator=gen))
    q, k, v = (
        torch.randn(ROWS, HEADS, HEAD_DIM, generator=gen).bfloat16() for _ in range(3)
    )
    attended = []

    def attend(q_, k_, v_):
        if fail_last and len(attended) == GROUPS - 1:
            raise RuntimeError("attention failed")
        out = (q_.float() * 2 + k_.float() - v_.float() * 0.5).bfloat16()
        attended.append((out, stack[-1], backend._TAIL_STREAM))
        return out

    out = state.pipelined_attention(q, k, v, attend, GROUPS, pack=_pack)
    received = [pool.outb[p].clone() for p in state.peers]
    flags = [pool.fout[p].clone() for p in state.peers]
    return out, received, flags, attended, state


@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("last", [["klast"], ["tlast"], ["klast", "tlast"]])
def test_last_group_options_deliver_the_default_bytes(
    monkeypatch, host_streams, rank, last
):
    ref_out, ref_received, ref_flags, _, _ = _run(monkeypatch, host_streams, rank, [])
    out, received, flags, _, _ = _run(monkeypatch, host_streams, rank, last)
    assert torch.equal(out, ref_out)
    for got, want in zip(received + flags, ref_received + ref_flags, strict=True):
        assert torch.equal(got, want)


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_one_kernel_send_writes_each_peers_block_at_its_slot(
    monkeypatch, host_streams, rank
):
    """klast: peer p's slot (group, rank) holds rows [p * ROWS, (p + 1) * ROWS)
    of the last group's output, written by one _foreach_copy_ in place of the
    per-peer SDMA copies; the earlier groups keep the SDMA copies."""
    calls = []
    foreach_copy = torch._foreach_copy_

    def spy(dsts, srcs):
        calls.append(len(dsts))
        return foreach_copy(dsts, srcs)

    monkeypatch.setattr(torch, "_foreach_copy_", spy)
    _, received, flags, attended, state = _run(
        monkeypatch, host_streams, rank, ["klast"]
    )
    assert calls == [WORLD - 1]
    g = GROUPS - 1
    last_out = attended[g][0]
    n = ROWS * GROUP_HEADS * HEAD_DIM
    for p, buf in zip(state.peers, received, strict=True):
        slot = buf[(g * WORLD + rank) * n : (g * WORLD + rank + 1) * n]
        assert torch.equal(slot, last_out[p * ROWS : (p + 1) * ROWS].reshape(-1))
    for flag in flags:
        # every group's flag to every peer carries this call's number
        assert flag.view(-1, WORLD)[:GROUPS, rank].tolist() == [1] * GROUPS
    chunk_out = n * 2
    data_sends = [c for c in state.hip.copies if c[1] == chunk_out]
    assert len(data_sends) == (GROUPS - 1) * (WORLD - 1)


def test_tail_stream_is_set_only_while_the_last_group_attends(
    monkeypatch, host_streams
):
    _, _, _, attended, state = _run(monkeypatch, host_streams, 0, ["tlast"])
    tail = state.tail_stream
    assert isinstance(tail, _Stream)
    assert [a[2] for a in attended] == [None] * (GROUPS - 1) + [tail]
    # the group attends on its own stream; the tail stream is the backend's to use
    assert [a[1] for a in attended] == state.group_streams[:GROUPS]
    assert host_streams[1]._TAIL_STREAM is None
    _, _, _, attended, _ = _run(monkeypatch, host_streams, 0, [])
    assert [a[2] for a in attended] == [None] * GROUPS


def test_tail_stream_is_cleared_when_the_last_group_fails(monkeypatch, host_streams):
    with pytest.raises(RuntimeError, match="attention failed"):
        _run(monkeypatch, host_streams, 0, ["tlast"], fail_last=True)
    assert host_streams[1]._TAIL_STREAM is None


def _rocm_gpus(n):
    return (
        torch.version.hip is not None
        and torch.cuda.is_available()
        and torch.cuda.device_count() >= n
    )


def _bitexact_worker(rank, world, port):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        SGLANG_DIFFUSION_ROCM_CE_A2A="1",
    )
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        from sglang.kernels.ops.diffusion.layout.ulysses_qkv_triton import (
            pack_qkv_destination_major,
        )
        from sglang.multimodal_gen.runtime.layers.attention.backends import (
            aiter as backend,
        )

        # MiniMax-H3 at 4 ranks: 9440 rows and 56 heads per rank, 7 groups of
        # 2 heads, the tail split planned for 7 heads as in the model
        rows, heads, hd, groups = 9440, 56, 128, 7
        total, used = world * rows, world * rows - 3
        backend._TAIL_SPLIT_HEADS = 7
        impl = backend.AITerImpl(
            num_heads=heads // world,
            head_size=hd,
            softmax_scale=hd**-0.5,
            packed_trailing_padding=True,
        )
        bounds = (0, used, total)
        cu = torch.tensor(bounds, dtype=torch.int32, device="cuda")

        def attend(q, k, v):
            return impl.forward_varlen(
                q, k, v, cu_seqlens=cu, max_seqlen=used, cu_seqlens_host=bounds
            )

        assert ce.rocm_ce_ready(dist.group.WORLD)
        state = ce.IPC_A2A_ROCM
        gen = torch.Generator(device="cuda").manual_seed(rank)
        inputs = [
            [
                torch.randn(rows, heads, hd, device="cuda", generator=gen).bfloat16()
                for _ in range(3)
            ]
            for _ in range(2)
        ]
        gh = heads // (world * groups)

        def run(last, fill_mode):
            ce._LAST = frozenset(last)
            ce._MAX_CONC = 2 if fill_mode else 0
            ce._FILL_FIRST = fill_mode == "first"
            outs = []
            for q, k, v in inputs:

                def fill(send, own, first, n, q=q, k=k, v=v):
                    for g in range(first, first + n):
                        for w in range(world):
                            blk = slice(
                                (w * groups + g) * gh, (w * groups + g + 1) * gh
                            )
                            dst = own[g, w] if w == rank else send[w, g]
                            torch.cat((q[:, blk], k[:, blk], v[:, blk]), -1, out=dst)

                outs.append(
                    state.pipelined_attention(
                        q,
                        k,
                        v,
                        attend,
                        groups,
                        pack=pack_qkv_destination_major,
                        fill=fill if fill_mode else None,
                    ).clone()
                )
            torch.cuda.synchronize()
            return outs

        ref = run([], None)
        for fill_mode in (None, "all", "first"):
            for last in ([], ["klast"], ["tlast"], ["klast", "tlast"]):
                outs = run(last, fill_mode)
                same = all(torch.equal(a, b) for a, b in zip(outs, ref, strict=True))
                flag = torch.tensor([int(same)], device="cuda")
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                assert flag.item() == 1, (fill_mode, last)
        ce._LAST, ce._MAX_CONC, ce._FILL_FIRST = frozenset(), 0, False
        state.drop()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not _rocm_gpus(4), reason="needs four ROCm GPUs")
def test_last_group_options_are_bitexact_on_four_rocm_gpus():
    """torch.equal on every rank against the default pipeline: pack and fill
    paths, group 0 filled first, klast / tlast / both, two calls back to back."""
    pytest.importorskip("aiter", reason="AITer is a ROCm-only dependency")
    mp.spawn(
        _bitexact_worker,
        args=(4, get_free_port_below_ephemeral()),
        nprocs=4,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
