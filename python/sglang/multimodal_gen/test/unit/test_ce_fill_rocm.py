# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3's QK RMSNorm + RoPE written into the ROCm copy-engine exchange.

CPU: the fill kernel's block layout under the Triton interpreter against a
pure-torch norm + RoPE + destination-major pack, the fill paths of the
copy-engine protocol on host threads (every rank a thread; a copy lands when
queued, a flag wait spins on the flag word), and the H3 gating. ROCm GPU: the
fill's bytes against ``aten_head_rmsnorm`` followed by the pack; the 4-rank
model-level parity test is single_test_file/test_minimax_h3_rocm_fill_multi_gpu.py.
"""

import contextlib
import ctypes
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.distributed.device_communicators import (
    ce_fill_rocm,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators import (
    ipc_a2a_rocm as ce,
)
from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as h3
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

WORLD, HEADS, HEAD_DIM, ROPE_DIM = 4, 56, 128, 96


def _pack(q, k, v, n):
    """Destination-major ``[n, rows, heads / n, 3 * head_dim]``, q|k|v per head."""
    rows, heads, head_dim = q.shape
    split = [t.reshape(rows, n, heads // n, head_dim) for t in (q, k, v)]
    return torch.cat(split, dim=-1).transpose(0, 1).contiguous()


def _norm_rope(x, weight, eps, cache, positions):
    """RMSNorm then NeoX RoPE with each product rounded to bf16, in torch."""
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = (weight.float() * (rstd * xf)).bfloat16()
    half = cache.shape[-1] // 2
    cos, sin = (t[:, None].float() for t in cache[positions].split(half, dim=-1))
    y1, y2 = y[..., :half].float(), y[..., half : 2 * half].float()
    r = lambda t: t.bfloat16().float()
    out = y.clone()
    out[..., :half] = (r(y1 * cos) - r(y2 * sin)).bfloat16()
    out[..., half : 2 * half] = (r(y2 * cos) + r(y1 * sin)).bfloat16()
    return out


def _fill_inputs(tokens, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    qkv = torch.randn(tokens, 3 * HEADS * HEAD_DIM, device=device, generator=gen)
    # q/k/v are views of the fused projection, as in MiniMax-H3
    q, k, v = (
        t.view(tokens, HEADS, HEAD_DIM)
        for t in qkv.bfloat16().split(HEADS * HEAD_DIM, dim=-1)
    )
    wq, wk = (
        (1 + 0.1 * torch.randn(HEAD_DIM, device=device, generator=gen)).bfloat16()
        for _ in range(2)
    )
    angles = 10 * torch.rand(tokens + 3, ROPE_DIM // 2, device=device, generator=gen)
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).bfloat16()
    positions = torch.randperm(tokens + 3, device=device, generator=gen)[:tokens]
    return q, k, v, wq, wk, cache, positions


def _check_fill(normed_reference, exact, device):
    """Every rank's fill of every group (one launch, and group 0 + the rest)
    against the reference's pack; untouched blocks stay NaN."""
    tokens, eps = 5, 1e-6
    q, k, v, wq, wk, cache, positions = _fill_inputs(tokens, device)
    nq, nk = (
        normed_reference(q, wq, eps, cache, positions),
        normed_reference(k, wk, eps, cache, positions),
    )
    for groups in (2, 7):
        gh = HEADS // (WORLD * groups)
        ref = _pack(nq, nk, v, WORLD * groups).view(WORLD, groups, tokens, gh, -1)
        for rank in range(WORLD):
            for launches in ([(0, groups)], [(0, 1), (1, groups - 1)]):
                send = torch.full(
                    (WORLD, groups, tokens, gh, 3 * HEAD_DIM),
                    float("nan"),
                    device=device,
                ).bfloat16()
                own = torch.full(
                    (groups, WORLD, tokens, gh, 3 * HEAD_DIM),
                    float("nan"),
                    device=device,
                ).bfloat16()
                for first, n in launches:
                    ce_fill_rocm.qkv_fill(
                        send,
                        own,
                        q,
                        k,
                        v,
                        wq,
                        wk,
                        eps,
                        eps,
                        cache,
                        positions,
                        rank=rank,
                        world=WORLD,
                        groups=groups,
                        first_group=first,
                        n_groups=n,
                    )
                for w in range(WORLD):
                    got, idle = (
                        (own[:, rank], send[rank])
                        if w == rank
                        else (send[w], own[:, w])
                    )
                    assert idle.isnan().all(), (groups, rank, w)
                    if exact:
                        assert torch.equal(got, ref[w]), (groups, rank, w)
                    else:
                        # the interpreter's fp32 sums may round an element one
                        # bf16 ulp (<= 2 ** -5 below 8) away from torch's
                        torch.testing.assert_close(
                            got[..., : 2 * HEAD_DIM],
                            ref[w, ..., : 2 * HEAD_DIM],
                            atol=0.07,
                            rtol=0,
                        )
                        assert torch.equal(
                            got[..., 2 * HEAD_DIM :], ref[w, ..., 2 * HEAD_DIM :]
                        )


def _interpreted_fill() -> int:
    with patch.object(
        ce_fill_rocm.torch,
        "get_device_module",
        lambda: SimpleNamespace(device=lambda d: contextlib.nullcontext()),
    ):
        _check_fill(_norm_rope, exact=False, device="cpu")
    print("INTERPRETED_FILL PASS", flush=True)
    return 0


def test_interpreted_fill_writes_each_block_to_its_destination():
    proc = subprocess.run(
        [sys.executable, __file__, "--interpreted-fill"],
        env={**os.environ, "TRITON_INTERPRET": "1"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert "INTERPRETED_FILL PASS" in proc.stdout, proc.stderr[-4000:]


@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="needs a ROCm GPU",
)
def test_fill_gives_the_bytes_of_the_norm_kernel_and_the_pack():
    from sglang.kernels.ops.diffusion import aten_head_rmsnorm

    q, k, v, wq, wk, cache, positions = _fill_inputs(5, "cuda")
    assert ce_fill_rocm.can_fill(q, k, v, wq, wk, cache, positions)
    assert not ce_fill_rocm.can_fill(q.float(), k, v, wq, wk, cache, positions)
    assert not ce_fill_rocm.can_fill(q, k, v, wq, wk, cache, positions.int())
    _check_fill(aten_head_rmsnorm, exact=True, device="cuda")


# ---- the fill paths of the copy-engine protocol, on host threads -----------


def _attend(q, k, v):
    # head-independent but row-order dependent, like attention
    return torch.cumsum(q.float() * k.flip(0).float(), dim=0).to(q.dtype) + v


def _reference(qf, kf, vf, groups):
    group_heads = qf.shape[1] // (WORLD * groups)
    out = torch.empty_like(qf)
    for block in range(WORLD * groups):
        h = slice(block * group_heads, (block + 1) * group_heads)
        out[:, h] = _attend(qf[:, h], kf[:, h], vf[:, h])
    return out


class _Stream:
    cuda_stream = 0

    def record_event(self):
        return object()

    def wait_event(self, event):
        pass

    def wait_stream(self, stream):
        pass


class _HostGroup:
    def __init__(self):
        self.sync = threading.Barrier(WORLD, timeout=60)
        self.slots = [None] * WORLD

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


def _host_fill(q, k, v, rank, groups):
    rows, heads, head_dim = q.shape
    packed = _pack(q, k, v, WORLD * groups).view(WORLD, groups, rows, -1, 3 * head_dim)

    def fill(send, own, first_group, n_groups):
        for g in range(first_group, first_group + n_groups):
            for w in range(WORLD):
                (own[g, rank] if w == rank else send[w, g]).copy_(packed[w, g])

    return fill


def _run_filled_ranks(inputs, groups, seq_table):
    group = _HostGroup()

    def rank_main(rank):
        state = ce.IpcA2ARocmState()
        state.group, state.rank, state.world = group, rank, WORLD
        state.hip = _HostHip(group)
        state.seqtab = torch.arange(seq_table, dtype=torch.int32)
        state.peers = [(rank + s) % WORLD for s in range(1, WORLD)]
        state.copy_streams = [_Stream() for _ in state.peers]
        state.inited = True
        outs = []
        try:
            for qf, kf, vf in inputs:
                rows = qf.shape[0] // WORLD
                q, k, v = (t[rank * rows : (rank + 1) * rows] for t in (qf, kf, vf))
                fill = _host_fill(q, k, v, rank, groups)
                outs.append(
                    state.pipelined_attention(
                        q, k, v, _attend, groups, fill=fill
                    ).clone()
                )
        except BaseException:
            group.sync.abort()
            raise
        return outs

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
        ThreadPoolExecutor(WORLD) as pool,
    ):
        futures = [pool.submit(rank_main, rank) for rank in range(WORLD)]
        return [f.result(timeout=300) for f in futures]


_VARIANTS = {
    "one_fill": {},
    "fill_per_group": {"_FILL_SPLIT": True},
    "fill_first_group": {"_FILL_FIRST": True},
    "merge_per_group": {"_GROUP_MERGE": True},
    "merged_output": {"_OUT_MERGED": True},
    "two_in_flight": {"_MAX_CONC": 2},
    "first_group_two_in_flight": {"_FILL_FIRST": True, "_MAX_CONC": 2},
}


@pytest.mark.parametrize("variant", list(_VARIANTS))
@pytest.mark.parametrize("groups", [2, 7])
def test_filled_exchange_gives_the_collective_bytes(groups, variant):
    # the 9-row calls outgrow the pool sized for 5 rows; the short table resets
    # the flag counters in between
    rows_per_call = [5, 5, 5, 9, 9, 9, 5]
    gen = torch.Generator().manual_seed(0)
    inputs = [
        tuple(
            torch.randn(WORLD * rows, HEADS, 8, generator=gen).bfloat16()
            for _ in range(3)
        )
        for rows in rows_per_call
    ]
    with contextlib.ExitStack() as stack:
        for name, value in _VARIANTS[variant].items():
            stack.enter_context(patch.object(ce, name, value))
        ranks = _run_filled_ranks(inputs, groups, seq_table=2 * groups + 2)
    for call, (qf, kf, vf) in enumerate(inputs):
        ref = _reference(qf, kf, vf, groups)
        rows = rows_per_call[call]
        for rank, outs in enumerate(ranks):
            assert torch.equal(outs[call], ref[rank * rows : (rank + 1) * rows]), (
                f"call {call} rank {rank}"
            )


# ---- MiniMax-H3 gating ------------------------------------------------------

_GO = dict(
    env="1",
    hip="7.0",
    verified=True,
    disabled=False,
    capturing=False,
    breakable_graph=False,
    eps=1e-6,
    norm=True,
    rope=True,
    flashinfer=False,
    groups=7,
    backend=AttentionBackendEnum.AITER,
    heads=HEADS,
    can_fill=True,
    ready=True,
)
_KNOCKOUTS = {
    "env_off": {"env": "0"},
    "not_rocm": {"hip": None},
    "norm_unverified": {"verified": False},
    "norm_disabled": {"disabled": True},
    "graph_capture": {"capturing": True},
    "breakable_cuda_graph": {"breakable_graph": True},
    "no_eps": {"eps": None},
    "norm_kernel_unsupported": {"norm": False},
    "rope_unsupported": {"rope": False},
    "flashinfer_ulysses": {"flashinfer": True},
    "sequential_exchange": {"groups": 0},
    "one_group": {"groups": 1},
    "other_backend": {"backend": AttentionBackendEnum.FA},
    "heads_not_divisible": {"heads": 60},
    "cannot_fill": {"can_fill": False},
    "copy_engine_not_ready": {"ready": False},
}


@contextlib.contextmanager
def _h3_conditions(monkeypatch, c):
    monkeypatch.setenv("SGLANG_DIFFUSION_ROCM_CE_FILL", c["env"])
    gate = h3._ATEN_HEAD_RMSNORM_ROPE
    group = object()
    with (
        patch.object(torch.version, "hip", c["hip"]),
        patch.object(gate, "verified", c["verified"]),
        patch.object(gate, "disabled", c["disabled"]),
        patch.object(
            h3.torch.cuda, "is_current_stream_capturing", return_value=c["capturing"]
        ),
        patch.object(
            h3, "is_in_breakable_cuda_graph", return_value=c["breakable_graph"]
        ),
        patch.object(h3, "can_use_aten_head_rmsnorm", return_value=c["norm"]),
        patch.object(h3, "can_use_aten_head_rmsnorm_rope", return_value=c["rope"]),
        patch.object(h3, "_minimax_h3_bind_attention_backend"),
        patch.object(h3.envs, "SGLANG_DIFFUSION_FLASHINFER_ULYSSES", c["flashinfer"]),
        patch.object(h3.envs, "SGLANG_DIFFUSION_ULYSSES_PIPELINE_GROUPS", c["groups"]),
        patch.object(h3, "get_ulysses_ctx", return_value=(WORLD, 0)),
        patch.object(
            parallel_state,
            "get_sp_group",
            return_value=SimpleNamespace(ulysses_group=group),
        ),
        patch.object(ce_fill_rocm, "can_fill", return_value=c["can_fill"]),
        patch.object(ce_fill_rocm, "qkv_fill") as qkv_fill,
        patch.object(ce, "rocm_ce_ready", return_value=c["ready"]),
        patch.object(ce.IPC_A2A_ROCM, "rank", 2, create=True),
        patch.object(ce.IPC_A2A_ROCM, "pipelined_attention", return_value="ce") as run,
    ):
        yield run, qkv_fill


def _h3_call(c):
    norm = lambda: SimpleNamespace(weight=torch.ones(HEAD_DIM).bfloat16(), eps=c["eps"])
    attention = SimpleNamespace(
        q_norm=norm(),
        k_norm=norm(),
        _attention_backend_enum=c["backend"],
        _attention_impl=MagicMock(),
    )
    q = torch.zeros(5, c["heads"], HEAD_DIM).bfloat16()
    rope_cache = (torch.zeros(5, ROPE_DIM).bfloat16(), torch.arange(5))
    bounds = (0, 4, 5)
    out = h3._minimax_h3_rocm_fill_pipelined_attention(
        attention,
        q,
        q,
        q,
        rope_cache,
        cu_seqlens=torch.tensor(bounds),
        cu_seqlens_host=bounds,
        max_seqlen=4,
    )
    return out, attention, q, rope_cache


@pytest.mark.parametrize("knockout", list(_KNOCKOUTS))
def test_h3_fill_falls_back_unless_its_path_would_run(monkeypatch, knockout):
    c = {**_GO, **_KNOCKOUTS[knockout]}
    with _h3_conditions(monkeypatch, c) as (run, qkv_fill):
        out, *_ = _h3_call(c)
    assert out is None
    run.assert_not_called()
    qkv_fill.assert_not_called()


@pytest.mark.parametrize("groups, used", [(7, 7), (-1, 2)], ids=["seven", "auto"])
def test_h3_fill_runs_the_copy_engine_pipeline(monkeypatch, groups, used):
    c = {**_GO, "groups": groups}
    with _h3_conditions(monkeypatch, c) as (run, qkv_fill):
        out, attention, q, (cache, positions) = _h3_call(c)
        assert out == "ce"
        args, kwargs = run.call_args
        assert args[4] == used and "pack" not in kwargs
        attend, fill = args[3], kwargs["fill"]
        attend(q, q, q)
        fill("send", "own", 1, 3)
    assert attention._attention_impl.forward_varlen.call_args.kwargs[
        "cu_seqlens_host"
    ] == (0, 4, 5)
    args, kwargs = qkv_fill.call_args
    assert args[:2] == ("send", "own")
    assert args[5] is attention.q_norm.weight and args[6] is attention.k_norm.weight
    assert args[9] is cache and args[10] is positions
    assert kwargs == dict(rank=2, world=WORLD, groups=used, first_group=1, n_groups=3)


@pytest.mark.parametrize(
    "ulysses, ring, filled",
    [
        (True, False, True),
        (True, False, False),
        (False, False, False),
        (True, True, False),
    ],
    ids=["filled", "fill_declined", "no_ulysses", "ring"],
)
def test_h3_attend_skips_the_norm_only_when_filled(ulysses, ring, filled):
    tokens = 5
    attention = SimpleNamespace(
        _use_fused_qknorm_rope=False,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        q_norm=None,
        k_norm=None,
        rotary_emb=None,
    )
    q = torch.zeros(tokens, HEADS, HEAD_DIM)
    out = torch.randn(tokens, HEADS, HEAD_DIM)

    class Norm(Exception):
        pass

    with (
        patch.object(
            h3,
            "_minimax_h3_rocm_fill_pipelined_attention",
            return_value=out if filled else None,
        ) as fill,
        patch.object(h3, "_apply_qk_norm_rope", side_effect=Norm),
    ):
        call = lambda: h3.MiniMaxH3Attention._attend(
            attention,
            q,
            q,
            q,
            q,
            rope_cache=(torch.zeros(tokens, ROPE_DIM), torch.arange(tokens)),
            cu_seqlens=torch.tensor([0, tokens]),
            cu_seqlens_host=None,
            max_seqlen=tokens,
            subblock_sparse_query_block_mask=None,
            ulysses_active=ulysses,
            ring_active=ring,
        )
        if filled:
            assert torch.equal(call(), out.reshape(tokens, HEADS * HEAD_DIM))
        else:
            with pytest.raises(Norm):
                call()
    assert fill.called == (ulysses and not ring)


if __name__ == "__main__":
    if "--interpreted-fill" in sys.argv:
        raise SystemExit(_interpreted_fill())
