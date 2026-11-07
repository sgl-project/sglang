"""Kimi Delta Attention in batch-invariant mode: a sequence's prefill (chunk_kda) and a
request's decode step (fused_recurrent_kda_packed_decode) must give the same bits alone
and inside larger batches, although both pick their kernels by batch size otherwise."""

import random
import sys
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention.fla import fused_recurrent as fused_recurrent_mod
from sglang.kernels.ops.attention.fla import kda as kda_mod
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_kda_packed_decode,
)
from sglang.kernels.ops.attention.fla.kda import chunk_kda
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

H, K, V = 32, 128, 128  # Kimi-Linear-48B-A3B at TP1


@pytest.fixture(autouse=True)
def _batch_invariant_mode():
    with (
        patch.object(kda_mod, "is_batch_invariant_mode_enabled", return_value=True),
        patch.object(
            fused_recurrent_mod, "is_batch_invariant_mode_enabled", return_value=True
        ),
    ):
        yield


def _randn(*shape, g, scale=1.0):
    return (torch.randn(*shape, generator=g, device="cuda") * scale).to(torch.bfloat16)


def _sequence(T, g):
    return [
        _randn(1, T, H, K, g=g),  # q
        _randn(1, T, H, K, g=g),  # k
        _randn(1, T, H, V, g=g),  # v
        _randn(1, T, H, K, g=g, scale=0.5),  # raw gate
        torch.rand(1, T, H, generator=g, device="cuda").to(torch.bfloat16),  # beta
    ]


def _prefill(seqs, A_log, dt_bias):
    q, k, v, a, beta = (torch.cat([s[i] for s in seqs], dim=1) for i in range(5))
    lens = [s[0].shape[1] for s in seqs]
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device="cuda")
    state = torch.zeros(len(seqs) + 1, H, V, K, dtype=torch.float32, device="cuda")
    idx = torch.arange(1, len(seqs) + 1, dtype=torch.int32, device="cuda")
    o = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=a,
        beta=beta,
        initial_state=state,
        initial_state_indices=idx,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu.to(torch.int32),
        A_log=A_log,
        dt_bias=dt_bias,
    )
    return (o[0] if isinstance(o, tuple) else o), state, cu


@pytest.mark.parametrize("target_len", [59, 200, 1000])
def test_chunk_kda_prefill_batch_invariant(target_len):
    g = torch.Generator(device="cuda").manual_seed(0)
    A_log = torch.randn(H, generator=g, device="cuda")
    dt_bias = torch.randn(H * K, generator=g, device="cuda")
    target = _sequence(target_len, g)
    alone, alone_state, _ = _prefill([target], A_log, dt_bias)
    rng = random.Random(target_len)
    for n in (2, 4, 8, 16, 32):
        pos = rng.randrange(n)
        seqs = [_sequence(rng.randint(16, 400), g) for _ in range(n - 1)]
        seqs.insert(pos, target)
        out, state, cu = _prefill(seqs, A_log, dt_bias)
        start = int(cu[pos])
        assert torch.equal(out[:, start : start + target_len], alone), (
            f"output, {n} seqs"
        )
        assert torch.equal(state[pos + 1], alone_state[1]), f"final state, {n} seqs"


def test_kda_packed_decode_batch_invariant():
    g = torch.Generator(device="cuda").manual_seed(1)
    B = 32
    mixed_qkv = _randn(B, 2 * H * K + H * V, g=g)
    a = _randn(B, H * K, g=g, scale=0.5)
    b = _randn(B, H, g=g)
    A_log = torch.randn(H, generator=g, device="cuda")
    dt_bias = torch.randn(H * K, generator=g, device="cuda")
    state = torch.randn(B, H, V, K, generator=g, device="cuda") * 0.1

    def run(n):
        s = state[:n].clone()
        out = mixed_qkv.new_empty(n, 1, H, V)
        fused_recurrent_kda_packed_decode(
            mixed_qkv=mixed_qkv[:n].contiguous(),
            a=a[:n].contiguous(),
            b=b[:n].contiguous(),
            A_log=A_log,
            dt_bias=dt_bias,
            scale=K**-0.5,
            initial_state=s,
            out=out,
            ssm_state_indices=torch.arange(n, dtype=torch.int32, device="cuda"),
            use_qk_l2norm_in_kernel=True,
        )
        return out, s

    ref_out, ref_state = run(1)
    for n in (2, 4, 7, 8, 16, 32):
        out, s = run(n)
        assert torch.equal(out[:1], ref_out), f"output, batch of {n}"
        assert torch.equal(s[:1], ref_state), f"state, batch of {n}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
