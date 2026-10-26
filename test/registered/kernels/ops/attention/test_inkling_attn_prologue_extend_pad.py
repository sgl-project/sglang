"""The fused attn prologue extend kernel must not store K/V for the padded tail
of a CUDA-graph launch (tokens at or beyond cu[B]); real tokens are still stored.
"""

import pytest
import torch

from sglang.kernels.ops.attention.inkling_attn_prologue import (
    inkling_attn_prologue_extend,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEAD = 128
W = 4
EPS = 1e-6
DQ = 1024
DKV = 256
SENTINEL = 7.0


@pytest.mark.parametrize("num_pad", [1, 8])
def test_extend_prologue_skips_padded_tokens(num_pad):
    torch.manual_seed(1234)
    dev = "cuda"
    bf16 = torch.bfloat16
    seq_lens = [5, 3]
    num_seqs = len(seq_lens)
    real = sum(seq_lens)
    t = real + num_pad
    row = DQ + 2 * DKV + 64

    qkvr = torch.randn(t, row, device=dev, dtype=bf16)
    qkvr[real:] = float("nan")  # stale non-finite rows of the padded bucket
    k_cache = torch.randn(64, W - 1, DKV, device=dev, dtype=bf16)
    v_cache = torch.randn(64, W - 1, DKV, device=dev, dtype=bf16)
    cache_indices = torch.tensor([3, 4], device=dev, dtype=torch.int32)
    cache_mask = torch.ones(num_seqs, device=dev, dtype=torch.bool)
    has_initial_state = torch.ones(num_seqs, device=dev, dtype=torch.bool)
    cu = torch.tensor([0, 5, real], device=dev, dtype=torch.int64)
    # Padding tokens are attributed to the last sequence, as the production launch does.
    si = torch.tensor(
        [0] * seq_lens[0] + [1] * (seq_lens[1] + num_pad), device=dev, dtype=torch.int32
    )
    k_weight = torch.randn(DKV, W, device=dev, dtype=bf16) * 0.3
    v_weight = torch.randn(DKV, W, device=dev, dtype=bf16) * 0.3
    q_gamma = torch.randn(HEAD, device=dev, dtype=bf16)
    k_gamma = torch.randn(HEAD, device=dev, dtype=bf16)

    slots = 64
    real_slots = torch.arange(10, 10 + real, device=dev, dtype=torch.int64)
    loc = torch.zeros(t, device=dev, dtype=torch.int64)  # padding maps to slot 0
    loc[:real] = real_slots
    k_buf = torch.zeros(slots, DKV // HEAD, HEAD, device=dev, dtype=bf16)
    v_buf = torch.zeros_like(k_buf)
    k_buf[0] = SENTINEL
    v_buf[0] = SENTINEL

    _, k_out, v_out, _ = inkling_attn_prologue_extend(
        qkvr=qkvr,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_indices=cache_indices,
        cache_mask=cache_mask,
        has_initial_state=has_initial_state,
        cu=cu,
        si=si,
        k_weight=k_weight,
        v_weight=v_weight,
        track_rows=torch.empty(0, W - 1, device=dev, dtype=torch.int64),
        track_mask=torch.empty(0, device=dev, dtype=torch.bool),
        track_dst=torch.empty(0, device=dev, dtype=torch.int64),
        q_gamma=q_gamma,
        k_gamma=k_gamma,
        eps=EPS,
        loc=loc,
        k_buf=k_buf,
        v_buf=v_buf,
        q_off=0,
        k_off=DQ,
        v_off=DQ + DKV,
        dq=DQ,
        dkv=DKV,
        activation=None,
        use_residual=True,
        do_store=True,
    )
    torch.cuda.synchronize()

    untouched = torch.ones(slots, device=dev, dtype=torch.bool)
    untouched[real_slots] = False
    sentinel = torch.full_like(k_buf[0], SENTINEL)
    assert torch.equal(k_buf[0], sentinel), "padded token K was stored into slot 0"
    assert torch.equal(v_buf[0], sentinel), "padded token V was stored into slot 0"
    assert not k_buf[untouched][1:].any(), "padded token K was stored into a slot"
    assert not v_buf[untouched][1:].any(), "padded token V was stored into a slot"

    # Real tokens are still stored, and equal to the kernel's own k/v outputs.
    k_real = k_out[:real].reshape(real, DKV)
    v_real = v_out[:real].reshape(real, DKV)
    torch.testing.assert_close(
        k_buf[real_slots].view(real, DKV), k_real, rtol=0, atol=0
    )
    torch.testing.assert_close(
        v_buf[real_slots].view(real, DKV), v_real, rtol=0, atol=0
    )
    assert torch.isfinite(k_buf[real_slots]).all()
    assert torch.isfinite(v_buf[real_slots]).all()
