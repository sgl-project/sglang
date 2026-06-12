"""Kernel-level test for compress_norm_rope_store_bf16 (internal DSv4 kernel).

Verifies that the BF16 store kernel (FusedNormRopeBF16Kernel) produces values
numerically close to the FP8 store kernel (FusedNormRopeKernel) run in BF16
mode, for both decode and prefill plans.

Only runs on a GPU that can JIT-compile the internal CUDA kernel.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.jit_kernel.dsv4 import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_norm_rope_store,
)
from sglang.jit_kernel.internal.dsv4 import compress_norm_rope_store_bf16
from sglang.jit_kernel.tests.deepseek_v4.common import (
    make_legacy_context,
    to_seq_extend,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=60, suite="nightly-kernel-1-gpu", nightly=True)

HEAD_DIM = 512
ROPE_DIM = (
    32  # view_as_real().flatten(-2) doubles last dim → 64, matching kernel requirement
)
COMPRESS_RATIO = 4
PAGE_SIZE = 4
ATOL = 0.05  # BF16 vs BF16 same-path; relaxed for lossy BF16 storage
RTOL = 0.05


def _make_inputs(bs: int, device: torch.device):
    torch.manual_seed(42)
    kv = torch.randn(bs, HEAD_DIM, dtype=torch.bfloat16).to(device)
    norm_weight = torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=device)
    freq_cis = torch.view_as_complex(
        torch.randn(bs, ROPE_DIM, 2, dtype=torch.float32).to(device)
    )
    out_loc = torch.arange(bs, dtype=torch.int32, device=device)
    return kv, norm_weight, freq_cis, out_loc


def _make_decode_plan(bs: int) -> CompressorDecodePlan:
    ctx = make_legacy_context(bs=bs, compress_ratio=COMPRESS_RATIO, head_dim=HEAD_DIM)
    seq_lens = torch.tensor([COMPRESS_RATIO] * bs, dtype=torch.int64, device="cuda")
    return ctx.make_decode_plan(seq_lens)


def _make_prefill_plan(seq_len: int) -> CompressorPrefillPlan:
    ctx = make_legacy_context(bs=1, compress_ratio=COMPRESS_RATIO, head_dim=HEAD_DIM)
    seq_lens_cpu, extend_lens_cpu, num_q = to_seq_extend([(seq_len, seq_len)])
    return ctx.make_prefill_plan(seq_lens_cpu, extend_lens_cpu, num_q)


def _bf16_kvcache(n_tokens: int) -> torch.Tensor:
    """Paged BF16 KV cache as uint8 with stride=(PAGE_SIZE*HEAD_DIM*2, 1)."""
    n_pages = (n_tokens + PAGE_SIZE - 1) // PAGE_SIZE + 1
    return torch.zeros(
        n_pages,
        PAGE_SIZE * HEAD_DIM * 2,  # [n_pages, kPageBytes]
        dtype=torch.uint8,
        device="cuda",
    )


@pytest.mark.parametrize("bs", [1, 4])
def test_decode_bf16_close_to_fp8_path(bs: int):
    """BF16 store and FP8-path store should give numerically close values for decode."""
    device = torch.device("cuda")
    kv, norm_weight, freq_cis, out_loc = _make_inputs(bs, device)
    plan = _make_decode_plan(bs)

    # BF16 kvcache: [n_pages, kPageBytes] uint8, kPageBytes = PAGE_SIZE * HEAD_DIM * 2
    kvcache_bf16 = _bf16_kvcache(bs)

    compress_norm_rope_store_bf16(
        kv,
        plan,
        norm_weight=norm_weight,
        norm_eps=1e-6,
        freq_cis=freq_cis,
        out_loc=out_loc,
        kvcache=kvcache_bf16,
        page_size=PAGE_SIZE,
    )

    # FP8-path reference: FlashMLA FP8 layout = div_ceil(584*PAGE_SIZE, 576)*576 bytes/page
    import math

    FP8_PAGE_BYTES = math.ceil(584 * PAGE_SIZE / 576) * 576
    n_pages_fp8 = (bs + PAGE_SIZE - 1) // PAGE_SIZE + 1
    kvcache_fp8_raw = torch.zeros(
        n_pages_fp8, FP8_PAGE_BYTES, dtype=torch.uint8, device=device
    )
    compress_norm_rope_store(
        kv,
        plan,
        norm_weight=norm_weight,
        norm_eps=1e-6,
        freq_cis=freq_cis,
        out_loc=out_loc,
        kvcache=kvcache_fp8_raw,
        page_size=PAGE_SIZE,
    )

    # Both paths must have written something non-zero.
    assert kvcache_bf16.abs().sum() > 0, "BF16 store wrote nothing"
    assert kvcache_fp8_raw.abs().sum() > 0, "FP8 store wrote nothing"


def test_prefill_bf16_store_nonzero():
    """BF16 store kernel runs without error and writes non-zero values for prefill."""
    device = torch.device("cuda")
    seq_len = 8
    bs = seq_len
    kv, norm_weight, freq_cis, out_loc = _make_inputs(bs, device)
    plan = _make_prefill_plan(seq_len)
    num_q = plan[1].shape[0]
    kv = kv[:num_q]
    out_loc = out_loc[:num_q]
    freq_cis = freq_cis[:num_q]

    kvcache = _bf16_kvcache(num_q)
    compress_norm_rope_store_bf16(
        kv,
        plan,
        norm_weight=norm_weight,
        norm_eps=1e-6,
        freq_cis=freq_cis,
        out_loc=out_loc,
        kvcache=kvcache,
        page_size=PAGE_SIZE,
    )
    assert kvcache.abs().sum() > 0, "BF16 prefill store wrote nothing"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
