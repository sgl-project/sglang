import unittest
from unittest.mock import patch

import torch
import triton
import triton.language as tl

import sglang.kernels.ops.layernorm.hy4_ihc as hy4_ihc
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=35, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@triton.jit
def _reference_hy4_ihc_pre_kernel(
    x_ptr,
    fn_ptr,
    scale_ptr,
    base_ptr,
    y_ptr,
    post_ptr,
    hidden_size: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    magnitude: tl.constexpr,
    norm_eps: tl.constexpr,
    hc_eps: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + pid * K_TOTAL
    m_idx = tl.arange(0, HC_POW2)
    m_mask = m_idx < HC_MULT

    sumsq = tl.zeros((), dtype=tl.float32)
    mix_pre = tl.zeros((HC_POW2,), dtype=tl.float32)
    mix_post = tl.zeros((HC_POW2,), dtype=tl.float32)
    for k_off in tl.range(0, K_TOTAL, BLOCK_K):
        k_offs = k_off + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_TOTAL
        x_tile = tl.load(x_row + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(x_tile * x_tile, axis=0)

        fn_offs = m_idx[:, None] * K_TOTAL + k_offs[None, :]
        fn_mask = m_mask[:, None] & k_mask[None, :]
        mix_pre += tl.sum(
            tl.load(fn_ptr + fn_offs, mask=fn_mask, other=0.0) * x_tile[None, :],
            axis=1,
        )
        mix_post += tl.sum(
            tl.load(
                fn_ptr + HC_MULT * K_TOTAL + fn_offs,
                mask=fn_mask,
                other=0.0,
            )
            * x_tile[None, :],
            axis=1,
        )

    rsqrt = tl.rsqrt(sumsq / K_TOTAL + norm_eps)
    scale_pre = tl.load(scale_ptr)
    scale_post = tl.load(scale_ptr + 1)
    base_pre = tl.load(base_ptr + m_idx, mask=m_mask, other=0.0)
    base_post = tl.load(base_ptr + HC_MULT + m_idx, mask=m_mask, other=0.0)

    pre = tl.sigmoid(mix_pre * rsqrt * scale_pre + base_pre) + hc_eps
    post = magnitude * tl.sigmoid(mix_post * rsqrt * scale_post + base_post) + hc_eps
    tl.store(post_ptr + pid * HC_MULT + m_idx, post, mask=m_mask)

    y_row = y_ptr + pid * hidden_size
    for d_off in tl.range(0, hidden_size, BLOCK_D):
        d_offs = d_off + tl.arange(0, BLOCK_D)
        d_mask = d_offs < hidden_size
        y_block = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for m in tl.static_range(HC_MULT):
            x_m = tl.load(x_row + m * hidden_size + d_offs, mask=d_mask, other=0.0)
            pre_m = tl.sum(tl.where(m_idx == m, pre, 0.0), axis=0)
            y_block += pre_m * x_m.to(tl.float32)
        tl.store(
            y_row + d_offs,
            y_block.to(y_ptr.dtype.element_ty),
            mask=d_mask,
        )


def _reference_hy4_ihc_pre(x, hc_fn, hc_scale, hc_base):
    num_tokens, hc_mult, hidden_size = x.shape
    k_total = hc_mult * hidden_size
    y = torch.empty((num_tokens, hidden_size), dtype=x.dtype, device=x.device)
    post = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=x.device)
    if num_tokens == 0:
        return y, post

    _reference_hy4_ihc_pre_kernel[(num_tokens,)](
        x,
        hc_fn,
        hc_scale,
        hc_base,
        y,
        post,
        hidden_size=hidden_size,
        HC_MULT=hc_mult,
        HC_POW2=triton.next_power_of_2(hc_mult),
        K_TOTAL=k_total,
        BLOCK_K=1024,
        BLOCK_D=1024,
        magnitude=2.0,
        norm_eps=1e-6,
        hc_eps=1e-6,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return y, post


@unittest.skipUnless(torch.cuda.is_available(), "HYV4 Triton kernels need CUDA")
class TestHy4DecodeKernels(CustomTestCase):
    def test_split_k_matches_single_cta(self):
        torch.manual_seed(0)
        for num_tokens, hidden_size in (
            (0, 6144),
            (1, 4096),
            (3, 4100),
            (31, 6144),
            (64, 6144),
        ):
            with self.subTest(num_tokens=num_tokens, hidden_size=hidden_size):
                hc_mult = 4
                k_total = hc_mult * hidden_size
                x = torch.randn(
                    num_tokens,
                    hc_mult,
                    hidden_size,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                hc_fn = (
                    torch.randn(
                        2 * hc_mult,
                        k_total,
                        device="cuda",
                        dtype=torch.float32,
                    )
                    * 0.02
                )
                hc_scale = torch.tensor([0.7, 1.3], device="cuda", dtype=torch.float32)
                hc_base = (
                    torch.randn(2 * hc_mult, device="cuda", dtype=torch.float32) * 0.5
                )

                expected = _reference_hy4_ihc_pre(x, hc_fn, hc_scale, hc_base)
                with patch.object(hy4_ihc, "_hpc_ihc_op", return_value=None):
                    actual = hy4_ihc.fused_hy4_ihc_pre(
                        x, hc_fn, hc_scale, hc_base, 2.0, 1e-6, 1e-6
                    )

                self.assertTrue(torch.equal(actual[0], expected[0]))
                self.assertTrue(torch.equal(actual[1], expected[1]))


if __name__ == "__main__":
    unittest.main()
