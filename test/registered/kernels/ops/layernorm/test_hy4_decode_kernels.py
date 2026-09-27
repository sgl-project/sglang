import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton
import triton.language as tl
from torch import nn

import sglang.kernels.ops.layernorm.hy4_ihc as hy4_ihc
from sglang.srt.models import hunyuan_v4
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

    def test_fused_ihc_failure_disables_each_path(self):
        class TupleLinear(nn.Module):
            def __init__(self, input_size, output_size):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(output_size, input_size))

            def forward(self, inputs):
                return nn.functional.linear(inputs, self.weight), None

        config = SimpleNamespace(
            hidden_size=8,
            hc_mult=2,
            hc_magnitude=2.0,
            hc_eps=1e-6,
            rms_norm_eps=1e-5,
        )
        counts = {"pre": 0, "post": 0, "post_pre": 0, "head": 0}

        def fail(name):
            def raise_error(*args, **kwargs):
                counts[name] += 1
                raise RuntimeError(name)

            return raise_error

        def make_linear(input_size, output_size, **kwargs):
            return TupleLinear(input_size, output_size)

        with patch.object(hunyuan_v4, "ReplicatedLinear", make_linear):
            pre_layer = hunyuan_v4.HYV4HCPreLayer(config, "pre").cuda()
            layer = hunyuan_v4.HYV4HCLayer(config, "layer").cuda()
            next_layer = hunyuan_v4.HYV4HCLayer(config, "next").cuda()
            head_layer = hunyuan_v4.HYV4HCHeadLayer(config, "head").cuda()

        next_layer.hc_pre._fused_ihc_pre_disabled = True
        norm = hunyuan_v4.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, force_native=True
        ).cuda()
        hidden_states = torch.randn(3, 2, 8, device="cuda")
        output = torch.randn(3, 8, device="cuda")
        residual = torch.randn(3, 2, 8, device="cuda")
        post = torch.randn(3, 2, device="cuda")

        with (
            patch.object(hy4_ihc, "fused_hy4_ihc_pre", fail("pre")),
            patch.object(hy4_ihc, "fused_hy4_ihc_post", fail("post")),
            patch.object(hy4_ihc, "fused_hy4_ihc_post_pre", fail("post_pre")),
            patch.object(hy4_ihc, "fused_hy4_ihc_head", fail("head")),
            patch.object(hunyuan_v4, "_hpc_ihc_available", return_value=True),
            self.assertLogs(hunyuan_v4.logger, level="WARNING") as logs,
        ):
            for _ in range(2):
                pre_layer(hidden_states)
                layer.post(output, residual, post)
                layer.post_pre(output, residual, post, next_layer, norm)
                head_layer(hidden_states)

        self.assertEqual(counts, {"pre": 1, "post": 1, "post_pre": 1, "head": 1})
        self.assertEqual(len(logs.records), 4)
        self.assertTrue(all(record.exc_info is not None for record in logs.records))


if __name__ == "__main__":
    unittest.main()
