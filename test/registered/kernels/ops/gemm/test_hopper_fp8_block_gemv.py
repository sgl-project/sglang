"""
Tests the Hopper block-FP8 GEMV on the K=6144 decode shapes of MiniMax M3.1 against
an fp64 reference, and the shape gate that sends a linear to it.
"""

import unittest

import torch

from sglang.kernels.ops.gemm.hopper_fp8_block_gemv import (
    hopper_fp8_block_gemv,
    use_hopper_fp8_block_gemv,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8_utils import (
    deepgemm_w8a8_block_fp8_linear_with_fallback,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

BLOCK = 128
# One rounding of the bf16 output is 2^-8 relative at worst.
MAX_REL_ERR = 5e-3


def _is_sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def _block_fp8_weight(n: int, k: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, device="cuda", generator=gen) * 0.05
    blocks = w.unfold(1, BLOCK, BLOCK).abs().amax(dim=-1)
    blocks = blocks.split(BLOCK, dim=0)
    scale = torch.stack([b.amax(dim=0) for b in blocks]) / 448.0
    expanded = scale.repeat_interleave(BLOCK, dim=0)[:n].repeat_interleave(BLOCK, dim=1)
    return (w / expanded).to(torch.float8_e4m3fn), scale


def _fp64_reference(xq, xs, wq, ws):
    n = wq.shape[0]
    x = xq.double() * xs.double().repeat_interleave(BLOCK, dim=1)
    w = wq.double() * ws.double().repeat_interleave(BLOCK, dim=0)[:n].repeat_interleave(
        BLOCK, dim=1
    )
    return x @ w.T


def _max_rel_err(out, ref):
    return ((out.double() - ref).norm(dim=1) / ref.norm(dim=1)).max().item()


class TestHopperFp8BlockGemvGate(CustomTestCase):
    def test_shape_gate(self):
        cases = [
            # M3.1 TP8: fused qkv+index, shared gate+up. Attention TP4 and TP2 fused.
            ((1, 1536, 6144), True),
            ((8, 768, 6144), True),
            ((4, 2560, 6144), True),
            ((8, 4928, 6144), True),
            # K above 4096 needs M <= 8, N <= 8192 and an 8-warp split (K % 1024 == 0).
            ((8, 8192, 6144), True),
            ((1, 1536, 5120), True),
            ((8, 4096, 8192), True),
            ((9, 1536, 6144), False),
            ((16, 768, 6144), False),
            ((1, 9856, 6144), False),
            ((1, 1536, 4608), False),
            ((8, 1536, 5632), False),
            ((1, 1536, 8320), False),
            # Odd K block counts: M3.1 TP8 shared down projection is the 384 one.
            ((1, 6144, 384), False),
            ((8, 1024, 384), False),
            ((1, 768, 1152), False),
            ((1, 768, 128), False),
            # Even block counts up to K=4096 keep their rule, Ling-3.1 TP8 included.
            ((8, 6144, 1024), True),
            ((8, 4096, 1280), True),
            ((1, 768, 256), True),
            ((16, 1536, 4096), True),
            ((16, 1600, 4096), False),
            # Not on the 128-wide scale grid.
            ((1, 1536, 6100), False),
        ]
        for (m, n, k), expected in cases:
            self.assertEqual(
                use_hopper_fp8_block_gemv(m, n, k, (BLOCK, BLOCK)), expected, (m, n, k)
            )


@unittest.skipIf(not _is_sm90(), "Hopper block-FP8 GEMV requires an SM90 GPU")
class TestHopperFp8BlockGemv(CustomTestCase):
    def test_wide_k_matches_reference(self):
        # The slab grows with K past the default dynamic shared memory limit. N=1544 and
        # 8184 leave a row tail in the last block; K=8192 is the largest admitted.
        for m, n, k in [
            (1, 1536, 6144),
            (2, 1544, 6144),
            (4, 2560, 6144),
            (8, 768, 6144),
            (8, 8184, 6144),
            (8, 1536, 5120),
            (8, 1536, 7168),
            (8, 1536, 8192),
        ]:
            gen = torch.Generator(device="cuda").manual_seed(m + n)
            x = torch.randn(m, k, device="cuda", generator=gen).to(torch.bfloat16)
            xq, xs = sglang_per_token_group_quant_fp8(x, BLOCK)
            wq, ws = _block_fp8_weight(n, k, seed=n + k)
            out = hopper_fp8_block_gemv(xq, xs, wq, ws)
            ref = _fp64_reference(xq, xs, wq, ws)
            self.assertLess(_max_rel_err(out, ref), MAX_REL_ERR, (m, n, k))

    def test_linear_dispatch_at_wide_k(self):
        # Through the linear entry point the flag has to reach the GEMV: same bf16
        # rounding bound on both routes, and the routes differ only in that rounding.
        m, n, k = 2, 1536, 6144
        x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        wq, ws = _block_fp8_weight(n, k, seed=3)
        xq, xs = sglang_per_token_group_quant_fp8(x, BLOCK)
        ref = _fp64_reference(xq, xs, wq, ws)
        block = [BLOCK, BLOCK]
        with envs.SGLANG_ENABLE_FP8_BLOCK_GEMV.override(False):
            off = deepgemm_w8a8_block_fp8_linear_with_fallback(x, wq, block, ws)
        with envs.SGLANG_ENABLE_FP8_BLOCK_GEMV.override(True):
            on = deepgemm_w8a8_block_fp8_linear_with_fallback(x, wq, block, ws)
        direct = hopper_fp8_block_gemv(xq, xs, wq, ws)
        self.assertTrue(torch.equal(on, direct))
        self.assertLess(_max_rel_err(on, ref), MAX_REL_ERR)
        self.assertLess(_max_rel_err(off, ref), MAX_REL_ERR)


if __name__ == "__main__":
    unittest.main()
