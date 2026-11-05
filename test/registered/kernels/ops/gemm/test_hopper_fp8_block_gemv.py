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

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

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
            # K=6144: M <= 8 up to N=8192. M3.1 TP8 fused qkv+index and shared gate+up.
            ((1, 1536, 6144), True),
            ((8, 768, 6144), True),
            ((8, 8192, 6144), True),
            ((9, 1536, 6144), False),
            ((1, 9856, 6144), False),
            # Other K above 4096.
            ((1, 1536, 5120), False),
            ((8, 1536, 8192), False),
            # Odd K block counts: M3.1 TP8 shared down is 6144x384. One block keeps its rule.
            ((1, 6144, 384), False),
            ((8, 1024, 1152), False),
            ((8, 768, 128), True),
            # Even block counts up to K=4096 keep their rule, Ling-3.1 TP8 dense down too.
            ((8, 4096, 1280), True),
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
        # The slab passes the default dynamic shared memory limit at K=6144, M=8.
        # N=1544 and 8184 leave a row tail in the last block.
        for m, n, k in [
            (1, 1536, 6144),
            (2, 1544, 6144),
            (3, 2560, 6144),
            (5, 768, 6144),
            (8, 8184, 6144),
        ]:
            gen = torch.Generator(device="cuda").manual_seed(m + n)
            x = torch.randn(m, k, device="cuda", generator=gen).to(torch.bfloat16)
            xq, xs = sglang_per_token_group_quant_fp8(x, BLOCK)
            wq, ws = _block_fp8_weight(n, k, seed=n + k)
            out = hopper_fp8_block_gemv(xq, xs, wq, ws)
            ref = _fp64_reference(xq, xs, wq, ws)
            self.assertLess(_max_rel_err(out, ref), MAX_REL_ERR, (m, n, k))

    def test_batch_invariant_up_to_m8(self):
        # M <= 8 all run one kernel config, so a row's output can't depend on M.
        n, k = 1536, 6144
        wq, ws = _block_fp8_weight(n, k, seed=5)
        x = torch.randn(8, k, device="cuda").to(torch.bfloat16)
        xq, xs = sglang_per_token_group_quant_fp8(x, BLOCK)
        full = hopper_fp8_block_gemv(xq, xs, wq, ws)
        for m in (1, 3, 7):
            out = hopper_fp8_block_gemv(xq[:m], xs[:m], wq, ws)
            self.assertTrue(torch.equal(out, full[:m]), m)

    def test_linear_dispatch_at_wide_k(self):
        # With the flag on, the linear entry point has to reach the GEMV at K=6144.
        m, n, k = 2, 1536, 6144
        x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        wq, ws = _block_fp8_weight(n, k, seed=3)
        xq, xs = sglang_per_token_group_quant_fp8(x, BLOCK)
        with envs.SGLANG_ENABLE_FP8_BLOCK_GEMV.override(True):
            on = deepgemm_w8a8_block_fp8_linear_with_fallback(x, wq, [BLOCK, BLOCK], ws)
        self.assertTrue(torch.equal(on, hopper_fp8_block_gemv(xq, xs, wq, ws)))
        self.assertLess(_max_rel_err(on, _fp64_reference(xq, xs, wq, ws)), MAX_REL_ERR)


if __name__ == "__main__":
    unittest.main()
