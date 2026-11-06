"""Quark W4A4 MXFP4 dense GEMM on gfx950 when K is not a whole number of AITER's BLOCK_SIZE_K:
outputs must match a dequantized reference whatever the previous kernel left in LDS."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

# Untuned at ROCm/aiter@285579b62, so DEFAULT.json picks BLOCK_SIZE_K=1024 for 8 < M <= 32;
# an M that is a multiple of 16 takes the stale-LDS path. No power of two >= 128 divides K=2880,
# so its fix keeps a K tail, in an LDS stage a full K block filled.
N = 7168
SHAPES = {1536: (16, 32), 2880: (16,)}
# Power of two at or above the ~105 KB of LDS this GEMM config uses.
_LDS_BYTES = 128 * 1024


def _make_lds_fill():
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    @gluon.jit
    def _lds_fill_kernel(out_ptr, value, NUM_WORDS: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout([NUM_WORDS // 256], [64], [4], [0])
        smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
        smem = gl.allocate_shared_memory(gl.int32, [NUM_WORDS], smem_layout)
        smem.store(gl.full([NUM_WORDS], value, gl.int32, layout))
        # Read back so the store is not dead.
        gl.store(out_ptr + gl.program_id(0), gl.sum(smem.load(layout), axis=0))

    def lds_fill(byte: int) -> None:
        """Leave ``byte`` in every CU's LDS, as a previous kernel would."""
        num_programs = 4096
        out = torch.empty(num_programs, dtype=torch.int32, device="cuda")
        value = int.from_bytes(bytes([byte] * 4), "little", signed=True)
        _lds_fill_kernel[(num_programs,)](out, value, _LDS_BYTES // 4, num_warps=4)

    return lds_fill


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 MXFP4 GEMM")
class TestQuarkAfp4wfp4KTail(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

        def dequant(q, scale):
            values = mxfp4_to_f32(q)
            scale = e8m0_to_f32(scale).repeat_interleave(32, dim=-1)
            return values * scale[:, : values.shape[-1]]

        torch.manual_seed(0)
        cls.inputs = {}
        for k, ms in SHAPES.items():
            w_q, w_s = dynamic_mxfp4_quant(
                (torch.randn(N, k, device="cuda") * 0.05).to(torch.bfloat16)
            )
            # Checkpoint weight scales are row-major; activation scales keep the quant layout.
            w_s = w_s.contiguous()
            w_ref = dequant(w_q, w_s)
            for m in ms:
                x_q, x_s = dynamic_mxfp4_quant(
                    (torch.randn(m, k, device="cuda") * 0.05).to(torch.bfloat16)
                )
                ref = dequant(x_q, x_s) @ w_ref.t()
                cls.inputs[k, m] = (x_q, x_s, w_q, w_s, ref)
        cls.lds_fill = staticmethod(_make_lds_fill())

    def _assert_matches_reference(self):
        from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
            _afp4wfp4_k_tail_in_unfilled_stage,
            _aiter_afp4wfp4_get_config,
            gemm_afp4wfp4,
        )

        for (k, m), (x_q, x_s, w_q, w_s, ref) in self.inputs.items():
            with self.subTest(k=k, m=m):
                aiter_config, _ = _aiter_afp4wfp4_get_config(m, N, k // 2)
                # Guard the premise: once AITER tunes this shape off the stale-LDS path, pick another.
                self.assertTrue(
                    _afp4wfp4_k_tail_in_unfilled_stage(
                        aiter_config["BLOCK_SIZE_K"], k, aiter_config["num_stages"]
                    )
                )
                y = torch.zeros((m, N), dtype=torch.bfloat16, device="cuda")
                # 0xFF is a NaN E8M0 scale; a K-tail that reads stale LDS turns it into NaN.
                self.lds_fill(0xFF)
                gemm_afp4wfp4(x_q, w_q, x_s, w_s, torch.bfloat16, y)
                err = (y.float() - ref).abs().nan_to_num(float("inf"))
                self.assertLessEqual(err.max().item(), 0.02 * ref.abs().max().item())

    def test_matches_dequantized_reference(self):
        self._assert_matches_reference()


if __name__ == "__main__":
    unittest.main()
