import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.kernels.ops.attention.utils import mla_quantize_without_rope_for_fp8
from sglang.kernels.ops.quantization.fp8_utils import to_fp8_satfinite
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# mla_quantize_without_rope_for_fp8 dispatches the CUDA-only concat JIT when
# sgl_kernel is importable, so build inputs on the device that path accepts.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestFP8KVSatfinite(CustomTestCase):
    def test_to_fp8_satfinite(self):
        values = torch.tensor(
            [
                464.0,
                500.0,
                -1000.0,
                torch.finfo(torch.bfloat16).max,
                -torch.finfo(torch.bfloat16).max,
                448.0,
                -448.0,
                1.5,
                0.0,
                -0.375,
                float("nan"),
            ],
            dtype=torch.bfloat16,
        )
        result = to_fp8_satfinite(values, torch.float8_e4m3fn)
        result_bytes = result.view(torch.uint8).tolist()

        assert result_bytes[:7] == [0x7E, 0x7E, 0xFE, 0x7E, 0xFE, 0x7E, 0xFE]
        assert torch.equal(
            result[7:10].view(torch.uint8),
            values[7:10].to(torch.float8_e4m3fn).view(torch.uint8),
        )
        assert not any((byte & 0x7F) == 0x7F for byte in result_bytes[:10])
        assert result_bytes[-1] in (0x7F, 0xFF)
        assert torch.isnan(result[-1].to(torch.float32))

    def test_mla_quantize_without_rope_satfinite(self):
        q_nope = torch.randn(1, 8, 512, dtype=torch.bfloat16, device=DEVICE)
        q_rope = torch.randn(1, 8, 64, dtype=torch.bfloat16, device=DEVICE)
        k_nope = torch.randn(1, 512, dtype=torch.bfloat16, device=DEVICE)
        k_rope = torch.randn(1, 64, dtype=torch.bfloat16, device=DEVICE)
        q_nope[0, 0, 0] = -1000.0
        q_rope[0, 0, -1] = 500.0
        k_nope[0, 0] = 500.0
        k_rope[0, -1] = -1000.0

        q, k_nope_fp8, k_rope_fp8 = mla_quantize_without_rope_for_fp8(
            q_nope, q_rope, k_nope, k_rope
        )
        q_ref = torch.cat([q_nope, q_rope], dim=-1).to(torch.float8_e4m3fn)
        k_nope_ref = k_nope.to(torch.float8_e4m3fn)
        k_rope_ref = k_rope.to(torch.float8_e4m3fn)

        assert int(q[0, 0, 0].view(torch.uint8)) == 0xFE
        assert int(q[0, 0, -1].view(torch.uint8)) == 0x7E
        assert int(k_nope_fp8[0, 0].view(torch.uint8)) == 0x7E
        assert int(k_rope_fp8[0, -1].view(torch.uint8)) == 0xFE

        q_mask = torch.ones_like(q, dtype=torch.bool)
        q_mask[0, 0, 0] = False
        q_mask[0, 0, -1] = False
        k_nope_mask = torch.ones_like(k_nope_fp8, dtype=torch.bool)
        k_nope_mask[0, 0] = False
        k_rope_mask = torch.ones_like(k_rope_fp8, dtype=torch.bool)
        k_rope_mask[0, -1] = False
        for actual, reference, mask in (
            (q, q_ref, q_mask),
            (k_nope_fp8, k_nope_ref, k_nope_mask),
            (k_rope_fp8, k_rope_ref, k_rope_mask),
        ):
            assert torch.equal(
                actual[mask].view(torch.uint8), reference[mask].view(torch.uint8)
            )
            assert not ((actual.view(torch.uint8) & 0x7F) == 0x7F).any()


if __name__ == "__main__":
    import unittest

    unittest.main()
