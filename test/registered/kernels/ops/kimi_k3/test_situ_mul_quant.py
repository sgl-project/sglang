"""SiTU (SoftCap-GLU) + FP8 per-group quant fused kernel (masked varlen path).

The fused kernel replaces the unfused chain of
  slice -> tanh/sigmoid softcap activation -> mul -> per-group fp8 quant
for MoE experts using the SiTU activation:

  gate_out = beta * tanh(gate / beta) * sigmoid(gate)
  up_out   = linear_beta * tanh(up / linear_beta)
  output   = gate_out * up_out

Correctness is pinned against an independent fp32 PyTorch reference: the
dequantized kernel output must match the reference activation within the
fp8-e4m3 per-group quantization step. This guards the activation formula,
gate/up half addressing, the varlen (masked_m) work distribution, and the
UE8M0 packed-transposed scale addressing.
"""

import unittest

import torch

from sglang.kernels.ops.kimi_k3 import situ_and_mul_masked_post_quant
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_GROUP_SIZE = 128
_FP8_MAX = 448.0
_BETA = 4.0
_LINEAR_BETA = 25.0


def _reference_situ(gate_up: torch.Tensor, beta: float, linear_beta: float):
    d = gate_up.shape[-1] // 2
    gate = gate_up[..., :d].float()
    up = gate_up[..., d:].float()
    gate_out = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up_out = linear_beta * torch.tanh(up / linear_beta)
    return gate_out * up_out


def _unpack_ue8m0_scales(packed: torch.Tensor, num_groups: int):
    # Physical layout [E, G//4, N] int32; each int32 packs the ue8m0 exponent
    # bytes of 4 consecutive groups (byte b <-> group g with g % 4 == b) for
    # one token. Returns fp32 scales with shape [E, N, G].
    E, G4, N = packed.shape
    assert G4 * 4 == num_groups
    exps = packed.contiguous().view(torch.uint8).view(E, G4, N, 4)
    exps = exps.permute(0, 2, 1, 3).reshape(E, N, num_groups)
    return torch.exp2(exps.to(torch.float32) - 127.0)


class TestSituMulQuantVarlen(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 90:
            raise unittest.SkipTest("Test requires CUDA SM 90 or higher")

    def _run_kernel(self, gate_up, masked_m, topk, scale_ue8m0, transposed):
        E, N, D2 = gate_up.shape
        D = D2 // 2
        G = D // _GROUP_SIZE
        device = gate_up.device

        output = torch.full((E, N, D), 0x7F, device=device, dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        if transposed:
            output_scale = torch.zeros((E, G // 4, N), device=device, dtype=torch.int32)
        else:
            output_scale = torch.zeros((E, N, G), device=device, dtype=torch.float32)

        situ_and_mul_masked_post_quant(
            input=gate_up,
            output=output,
            output_scale=output_scale,
            quant_group_size=_GROUP_SIZE,
            masked_m=masked_m,
            beta=_BETA,
            linear_beta=_LINEAR_BETA,
            scale_ue8m0=scale_ue8m0,
            topk=topk,
            transposed=transposed,
        )
        torch.cuda.synchronize()

        if transposed:
            scales = _unpack_ue8m0_scales(output_scale, G)
        else:
            scales = output_scale
        return output, scales

    def _assert_matches_reference(self, gate_up, masked_m, output, scales):
        E, N, D2 = gate_up.shape
        D = D2 // 2
        ref = _reference_situ(gate_up, _BETA, _LINEAR_BETA)
        scales_exp = scales.repeat_interleave(_GROUP_SIZE, dim=-1)
        dequant = output.float() * scales_exp

        # fp8-e4m3 has a 3-bit mantissa: after scaling into [-448, 448] the
        # worst-case rounding error is half the top-bin spacing (32/2 = 16)
        # in scaled units. 17x leaves margin for fast-math tanh/exp drift.
        bound = scales_exp * 17.0
        for e in range(E):
            m = int(masked_m[e].item())
            if m == 0:
                continue
            err = (dequant[e, :m] - ref[e, :m]).abs()
            self.assertTrue(
                bool((err <= bound[e, :m]).all()),
                f"expert {e}: max quant error {err.max().item():.4f} exceeds bound",
            )

    def _assert_padding_untouched(self, output, masked_m):
        # Rows at or beyond masked_m[e] must keep the 0x7F sentinel: the
        # varlen work distribution must never write outside valid tokens.
        E = output.shape[0]
        raw = output.view(torch.uint8)
        for e in range(E):
            m = int(masked_m[e].item())
            self.assertTrue(
                bool((raw[e, m:] == 0x7F).all()),
                f"expert {e}: kernel wrote past masked_m",
            )

    def test_matches_reference(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        device = torch.device("cuda")

        # (E, N, D, topk, mm_max): small shape + the target-model decode
        # shape (112 local experts under EP8, D = 2 x moe_intermediate_size).
        # mm_max keeps sum(masked_m) <= N * topk, the kernel's grid extent.
        cases = [
            (8, 64, 1024, 16, 64),
            (112, 64, 6144, 16, 6),
        ]
        for E, N, D, topk, mm_max in cases:
            gate_up = (
                torch.randn(E, N, D * 2, device=device, dtype=torch.float32) * 2.0
            ).to(torch.bfloat16)
            # Include empty (0) and full (N) experts to exercise the
            # prefix-scan early-exit and the max-extent boundary.
            masked_m = torch.randint(
                0, mm_max + 1, (E,), device=device, dtype=torch.int32
            )
            masked_m[0] = 0
            masked_m[-1] = N
            self.assertLessEqual(int(masked_m.sum().item()), N * topk)

            for scale_ue8m0, transposed in [(False, False), (True, True)]:
                with self.subTest(
                    E=E, N=N, D=D, scale_ue8m0=scale_ue8m0, transposed=transposed
                ):
                    output, scales = self._run_kernel(
                        gate_up, masked_m, topk, scale_ue8m0, transposed
                    )
                    self._assert_matches_reference(gate_up, masked_m, output, scales)
                    self._assert_padding_untouched(output, masked_m)

    def test_softcap_saturation(self):
        # Extreme inputs must saturate to the softcap bound instead of
        # overflowing fp8: |output| <= beta * linear_beta = 100 < 448.
        device = torch.device("cuda")
        E, N, D, topk = 8, 32, 1024, 16

        gate_up = torch.zeros(E, N, D * 2, device=device, dtype=torch.bfloat16)
        gate_up[..., :D] = 1000.0  # gate -> +beta
        gate_up[..., D : D + D // 2] = 1000.0  # up -> +linear_beta
        gate_up[..., D + D // 2 :] = -1000.0  # up -> -linear_beta
        masked_m = torch.full((E,), N, device=device, dtype=torch.int32)

        output, scales = self._run_kernel(
            gate_up, masked_m, topk, scale_ue8m0=True, transposed=True
        )
        dequant = output.float() * scales.repeat_interleave(_GROUP_SIZE, dim=-1)

        expected = torch.full_like(dequant, _BETA * _LINEAR_BETA)
        expected[..., D // 2 :] = -_BETA * _LINEAR_BETA
        torch.testing.assert_close(dequant, expected, atol=4.0, rtol=0.05)


if __name__ == "__main__":
    unittest.main()
