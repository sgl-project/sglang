import unittest

import torch

from sglang.kernels.ops.moe.ep_moe_kernels import (
    silu_and_mul_masked_post_quant_fwd,
    silu_and_mul_masked_post_quant_packed_fwd,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
GROUP = 128
E, M_MAX, SIZE_N = 4, 256, 512  # SIZE_N = half-width (gate/up each); 4 groups
MASKED_M = [7, 100, 200, 256]
LIMIT = 7.0


def _build_inputs():
    """[E, M, 2*SIZE_N] bf16 with saturating values in both halves."""
    torch.manual_seed(31)
    gateup = torch.randn(E, M_MAX, 2 * SIZE_N) * 1.5
    for g, m in enumerate(MASKED_M):
        for r in range(0, m, 3):  # every 3rd valid row saturates
            gateup[g, r, torch.arange(0, SIZE_N, 17)] = 9.0 + (r % 5)  # gate > limit
            gateup[g, r, SIZE_N + torch.arange(0, SIZE_N, 23)] = -(
                8.0 + (r % 7)
            )  # up < -limit
    return gateup.to(torch.bfloat16)


def _reference_dequant(gateup, limit, ue8m0):
    """fp32 reference of silu-mul + per-token-group quant-dequant.

    Mirrors the kernel's op order: silu in fp32, optional clamp, bf16 cast
    boundary, bf16 product, group absmax scale.
    """
    gate = gateup[..., :SIZE_N].float()
    up = gateup[..., SIZE_N:].float()
    gate = gate / (1 + torch.exp(-gate))  # silu
    if limit > 0:
        # _swiglu_silu_clamp_mul semantics: clamp after silu on gate (max
        # only), symmetric clamp on up.
        gate = torch.minimum(gate, torch.tensor(limit))
        up = up.clamp(-limit, limit)
    gate_up = (gate.bfloat16() * up.bfloat16()).float()

    grouped = gate_up.reshape(*gate_up.shape[:-1], SIZE_N // GROUP, GROUP)
    absmax = grouped.abs().amax(dim=-1).clamp_min(1e-10)
    s = absmax / FP8_MAX
    if ue8m0:
        s = torch.exp2(torch.ceil(torch.log2(s)))
    q = (grouped / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    deq = q * s.unsqueeze(-1)
    return deq.reshape(*gate_up.shape)


def _kernel_dequant(output_q, output_s, ue8m0_exponent_packed=False):
    if ue8m0_exponent_packed:
        # output_s: [E, SIZE_N//(4*GROUP), M] int32, 4 exponent bytes LE per int32
        bytes_ = output_s.view(torch.uint8)  # [E, G4, M*4]
        expo = bytes_.reshape(E, -1, M_MAX, 4).permute(0, 2, 1, 3)
        expo = expo.reshape(E, M_MAX, -1).float()
        s = torch.exp2(expo - 127.0)  # [E, M, groups]
    else:
        s = output_s  # [E, M, groups]
    grouped = output_q.float().reshape(*output_q.shape[:-1], SIZE_N // GROUP, GROUP)
    return (grouped * s.unsqueeze(-1)).reshape(*output_q.shape)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestSiluMulPostQuantClamp(CustomTestCase):
    def _assert_dequant_close(self, got, want, msg):
        # fp8 e4m3 has 3 mantissa bits: relative quantization error <= 2^-4,
        # plus the kernel's bf16 product rounding. Compare per element.
        diff = (got - want).abs()
        tol = 0.25 + 0.08 * want.abs()
        self.assertTrue(
            bool((diff <= tol).all()),
            f"{msg}: max diff {diff.max().item():.4f} exceeds fp8 rounding bound",
        )

    def _run_main_kernel(self, gateup, limit):
        output_q = torch.empty(
            E, M_MAX, SIZE_N, dtype=torch.float8_e4m3fn, device="cuda"
        )
        output_s = torch.empty(
            E, M_MAX, SIZE_N // GROUP, dtype=torch.float32, device="cuda"
        )
        masked_m = torch.tensor(MASKED_M, dtype=torch.int32, device="cuda")
        silu_and_mul_masked_post_quant_fwd(
            gateup.cuda(),
            output_q,
            output_s,
            GROUP,
            masked_m,
            scale_ue8m0=False,
            gemm1_alpha=0.0,  # alpha-less branch: the clamp must still apply
            gemm1_clamp_limit=limit,
        )
        return output_q.cpu(), output_s.cpu()

    def _valid_rows(self):
        for e in range(E):
            yield e, slice(0, MASKED_M[e])

    def test_clamp_limit_applied_without_alpha(self):
        """Regression: gemm1_clamp_limit without gemm1_alpha was silently dropped."""
        gateup = _build_inputs()
        q, s = self._run_main_kernel(gateup, LIMIT)
        got = _kernel_dequant(q, s)
        want = _reference_dequant(gateup, LIMIT, ue8m0=False)
        for e, rows in self._valid_rows():
            self._assert_dequant_close(
                got[e, rows],
                want[e, rows],
                f"expert {e}: clamped kernel output diverges from reference",
            )
            # Structural: clamped output must never exceed limit^2 (=49).
            self.assertLessEqual(
                got[e, rows].abs().max().item(),
                LIMIT * LIMIT * 1.05,
                f"expert {e}: clamp not engaged",
            )

    def test_zero_limit_path_unchanged(self):
        """limit=0 layers must compute the plain silu-mul (no clamp)."""
        gateup = _build_inputs()
        q, s = self._run_main_kernel(gateup, 0.0)
        got = _kernel_dequant(q, s)
        want = _reference_dequant(gateup, 0.0, ue8m0=False)
        for e, rows in self._valid_rows():
            self._assert_dequant_close(
                got[e, rows],
                want[e, rows],
                f"expert {e}: limit=0 path altered",
            )
            # Divergence detector: with saturation present, unclamped output
            # must exceed limit^2 somewhere.
            self.assertGreater(
                want[e, rows].abs().max().item(),
                LIMIT * LIMIT,
                "test inputs fail to exercise the clamp",
            )

    def test_packed_kernel_clamp_limit(self):
        """Same regression on the packed UE8M0 variant."""
        gateup = _build_inputs().cuda()
        output_q = torch.empty(
            E, M_MAX, SIZE_N, dtype=torch.float8_e4m3fn, device="cuda"
        )
        scale_packed = torch.empty(
            E, SIZE_N // (4 * GROUP), M_MAX, dtype=torch.int32, device="cuda"
        )
        masked_m = torch.tensor(MASKED_M, dtype=torch.int32, device="cuda")
        silu_and_mul_masked_post_quant_packed_fwd(
            gateup,
            output_q,
            scale_packed,
            GROUP,
            masked_m,
            # grid covers sum(masked_m) work items (one per valid token here)
            num_real_tokens=sum(MASKED_M),
            topk=1,
            gemm1_alpha=0.0,
            gemm1_clamp_limit=LIMIT,
        )
        got = _kernel_dequant(
            output_q.cpu(), scale_packed.cpu(), ue8m0_exponent_packed=True
        )
        want = _reference_dequant(gateup.cpu(), LIMIT, ue8m0=True)
        for e, rows in self._valid_rows():
            self._assert_dequant_close(
                got[e, rows],
                want[e, rows],
                f"expert {e}: packed clamped output diverges from reference",
            )
            self.assertLessEqual(
                got[e, rows].abs().max().item(),
                LIMIT * LIMIT * 1.05,
                f"expert {e}: packed kernel clamp not engaged",
            )


if __name__ == "__main__":
    unittest.main()
