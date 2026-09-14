"""Per-token FP8 linear consumes the fused add-RMSNorm's pre-quantized activation.

On ROCm gfx95 the Gemma fused add-RMSNorm kernel can emit ``out._fp8_qinput``
(fp8 tensor, per-token scale). ``apply_fp8_linear`` with a per-channel fp8
weight and dynamic per-token activations must use that pair instead of
re-quantizing, and produce the same result as the separate quant path.
"""

import unittest

import torch

from sglang.srt.environ import envs

# the aiter per-token FP8 GEMM path is selected at import time
if not envs.SGLANG_USE_AITER.is_set():
    envs.SGLANG_USE_AITER.set(True)

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")

try:
    from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import (
        gemma_fused_add_rmsnorm,
    )
    from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

    _HAS_DEPS = True
except ImportError:  # pragma: no cover - CI without the ROCm kernels
    _HAS_DEPS = False


def _fp8_per_channel_weight(n: int, k: int, device: torch.device):
    from aiter.ops.shuffle import shuffle_weight

    w = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.02
    scale = (w.float().abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    wq = (w.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    # Layer storage: shuffled (N, K) stored transposed; apply_fp8_linear passes weight.T
    return shuffle_weight(wq, (16, 16)).t(), scale.float(), w


@unittest.skipUnless(
    is_hip() and torch.cuda.is_available() and _HAS_DEPS,
    "ROCm gfx95 fused norm + aiter fp8 GEMM only",
)
class TestFusedNormQuantConsume(CustomTestCase):
    def _run(self, m: int, n: int = 2304, k: int = 6144):
        device = torch.device("cuda")
        torch.manual_seed(0)
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        residual = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        norm_w = torch.randn(k, device=device, dtype=torch.bfloat16) * 0.1
        weight, w_scale, w_bf16 = _fp8_per_channel_weight(n, k, device)

        with envs.SGLANG_FUSED_NORM_FP8_QUANT_MAX_M.override(16384):
            normed, _ = gemma_fused_add_rmsnorm(
                x.clone(), residual.clone(), norm_w, 1e-6, emit_fp8=True
            )
        self.assertTrue(hasattr(normed, "_fp8_qinput"))
        fused = apply_fp8_linear(
            input=normed,
            weight=weight,
            weight_scale=w_scale,
            input_scale=None,
            cutlass_fp8_supported=False,
            use_per_token_if_dynamic=True,
        )
        plain_in = normed.clone()  # drops the _fp8_qinput attribute
        self.assertFalse(hasattr(plain_in, "_fp8_qinput"))
        separate = apply_fp8_linear(
            input=plain_in,
            weight=weight,
            weight_scale=w_scale,
            input_scale=None,
            cutlass_fp8_supported=False,
            use_per_token_if_dynamic=True,
        )
        ref = torch.nn.functional.linear(normed, w_bf16)
        # Both fp8 paths quantize per token with the same absmax scale; they
        # should agree closely with each other and be within fp8 error of bf16.
        rel_fused_vs_sep = (fused - separate).abs().max() / separate.abs().max()
        rel_fused_vs_ref = (fused.float() - ref.float()).abs().mean() / ref.abs().mean()
        self.assertLess(rel_fused_vs_sep.item(), 2e-2)
        self.assertLess(rel_fused_vs_ref.item(), 8e-2)

    def test_decode_rows(self):
        self._run(m=24)

    def test_prefill_rows(self):
        self._run(m=1024, n=256)


if __name__ == "__main__":
    unittest.main()
