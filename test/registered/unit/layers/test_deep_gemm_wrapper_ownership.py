"""Ownership guard tests for the DeepGEMM scale-layout wrappers.

sgl-deep-gemm's check-only fast paths can return a NON-OWNING alias of the
input across the TVM-FFI boundary (sgl-project/sglang#39684). The wrappers in
``sglang.srt.layers.deep_gemm_wrapper.entrypoint`` must hand back the input
itself in that case so the caller keeps the storage alive.

Requires one GPU and sgl-deep-gemm.
"""

import gc
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

try:
    from sglang.srt.layers import deep_gemm_wrapper

    _HAS_WRAPPER = hasattr(deep_gemm_wrapper, "transform_sf_into_required_layout")
except Exception:
    _HAS_WRAPPER = False


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(_HAS_WRAPPER, "sgl-deep-gemm wrapper unavailable")
class TestDeepGemmWrapperOwnership(CustomTestCase):
    def _assert_output_survives_input_free(self, input_factory, make_out, expected_value):
        # input_factory keeps exactly one reference alive inside this frame,
        # mirroring callers that rebind the result over their only reference.
        sf = input_factory()
        out = make_out(sf)
        self.assertIsInstance(out, torch.Tensor)
        torch.cuda.synchronize()

        del sf
        gc.collect()
        # Force the caching allocator to recycle the freed block; NaN-filled
        # buffers act as an overwrite sentinel.
        sentinels = [
            torch.full((1, 128, 40), float("nan"), device="cuda") for _ in range(20)
        ]
        torch.cuda.synchronize()
        del sentinels

        self.assertTrue(
            torch.isfinite(out).all().item(),
            "wrapper returned a non-owning alias; output was corrupted after "
            "the input was freed and its storage recycled",
        )
        self.assertTrue(torch.allclose(out, torch.full_like(out, expected_value)))

    def test_transform_sf_into_required_layout_preserves_ownership(self):
        # (FP32, 128, 128) recipe on SM90 hits the check-only pass-through
        # path, which returns the input itself (an alias). The wrapper must
        # return the input object rather than the alias.
        self._assert_output_survives_input_free(
            lambda: torch.full((1, 128, 40), 0.0005, device="cuda", dtype=torch.float32),
            lambda sf: deep_gemm_wrapper.transform_sf_into_required_layout(
                sf, 16384, 5120, (1, 128, 128),
                num_groups=1, is_sfa=False, disable_ue8m0_cast=True,
            ),
            0.0005,
        )

    def test_get_mn_major_tma_aligned_tensor_preserves_ownership(self):
        # Feed a tensor already in MN-major TMA-aligned layout so the
        # already-aligned fast path returns the input itself.
        def make_aligned():
            t = torch.empty_strided(
                (1, 128, 40), (128 * 40, 1, 128), device="cuda", dtype=torch.float32
            )
            t.fill_(0.0005)
            return t

        self._assert_output_survives_input_free(
            make_aligned,
            lambda sf: deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(sf),
            0.0005,
        )


if __name__ == "__main__":
    unittest.main()
