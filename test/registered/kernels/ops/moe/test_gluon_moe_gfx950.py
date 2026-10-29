"""Numerical coverage of every gfx950 GLM MXFP4 Gluon MoE shape family."""

import importlib
import unittest
from functools import lru_cache

import torch
from parameterized import parameterized

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kernels.gluon_moe_reference import GluonMoeFixture
from sglang.test.test_utils import CustomTestCase, empty_gpu_cache

register_amd_ci(est_time=180, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")


# Each row names the expected module explicitly: a dispatcher regression must
# not silently change which implementation gets numerical coverage.
FAMILY_CASES = (
    ("tp8_m1_16", 8, 1, 0, False, (1, 2, 3, 4, 6, 8, 9, 12, 16)),
    ("tp8_m32_128", 8, 1, 0, False, (17, 32, 33, 255)),
    ("tp8_m256", 8, 1, 0, False, (256, 1023)),
    ("tp8_m1024_4192", 8, 1, 0, False, (1024, 2046, 3366, 4192)),
    ("tp8_m4193_16768", 8, 1, 0, False, (4193, 16769, 32768)),
    ("tp8_m1_16_mtp", 8, 1, 0, True, (9, 15)),
    ("tp8_m32_256_mtp", 8, 1, 0, True, (17, 18, 24, 96, 1023)),
    ("tp8_m1024_4192_mtp", 8, 1, 0, True, (1024, 2046, 3366, 4192)),
    ("tp4_m1_16", 4, 1, 0, False, (1, 2, 4, 6, 8, 16)),
    ("tp4_m32_64", 4, 1, 0, False, (17, 32, 33, 64)),
    ("tp4_m128_4192", 4, 1, 0, False, (65, 2046, 3366, 4192)),
    ("tp4_m16769_32768_shared", 4, 1, 0, False, (4193, 16769, 32768)),
    ("tp4_m4193_16768_shared", 8, 4, 2, False, (4193, 32768)),
    # Rank-local EP output, including nonzero expert offsets and native shared.
    ("tp4_m1_16", 8, 2, 1, False, (1,)),
    ("tp4_m32_64", 8, 2, 1, False, (33,)),
    ("tp4_m128_4192", 8, 2, 1, False, (65,)),
    ("tp4_m4193_16768_shared", 8, 2, 1, False, (4193,)),
    ("tp4_m1_16", 8, 4, 2, False, (1,)),
    ("tp4_m32_64", 8, 4, 2, False, (33,)),
    ("tp4_m128_4192", 8, 4, 2, False, (65,)),
    ("tp4_m1_16", 8, 8, 7, False, (1,)),
    ("tp4_m32_64", 8, 8, 7, False, (33,)),
    ("tp4_m128_4192", 8, 8, 7, False, (65,)),
    ("tp4_m4193_16768_shared", 8, 8, 7, False, (4193,)),
    ("tp4_m128_4192", 4, 4, 3, False, (65,)),
    # TP4 NextN uses the same kernels with a zero appended shared-expert slot.
    ("tp4_m1_16", 4, 1, 0, True, (4,)),
    ("tp4_m32_64", 4, 1, 0, True, (17,)),
    ("tp4_m128_4192", 4, 1, 0, True, (65,)),
    ("tp4_m16769_32768_shared", 4, 1, 0, True, (4193,)),
)


@lru_cache(maxsize=2)
def _fixture(tp, ep, rank, nextn):
    fixture = GluonMoeFixture(
        local_experts=256 // ep,
        intermediate=2048 // (tp // ep),
        expert_start=rank * (256 // ep),
        shared=ep == 1 and not nextn,
    )
    return fixture, fixture.reference(routed_scaling_factor=1.75)


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
    "requires one gfx950 GPU",
)
class TestGluonMoeGfx950(CustomTestCase):
    @classmethod
    def tearDownClass(cls):
        _fixture.cache_clear()
        empty_gpu_cache()
        super().tearDownClass()

    @parameterized.expand(
        [
            (
                f"{family}_tp{tp}_ep{ep}_rank{rank}_nextn{nextn}_m{m}",
                family,
                tp,
                ep,
                rank,
                nextn,
                m,
            )
            for family, tp, ep, rank, nextn, sizes in FAMILY_CASES
            for m in sizes
        ]
    )
    def test_family_matches_reference(self, name, family, tp, ep, rank, nextn, m):
        from sglang.srt.layers.moe.glm_mxfp4_gluon import _KERNEL_PACKAGE, _kernel_name

        module_name = f"fused_moe_{family}"
        self.assertEqual(_kernel_name(tp, ep, nextn, m), module_name)
        fixture, expected_tokens = _fixture(tp, ep, rank, nextn)
        x = fixture.inputs(m)
        period = expected_tokens.shape[0]
        expected = expected_tokens.repeat(((m + period - 1) // period, 1))[:m]
        kwargs = {"routed_scaling_factor": 1.75}
        if tp == 4 or ep > 1:
            kwargs.update(
                expert_start=fixture.expert_start,
                fuse_shared_expert=fixture.shared,
            )
        kernel = importlib.import_module(f"{_KERNEL_PACKAGE}.{module_name}").fused_moe
        output = kernel(x, fixture.router, fixture.bias, *fixture.packed, **kwargs)
        torch.cuda.synchronize()
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output).all().item(), name)
        if ep == 1 or m > 1:
            self.assertGreater(expected.float().square().mean().item(), 0, name)
        # Different families may round routed down-projection partials to BF16.
        # Check every element, including cancellation and zero-input rows.
        rms = expected.float().square().mean().sqrt().item()
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02 * rms)
        relative_rmse = (
            output.float() - expected.float()
        ).square().mean().sqrt().item() / max(rms, 1e-10)
        self.assertLess(relative_rmse, 0.005, name)
        print(f"NUMERICS {name}: relative_rmse={relative_rmse:.6g}")

    @parameterized.expand(
        [
            ("tp4_m1_16", 4),
            ("tp4_m32_64", 33),
            ("tp4_m128_4192", 65),
            ("tp4_m4193_16768_shared", 4193),
        ]
    )
    def test_ep_nonlocal_routes_produce_zero(self, family, m):
        """A rank with no selected local experts must not read unwritten partials."""
        fixture, _ = _fixture(8, 4, 2, False)
        router = torch.zeros_like(fixture.router)
        bias = torch.zeros_like(fixture.bias)
        bias[:8] = 1
        kernel = importlib.import_module(
            f"sglang.kernels.ops.moe.gluon_moe_gfx950.fused_moe_{family}"
        ).fused_moe
        output = kernel(
            fixture.inputs(m),
            router,
            bias,
            *fixture.packed,
            expert_start=fixture.expert_start,
            fuse_shared_expert=False,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)

    def test_tp8_m1_kernel_compiles_and_runs(self):
        """Keep relocation and ABI regressions from hiding behind dispatcher-only tests."""
        from sglang.kernels.ops.moe.gluon_moe_gfx950.fused_moe_tp8_m1_16 import (
            fused_moe,
        )

        torch.manual_seed(0)
        device = torch.device("cuda")
        x = torch.randn((1, 6144), dtype=torch.bfloat16, device=device)
        router = torch.randn((256, 6144), dtype=torch.bfloat16, device=device)
        correction_bias = torch.zeros((256,), dtype=torch.bfloat16, device=device)
        w13 = torch.zeros((257, 512, 3072), dtype=torch.uint8, device=device)
        w13_scale = torch.full((257, 512, 192), 127, dtype=torch.uint8, device=device)
        w2 = torch.zeros((257, 6144, 128), dtype=torch.uint8, device=device)
        w2_scale = torch.full((257, 6144, 8), 127, dtype=torch.uint8, device=device)

        try:
            output = fused_moe(
                x,
                router,
                correction_bias,
                w13,
                w13_scale,
                w2,
                w2_scale,
            )
            torch.cuda.synchronize()
            self.assertEqual(output.shape, x.shape)
            self.assertEqual(output.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(output).all().item())
            self.assertEqual(torch.count_nonzero(output).item(), 0)
        finally:
            del x, router, correction_bias, w13, w13_scale, w2, w2_scale
            empty_gpu_cache()


if __name__ == "__main__":
    unittest.main()
