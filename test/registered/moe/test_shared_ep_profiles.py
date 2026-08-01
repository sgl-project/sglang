import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.shared_ep import SharedEpQuantization
from sglang.srt.layers.moe.shared_ep.profiles import (
    make_pull_cache_prefill_profile,
    resolve_prefill_capacity,
    select_profile,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")


def _config(
    *,
    hidden_size,
    top_k,
    intermediate_size=2048,
    num_experts=256,
    num_local_experts=32,
):
    return MoeRunnerConfig(
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        top_k=top_k,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        params_dtype=torch.bfloat16,
    )


class TestSharedEpProfiles(unittest.TestCase):
    def test_exact_glm_and_dsv4_shapes_select_registered_profiles(self):
        """The release registry must cover both promised model contracts."""
        glm = select_profile(
            _config(hidden_size=6144, top_k=8),
            platform="cuda",
            capability=(9, 0),
            ep_size=8,
            block_shape=(128, 128),
            max_tokens_per_rank=32,
        )
        dsv4 = select_profile(
            _config(hidden_size=4096, top_k=6),
            platform="cuda",
            capability=(9, 0),
            ep_size=8,
            block_shape=(128, 128),
            max_tokens_per_rank=32,
        )

        self.assertEqual(glm.name, "glm52")
        self.assertEqual(dsv4.name, "dsv4_flash")
        self.assertEqual(glm.route_kernel_config, {"num_threads": 1024})
        self.assertEqual(dsv4.route_kernel_config, {"num_threads": 1024})
        self.assertEqual(dsv4.kernel_config(4)["BLOCK_SIZE_N"], 64)
        self.assertEqual(dsv4.kernel_config(8)["BLOCK_SIZE_N"], 128)

    def test_w13_and_w2_profiles_are_selected_independently(self):
        """Measured stage profiles must not be collapsed into one compromise."""
        glm = select_profile(
            _config(hidden_size=6144, top_k=8),
            platform="cuda",
            capability=(9, 0),
            ep_size=8,
            block_shape=(128, 128),
            max_tokens_per_rank=32,
        )
        dsv4 = select_profile(
            _config(hidden_size=4096, top_k=6),
            platform="cuda",
            capability=(9, 0),
            ep_size=8,
            block_shape=(128, 128),
            max_tokens_per_rank=32,
        )

        expected = {
            ("dsv4", 0): ((64, 4, 3), (64, 4, 3)),
            ("dsv4", 4): ((64, 4, 3), (64, 4, 3)),
            ("dsv4", 8): ((128, 4, 4), (64, 4, 3)),
            ("dsv4", 32): ((128, 4, 4), (64, 4, 3)),
            ("glm", 0): ((64, 4, 3), (128, 4, 3)),
            ("glm", 4): ((64, 4, 3), (128, 4, 3)),
            ("glm", 8): ((128, 8, 4), (128, 4, 3)),
            ("glm", 32): ((128, 8, 4), (128, 4, 3)),
        }
        for model, profile in (("dsv4", dsv4), ("glm", glm)):
            for tokens in (0, 4, 8, 32):
                with self.subTest(model=model, tokens=tokens):
                    w13 = profile.w13_kernel_config(tokens)
                    w2 = profile.w2_kernel_config(tokens)
                    self.assertEqual(
                        (
                            w13["BLOCK_SIZE_N"],
                            w13["num_warps"],
                            w13["num_stages"],
                        ),
                        expected[(model, tokens)][0],
                    )
                    self.assertEqual(
                        (
                            w2["BLOCK_SIZE_N"],
                            w2["num_warps"],
                            w2["num_stages"],
                        ),
                        expected[(model, tokens)][1],
                    )

        for profile in (dsv4, glm):
            for tokens in (-1, 33):
                with self.subTest(profile=profile.name, tokens=tokens):
                    with self.assertRaisesRegex(ValueError, "profile capacity"):
                        profile.w13_kernel_config(tokens)
                    with self.assertRaisesRegex(ValueError, "profile capacity"):
                        profile.w2_kernel_config(tokens)

    def test_gfx950_profiles_use_rocm_keys_and_conservative_configs(self):
        for hidden_size, top_k, expected_name in (
            (6144, 8, "glm52_gfx950"),
            (4096, 6, "dsv4_flash_gfx950"),
        ):
            with self.subTest(expected_name=expected_name):
                profile = select_profile(
                    _config(hidden_size=hidden_size, top_k=top_k),
                    platform="rocm",
                    capability=(9, 5),
                    ep_size=8,
                    block_shape=(128, 128),
                    max_tokens_per_rank=32,
                )
                self.assertEqual(profile.name, expected_name)
                self.assertEqual(profile.platform, "rocm")
                self.assertEqual(profile.route_kernel_config, {"num_threads": 256})
                for config in (
                    profile.w13_kernel_config(4),
                    profile.w13_kernel_config(32),
                    profile.w2_kernel_config(4),
                    profile.w2_kernel_config(32),
                ):
                    self.assertEqual(config["BLOCK_SIZE_N"], 64)
                    self.assertEqual(config["BLOCK_SIZE_K"], 128)
                    self.assertLessEqual(config["num_warps"], 4)
                    self.assertLessEqual(config["num_stages"], 2)

        with patch(
            "sglang.srt.layers.moe.shared_ep.profiles._runtime_platform_key",
            return_value="rocm",
        ):
            inferred = select_profile(
                _config(hidden_size=4096, top_k=6),
                capability=(9, 5),
                ep_size=8,
                block_shape=(128, 128),
                max_tokens_per_rank=32,
            )
        self.assertEqual(inferred.name, "dsv4_flash_gfx950")

    def test_exact_dsv4_pro_mxfp4_gfx950_profile_is_quantization_gated(self):
        config = _config(
            hidden_size=7168,
            intermediate_size=3072,
            top_k=6,
            num_experts=384,
            num_local_experts=48,
        )
        profile = select_profile(
            config,
            platform="rocm",
            capability=(9, 5),
            ep_size=8,
            quantization=SharedEpQuantization.MXFP4,
            block_shape=(1, 32),
            max_tokens_per_rank=32,
        )

        self.assertEqual(profile.name, "dsv4_pro_mxfp4_gfx950")
        self.assertIs(profile.quantization, SharedEpQuantization.MXFP4)
        self.assertEqual(profile.block_shape, (1, 32))
        self.assertEqual(profile.num_local_experts, 48)
        self.assertEqual(profile.block_size_m, 128)
        self.assertEqual(profile.w13_kernel_config(32)["BLOCK_SIZE_K"], 128)
        self.assertEqual(profile.w2_kernel_config(32)["BLOCK_SIZE_N"], 128)

        for quantization, block_shape in (
            (SharedEpQuantization.BLOCK_FP8, (1, 32)),
            (SharedEpQuantization.MXFP4, (128, 128)),
        ):
            with self.subTest(
                quantization=quantization,
                block_shape=block_shape,
            ):
                with self.assertRaisesRegex(ValueError, "observed=.*supported="):
                    select_profile(
                        config,
                        platform="rocm",
                        capability=(9, 5),
                        ep_size=8,
                        quantization=quantization,
                        block_shape=block_shape,
                        max_tokens_per_rank=32,
                    )

    def test_pull_cache_prefill_profiles_are_rocm_block_fp8_only(self):
        for hidden_size, top_k in ((6144, 8), (4096, 6)):
            with self.subTest(hidden_size=hidden_size):
                decode = select_profile(
                    _config(hidden_size=hidden_size, top_k=top_k),
                    platform="rocm",
                    capability=(9, 5),
                    ep_size=8,
                    block_shape=(128, 128),
                    max_tokens_per_rank=32,
                )
                prefill = make_pull_cache_prefill_profile(decode, 1024)
                self.assertIsNotNone(prefill)
                self.assertEqual(prefill.max_tokens_per_rank, 1024)
                self.assertEqual(prefill.block_size_m, 64)
                self.assertEqual(prefill.w13_kernel_config(1024)["BLOCK_SIZE_M"], 64)
                self.assertEqual(prefill.w2_kernel_config(1024)["BLOCK_SIZE_M"], 64)

        cuda_profile = select_profile(
            _config(hidden_size=4096, top_k=6),
            platform="cuda",
            capability=(9, 0),
            ep_size=8,
            block_shape=(128, 128),
            max_tokens_per_rank=32,
        )
        self.assertIsNone(make_pull_cache_prefill_profile(cuda_profile, 1024))

        mxfp4_profile = select_profile(
            _config(
                hidden_size=7168,
                intermediate_size=3072,
                top_k=6,
                num_experts=384,
                num_local_experts=48,
            ),
            platform="rocm",
            capability=(9, 5),
            ep_size=8,
            quantization=SharedEpQuantization.MXFP4,
            block_shape=(1, 32),
            max_tokens_per_rank=32,
        )
        self.assertIsNone(make_pull_cache_prefill_profile(mxfp4_profile, 1024))

    def test_prefill_capacity_is_dp_local_and_bounded(self):
        self.assertEqual(resolve_prefill_capacity(1), 1)
        self.assertEqual(resolve_prefill_capacity(1024), 1024)
        for invalid in (None, -1, 0, 1025):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    resolve_prefill_capacity(invalid)

    def test_every_unsupported_admission_dimension_is_rejected(self):
        """A near-match must fail before graph capture instead of misdispatching."""
        base = dict(
            config=_config(hidden_size=4096, top_k=6),
            platform="cuda",
            capability=(9, 0),
            ep_size=8,
            block_shape=(128, 128),
            max_tokens_per_rank=32,
        )
        mutations = (
            {"capability": (10, 0)},
            {"platform": "rocm"},
            {"ep_size": 4},
            {"block_shape": (64, 128)},
            {"max_tokens_per_rank": 64},
            {"config": _config(hidden_size=4096, top_k=8)},
            {"config": _config(hidden_size=4096, top_k=6, intermediate_size=2560)},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                observed = {**base, **mutation}
                with self.assertRaisesRegex(
                    ValueError,
                    "observed=.*supported=",
                ):
                    select_profile(**observed)


if __name__ == "__main__":
    unittest.main()
