import unittest

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.shared_ep.profiles import select_profile
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


def _config(*, hidden_size, top_k, intermediate_size=2048):
    return MoeRunnerConfig(
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        top_k=top_k,
        num_experts=256,
        num_local_experts=32,
        params_dtype=torch.bfloat16,
    )


class TestSharedEpProfiles(unittest.TestCase):
    def test_exact_glm_and_dsv4_shapes_select_registered_profiles(self):
        """The release registry must cover both promised model contracts."""
        glm = select_profile(
            _config(hidden_size=6144, top_k=8),
            capability=(9, 0),
            ep_size=8,
            block_shape=(128, 128),
            max_tokens_per_rank=32,
        )
        dsv4 = select_profile(
            _config(hidden_size=4096, top_k=6),
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
            capability=(9, 0),
            ep_size=8,
            block_shape=(128, 128),
            max_tokens_per_rank=32,
        )
        dsv4 = select_profile(
            _config(hidden_size=4096, top_k=6),
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

    def test_every_unsupported_admission_dimension_is_rejected(self):
        """A near-match must fail before graph capture instead of misdispatching."""
        base = dict(
            config=_config(hidden_size=4096, top_k=6),
            capability=(9, 0),
            ep_size=8,
            block_shape=(128, 128),
            max_tokens_per_rank=32,
        )
        mutations = (
            {"capability": (10, 0)},
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
