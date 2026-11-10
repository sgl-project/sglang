"""Reject explicit AITER fusion requests that deterministic inference cannot honor."""

import unittest

from sglang.srt.arg_groups.attention_hook import handle_deterministic_inference
from sglang.srt.arg_groups.overrides import declare_resolution, resolution_result
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeterministicAiterFusion(CustomTestCase):
    def test_explicit_fusion_is_rejected(self):
        """Deterministic mode must not silently discard an explicit fusion request."""
        for deterministic_args in (
            {"enable_deterministic_inference": True},
            {"rl_on_policy_target": "fsdp"},
        ):
            with (
                self.subTest(deterministic_args=deterministic_args),
                envs.SGLANG_VLM_CACHE_SIZE_MB.override(
                    envs.SGLANG_VLM_CACHE_SIZE_MB.get()
                ),
                envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(
                    envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
                ),
            ):
                args = ServerArgs(
                    model_path="instance://127.0.0.1:8000/dummy",
                    attention_backend="triton",
                    enable_aiter_allreduce_fusion=True,
                    **deterministic_args,
                )
                with self.assertRaisesRegex(
                    ValueError, "--enable-aiter-allreduce-fusion.*deterministic"
                ):
                    handle_deterministic_inference(args)

    def test_automatic_fusion_is_still_disabled(self):
        """A model-selected fusion default must not make deterministic mode fail."""
        for fusion_args in ({}, {"enable_aiter_allreduce_fusion": False}):
            with self.subTest(fusion_args=fusion_args):
                args = ServerArgs(
                    model_path="instance://127.0.0.1:8000/dummy",
                    attention_backend="triton",
                    enable_deterministic_inference=True,
                    **fusion_args,
                )
                declare_resolution(
                    args,
                    "_handle_model_specific_adjustments",
                    enable_aiter_allreduce_fusion=True,
                )
                handle_deterministic_inference(args)
                self.assertFalse(
                    resolution_result(args, "enable_aiter_allreduce_fusion")
                )

    def test_fusion_without_deterministic_inference_is_unchanged(self):
        args = ServerArgs(model_path="dummy", enable_aiter_allreduce_fusion=True)
        handle_deterministic_inference(args)
        self.assertTrue(resolution_result(args, "enable_aiter_allreduce_fusion"))


if __name__ == "__main__":
    unittest.main()
