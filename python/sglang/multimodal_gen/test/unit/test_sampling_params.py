import argparse
import math
import unittest

from sglang.multimodal_gen.configs.sample.diffusers_generic import (
    DiffusersGenericSamplingParams,
)
from sglang.multimodal_gen.configs.sample.flux import FluxSamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


class TestSamplingParamsValidate(unittest.TestCase):
    def test_prompt_path_suffix(self):
        with self.assertRaisesRegex(ValueError, r"prompt_path"):
            SamplingParams(prompt_path="bad.png")

    def test_num_outputs_per_prompt_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, r"num_outputs_per_prompt"):
            SamplingParams(num_outputs_per_prompt=0)

    def test_fps_must_be_positive_int(self):
        with self.assertRaisesRegex(ValueError, r"\bfps\b"):
            SamplingParams(fps=0)
        with self.assertRaisesRegex(ValueError, r"\bfps\b"):
            SamplingParams(fps=None)  # type: ignore[arg-type]

    def test_num_inference_steps_optional_but_if_set_must_be_positive(self):
        SamplingParams(num_inference_steps=None)
        with self.assertRaisesRegex(ValueError, r"num_inference_steps"):
            SamplingParams(num_inference_steps=-1)

    def test_guidance_scale_must_be_finite_non_negative_if_set(self):
        SamplingParams(guidance_scale=None)
        with self.assertRaisesRegex(ValueError, r"guidance_scale"):
            SamplingParams(guidance_scale=math.nan)
        with self.assertRaisesRegex(ValueError, r"guidance_scale"):
            SamplingParams(guidance_scale=-0.1)

    def test_guidance_rescale_must_be_finite_non_negative(self):
        with self.assertRaisesRegex(ValueError, r"guidance_rescale"):
            SamplingParams(guidance_rescale=-1.0)
        with self.assertRaisesRegex(ValueError, r"guidance_rescale"):
            SamplingParams(guidance_rescale=math.inf)

    def test_boundary_ratio_range(self):
        SamplingParams(boundary_ratio=None)
        with self.assertRaisesRegex(ValueError, r"boundary_ratio"):
            SamplingParams(boundary_ratio=1.5)
        with self.assertRaisesRegex(ValueError, r"boundary_ratio"):
            SamplingParams(boundary_ratio=math.nan)


class TestSamplingParamsSubclass(unittest.TestCase):
    def test_flux_defaults_resolution_when_not_provided(self):
        params = FluxSamplingParams()

        self.assertEqual(params.height, 1024)
        self.assertEqual(params.width, 1024)

    def test_flux_preserves_user_resolution(self):
        params = FluxSamplingParams(height=640, width=768)

        self.assertEqual(params.height, 640)
        self.assertEqual(params.width, 768)

    def test_diffusers_generic_calls_base_post_init(self):
        with self.assertRaises(AssertionError):
            DiffusersGenericSamplingParams(num_frames=0)


class TestNegativePromptMerge(unittest.TestCase):
    """Regression tests for negative_prompt not being passed through CLI"""

    def test_get_cli_args_filters_none(self):
        ns = argparse.Namespace(negative_prompt=None, prompt="hello")
        result = SamplingParams.get_cli_args(ns)
        self.assertNotIn("negative_prompt", result)
        self.assertEqual(result["prompt"], "hello")

    def test_get_cli_args_keeps_explicit_value(self):
        ns = argparse.Namespace(negative_prompt="ugly, blurry")
        result = SamplingParams.get_cli_args(ns)
        self.assertEqual(result["negative_prompt"], "ugly, blurry")

    def test_merge_preserves_subclass_default_when_user_unchanged(self):
        """When user doesn't pass --negative-prompt, the subclass default
        (empty string for DiffusersGeneric) should be preserved."""
        target = DiffusersGenericSamplingParams()
        self.assertEqual(target.negative_prompt, "")

        user = SamplingParams()  # uses base default
        target._merge_with_user_params(user)
        self.assertEqual(target.negative_prompt, "")

    def test_merge_applies_user_negative_prompt(self):
        """When user explicitly passes --negative-prompt, it should override
        the subclass default."""
        target = DiffusersGenericSamplingParams()
        user = SamplingParams(negative_prompt="ugly, blurry")
        target._merge_with_user_params(user)
        self.assertEqual(target.negative_prompt, "ugly, blurry")

    def test_cli_roundtrip_no_negative_prompt(self):
        """Simulate CLI without --negative-prompt: subclass default is kept."""
        ns = argparse.Namespace(negative_prompt=None, width=512, height=512)
        kwargs = SamplingParams.get_cli_args(ns)
        user = SamplingParams(**kwargs)

        target = DiffusersGenericSamplingParams()
        target._merge_with_user_params(user)
        self.assertEqual(target.negative_prompt, "")

    def test_cli_roundtrip_with_negative_prompt(self):
        """Simulate CLI with --negative-prompt: user value is applied."""
        user_neg = "bad quality, watermark"
        ns = argparse.Namespace(negative_prompt=user_neg, width=512, height=512)
        kwargs = SamplingParams.get_cli_args(ns)
        user = SamplingParams(**kwargs)

        target = DiffusersGenericSamplingParams()
        target._merge_with_user_params(user)
        self.assertEqual(target.negative_prompt, user_neg)


if __name__ == "__main__":
    unittest.main()
