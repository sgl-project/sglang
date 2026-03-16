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

    def test_merge_preserves_subclass_default_when_not_explicit(self):
        """Without explicit_fields, value matching base default is not merged,
        so the subclass default (empty string) is preserved."""
        target = DiffusersGenericSamplingParams()
        self.assertEqual(target.negative_prompt, "")

        user = SamplingParams()
        target._merge_with_user_params(user)
        self.assertEqual(target.negative_prompt, "")

    def test_merge_applies_different_negative_prompt(self):
        target = DiffusersGenericSamplingParams()
        user = SamplingParams(negative_prompt="ugly, blurry")
        target._merge_with_user_params(user)
        self.assertEqual(target.negative_prompt, "ugly, blurry")

    def test_merge_explicit_field_matching_base_default(self):
        """Even when the user value matches the base-class default, it should
        still be applied if listed in explicit_fields."""
        base_default = SamplingParams.negative_prompt
        target = DiffusersGenericSamplingParams()
        self.assertEqual(target.negative_prompt, "")

        user = SamplingParams(negative_prompt=base_default)
        target._merge_with_user_params(user, explicit_fields={"negative_prompt"})
        self.assertEqual(target.negative_prompt, base_default)

    def test_cli_roundtrip_no_negative_prompt(self):
        """Simulate CLI without --negative-prompt: subclass default is kept."""
        ns = argparse.Namespace(negative_prompt=None, width=512, height=512)
        kwargs = SamplingParams.get_cli_args(ns)
        self.assertNotIn("negative_prompt", kwargs)

        user = SamplingParams(**kwargs)
        target = DiffusersGenericSamplingParams()
        target._merge_with_user_params(user, explicit_fields=set(kwargs.keys()))
        self.assertEqual(target.negative_prompt, "")

    def test_cli_roundtrip_with_negative_prompt(self):
        """Simulate CLI with --negative-prompt: user value is applied."""
        user_neg = "bad quality, watermark"
        ns = argparse.Namespace(negative_prompt=user_neg, width=512, height=512)
        kwargs = SamplingParams.get_cli_args(ns)
        user = SamplingParams(**kwargs)

        target = DiffusersGenericSamplingParams()
        target._merge_with_user_params(user, explicit_fields=set(kwargs.keys()))
        self.assertEqual(target.negative_prompt, user_neg)

    def test_cli_roundtrip_with_base_default_negative_prompt(self):
        """Simulate CLI where --negative-prompt value matches the base default:
        user value should still be applied (not dropped)."""
        base_default = SamplingParams.negative_prompt
        ns = argparse.Namespace(negative_prompt=base_default, width=512, height=512)
        kwargs = SamplingParams.get_cli_args(ns)
        self.assertIn("negative_prompt", kwargs)

        user = SamplingParams(**kwargs)
        target = DiffusersGenericSamplingParams()
        target._merge_with_user_params(user, explicit_fields=set(kwargs.keys()))
        self.assertEqual(target.negative_prompt, base_default)


if __name__ == "__main__":
    unittest.main()
