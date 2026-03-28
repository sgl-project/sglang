import argparse
import math
import unittest

from sglang.multimodal_gen.configs.sample.diffusers_generic import (
    DiffusersGenericSamplingParams,
)
from sglang.multimodal_gen.configs.sample.flux import FluxSamplingParams
from sglang.multimodal_gen.configs.sample.qwenimage import QwenImageSamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import (
    SamplingParams,
    _json_safe,
)
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams
from sglang.multimodal_gen.configs.sample.wan import (
    WanI2V_14B_480P_SamplingParam,
    WanI2V_14B_720P_SamplingParam,
    WanT2V_1_3B_SamplingParams,
    WanT2V_14B_SamplingParams,
)


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

    def test_output_file_name_supports_callable_teacache_params(self):
        def coefficients_callback(_: TeaCacheParams) -> list[float]:
            return [1.0, 2.0, 3.0, 4.0, 5.0]

        params = SamplingParams(
            prompt="callable teacache",
            teacache_params=TeaCacheParams(
                coefficients_callback=coefficients_callback,
            ),
        )

        params._set_output_file_name()

        self.assertTrue(params.output_file_name.endswith(".mp4"))
        self.assertIn(
            "test_sampling_params.TestSamplingParamsSubclass.test_output_file_name_supports_callable_teacache_params",
            _json_safe(coefficients_callback),
        )

    def test_teacache_callback_takes_precedence_over_static_coefficients(self):
        def coefficients_callback(_: TeaCacheParams) -> list[float]:
            return [9.0, 8.0, 7.0, 6.0, 5.0]

        params = TeaCacheParams(
            coefficients=[1.0, 2.0, 3.0, 4.0, 5.0],
            coefficients_callback=coefficients_callback,
        )

        self.assertEqual(params.get_coefficients(), [9.0, 8.0, 7.0, 6.0, 5.0])

    def test_wan_teacache_boundaries_match_legacy_behavior(self):
        legacy_equivalent_cases = [
            (WanT2V_1_3B_SamplingParams().teacache_params, False, (5, 50)),
            (WanT2V_1_3B_SamplingParams().teacache_params, True, (10, 100)),
            (WanT2V_14B_SamplingParams().teacache_params, False, (1, 49)),
            (WanT2V_14B_SamplingParams().teacache_params, True, (2, 98)),
            (WanI2V_14B_480P_SamplingParam().teacache_params, False, (5, 50)),
            (WanI2V_14B_480P_SamplingParam().teacache_params, True, (10, 100)),
            (WanI2V_14B_720P_SamplingParam().teacache_params, False, (5, 50)),
            (WanI2V_14B_720P_SamplingParam().teacache_params, True, (10, 100)),
        ]

        for teacache_params, do_cfg, expected in legacy_equivalent_cases:
            with self.subTest(
                use_ret_steps=teacache_params.use_ret_steps,
                do_cfg=do_cfg,
                expected=expected,
            ):
                self.assertEqual(
                    teacache_params.get_skip_boundaries(50, do_cfg),
                    expected,
                )


class TestSamplingParamsCliArgs(unittest.TestCase):
    def _parse_cli_kwargs(self, argv: list[str]) -> dict:
        parser = argparse.ArgumentParser()
        SamplingParams.add_cli_args(parser)
        args = parser.parse_args(argv)
        return SamplingParams.get_cli_args(args)

    def _make_qwen_image_params(self, argv: list[str]) -> QwenImageSamplingParams:
        return QwenImageSamplingParams(**self._parse_cli_kwargs(argv))

    def test_get_cli_args_drops_unset_sampling_params(self):
        self.assertEqual(self._parse_cli_kwargs([]), {})

    def test_get_cli_args_keeps_explicit_sampling_params(self):
        kwargs = self._parse_cli_kwargs(
            [
                "--guidance-scale",
                str(SamplingParams.guidance_scale),
                "--negative-prompt",
                SamplingParams.negative_prompt,
                "--save-output",
            ]
        )

        self.assertEqual(kwargs["guidance_scale"], SamplingParams.guidance_scale)
        self.assertEqual(kwargs["negative_prompt"], SamplingParams.negative_prompt)
        self.assertTrue(kwargs["save_output"])

    def test_qwen_image_cli_path_preserves_model_defaults(self):
        params = self._make_qwen_image_params([])

        self.assertEqual(params.negative_prompt, " ")
        self.assertEqual(params.guidance_scale, 4.0)

    def test_qwen_image_cli_path_allows_explicit_override_to_base_defaults(self):
        params = self._make_qwen_image_params(
            [
                "--guidance-scale",
                str(SamplingParams.guidance_scale),
                "--negative-prompt",
                SamplingParams.negative_prompt,
            ]
        )

        self.assertEqual(params.guidance_scale, SamplingParams.guidance_scale)
        self.assertEqual(params.negative_prompt, SamplingParams.negative_prompt)

    def test_merge_allows_explicit_field_matching_base_default(self):
        target = DiffusersGenericSamplingParams()
        user = SamplingParams(negative_prompt=SamplingParams.negative_prompt)

        target._merge_with_user_params(user, explicit_fields={"negative_prompt"})

        self.assertEqual(target.negative_prompt, SamplingParams.negative_prompt)


if __name__ == "__main__":
    unittest.main()
