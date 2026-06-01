import argparse
import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
    LTX2PipelineConfig,
    is_ltx23_native_variant,
    sync_ltx23_runtime_vae_markers,
)
from sglang.multimodal_gen.configs.sample.diffusers_generic import (
    DiffusersGenericSamplingParams,
)
from sglang.multimodal_gen.configs.sample.flux import (
    Flux2KleinSamplingParams,
    Flux2SamplingParams,
    FluxSamplingParams,
)
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

    def test_seed_accepts_int_or_non_empty_int_list(self):
        self.assertEqual(SamplingParams(seed=7).seed, 7)
        self.assertEqual(SamplingParams(seed=[7, 8]).seed, [7, 8])
        with self.assertRaisesRegex(ValueError, r"seed list"):
            SamplingParams(seed=[])
        with self.assertRaisesRegex(ValueError, r"seed"):
            SamplingParams(seed=[1, -1])

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

    def test_flux_guidance_defaults_match_model_defaults(self):
        self.assertEqual(FluxSamplingParams().guidance_scale, 3.5)
        self.assertEqual(Flux2SamplingParams().guidance_scale, 4.0)
        self.assertEqual(Flux2KleinSamplingParams().guidance_scale, 1.0)

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

    def test_ltx23_runtime_vae_markers_sync_variant_and_decoder_metadata(self):
        arch_config = LTX2PipelineConfig().vae_config.arch_config

        self.assertFalse(is_ltx23_native_variant(arch_config))
        self.assertEqual(arch_config.video_decoder_variant, "ltx_2")
        self.assertEqual(arch_config.condition_encoder_subdir, "")

        sync_ltx23_runtime_vae_markers(
            arch_config,
            SimpleNamespace(
                arch_config=SimpleNamespace(
                    ltx_variant="ltx_2_3",
                    condition_encoder_subdir="ltx23_image_encoder",
                    video_decoder_variant="ltx_2_3",
                    video_decoder_config={"_class_name": "AutoencoderKLLTX2Video"},
                )
            ),
        )

        self.assertTrue(is_ltx23_native_variant(arch_config))
        self.assertEqual(arch_config.condition_encoder_subdir, "ltx23_image_encoder")
        self.assertEqual(arch_config.video_decoder_variant, "ltx_2_3")
        self.assertEqual(
            arch_config.video_decoder_config,
            {"_class_name": "AutoencoderKLLTX2Video"},
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

    def test_get_cli_args_accepts_seed_list(self):
        self.assertEqual(self._parse_cli_kwargs(["--seed", "7"])["seed"], 7)
        self.assertEqual(
            self._parse_cli_kwargs(["--seed", "7", "8"])["seed"],
            [7, 8],
        )

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

    def test_cli_path_tracks_explicit_width_height_fields(self):
        server_args = MagicMock()
        server_args.backend = "sglang"
        server_args.model_id = None
        server_args.pipeline_config = MagicMock()

        with patch.object(
            SamplingParams,
            "from_pretrained",
            side_effect=lambda *args, **kwargs: Flux2SamplingParams(),
        ):
            implicit_size = SamplingParams.from_user_sampling_params_args(
                "dummy-model",
                server_args=server_args,
                prompt="p",
                image_path="/tmp/in.png",
            )
            explicit_size = SamplingParams.from_user_sampling_params_args(
                "dummy-model",
                server_args=server_args,
                prompt="p",
                image_path="/tmp/in.png",
                width=768,
                height=512,
            )

        implicit_fields = set(implicit_size.build_request_extra()["explicit_fields"])
        explicit_fields = set(explicit_size.build_request_extra()["explicit_fields"])

        self.assertNotIn("width", implicit_fields)
        self.assertNotIn("height", implicit_fields)
        self.assertIn("width", explicit_fields)
        self.assertIn("height", explicit_fields)

    def test_dataclasses_replace_preserves_explicit_fields(self):
        """`dataclasses.replace` drops `_explicit_fields`; DiffGenerator must restore it."""
        import dataclasses

        server_args = MagicMock()
        server_args.backend = "sglang"
        server_args.model_id = None
        server_args.pipeline_config = MagicMock()

        with patch.object(
            SamplingParams,
            "from_pretrained",
            side_effect=lambda *args, **kwargs: Flux2SamplingParams(),
        ):
            sampling_params_orig = SamplingParams.from_user_sampling_params_args(
                "dummy-model",
                server_args=server_args,
                prompt="orig",
                image_path="/tmp/in.png",
                width=768,
                height=512,
            )

        self.assertIn("width", sampling_params_orig._explicit_fields)
        self.assertIn("height", sampling_params_orig._explicit_fields)

        cloned = dataclasses.replace(
            sampling_params_orig,
            prompt="new",
            output_file_name=None,
            image_path="/tmp/in2.png",
        )
        self.assertFalse(hasattr(cloned, "_explicit_fields"))

        # Mirror the restore done in DiffGenerator.generate().
        cloned._explicit_fields = getattr(
            sampling_params_orig, "_explicit_fields", set()
        ) | {"prompt", "output_file_name", "image_path"}

        explicit = set(cloned.build_request_extra()["explicit_fields"])
        self.assertIn("width", explicit)
        self.assertIn("height", explicit)
        self.assertIn("prompt", explicit)
        self.assertIn("image_path", explicit)


class TestProgressiveSamplingParams(unittest.TestCase):
    """Tests for progressive resolution growing fields on SamplingParams."""

    def test_defaults(self):
        p = SamplingParams()
        self.assertEqual(p.progressive_mode, "fullres")
        self.assertEqual(p.progressive_levels, 1)
        self.assertAlmostEqual(p.progressive_delta, 0.01)

    def test_valid_modes_accepted(self):
        for mode in ("fullres", "dct", "dct_rewind"):
            with self.subTest(mode=mode):
                p = SamplingParams(progressive_mode=mode)
                self.assertEqual(p.progressive_mode, mode)

    def test_batch_sig_exclude_on_all_progressive_fields(self):
        import dataclasses

        fields = {f.name: f for f in dataclasses.fields(SamplingParams)}
        for fname in ("progressive_mode", "progressive_levels", "progressive_delta"):
            with self.subTest(field=fname):
                self.assertTrue(
                    fields[fname].metadata.get("batch_sig_exclude"),
                    f"{fname} must have batch_sig_exclude=True so different requests "
                    "can mix progressive and fullres modes in the same server",
                )

    def test_cli_progressive_mode(self):
        parser = argparse.ArgumentParser()
        SamplingParams.add_cli_args(parser)
        args, _ = parser.parse_known_args(["--progressive-mode", "dct_rewind"])
        self.assertEqual(args.progressive_mode, "dct_rewind")

    def test_cli_progressive_levels(self):
        parser = argparse.ArgumentParser()
        SamplingParams.add_cli_args(parser)
        args, _ = parser.parse_known_args(["--progressive-levels", "2"])
        self.assertEqual(args.progressive_levels, 2)

    def test_cli_progressive_delta(self):
        parser = argparse.ArgumentParser()
        SamplingParams.add_cli_args(parser)
        args, _ = parser.parse_known_args(["--progressive-delta", "0.05"])
        self.assertAlmostEqual(args.progressive_delta, 0.05)

    def test_cli_unset_progressive_fields_suppressed(self):
        """Without --progressive-* flags, progressive fields are absent from get_cli_args
        (add_cli_args uses argparse.SUPPRESS so unset flags don't appear in the namespace)."""
        parser = argparse.ArgumentParser()
        SamplingParams.add_cli_args(parser)
        args = parser.parse_args([])
        kwargs = SamplingParams.get_cli_args(args)
        self.assertNotIn("progressive_mode", kwargs)
        self.assertNotIn("progressive_levels", kwargs)
        self.assertNotIn("progressive_delta", kwargs)

    def test_cli_choices_reject_invalid_mode(self):
        parser = argparse.ArgumentParser()
        SamplingParams.add_cli_args(parser)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--progressive-mode", "wavelet"])


if __name__ == "__main__":
    unittest.main()
