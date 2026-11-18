import argparse
import logging
import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.configs.pipeline_configs.glm_image import (
    GlmImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
    LTX2PipelineConfig,
    is_ltx23_native_variant,
    sync_ltx23_runtime_vae_markers,
)
from sglang.multimodal_gen.configs.sample.diffusers_generic import (
    DiffusersGenericSamplingParams,
)
from sglang.multimodal_gen.configs.sample.flux import (
    Flux2SamplingParams,
    FluxSamplingParams,
)
from sglang.multimodal_gen.configs.sample.glmimage import (
    GlmImageSamplingParams,
    align_glm_image_dimension,
)
from sglang.multimodal_gen.configs.sample.qwenimage import QwenImageSamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import (
    QUALITY_LEVELS,
    SamplingParams,
    SkipSoftmaxParams,
    _json_safe,
    normalize_quality,
    quality_allows,
    resolve_skip_softmax_params,
)
from sglang.multimodal_gen.configs.sample.spectrum import SpectrumParams
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams
from sglang.multimodal_gen.configs.sample.wan import (
    FastWanT2V480PConfig,
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

    def test_quality_defaults_to_lossless(self):
        # The default runs every fast path that keeps the reference math;
        # "exact" is the opt-in for bit-identical output.
        self.assertEqual(SamplingParams().quality, "lossless")

    def test_quality_levels_are_cumulative(self):
        self.assertEqual(QUALITY_LEVELS, ("exact", "lossless", "high"))
        for quality in QUALITY_LEVELS:
            with self.subTest(quality=quality):
                self.assertEqual(SamplingParams(quality=quality).quality, quality)

        # A request admits the fast paths of its own tier and of every
        # stricter one, so the levels stay cumulative.
        for request, admits in (
            ("exact", {"exact"}),
            ("lossless", {"exact", "lossless"}),
            ("high", {"exact", "lossless", "high"}),
        ):
            for tier in QUALITY_LEVELS:
                with self.subTest(request=request, tier=tier):
                    self.assertEqual(quality_allows(request, tier), tier in admits)

    def test_extra_high_is_accepted_as_the_former_name_of_lossless(self):
        self.assertEqual(SamplingParams(quality="extra-high").quality, "lossless")
        self.assertEqual(normalize_quality("extra-high"), "lossless")

    def test_quality_rejects_invalid_values(self):
        for bad in ("ultra", "draft", "fast", "", True, 1):
            with self.assertRaisesRegex(ValueError, r"quality must be one of"):
                SamplingParams(quality=bad)  # type: ignore[arg-type]

    def test_skip_softmax_params(self):
        params = {"threshold_scale_factor": 500, "start_step": 14}
        self.assertEqual(
            resolve_skip_softmax_params(params),
            SkipSoftmaxParams(threshold_scale_factor=500.0, start_step=14),
        )
        self.assertEqual(
            SamplingParams(skip_softmax_params=params).skip_softmax_params, params
        )

    def test_skip_softmax_params_reject_invalid_values(self):
        invalid = (
            {},
            {"threshold_scale_factor": 0},
            {"threshold_scale_factor": math.inf},
            {"threshold_scale_factor": 1, "start_step": -1},
            {"threshold_scale_factor": 1, "unknown": True},
        )
        for params in invalid:
            with self.subTest(params=params):
                with self.assertRaisesRegex(ValueError, "skip_softmax_params"):
                    SamplingParams(skip_softmax_params=params)

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

    def test_postprocess_scale_ranges(self):
        """Out-of-range upscaling/interpolation scales must not reach the kernels."""
        for field, value in [
            ("frame_interpolation_scale", 0.0),
            ("frame_interpolation_scale", math.nan),
            ("frame_interpolation_scale", 100.0),
            ("upscaling_scale", 0),
            ("upscaling_scale", 1000),
            ("upscaling_scale", True),
        ]:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, field):
                    SamplingParams(**{field: value})
        SamplingParams(frame_interpolation_scale=0.25, upscaling_scale=1)
        SamplingParams(frame_interpolation_scale=4.0, upscaling_scale=8)

    def test_teacache_and_spectrum_are_mutually_exclusive(self):
        with self.assertRaisesRegex(
            ValueError, r"enable_teacache and enable_spectrum are mutually exclusive"
        ):
            SamplingParams(enable_teacache=True, enable_spectrum=True)

    def test_spectrum_params_reject_invalid_controls(self):
        invalid_controls = (
            {"window_size": 0},
            {"flex_window": -0.1},
            {"w": 1.1},
            {"lam": -0.1},
            {"warmup_steps": -1},
            {"m": 0},
            {"history_size": 0},
            {"tau_num_steps": 0},
            {"taylor_order": 4},
        )
        for kwargs in invalid_controls:
            with self.assertRaises(ValueError):
                SpectrumParams(**kwargs)

    def test_spectrum_dict_is_validated_when_sampling_params_constructs_it(self):
        with self.assertRaisesRegex(ValueError, "history_size"):
            SamplingParams(
                enable_spectrum=True,
                spectrum_params={"history_size": 0},
            )


class TestSamplingParamsSubclass(unittest.TestCase):
    def test_glm_image_rounds_resolution_up_to_multiple_of_32(self):
        server_args = SimpleNamespace(
            pipeline_config=GlmImagePipelineConfig(),
            output_path=None,
            comfyui_mode=True,
        )
        cases = [
            ((500, 500), (512, 512)),
            ((1024, 600), (1024, 608)),
            ((500, 600), (512, 608)),
            ((550, 1009), (576, 1024)),
            ((1280, 720), (1280, 736)),
        ]

        for requested, expected in cases:
            with self.subTest(requested=requested):
                params = GlmImageSamplingParams(
                    width=requested[0],
                    height=requested[1],
                )

                with patch(
                    "sglang.multimodal_gen.configs.sample.glmimage.logger.warning"
                ) as mock_warning:
                    params._adjust(server_args)

                self.assertEqual((params.width, params.height), expected)
                self.assertEqual(
                    (params.requested_width, params.requested_height), requested
                )
                mock_warning.assert_called_once_with(
                    "GLM-Image requires dimensions divisible by %s; adjusted "
                    "requested resolution from %sx%s to %sx%s",
                    32,
                    requested[0],
                    requested[1],
                    expected[0],
                    expected[1],
                )

    def test_glm_image_resolution_rounds_up(self):
        self.assertEqual(align_glm_image_dimension(560), 576)

    def test_glm_image_resolution_keeps_minimum_alignment(self):
        self.assertEqual(align_glm_image_dimension(0), 32)
        self.assertEqual(align_glm_image_dimension(-1), 32)

    def test_glm_image_does_not_warn_for_aligned_resolution(self):
        server_args = SimpleNamespace(
            pipeline_config=GlmImagePipelineConfig(),
            output_path=None,
            comfyui_mode=True,
        )
        params = GlmImageSamplingParams(width=1024, height=1024)

        with patch(
            "sglang.multimodal_gen.configs.sample.glmimage.logger.warning"
        ) as mock_warning:
            params._adjust(server_args)

        mock_warning.assert_not_called()

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

    def test_fastwan_480p_default_resolution_is_supported(self):
        params = FastWanT2V480PConfig()

        self.assertEqual((params.width, params.height), (832, 480))
        self.assertIn((params.width, params.height), params.supported_resolutions)

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

    def test_wan_teacache_skip_step_ranges(self):
        cases = [
            (WanT2V_1_3B_SamplingParams().teacache_params, (5, 50)),
            (WanT2V_14B_SamplingParams().teacache_params, (1, 49)),
            (WanI2V_14B_480P_SamplingParam().teacache_params, (5, 50)),
            (WanI2V_14B_720P_SamplingParam().teacache_params, (5, 50)),
        ]

        for teacache_params, expected in cases:
            with self.subTest(
                use_ret_steps=teacache_params.use_ret_steps,
                expected=expected,
            ):
                self.assertEqual(
                    teacache_params.get_skip_step_range(50),
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

    def test_cli_accepts_url_postprocess_model_paths(self):
        """URL weights stay supported outside HTTP; only network entry points reject them."""
        url = "https://example.com/weights.pth"
        params = SamplingParams(
            **self._parse_cli_kwargs(
                ["--upscaling-model-path", url, "--frame-interpolation-model-path", url]
            )
        )
        self.assertEqual(params.upscaling_model_path, url)
        self.assertEqual(params.frame_interpolation_model_path, url)

    def test_get_cli_args_accepts_seed_list(self):
        self.assertEqual(self._parse_cli_kwargs(["--seed", "7"])["seed"], 7)
        self.assertEqual(
            self._parse_cli_kwargs(["--seed", "7", "8"])["seed"],
            [7, 8],
        )

    def test_quality_is_request_scoped_cli_arg(self):
        self.assertNotIn("quality", self._parse_cli_kwargs([]))
        for quality in ("exact", "lossless", "high", "extra-high"):
            with self.subTest(quality=quality):
                self.assertEqual(
                    self._parse_cli_kwargs(["--quality", quality])["quality"], quality
                )

    def test_get_cli_args_maps_spectrum_prefixed_flags(self):
        kwargs = self._parse_cli_kwargs(
            [
                "--enable-spectrum",
                "--spectrum-window-size",
                "2.5",
                "--spectrum-flex-window",
                "0.9",
                "--spectrum-warmup-steps",
                "6",
                "--spectrum-m",
                "3",
                "--spectrum-lam",
                "0.2",
                "--spectrum-tau-num-steps",
                "42",
            ]
        )

        self.assertTrue(kwargs["enable_spectrum"])
        self.assertEqual(
            kwargs["spectrum_params"],
            {
                "window_size": 2.5,
                "flex_window": 0.9,
                "warmup_steps": 6,
                "m": 3,
                "lam": 0.2,
                "tau_num_steps": 42,
            },
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

    def test_cli_path_preserves_diffusers_kwargs_in_request_extra(self):
        server_args = MagicMock()
        server_args.backend = "sglang"
        server_args.model_id = None
        server_args.pipeline_config = MagicMock()
        diffusers_kwargs = {"camera_to_world_path": "/tmp/camera.npy"}

        with patch.object(
            SamplingParams,
            "from_pretrained",
            side_effect=lambda *args, **kwargs: Flux2SamplingParams(),
        ):
            params = SamplingParams.from_user_sampling_params_args(
                "dummy-model",
                server_args=server_args,
                prompt="p",
                image_path="/tmp/in.png",
                diffusers_kwargs=diffusers_kwargs,
            )

        self.assertEqual(params.diffusers_kwargs, diffusers_kwargs)
        self.assertEqual(
            params.build_request_extra()["diffusers_kwargs"],
            diffusers_kwargs,
        )

    def test_dataclasses_replace_preserves_explicit_fields(self):
        """Per-prompt clones retain explicit and model-internal fields."""
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
            _replace_sampling_params_for_prompt,
        )

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

        cloned = _replace_sampling_params_for_prompt(
            sampling_params_orig,
            prompt="new",
            output_file_name=None,
            image_path="/tmp/in2.png",
        )

        explicit = set(cloned.build_request_extra()["explicit_fields"])
        self.assertIn("width", explicit)
        self.assertIn("height", explicit)
        self.assertIn("prompt", explicit)
        self.assertIn("image_path", explicit)

    def test_per_prompt_clone_preserves_glm_image_crop_size(self):
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
            _replace_sampling_params_for_prompt,
        )

        sampling_params_orig = GlmImageSamplingParams(
            prompt="orig",
            width=1024,
            height=1024,
        )
        sampling_params_orig.requested_width = 1000
        sampling_params_orig.requested_height = 999

        cloned = _replace_sampling_params_for_prompt(
            sampling_params_orig,
            prompt="new",
            output_file_name=None,
            image_path=None,
        )

        self.assertEqual((cloned.width, cloned.height), (1024, 1024))
        self.assertEqual((cloned.requested_width, cloned.requested_height), (1000, 999))


# Request sizes fed to every model below; "float" pins how a non-int size is handled.
_SIZE_CASES = {
    "aligned": (1024, 512),
    "unaligned": (1000, 520),
    "below_multiple": (10, 1000),
    "width_none": (None, 520),
    "height_none": (1000, None),
    "both_none": (None, None),
    "zero": (0, 520),
    "negative": (-16, 520),
    "negative_and_none": (-16, None),
    "float": (1000.0, 520),
}
_SIZE_CASES_WITHOUT_NONE = (
    "aligned",
    "unaligned",
    "below_multiple",
    "zero",
    "negative",
)

# Final (width, height, warned) per case.
_UNCHANGED = {name: (*size, False) for name, size in _SIZE_CASES.items()}
# A missing or non-positive side leaves both sides alone.
_KEEP_16 = {
    **_UNCHANGED,
    "unaligned": (992, 512, True),
    "below_multiple": (16, 992, True),
    "float": (992.0, 512, True),
}
# MOSS-Image leaves a non-int size to request validation.
_KEEP_16_INT_ONLY = {**_KEEP_16, "float": (1000.0, 520, False)}
# The missing side is derived from 16:9 first, as InputValidationStage does.
_FILL_16 = {
    **_KEEP_16,
    "width_none": (912, 512, True),
    "height_none": (992, 560, True),
    "negative_and_none": (-16, -9, False),
}
# Each set, positive side is aligned on its own.
_PER_SIDE_16 = {
    **_KEEP_16,
    "width_none": (None, 512, True),
    "height_none": (992, None, True),
    "zero": (0, 512, True),
    "negative": (-16, 512, True),
}
_PER_SIDE_32 = {**_PER_SIDE_16, "below_multiple": (32, 992, True)}


def _size_alignment_models():
    """(row id, label, multiple, sampling params cls, pipeline config, {path: expected})."""
    from sglang.multimodal_gen.configs.pipeline_configs.anima import (
        AnimaPipelineConfig,
    )
    from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
    from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview import (
        Cosmos3MultiviewConfig,
    )
    from sglang.multimodal_gen.configs.pipeline_configs.diffusers_generic import (
        DiffusersGenericPipelineConfig,
    )
    from sglang.multimodal_gen.configs.pipeline_configs.flux3_image import (
        Flux3ImagePipelineConfig,
        Flux3ImageSamplingParams,
    )
    from sglang.multimodal_gen.configs.pipeline_configs.moss_image import (
        MOSSImagePipelineConfig,
    )
    from sglang.multimodal_gen.configs.pipeline_configs.qwen_image21 import (
        QwenImage21PipelineConfig,
    )
    from sglang.multimodal_gen.configs.pipeline_configs.zimage import (
        ZImagePipelineConfig,
    )
    from sglang.multimodal_gen.configs.sample.anima import AnimaSamplingParams
    from sglang.multimodal_gen.configs.sample.cosmos3 import Cosmos3SamplingParams
    from sglang.multimodal_gen.configs.sample.moss_image import (
        MOSSImageSamplingParams,
    )
    from sglang.multimodal_gen.configs.sample.qwenimage21 import (
        QwenImage21SamplingParams,
    )
    from sglang.multimodal_gen.configs.sample.zimage import ZImageSamplingParams

    diffusers = DiffusersGenericPipelineConfig()

    def both(table):
        return {"adjust": table, "warmup": table, "init": _UNCHANGED}

    unchanged = both(_UNCHANGED)
    return [
        (
            "zimage",
            "Z-Image",
            16,
            ZImageSamplingParams,
            ZImagePipelineConfig(),
            both(_FILL_16),
        ),
        ("zimage-diffusers", "Z-Image", 16, ZImageSamplingParams, diffusers, unchanged),
        (
            "anima",
            "Anima",
            16,
            AnimaSamplingParams,
            AnimaPipelineConfig(),
            both(_KEEP_16),
        ),
        ("anima-diffusers", "Anima", 16, AnimaSamplingParams, diffusers, unchanged),
        (
            "moss",
            "MOSS-Image",
            16,
            MOSSImageSamplingParams,
            MOSSImagePipelineConfig(),
            both(_KEEP_16_INT_ONLY),
        ),
        (
            "moss-diffusers",
            "MOSS-Image",
            16,
            MOSSImageSamplingParams,
            diffusers,
            unchanged,
        ),
        (
            "cosmos3",
            "Cosmos3",
            16,
            Cosmos3SamplingParams,
            Cosmos3Config(),
            both(_KEEP_16),
        ),
        # The simulation pipelines subclass Cosmos3Config and size their own canvas.
        (
            "cosmos3-multiview",
            "Cosmos3",
            16,
            Cosmos3SamplingParams,
            Cosmos3MultiviewConfig(),
            unchanged,
        ),
        # Qwen-Image 2.1 aligns at construction and warmup on every backend, not in _adjust.
        (
            "qwen21",
            "Qwen-Image 2.1",
            32,
            QwenImage21SamplingParams,
            QwenImage21PipelineConfig(),
            {"adjust": _UNCHANGED, "warmup": _PER_SIDE_32, "init": _PER_SIDE_32},
        ),
        (
            "qwen21-diffusers",
            "Qwen-Image 2.1",
            32,
            QwenImage21SamplingParams,
            diffusers,
            {"adjust": _UNCHANGED, "warmup": _PER_SIDE_32, "init": _PER_SIDE_32},
        ),
        (
            "flux3",
            "FLUX.3",
            16,
            Flux3ImageSamplingParams,
            Flux3ImagePipelineConfig(),
            both(_PER_SIDE_16),
        ),
        (
            "flux3-diffusers",
            "FLUX.3",
            16,
            Flux3ImageSamplingParams,
            diffusers,
            unchanged,
        ),
    ]


class _WarningCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        message = record.getMessage()
        if "requires dimensions divisible by" in message:
            self.messages.append(message)


class TestRequestSizeAlignment(unittest.TestCase):
    """Characterizes request-size alignment for every model that rounds sizes to a grid.

    Each model rounds an unaligned request down so its DiT does not reject the size
    with an HTTP 500. Covers the _adjust hook, the warmup hook (which overwrites sizes
    after _adjust) and construction, on the native and the non-matching backends.
    """

    def _run(self, path, cls, pipeline_config, size):
        width, height = size
        if path == "init":
            return cls(width=width, height=height)
        params = cls()
        params.width, params.height = width, height
        if path == "adjust":
            server_args = SimpleNamespace(
                pipeline_config=pipeline_config, num_gpus=1, comfyui_mode=True
            )
            params._adjust_visual_fields(server_args, pipeline_config)
        else:
            req = SimpleNamespace(extra={}, image_path=None)
            server_args = SimpleNamespace(pipeline_config=pipeline_config)
            params.prepare_synthetic_warmup_request_for_queue(req, server_args)
        return params

    def test_request_size_alignment(self):
        capture = _WarningCapture()
        root = logging.getLogger("sglang.multimodal_gen")
        root.addHandler(capture)
        self.addCleanup(root.removeHandler, capture)
        for (
            row_id,
            label,
            multiple,
            cls,
            pipeline_config,
            expected_by_path,
        ) in _size_alignment_models():
            for path, expected_table in expected_by_path.items():
                case_names = _SIZE_CASES_WITHOUT_NONE if path == "init" else _SIZE_CASES
                for case in case_names:
                    with self.subTest(model=row_id, path=path, case=case):
                        capture.messages.clear()
                        params = self._run(
                            path, cls, pipeline_config, _SIZE_CASES[case]
                        )
                        width, height, warned = expected_table[case]
                        defaults = cls().supported_resolutions
                        if path == "adjust" and case == "both_none" and defaults:
                            # _adjust fills an unset size from the first supported one.
                            width, height = defaults[0]
                        actual = (params.width, params.height)
                        self.assertEqual(actual, (width, height))
                        self.assertEqual(
                            [type(v) for v in actual], [type(width), type(height)]
                        )
                        if not warned:
                            self.assertEqual(capture.messages, [])
                            continue
                        requested = _SIZE_CASES[case]
                        self.assertEqual(
                            capture.messages,
                            [
                                f"{label} requires dimensions divisible by {multiple}; "
                                f"adjusted requested resolution from "
                                f"{requested[0]}x{requested[1]} to {width}x{height}"
                            ],
                        )


if __name__ == "__main__":
    unittest.main()
