import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.cli.utils import get_is_diffusion_model
from sglang.multimodal_gen.configs.pipeline_configs.bagel import (
    BagelEditPipelineConfig,
    BagelPipelineConfig,
    BagelThinkingPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.bagel import (
    BagelEditSamplingParams,
    BagelSamplingParams,
    BagelThinkingSamplingParams,
)
from sglang.multimodal_gen.registry import (
    _get_config_info,
    get_model_info,
    get_non_diffusers_pipeline_name,
    get_pipeline_config_classes,
    has_registered_diffusion_model_path,
    is_known_non_diffusers_multimodal_model,
)
from sglang.utils import (
    BAGEL_LOCAL_CHECKPOINT_MARKERS,
    BAGEL_MODEL_ID,
    BAGEL_PIPELINE_NAME,
    is_known_non_diffusers_diffusion_model,
    resolve_non_diffusers_diffusion_pipeline,
)


class TestBagelNonDiffusersResolution(unittest.TestCase):
    """Verify BAGEL routing without a Diffusers model_index.json."""

    def test_canonical_id_and_hf_cache_path_resolve(self):
        cache_path = (
            "/cache/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/5019f57d"
        )

        for model_path in (BAGEL_MODEL_ID, cache_path):
            with self.subTest(model_path=model_path):
                self.assertEqual(
                    resolve_non_diffusers_diffusion_pipeline(model_path),
                    BAGEL_PIPELINE_NAME,
                )
                self.assertTrue(is_known_non_diffusers_diffusion_model(model_path))
                self.assertTrue(is_known_non_diffusers_multimodal_model(model_path))
                self.assertTrue(has_registered_diffusion_model_path(model_path))
                self.assertEqual(
                    get_non_diffusers_pipeline_name(model_path), BAGEL_PIPELINE_NAME
                )

    def test_arbitrary_local_directory_requires_all_markers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "arbitrary-checkpoint-name"
            checkpoint_dir.mkdir()

            for marker in BAGEL_LOCAL_CHECKPOINT_MARKERS:
                (checkpoint_dir / marker).touch()

            self.assertEqual(
                resolve_non_diffusers_diffusion_pipeline(str(checkpoint_dir)),
                BAGEL_PIPELINE_NAME,
            )
            self.assertTrue(get_is_diffusion_model(str(checkpoint_dir)))

    def test_each_partial_marker_set_is_rejected(self):
        for missing_marker in BAGEL_LOCAL_CHECKPOINT_MARKERS:
            with self.subTest(missing_marker=missing_marker):
                with tempfile.TemporaryDirectory() as tmpdir:
                    checkpoint_dir = Path(tmpdir) / "BAGEL-7B-MoT"
                    checkpoint_dir.mkdir()
                    for marker in BAGEL_LOCAL_CHECKPOINT_MARKERS - {missing_marker}:
                        (checkpoint_dir / marker).touch()

                    self.assertIsNone(
                        resolve_non_diffusers_diffusion_pipeline(str(checkpoint_dir))
                    )
                    self.assertFalse(get_is_diffusion_model(str(checkpoint_dir)))
                    self.assertFalse(
                        has_registered_diffusion_model_path(str(checkpoint_dir))
                    )

    def test_similar_names_do_not_create_false_positives(self):
        for model_path in (
            "/models/BAGEL-7B-MoT",
            "/cache/models--ByteDance-Seed--BAGEL-7B-MoT-extra/snapshots/rev",
            "someone-else/BAGEL-7B-MoT",
        ):
            with self.subTest(model_path=model_path):
                self.assertIsNone(resolve_non_diffusers_diffusion_pipeline(model_path))


class TestBagelRegistryConfig(unittest.TestCase):
    """Verify official and marker-based paths share the BAGEL config classes."""

    def tearDown(self):
        _get_config_info.cache_clear()
        get_model_info.cache_clear()

    def test_official_model_id_is_registered(self):
        config_info = _get_config_info(BAGEL_MODEL_ID)

        self.assertIsNotNone(config_info)
        self.assertIs(config_info.pipeline_config_cls, BagelPipelineConfig)
        self.assertIs(config_info.sampling_param_cls, BagelSamplingParams)

    def test_official_model_resolves_native_pipeline(self):
        model_info = get_model_info(BAGEL_MODEL_ID, backend="sglang")

        self.assertIsNotNone(model_info)
        self.assertEqual(model_info.pipeline_cls.__name__, BAGEL_PIPELINE_NAME)
        self.assertIs(model_info.pipeline_config_cls, BagelPipelineConfig)
        self.assertIs(model_info.sampling_param_cls, BagelSamplingParams)

    def test_local_marker_resolution_does_not_read_model_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "downloaded-checkpoint"
            checkpoint_dir.mkdir()
            for marker in BAGEL_LOCAL_CHECKPOINT_MARKERS:
                (checkpoint_dir / marker).touch()

            with patch(
                "sglang.multimodal_gen.registry.maybe_download_model_index",
                side_effect=AssertionError("BAGEL must not read model_index.json"),
            ):
                model_info = get_model_info(str(checkpoint_dir), backend="sglang")

        self.assertIsNotNone(model_info)
        self.assertEqual(model_info.pipeline_cls.__name__, BAGEL_PIPELINE_NAME)
        self.assertIs(model_info.pipeline_config_cls, BagelPipelineConfig)
        self.assertIs(model_info.sampling_param_cls, BagelSamplingParams)

    def test_explicit_editing_pipeline_registers_its_config_classes(self):
        config_classes = get_pipeline_config_classes("BagelEditPipeline")

        self.assertIsNotNone(config_classes)
        self.assertIs(config_classes[0], BagelEditPipelineConfig)
        self.assertIs(config_classes[1], BagelEditSamplingParams)

    def test_explicit_thinking_pipeline_registers_its_config_classes(self):
        config_classes = get_pipeline_config_classes("BagelThinkingPipeline")

        self.assertIsNotNone(config_classes)
        self.assertIs(config_classes[0], BagelThinkingPipelineConfig)
        self.assertIs(config_classes[1], BagelThinkingSamplingParams)


if __name__ == "__main__":
    unittest.main()
