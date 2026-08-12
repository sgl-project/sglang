"""Unit tests for independent diffusion auto-residency flags."""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@contextmanager
def _high_memory_cuda_platform():
    with (
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.is_cpu",
            return_value=False,
        ),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.is_mps",
            return_value=False,
        ),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.is_cuda",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.get_device_total_memory",
            return_value=80 * 1024**3,
        ),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.get_available_gpu_memory",
            return_value=80,
        ),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.enable_dit_layerwise_offload_by_default",
            return_value=True,
        ),
        patch.object(
            PipelineConfig,
            "from_kwargs",
            return_value=QwenImagePipelineConfig(),
        ),
    ):
        yield


class TestDiffusionAutoResidency(CustomTestCase):
    def test_explicit_layerwise_false_keeps_independent_auto_residency(self):
        with _high_memory_cuda_platform():
            args = ServerArgs.from_dict(
                {
                    "model_path": "/fake/model",
                    "performance_mode": "auto",
                    "dit_layerwise_offload": False,
                }
            )

        self.assertFalse(args.dit_layerwise_offload)
        self.assertFalse(args.dit_cpu_offload)

    def test_explicit_dit_cpu_offload_is_still_preserved(self):
        with _high_memory_cuda_platform():
            args = ServerArgs.from_dict(
                {
                    "model_path": "/fake/model",
                    "performance_mode": "auto",
                    "dit_layerwise_offload": False,
                    "dit_cpu_offload": True,
                }
            )

        self.assertFalse(args.dit_layerwise_offload)
        self.assertTrue(args.dit_cpu_offload)

    def test_explicit_layerwise_true_preserves_initial_dit_cpu_residency(self):
        with _high_memory_cuda_platform():
            args = ServerArgs.from_dict(
                {
                    "model_path": "/fake/model",
                    "performance_mode": "auto",
                    "dit_layerwise_offload": True,
                }
            )

        self.assertTrue(args.dit_layerwise_offload)
        self.assertTrue(args.dit_cpu_offload)

    def test_explicit_vae_cpu_offload_is_preserved_independently(self):
        with _high_memory_cuda_platform():
            args = ServerArgs.from_dict(
                {
                    "model_path": "/fake/model",
                    "performance_mode": "auto",
                    "dit_layerwise_offload": False,
                    "vae_cpu_offload": True,
                }
            )

        self.assertFalse(args.dit_cpu_offload)
        self.assertTrue(args.vae_cpu_offload)

    def test_explicit_unified_cpu_offload_selection_is_preserved(self):
        with _high_memory_cuda_platform():
            args = ServerArgs.from_dict(
                {
                    "model_path": "/fake/model",
                    "performance_mode": "auto",
                    "cpu_offload_components": ["dit", "vae"],
                }
            )

        self.assertTrue(args.dit_cpu_offload)
        self.assertTrue(args.vae_cpu_offload)

    def test_explicit_dit_layerwise_component_preserves_initial_residency(self):
        with _high_memory_cuda_platform():
            args = ServerArgs.from_dict(
                {
                    "model_path": "/fake/model",
                    "performance_mode": "auto",
                    "layerwise_offload_components": ["dit"],
                }
            )

        self.assertTrue(args.dit_cpu_offload)
        self.assertEqual(args.layerwise_offload_components, ["dit"])


if __name__ == "__main__":
    unittest.main()
