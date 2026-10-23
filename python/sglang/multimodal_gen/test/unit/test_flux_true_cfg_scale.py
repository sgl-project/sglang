import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.configs.pipeline_configs.flux import (
    Flux2PipelineConfig,
    FluxPipelineConfig,
)


class TestFluxTrueCFGScale(unittest.TestCase):
    def test_true_cfg_scale_is_the_cfg_weight(self):
        for config in (FluxPipelineConfig(), Flux2PipelineConfig()):
            with self.subTest(config=type(config).__name__):
                batch = SimpleNamespace(true_cfg_scale=4.0)
                self.assertEqual(
                    config.get_classifier_free_guidance_scale(batch, 3.5), 4.0
                )

    def test_guidance_scale_is_the_cfg_weight_without_true_cfg_scale(self):
        batch = SimpleNamespace(true_cfg_scale=None)
        self.assertEqual(
            FluxPipelineConfig().get_classifier_free_guidance_scale(batch, 3.5), 3.5
        )


if __name__ == "__main__":
    unittest.main()
