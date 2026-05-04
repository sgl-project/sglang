# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang.multimodal_gen.configs.pipeline_configs.ug import UGPipelineConfig
from sglang.multimodal_gen.configs.sample.ug import UGSamplingParams
from sglang.multimodal_gen.registry import (
    _get_config_info,
    get_model_info,
    get_non_diffusers_pipeline_name,
)
from sglang.multimodal_gen.runtime.pipelines.ug import UGPipeline


class TestUGRegistry(unittest.TestCase):
    def setUp(self):
        _get_config_info.cache_clear()
        get_model_info.cache_clear()

    def test_fake_ug_resolves_without_model_index(self):
        info = get_model_info("sglang-internal/fake-ug", backend="sglang")

        self.assertIsNotNone(info)
        self.assertIs(info.pipeline_cls, UGPipeline)
        self.assertIs(info.pipeline_config_cls, UGPipelineConfig)
        self.assertIs(info.sampling_param_cls, UGSamplingParams)

    def test_bagel_resolves_without_model_index(self):
        for model_path in (
            "sglang-internal/mock-bagel",
            "ByteDance-Seed/BAGEL-7B-MoT",
        ):
            with self.subTest(model_path=model_path):
                info = get_model_info(model_path, backend="sglang")

                self.assertIsNotNone(info)
                self.assertIs(info.pipeline_cls, UGPipeline)
                self.assertIs(info.pipeline_config_cls, UGPipelineConfig)
                self.assertIs(info.sampling_param_cls, UGSamplingParams)

    def test_fake_ug_resolves_to_ug_pipeline_name(self):
        self.assertEqual(
            get_non_diffusers_pipeline_name("sglang-internal/fake-ug"),
            "UGPipeline",
        )

    def test_bagel_resolves_to_ug_pipeline_name(self):
        self.assertEqual(
            get_non_diffusers_pipeline_name("ByteDance-Seed/BAGEL-7B-MoT"),
            "UGPipeline",
        )


if __name__ == "__main__":
    unittest.main()
