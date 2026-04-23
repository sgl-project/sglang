import unittest

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig
from sglang.multimodal_gen.configs.sample.ltx_2 import LTX23HQSamplingParams
from sglang.multimodal_gen.registry import get_pipeline_config_classes
from sglang.multimodal_gen.runtime.server_args import is_ltx2_two_stage_pipeline_name


class TestLTX23HQPipeline(unittest.TestCase):
    def test_hq_pipeline_registration_and_sampling_params(self):
        config_classes = get_pipeline_config_classes("LTX2TwoStageHQPipeline")

        self.assertEqual(config_classes, (LTX2PipelineConfig, LTX23HQSamplingParams))
        self.assertTrue(is_ltx2_two_stage_pipeline_name("LTX2TwoStageHQPipeline"))

        params = LTX23HQSamplingParams(width=768, height=512, num_frames=121)
        extra = params.build_request_extra()
        self.assertEqual(extra["ltx2_distilled_lora_strength_stage_1"], 0.25)
        self.assertEqual(extra["ltx2_distilled_lora_strength_stage_2"], 0.5)
        self.assertEqual(extra["ltx2_stage1_guider_params"]["video_stg_scale"], 0.0)
        self.assertEqual(extra["ltx2_stage1_guider_params"]["audio_stg_blocks"], [])


if __name__ == "__main__":
    unittest.main()
