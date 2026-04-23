import argparse
import unittest

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig
from sglang.multimodal_gen.configs.sample.ltx_2 import LTX23HQSamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.registry import get_pipeline_config_classes
from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import (
    LTX2TwoStageHQPipeline,
)
from sglang.multimodal_gen.runtime.server_args import is_ltx2_two_stage_pipeline_name


class TestLTX23HQPipeline(unittest.TestCase):
    def test_hq_pipeline_registration_and_sampling_params(self):
        config_classes = get_pipeline_config_classes("LTX2TwoStageHQPipeline")

        self.assertEqual(config_classes, (LTX2PipelineConfig, LTX23HQSamplingParams))
        self.assertTrue(is_ltx2_two_stage_pipeline_name("LTX2TwoStageHQPipeline"))
        self.assertEqual(LTX2TwoStageHQPipeline.STAGE_1_DENOISING_SAMPLER_NAME, "res2s")
        self.assertEqual(LTX2TwoStageHQPipeline.STAGE_2_DENOISING_SAMPLER_NAME, "res2s")

        params = LTX23HQSamplingParams(width=768, height=512, num_frames=121)
        extra = params.build_request_extra()
        self.assertEqual(extra["ltx2_distilled_lora_strength_stage_1"], 0.25)
        self.assertEqual(extra["ltx2_distilled_lora_strength_stage_2"], 0.5)
        self.assertEqual(extra["ltx2_stage1_guider_params"]["video_stg_scale"], 0.0)
        self.assertEqual(extra["ltx2_stage1_guider_params"]["audio_stg_blocks"], [])

        parser = argparse.ArgumentParser()
        SamplingParams.add_cli_args(parser)
        args = parser.parse_args(
            [
                "--video-stg-guidance-scale",
                "0.25",
                "--audio-stg-blocks",
                "2",
                "4",
                "--distilled-lora-strength-stage-1",
                "0.125",
            ]
        )
        kwargs = LTX23HQSamplingParams.get_cli_args(args)
        self.assertEqual(kwargs["video_stg_scale"], 0.25)
        self.assertEqual(kwargs["audio_stg_blocks"], [2, 4])
        self.assertEqual(kwargs["distilled_lora_strength_stage_1"], 0.125)


if __name__ == "__main__":
    unittest.main()
