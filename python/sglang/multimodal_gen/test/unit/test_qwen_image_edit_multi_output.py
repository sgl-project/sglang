import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageEditPipelineConfig,
    QwenImageEditPlus_2511_PipelineConfig,
    QwenImageEditPlusPipelineConfig,
    QwenImageLayeredPipelineConfig,
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)


def _sequential_requests(config, num_outputs):
    req = Req(
        sampling_params=SamplingParams(
            request_id="rid",
            prompt="edit",
            num_outputs_per_prompt=num_outputs,
            seed=0,
            generator_device="cpu",
        )
    )
    server_args = SimpleNamespace(pipeline_config=config)
    return list(InputValidationStage().iter_sequential_requests(req, server_args))


class TestQwenImageEditMultipleOutputs(unittest.TestCase):
    def test_edit_outputs_run_as_single_sample_requests(self):
        for config_cls in (
            QwenImageEditPipelineConfig,
            QwenImageEditPlusPipelineConfig,
            QwenImageEditPlus_2511_PipelineConfig,
        ):
            with self.subTest(config_cls.__name__):
                outputs = _sequential_requests(config_cls(), 2)
                self.assertEqual([r.num_outputs_per_prompt for r in outputs], [1, 1])
                self.assertEqual([r.seeds for r in outputs], [[0], [1]])

    def test_text_to_image_and_layered_keep_batched_outputs(self):
        for config_cls in (QwenImagePipelineConfig, QwenImageLayeredPipelineConfig):
            with self.subTest(config_cls.__name__):
                config = config_cls()
                self.assertFalse(config.supports_sequential_multi_output_inference())
                outputs = _sequential_requests(config, 2)
                self.assertEqual([r.num_outputs_per_prompt for r in outputs], [2])


if __name__ == "__main__":
    unittest.main()
