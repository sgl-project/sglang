import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.configs.pipeline_configs.zimage import (
    ZImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)


class TestZImageMultipleOutputs(unittest.TestCase):
    def test_outputs_run_as_single_sample_requests(self):
        # The Z-Image DiT pairs one caption embedding with each latent, so n > 1
        # must not reach it as one batch of latents against a single caption.
        req = Req(
            sampling_params=SamplingParams(
                request_id="rid",
                prompt="a cat",
                num_outputs_per_prompt=2,
                seed=0,
                generator_device="cpu",
            )
        )
        server_args = SimpleNamespace(pipeline_config=ZImagePipelineConfig())
        outputs = list(
            InputValidationStage().iter_sequential_requests(req, server_args)
        )
        self.assertEqual([r.num_outputs_per_prompt for r in outputs], [1, 1])
        self.assertEqual([r.seeds for r in outputs], [[0], [1]])


if __name__ == "__main__":
    unittest.main()
