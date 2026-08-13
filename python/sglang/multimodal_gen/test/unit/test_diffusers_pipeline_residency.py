import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline import (
    DiffusersPipeline,
)


class TestDiffusersPipelineResidency(unittest.TestCase):
    def test_effective_residency_overrides_legacy_offload_flags(self):
        server_args = SimpleNamespace(
            trust_remote_code=False,
            revision=None,
            pipeline_config=SimpleNamespace(quantization_config=None),
            dit_cpu_offload=True,
            text_encoder_cpu_offload=False,
            image_encoder_cpu_offload=False,
            vae_cpu_offload=False,
            should_cpu_offload_component=MagicMock(return_value=False),
        )
        diffusers_pipe = MagicMock()
        diffusers_pipe.to.return_value = diffusers_pipe
        pipeline = DiffusersPipeline.__new__(DiffusersPipeline)
        pipeline._get_dtype = MagicMock(return_value=torch.float16)
        pipeline._apply_vae_optimizations = MagicMock()
        pipeline._apply_attention_backend = MagicMock()
        pipeline._apply_cache_dit = MagicMock(return_value=diffusers_pipe)
        pipeline._apply_torch_compile = MagicMock(return_value=diffusers_pipe)

        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline.maybe_download_model",
                return_value="/model",
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline.DiffusionPipeline.from_pretrained",
                return_value=diffusers_pipe,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline.get_local_torch_device",
                return_value=torch.device("cpu"),
            ),
        ):
            loaded_pipe = pipeline._load_diffusers_pipeline("model", server_args)

        self.assertIs(loaded_pipe, diffusers_pipe)
        diffusers_pipe.enable_model_cpu_offload.assert_not_called()
        diffusers_pipe.to.assert_called_once_with(torch.device("cpu"))
        server_args.should_cpu_offload_component.assert_has_calls(
            [
                call("transformer"),
                call("text_encoder"),
                call("image_encoder"),
                call("vae"),
            ]
        )


if __name__ == "__main__":
    unittest.main()
