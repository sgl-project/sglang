import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import transformers

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.longcat_image import (
    LongCatPromptRewriteStage,
)


class TestLongCatImageResidency(unittest.TestCase):
    def test_text_encoder_initial_device_uses_effective_residency(self):
        accelerator = torch.device("cuda", 0)

        for stage_on_cpu, expected_device in (
            (True, torch.device("cpu")),
            (False, accelerator),
        ):
            with self.subTest(stage_on_cpu=stage_on_cpu):
                server_args = SimpleNamespace(
                    should_load_component_on_cpu=MagicMock(return_value=stage_on_cpu)
                )
                text_encoder = MagicMock()
                text_encoder.to.return_value = text_encoder

                with (
                    patch(
                        "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args",
                        return_value=server_args,
                    ),
                    patch(
                        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.longcat_image.get_local_torch_device",
                        return_value=accelerator,
                    ),
                    patch.object(
                        transformers.Qwen2_5_VLForConditionalGeneration,
                        "from_pretrained",
                        return_value=text_encoder,
                    ),
                ):
                    stage = LongCatPromptRewriteStage(
                        tokenizer=object(),
                        text_processor=object(),
                        model_path="model",
                        text_encoder_dtype=torch.bfloat16,
                    )

                self.assertIs(stage.text_encoder, text_encoder)
                server_args.should_load_component_on_cpu.assert_called_once_with(
                    "text_encoder"
                )
                self.assertEqual(
                    text_encoder.to.call_args_list[0].args, (expected_device,)
                )


if __name__ == "__main__":
    unittest.main()
