import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalInputFormat
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.phi4mm import Phi4MMMultimodalProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPhi4MMProcessorOutput(unittest.TestCase):
    @staticmethod
    def _make_processor() -> Phi4MMMultimodalProcessor:
        processor = object.__new__(Phi4MMMultimodalProcessor)
        processor.ATTR_NAME_TO_MODALITY = {
            "pixel_values": Modality.IMAGE,
            "audio_features": Modality.AUDIO,
            "audio_feature_lens": Modality.AUDIO,
        }
        processor.FEATURE_NAMES = ["pixel_values", "audio_features"]
        processor._processor = object()
        processor._tokenizer = lambda *args, **kwargs: SimpleNamespace(
            input_ids=torch.tensor([200010, 200011])
        )
        processor.use_cuda_ipc = False
        processor.precompute_hash_before_cpu_transfer = False
        return processor

    def test_offline_processor_output_uses_hf_key_renames(self) -> None:
        image_features = torch.arange(4).reshape(1, 4)
        audio_features = torch.arange(6).reshape(1, 6)
        audio_feature_lens = torch.tensor([6])
        processor_output = {
            "format": "processor_output",
            "input_ids": [200010, 200011],
            "input_image_embeds": image_features,
            "input_audio_embeds": audio_features,
            "audio_embed_sizes": audio_feature_lens,
        }
        processor = self._make_processor()

        items, input_ids, _ = processor.process_and_combine_mm_data(
            base_output=BaseMultiModalProcessorOutput(
                input_text="",
                images=[processor_output],
            ),
            mm_tokens=MultimodalSpecialTokens(
                image_token_id=200010,
                audio_token_id=200011,
            ),
        )
        items_by_modality = {item.modality: item for item in items}
        self.assertEqual(input_ids.tolist(), [200010, 200011])
        self.assertEqual(
            items_by_modality[Modality.IMAGE].format,
            MultimodalInputFormat.PROCESSOR_OUTPUT,
        )
        self.assertIs(items_by_modality[Modality.IMAGE].feature, image_features)
        self.assertIs(items_by_modality[Modality.AUDIO].feature, audio_features)
        self.assertIs(
            items_by_modality[Modality.AUDIO].audio_feature_lens,
            audio_feature_lens,
        )


if __name__ == "__main__":
    unittest.main()
