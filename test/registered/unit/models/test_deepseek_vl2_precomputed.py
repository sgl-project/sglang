import unittest

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
)
from sglang.srt.models.deepseek_vl2 import DeepseekVL2ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepseekVL2PrecomputedEmbedding(CustomTestCase):
    def test_precomputed_embeddings_bypass_vision_encoder(self) -> None:
        features = [
            torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
            torch.arange(12, 18, dtype=torch.float32).reshape(2, 3),
        ]
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
                feature=feature,
            )
            for feature in features
        ]
        model = DeepseekVL2ForCausalLM.__new__(DeepseekVL2ForCausalLM)

        image_features = model.get_image_feature(items)

        torch.testing.assert_close(
            image_features,
            torch.cat([feature.reshape(-1, feature.shape[-1]) for feature in features]),
        )


if __name__ == "__main__":
    unittest.main()
