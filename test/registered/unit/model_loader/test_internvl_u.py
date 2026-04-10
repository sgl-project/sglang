import unittest
from unittest.mock import patch

import torch

from sglang.srt.models.internvl import InternVLChatModel
from sglang.srt.models.internvl_u import InternVLUChatModel


class TestInternVLUChatModel(unittest.TestCase):
    def test_load_weights_skips_special_token_embedding(self):
        model = InternVLUChatModel.__new__(InternVLUChatModel)
        weights = [
            ("language_model.model.embed_tokens.weight", torch.ones(1)),
            ("special_token_embedding.weight", torch.zeros(1)),
            ("vision_model.embeddings.class_embedding", torch.full((1,), 2.0)),
        ]

        with patch.object(
            InternVLChatModel, "load_weights", autospec=True
        ) as mock_load_weights:
            InternVLUChatModel.load_weights(model, weights)

        self.assertEqual(mock_load_weights.call_count, 1)
        forwarded_weights = list(mock_load_weights.call_args.args[1])
        forwarded_names = [name for name, _ in forwarded_weights]

        self.assertEqual(
            forwarded_names,
            [
                "language_model.model.embed_tokens.weight",
                "vision_model.embeddings.class_embedding",
            ],
        )
        self.assertNotIn("special_token_embedding.weight", forwarded_names)


if __name__ == "__main__":
    unittest.main()
