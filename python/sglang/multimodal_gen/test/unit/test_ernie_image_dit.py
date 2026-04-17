import unittest

import torch

from sglang.multimodal_gen.runtime.models.dits.ernie_image import (
    EmbedND3,
    ErnieImageTransformer2DModel,
)


class TestErnieImageDiT(unittest.TestCase):
    def _make_model(self):
        model = ErnieImageTransformer2DModel.__new__(ErnieImageTransformer2DModel)
        torch.nn.Module.__init__(model)
        model.pos_embed = EmbedND3(dim=64, theta=10000, axes_dim=(16, 56, 56))
        model._rotary_pos_emb_cache = {}
        model.eval()
        return model

    def test_rotary_pos_emb_cache_reuses_same_shape(self):
        model = self._make_model()
        device = torch.device("cpu")

        first = model._get_rotary_pos_emb(1, 2, 3, 4, device)
        second = model._get_rotary_pos_emb(1, 2, 3, 4, device)
        expected = model._build_rotary_pos_emb(1, 2, 3, 4, device)

        self.assertIs(first, second)
        self.assertEqual(len(model._rotary_pos_emb_cache), 1)
        torch.testing.assert_close(first, expected)

    def test_rotary_pos_emb_cache_keys_include_text_length(self):
        model = self._make_model()
        device = torch.device("cpu")

        first = model._get_rotary_pos_emb(1, 2, 3, 4, device)
        second = model._get_rotary_pos_emb(1, 2, 3, 5, device)

        self.assertIsNot(first, second)
        self.assertEqual(len(model._rotary_pos_emb_cache), 2)


if __name__ == "__main__":
    unittest.main()
