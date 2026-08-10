"""Z-Image BCG must capture at the native caption length, not a text bucket.

Regression for the numeric half of sgl-project/sglang#34183: Z-Image attends
its caption slots UNMASKED — the pipeline pads captions to the native length
(a multiple of 32) with learned pad-token embeddings that act as attended
registers, and the DiT derives the attention span from the full padded tensor
(``lens == target`` → no mask). Padding further to a BCG text bucket changes
how many registers every token attends: the first caption self-attention
block already differs materially (maxdiff ≈ 6 on the valid rows, PSNR ≈ 21 dB
after 9 distilled steps). Capturing at the incoming (native) length is the
only bit-exact choice, so the padder must never extend the caption.
"""

import unittest

import torch

from sglang.multimodal_gen.runtime.breakable_cuda_graph.model_padders.zimage import (
    pad_zimage_prompt_kwargs,
)


class _FakeRotary:
    def __call__(self, pos_ids):
        n = pos_ids.shape[0]
        return torch.ones(n, 8), torch.zeros(n, 8)


class _FakeZImageTransformer2DModel:
    # transformer_class_name_matches() checks the class NAME for "zimage".
    rotary_emb = _FakeRotary()


_FakeZImageTransformer2DModel.__name__ = "ZImageTransformer2DModel"


class TestZImageNativeLengthCapture(unittest.TestCase):
    def test_padder_never_extends_the_caption(self):
        nat_len, hidden = 32, 16
        caption = torch.randn(nat_len, hidden)
        cap_freqs = (torch.ones(nat_len, 8), torch.zeros(nat_len, 8))
        img_freqs = (torch.ones(64, 8), torch.zeros(64, 8))
        kwargs = {
            "hidden_states": torch.randn(1, 4, 8, 8),
            "timestep": torch.zeros(1),
            "encoder_hidden_states": [caption],
            "freqs_cis": (cap_freqs, img_freqs),
        }
        out = pad_zimage_prompt_kwargs(
            dict(kwargs), _FakeZImageTransformer2DModel(), buckets=(64, 128)
        )

        padded_caption = out["encoder_hidden_states"][0]
        # The caption keeps its native length: buckets larger than the
        # incoming length must not add attended pad registers.
        self.assertEqual(padded_caption.shape[0], nat_len)
        self.assertTrue(torch.equal(padded_caption, caption))
        cap_cos = out["freqs_cis"][0][0]
        self.assertEqual(cap_cos.shape[0], nat_len)
        self.assertEqual(
            out["caption_valid_lens"].tolist(), [nat_len]
        )  # mask covers exactly the native slots


if __name__ == "__main__":
    unittest.main()
