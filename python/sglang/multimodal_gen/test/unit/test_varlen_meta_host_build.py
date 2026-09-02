"""Host-built varlen metadata must match the mask-based (nonzero) builder."""

import unittest

import torch

from sglang.multimodal_gen.runtime.layers.attention.layer import (
    build_varlen_mask_meta,
    build_varlen_mask_meta_from_ranges,
)


class TestHostVarlenMetaEquivalence(unittest.TestCase):
    def test_prefix_text_plus_full_image_matches_nonzero_builder(self):
        txt_len, img_len = 7, 5
        txt_seq_lens = [3, 7, 0]
        bs = len(txt_seq_lens)
        mask = torch.zeros(bs, txt_len + img_len, dtype=torch.bool)
        for row, n in enumerate(txt_seq_lens):
            mask[row, :n] = True
            mask[row, txt_len:] = True

        ref = build_varlen_mask_meta(mask)
        host = build_varlen_mask_meta_from_ranges(
            [[(0, n), (txt_len, txt_len + img_len)] for n in txt_seq_lens],
            max_seqlen=txt_len + img_len,
            device=mask.device,
        )

        for key in ("cu_seqlens", "indices", "inv_indices"):
            torch.testing.assert_close(host[key], ref[key], rtol=0, atol=0)
        self.assertEqual(host["max_seqlen"], ref["max_seqlen"])


if __name__ == "__main__":
    unittest.main()
