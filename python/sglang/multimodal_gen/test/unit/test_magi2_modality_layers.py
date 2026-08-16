# SPDX-License-Identifier: Apache-2.0
import unittest

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.dits.magi2_common import (
    Magi2ModalityLinear,
    Magi2ModalityRMSNorm,
)

NUM_MODALITY = 3


def _modality_ids(num_tokens, *, seed=0):
    torch.manual_seed(seed)
    ids = torch.randint(0, NUM_MODALITY, (num_tokens,))
    ids[:NUM_MODALITY] = torch.arange(NUM_MODALITY)
    return ids


class TestMagi2ModalityLinear(unittest.TestCase):
    def test_matches_a_per_row_loop_over_each_rows_own_weight_block(self):
        in_features, out_features, num_tokens = 8, 5, 12
        layer = Magi2ModalityLinear(
            in_features, out_features, num_modality=NUM_MODALITY
        ).requires_grad_(False)
        with torch.no_grad():
            layer.weight.copy_(torch.randn(NUM_MODALITY * out_features, in_features))

        x = torch.randn(num_tokens, in_features)
        ids = _modality_ids(num_tokens)
        blocks = layer.weight.view(NUM_MODALITY, out_features, in_features)
        want = torch.stack(
            [F.linear(x[row], blocks[int(ids[row])]) for row in range(num_tokens)]
        )

        self.assertTrue(torch.allclose(layer(x, ids), want, atol=1e-6))

    def test_different_modalities_use_different_weights(self):
        layer = Magi2ModalityLinear(4, 4, num_modality=NUM_MODALITY)
        with torch.no_grad():
            layer.weight.copy_(torch.randn(NUM_MODALITY * 4, 4))

        row = torch.randn(1, 4)
        x = row.repeat(NUM_MODALITY, 1)
        out = layer(x, torch.arange(NUM_MODALITY))
        for modality in range(1, NUM_MODALITY):
            self.assertFalse(torch.equal(out[0], out[modality]))


class TestMagi2ModalityRMSNorm(unittest.TestCase):
    def test_broadcasts_per_token_weight_over_a_qk_norm_head_axis(self):
        tokens, heads, head_dim = 10, 4, 8
        norm = Magi2ModalityRMSNorm(head_dim, num_modality=NUM_MODALITY)
        with torch.no_grad():
            norm.weight.copy_(torch.randn(NUM_MODALITY * head_dim) * 0.3 + 1.0)

        x = torch.randn(tokens, heads, head_dim)
        ids = _modality_ids(tokens)
        out = norm(x, ids)
        self.assertEqual(out.shape, x.shape)

        blocks = norm.weight.detach().view(NUM_MODALITY, head_dim)
        for token in range(tokens):
            row = x[token].float()
            row = row * torch.rsqrt(row.pow(2).mean(-1, keepdim=True) + norm.eps)
            want = row * blocks[int(ids[token])]
            self.assertTrue(torch.allclose(out[token], want, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
