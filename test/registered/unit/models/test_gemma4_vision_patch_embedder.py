import unittest

import torch
import torch.nn.functional as F
from transformers import Gemma4VisionConfig

from sglang.srt.models.gemma4_vision import Gemma4VisionPatchEmbedder
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _ref_position_embeddings(
    table: torch.Tensor,
    patch_positions: torch.Tensor,
    padding_positions: torch.Tensor,
) -> torch.Tensor:
    clamped = patch_positions.clamp(min=0)
    one_hot = F.one_hot(clamped.long(), num_classes=table.shape[1])
    one_hot = one_hot.permute(0, 2, 1, 3).to(table)
    position_embeddings = (one_hot @ table).sum(dim=1)
    return torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)


def _make_grid_positions(grid_w: int, grid_h: int, total_patches: int) -> torch.Tensor:
    xs = torch.arange(grid_w).repeat(grid_h)
    ys = torch.arange(grid_h).repeat_interleave(grid_w)
    pos = torch.stack([xs, ys], dim=-1)
    pad = total_patches - pos.shape[0]
    if pad > 0:
        pos = torch.cat([pos, torch.full((pad, 2), -1, dtype=pos.dtype)], dim=0)
    return pos


class TestGemma4VisionPatchEmbedder(CustomTestCase):
    @torch.no_grad()
    def test_position_embeddings_matches_one_hot(self):
        torch.manual_seed(0)
        config = Gemma4VisionConfig(
            hidden_size=16,
            num_hidden_layers=0,
            patch_size=1,
            position_embedding_size=64,
        )
        embedder = Gemma4VisionPatchEmbedder(config)
        base_table = torch.randn_like(embedder.position_embedding_table)

        pos_batch = torch.stack(
            [
                _make_grid_positions(6, 3, 27),
                _make_grid_positions(3, 6, 27),
                torch.full((27, 2), -1, dtype=torch.int64),
            ],
            dim=0,
        )
        padding = (pos_batch == -1).all(dim=-1)

        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            embedder.float()
            embedder.position_embedding_table.copy_(base_table)
            embedder.to(dtype=dtype)
            for pos_dtype in (torch.int64, torch.int32):
                with self.subTest(dtype=dtype, pos_dtype=pos_dtype):
                    pos = pos_batch.to(dtype=pos_dtype)
                    actual = embedder._position_embeddings(pos, padding)
                    expected = _ref_position_embeddings(
                        embedder.position_embedding_table, pos, padding
                    )
                    self.assertEqual(actual.dtype, expected.dtype)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @torch.no_grad()
    def test_forward_matches_reference(self):
        torch.manual_seed(0)
        config = Gemma4VisionConfig(
            hidden_size=16,
            num_hidden_layers=0,
            patch_size=1,
            position_embedding_size=64,
        )
        embedder = Gemma4VisionPatchEmbedder(config)
        embedder.position_embedding_table.normal_()

        pos_batch = torch.stack(
            [_make_grid_positions(6, 3, 27), _make_grid_positions(3, 6, 27)],
            dim=0,
        )
        padding = (pos_batch == -1).all(dim=-1)
        pixels = torch.rand(2, 27, 3, dtype=torch.float32)
        pixels_orig = pixels.clone()

        weight = embedder.input_proj.weight
        ref_hidden = F.linear((2 * (pixels - 0.5)).to(weight.dtype), weight)
        ref_pos = _ref_position_embeddings(
            embedder.position_embedding_table, pos_batch, padding
        )
        expected = ref_hidden + ref_pos
        actual = embedder(pixels, pos_batch, padding)

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(pixels, pixels_orig, atol=0, rtol=0)

        empty_out = embedder(
            torch.empty((0, 9, 3), dtype=torch.float32),
            torch.empty((0, 9, 2), dtype=torch.int64),
            torch.empty((0, 9), dtype=torch.bool),
        )
        self.assertEqual(empty_out.shape, (0, 9, config.hidden_size))
        self.assertEqual(empty_out.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
