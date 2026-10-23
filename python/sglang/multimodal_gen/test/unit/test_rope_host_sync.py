import unittest
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.runtime.layers import layernorm
from sglang.multimodal_gen.runtime.layers.rotary_embedding import utils as rope_utils
from sglang.multimodal_gen.runtime.layers.rotary_embedding.mrope import FluxPosEmbed
from sglang.multimodal_gen.test.unit.host_sync_guard import SyncOnRead


def _grid_ids(height: int, width: int) -> torch.Tensor:
    ids = torch.zeros(height, width, 3)
    ids[..., 1] = torch.arange(height)[:, None]
    ids[..., 2] = torch.arange(width)[None, :]
    return ids.reshape(-1, 3)


class TestNDRopeHostSync(unittest.TestCase):
    def test_never_reads_positions_on_host(self):
        embed = FluxPosEmbed(theta=10000, axes_dim=[16, 56, 56])
        ids = _grid_ids(6, 8)
        expected = embed(ids)
        actual = embed(ids.as_subclass(SyncOnRead))
        for result, reference in zip(actual, expected, strict=True):
            self.assertTrue(torch.equal(result.as_subclass(torch.Tensor), reference))


class TestRopePositions(unittest.TestCase):
    def setUp(self):
        rope_utils._shared_rope_positions.cache_clear()
        self.addCleanup(rope_utils._shared_rope_positions.cache_clear)

    def test_shared_across_blocks(self):
        device = torch.device("cpu")
        with patch.object(
            rope_utils,
            "_build_rope_positions",
            wraps=rope_utils._build_rope_positions,
        ) as build:
            first = rope_utils.rope_positions(3, 5, 2, device)
            second = rope_utils.rope_positions(3, 5, 2, device)
        self.assertIs(first, second)
        build.assert_called_once()
        self.assertTrue(torch.equal(first, torch.arange(3, 8).repeat(2)))

    def test_compile_and_capture_build_their_own_positions(self):
        build = Mock(side_effect=lambda *args: object())
        modes = (
            (torch.compiler, "is_compiling", torch.device("cpu")),
            (torch.cuda, "is_current_stream_capturing", torch.device("cuda")),
        )
        for owner, name, device in modes:
            with self.subTest(mode=name):
                with (
                    patch.object(rope_utils, "_build_rope_positions", build),
                    patch.object(owner, name, return_value=True),
                ):
                    first = rope_utils.rope_positions(0, 4, 1, device)
                    second = rope_utils.rope_positions(0, 4, 1, device)
                self.assertIsNot(first, second)
        self.assertEqual(rope_utils._shared_rope_positions.cache_info().currsize, 0)

    def test_qk_norm_rope_takes_shared_positions(self):
        torch.manual_seed(0)
        q = torch.randn(2, 6, 2, 8)
        k = torch.randn(2, 6, 2, 8)
        cos_sin_cache = torch.randn(16, 8)
        q_norm = layernorm.RMSNorm(8)
        k_norm = layernorm.RMSNorm(8)
        expected = layernorm.apply_qk_norm_rope(
            q.clone(),
            k.clone(),
            q_norm,
            k_norm,
            8,
            cos_sin_cache,
            positions=torch.arange(1, 7).repeat(2),
        )
        with patch.object(
            layernorm, "rope_positions", wraps=rope_utils.rope_positions
        ) as positions:
            actual = layernorm.apply_qk_norm_rope(
                q.clone(),
                k.clone(),
                q_norm,
                k_norm,
                8,
                cos_sin_cache,
                position_offset=1,
            )
        positions.assert_called_once_with(1, 6, 2, q.device)
        for result, reference in zip(actual, expected, strict=True):
            self.assertTrue(torch.equal(result, reference))


if __name__ == "__main__":
    unittest.main()
