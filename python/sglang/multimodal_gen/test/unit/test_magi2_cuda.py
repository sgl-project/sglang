# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.multimodal_gen.configs.models.dits.magi2 import Magi2PreviewArchConfig
from sglang.multimodal_gen.runtime.layers.attention.magi2_block_grid_attention import (
    Magi2BlockGrid,
    Magi2BlockGridAttention,
    cached_block_mask,
)

_HEADS = 4
_HEAD_DIM = 64
_LATENT_THW = (4, 8, 8)
_TAIL_ROOM = 256


def _grid(*, num_tail_tokens: int = 3, num_pad_tokens: int = 0) -> Magi2BlockGrid:
    return Magi2BlockGrid(
        latent_thw=_LATENT_THW,
        block_thw=(2, 2, 2),
        radius_thw=(1, 1, 1),
        num_tail_tokens=num_tail_tokens,
        num_pad_tokens=num_pad_tokens,
    )


def _bucketed_grid(*, num_tail_tokens: int) -> Magi2BlockGrid:
    """Same total length, different valid length: what bucketing produces."""
    return _grid(
        num_tail_tokens=num_tail_tokens,
        num_pad_tokens=_TAIL_ROOM - num_tail_tokens,
    )


def _qkv(grid: Magi2BlockGrid, *, device):
    generator = torch.Generator(device=device).manual_seed(0)
    return [
        torch.randn(grid.seq_len, _HEADS, _HEAD_DIM, device=device, generator=generator)
        for _ in range(3)
    ]


def _attend(attention, tensors, grid: Magi2BlockGrid) -> torch.Tensor:
    q, k, v = (t.transpose(0, 1)[None] for t in tensors)
    return attention._attend(q=q, k=k, v=v, grid=grid)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFlexAttentionPath(unittest.TestCase):
    """Covers the compiled path; the mask tests only evaluate ``mask_mod``."""

    def setUp(self):
        self.device = torch.device("cuda")
        self.attention = Magi2BlockGridAttention(num_heads=_HEADS, head_dim=_HEAD_DIM)
        cached_block_mask.cache_clear()

    def test_sparse_block_mask_matches_the_dense_reference(self):
        # The tile occupancy handed to from_kv_blocks is a superset that mask_mod
        # refines, so too tight a superset silently drops keys.
        grid = _grid()
        tensors = _qkv(grid, device=self.device)
        dense = Magi2BlockGridAttention(
            num_heads=_HEADS, head_dim=_HEAD_DIM, dense_fallback=True
        )
        torch.testing.assert_close(
            _attend(self.attention, tensors, grid),
            _attend(dense, tensors, grid),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_full_blocks_match_the_dense_reference(self):
        # Block volume 8*4*4 = 128 is exactly one tile, so most allowed pairs are
        # fully unmasked and take from_kv_blocks' full_* path, which skips mask_mod
        # entirely. The other tests use an 8-token block, where no pair is ever full,
        # so without this one that path would be uncovered.
        grid = Magi2BlockGrid(
            latent_thw=(16, 16, 16),
            block_thw=(8, 4, 4),
            radius_thw=(2, 2, 2),
            num_tail_tokens=200,
        )
        mask = cached_block_mask(grid=grid, device=self.device)
        self.assertIsNotNone(mask.full_kv_num_blocks)
        self.assertGreater(int(mask.full_kv_num_blocks.sum()), 0)

        tensors = _qkv(grid, device=self.device)
        dense = Magi2BlockGridAttention(
            num_heads=_HEADS, head_dim=_HEAD_DIM, dense_fallback=True
        )
        torch.testing.assert_close(
            _attend(self.attention, tensors, grid),
            _attend(dense, tensors, grid),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_padded_rows_do_not_reach_a_valid_row(self):
        # Pad rows stay attendable as queries to avoid an all -inf softmax, so they
        # must not leak into the rows that are gathered back.
        plain = _grid()
        padded = _grid(num_pad_tokens=2)
        tensors = _qkv(plain, device=self.device)
        base = _attend(self.attention, tensors, plain)

        padded_tensors = [
            torch.cat([t, t[-1:].expand(2, *t.shape[1:])], dim=0) for t in tensors
        ]
        got = _attend(self.attention, padded_tensors, padded)

        self.assertFalse(torch.isnan(got).any())
        torch.testing.assert_close(
            got[:, :, : plain.seq_len], base, rtol=1e-4, atol=1e-4
        )

    def test_bucketed_lengths_share_one_compiled_graph(self):
        # dynamo guards a captured int by value, so build_mask_mod carries the valid
        # length as a tensor. Bucketing fixes the shape while the valid length still
        # varies per prompt; as an int each value here would be its own graph and
        # nine would trip the default recompile_limit of 8.
        import torch._dynamo as dynamo

        dynamo.reset()
        dynamo.utils.counters.clear()

        lengths = (3, 17, 64, 129, 200, 231, 244, 250, 255)
        seq_lens = set()
        for num_tail in lengths:
            grid = _bucketed_grid(num_tail_tokens=num_tail)
            seq_lens.add(grid.seq_len)
            _attend(self.attention, _qkv(grid, device=self.device), grid)

        # Guard the premise: the point is one shape with many valid lengths.
        self.assertEqual(len(seq_lens), 1)
        self.assertEqual(len(lengths), 9)
        self.assertEqual(dynamo.utils.counters["stats"]["unique_graphs"], 1)


NUM_STREAM = 4
HIDDEN = 128
TOKENS = 37


def _inputs(device):
    torch.manual_seed(0)
    streams = torch.randn(
        TOKENS, NUM_STREAM, HIDDEN, device=device, dtype=torch.bfloat16
    )
    block_out = torch.randn(TOKENS, HIDDEN, device=device, dtype=torch.bfloat16)
    post = torch.rand(TOKENS, NUM_STREAM, device=device) * 2.0
    res = torch.rand(TOKENS, NUM_STREAM, NUM_STREAM, device=device)
    return streams, block_out, post, res / res.sum(-1, keepdim=True)


def _reference(streams, block_out, post, res):
    """fp32 throughout: the kernel accumulates in fp32, the einsum spelling does not."""
    return torch.einsum("tij,tjc->tic", res.float(), streams.float()) + (
        post.float()[..., None] * block_out.float()[:, None, :]
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestMhcMixOutputKernel(unittest.TestCase):
    def test_matches_an_fp32_reference(self):
        from sglang.multimodal_gen.runtime.layers.magi2_mhc_kernel import (
            mhc_mix_output,
        )

        args = _inputs(torch.device("cuda"))
        got = mhc_mix_output(*args)
        self.assertEqual(got.shape, args[0].shape)
        self.assertEqual(got.dtype, args[0].dtype)
        # bf16 output spacing at these magnitudes is ~0.06.
        torch.testing.assert_close(got.float(), _reference(*args), rtol=0, atol=0.07)

    def test_hidden_size_wider_than_one_column_block(self):
        from sglang.multimodal_gen.runtime.layers.magi2_mhc_kernel import (
            mhc_mix_output,
        )

        hidden = Magi2PreviewArchConfig().hidden_size
        dev = torch.device("cuda")
        torch.manual_seed(1)
        streams = torch.randn(8, NUM_STREAM, hidden, device=dev, dtype=torch.bfloat16)
        block_out = torch.randn(8, hidden, device=dev, dtype=torch.bfloat16)
        post = torch.rand(8, NUM_STREAM, device=dev) * 2.0
        res = torch.rand(8, NUM_STREAM, NUM_STREAM, device=dev)
        res = res / res.sum(-1, keepdim=True)

        torch.testing.assert_close(
            mhc_mix_output(streams, block_out, post, res).float(),
            _reference(streams, block_out, post, res),
            rtol=0,
            atol=0.07,
        )

    def test_registered_as_an_op_so_compile_emits_an_opaque_call(self):
        # A raw Triton launch inside a compiled region breaks the graph and pays
        # Triton's JIT per block: measured as +330s on the first clip.
        self.assertTrue(hasattr(torch.ops.sglang, "magi2_mhc_mix_output"))

    def test_module_uses_the_kernel_and_tracks_the_einsum_fallback(self):
        from sglang.multimodal_gen.runtime.models.dits.magi2_mhc import Magi2MHC

        dev = torch.device("cuda")
        mhc = Magi2MHC(num_stream=NUM_STREAM, hidden_size=HIDDEN).to(dev)
        streams, block_out, _, _ = _inputs(dev)
        h_post = torch.randn(TOKENS, NUM_STREAM, device=dev)
        h_res = torch.randn(TOKENS, NUM_STREAM, NUM_STREAM, device=dev)

        got = mhc.mix_output(streams, block_out, h_post, h_res)
        # The CPU branch keeps the einsum spelling, so this pins the fallback too.
        cpu = mhc.cpu().mix_output(
            streams.cpu(), block_out.cpu(), h_post.cpu(), h_res.cpu()
        )
        self.assertEqual(got.shape, streams.shape)
        torch.testing.assert_close(got.cpu().float(), cpu.float(), rtol=0, atol=0.07)


if __name__ == "__main__":
    unittest.main()
