# SPDX-License-Identifier: Apache-2.0
"""SubBlock block-sparse attention backend.

The schedule and adapter tests are pure CPU. The numerical tests need either
an SM90 GPU with SGLang's CuTe-DSL dependencies, an SM100 GPU with
FlashInfer's ``bsa_attn_blk64_fwd``, or an SM120 GPU with FlashInfer's
``bsa_attn_sm120_blk64_fwd`` and are skipped otherwise.

The trick that makes the sparse kernel checkable against dense attention: at
``sparsity`` just above 0 every block is inside the budget, so the block-sparse
result must reproduce dense attention up to bf16 rounding. That covers the
routing indices, the tail block sizes, and the softmax scale in one assertion,
none of which an accuracy-only comparison at real sparsity would pin down.
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse.router import (
    _snap_up_to_8,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn import (
    SubBlockSparseAttentionBackend,
    SubBlockSparseAttentionImpl,
    SubBlockSparseSchedule,
    _dit_layer_index,
    _run_subblock_sparse_attention,
    _sm90_sparse_attention,
)

HEAD_DIM = 128
NUM_HEADS = 4


def _subblock_kernel_available() -> bool:
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability(0)
    try:
        if capability == (9, 0):
            from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn import (
                _load_sm90_block_sparse_attention,
            )

            _load_sm90_block_sparse_attention()
        elif capability == (10, 0):
            from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse import (
                load_bsa_attn_blk64_fwd,
            )

            load_bsa_attn_blk64_fwd()
        elif capability == (12, 0):
            from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse import (
                load_bsa_attn_sm120_blk64_fwd,
            )

            load_bsa_attn_sm120_blk64_fwd()
        else:
            return False
    except Exception:
        return False
    return True


requires_subblock_kernel = unittest.skipUnless(
    _subblock_kernel_available(),
    "needs an SM90, SM100, or SM120 SubBlock attention kernel",
)


class _FakeServerArgs:
    def __init__(self, config):
        self.attention_backend_config = config


def _patch_schedule(config):
    return patch(
        "sglang.multimodal_gen.runtime.server_args.get_global_server_args",
        return_value=_FakeServerArgs(config),
    )


def _patch_step(step: int):
    class _Ctx:
        current_timestep = step

    return patch(
        "sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn.get_forward_context",
        return_value=_Ctx(),
    )


def _dense_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    """q, k, v: [1, S, H, D]."""
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale
    ).transpose(1, 2)


def _structured_qkv(
    seq_len: int, device: torch.device, n_topic: int = 64, seed: int = 0
):
    """Attention-like q/k/v: ``[1, S, H, 128]`` bf16.

    Random q/k produces a near-uniform attention map, and under a uniform map
    no block-selection rule can work -- dropping blocks drops mass wherever you
    cut. Real video attention concentrates on a limited region of keys, so each
    query here prefers one narrow key topic (a couple of 64-token blocks), and
    each topic carries its own value, so attending to the wrong blocks gives a
    visibly wrong answer instead of the same blurred average.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    shape = (n_topic, NUM_HEADS, HEAD_DIM)
    key_centers = torch.randn(shape, device=device, generator=gen)
    value_centers = torch.randn(shape, device=device, generator=gen)
    topic = torch.arange(seq_len, device=device) * n_topic // seq_len

    def _noise(scale: float) -> torch.Tensor:
        return scale * torch.randn(
            seq_len, NUM_HEADS, HEAD_DIM, device=device, generator=gen
        )

    q = key_centers[topic] + _noise(0.3)
    # the topic a query wants sits half a sequence away, so a rule that simply
    # keeps the diagonal cannot pass
    k = key_centers[(topic + n_topic // 2) % n_topic] + _noise(0.3)
    v = value_centers[topic] + _noise(0.1)
    return (
        q[None].to(torch.bfloat16),
        k[None].to(torch.bfloat16),
        v[None].to(torch.bfloat16),
    )


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            a.float().flatten(), b.float().flatten(), dim=0
        )
    )


class TestSubBlockSparseSchedule(unittest.TestCase):
    def test_dit_layer_index_only_matches_top_level_blocks(self):
        self.assertEqual(_dit_layer_index("blocks.7.attn"), 7)
        self.assertEqual(_dit_layer_index("blocks.0.attn"), 0)
        self.assertIsNone(_dit_layer_index("token_refiner.blocks.1.attn"))
        self.assertIsNone(_dit_layer_index(""))
        self.assertIsNone(_dit_layer_index("blocks_extra.3.attn"))

    def test_defaults_when_config_is_empty(self):
        with _patch_schedule({}):
            schedule = SubBlockSparseSchedule.from_server_args()
        self.assertEqual(schedule.sparsity, 0.75)
        self.assertEqual(schedule.skip_first_steps, 10)
        # Depth is not protected by default; the early steps are. See the
        # sweep recorded next to the constants.
        self.assertEqual(schedule.skip_first_layers, 0)
        self.assertEqual(schedule.n_k, 4)
        self.assertEqual(schedule.n_q, 4)

    def test_rejects_out_of_range_values(self):
        for config in ({"sparsity": 1.0}, {"n_k": 3}, {"skip_first_steps": -1}):
            with self.subTest(config=config), _patch_schedule(config):
                with self.assertRaises(ValueError):
                    SubBlockSparseSchedule.from_server_args()


class TestBudgetGranularity(unittest.TestCase):
    """The kernel bills in groups of 8 blocks; the budget should collect them."""

    def test_snaps_up_to_the_billed_count(self):
        for topk, expected in ((148, 152), (118, 120), (1, 8), (0, 8)):
            with self.subTest(topk=topk):
                self.assertEqual(_snap_up_to_8(topk, 590), expected)

    def test_never_exceeds_the_blocks_that_exist(self):
        """The cap wins over the granularity: 590 blocks means at most 590."""
        self.assertEqual(_snap_up_to_8(586, 590), 590)
        self.assertEqual(_snap_up_to_8(3, 5), 5)


class TestSubBlockSparseBackend(unittest.TestCase):
    def test_the_advertised_builder_can_be_built(self):
        """`AttentionMetadataBuilder.__init__` is abstract; a builder that does
        not override it makes `get_builder_cls()()` a TypeError."""
        builder = SubBlockSparseAttentionBackend.get_builder_cls()()
        builder.prepare()
        metadata = builder.build(current_timestep=7)
        self.assertIsInstance(
            metadata, SubBlockSparseAttentionBackend.get_metadata_cls()
        )
        self.assertEqual(metadata.current_timestep, 7)

    def test_sm90_adapter_uses_presorted_indices_and_64x64_blocks(self):
        captured = {}

        class _FakeBlockSparseTensors:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.__dict__.update(kwargs)

        def fake_flash_attn_func(q, k, v, **kwargs):
            captured.update(kwargs)
            return q, None

        index = torch.tensor([[[[1, 3, 5, 7]]]], dtype=torch.int32)
        q = torch.empty(1, 64, 1, HEAD_DIM, dtype=torch.bfloat16)
        with patch(
            "sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn._load_sm90_block_sparse_attention",
            return_value=(_FakeBlockSparseTensors, fake_flash_attn_func),
        ):
            out = _sm90_sparse_attention(q, q, q, index, 4, HEAD_DIM**-0.5)

        self.assertIs(out, q)
        torch.testing.assert_close(
            captured["mask_block_idx"],
            torch.tensor([[[[1, 3, 5, 7]]]], dtype=torch.int32),
        )
        self.assertEqual(captured["mask_block_cnt"].item(), 4)
        self.assertIsNone(captured["full_block_cnt"])
        self.assertIsNone(captured["full_block_idx"])
        self.assertEqual(captured["block_size"], (64, 64))
        self.assertIs(
            captured["block_sparse_tensors"].mask_block_idx,
            captured["mask_block_idx"],
        )


class TestSubBlockGating(unittest.TestCase):
    """The schedule must decide sparsity from the layer and the step alone."""

    def _impl(self, prefix: str, **config) -> SubBlockSparseAttentionImpl:
        with (
            _patch_schedule(config),
            patch.object(
                SubBlockSparseAttentionImpl, "_build_dense_impl", return_value=None
            ),
        ):
            return SubBlockSparseAttentionImpl(
                num_heads=NUM_HEADS,
                head_size=HEAD_DIM,
                causal=False,
                softmax_scale=HEAD_DIM**-0.5,
                prefix=prefix,
            )

    def test_skip_first_layers_gates_the_bottom_of_the_stack(self):
        for prefix, expected in (
            ("blocks.0.attn", False),
            ("blocks.1.attn", False),
            ("blocks.2.attn", True),
        ):
            with self.subTest(prefix=prefix):
                impl = self._impl(prefix, skip_first_layers=2)
                self.assertEqual(impl.layer_enabled, expected)

    def test_token_refiner_is_dense(self):
        self.assertFalse(self._impl("token_refiner.blocks.0.attn").layer_enabled)

    def test_head_dim_other_than_128_is_dense(self):
        with (
            _patch_schedule({}),
            patch.object(
                SubBlockSparseAttentionImpl, "_build_dense_impl", return_value=None
            ),
        ):
            impl = SubBlockSparseAttentionImpl(
                num_heads=NUM_HEADS,
                head_size=64,
                causal=False,
                softmax_scale=64**-0.5,
                prefix="blocks.9.attn",
            )
        self.assertFalse(impl.layer_enabled)

    def test_first_steps_are_dense(self):
        impl = self._impl("blocks.9.attn")
        for step, expected in ((0, False), (9, False), (10, True), (49, True)):
            with self.subTest(step=step), _patch_step(step):
                self.assertEqual(impl._step_enabled(), expected)

    def test_short_sequences_are_dense(self):
        impl = self._impl("blocks.9.attn")
        q = torch.empty(1, 1024, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16)
        with _patch_step(20):
            self.assertFalse(impl._sparse_ready(q, q))

    def test_fp32_is_dense(self):
        impl = self._impl("blocks.9.attn")
        q = torch.empty(1, 8192, NUM_HEADS, HEAD_DIM, dtype=torch.float32)
        with _patch_step(20):
            self.assertFalse(impl._sparse_ready(q, q))


@requires_subblock_kernel
class TestSubBlockNumerics(unittest.TestCase):
    seq_len = 8192

    def _impl(self, **config) -> SubBlockSparseAttentionImpl:
        with _patch_schedule(config):
            return SubBlockSparseAttentionImpl(
                num_heads=NUM_HEADS,
                head_size=HEAD_DIM,
                causal=False,
                softmax_scale=HEAD_DIM**-0.5,
                prefix="blocks.9.attn",
            )

    def test_full_budget_reproduces_dense(self):
        device = torch.device("cuda")
        q, k, v = _structured_qkv(self.seq_len, device)
        impl = self._impl(sparsity=1e-6)
        with _patch_step(20):
            out = impl.forward(q, k, v, None)
        ref = _dense_reference(q, k, v, HEAD_DIM**-0.5)
        self.assertGreater(_cosine(out, ref), 0.999)

    def test_ragged_tail_reproduces_dense(self):
        """A sequence that is not a multiple of the 64-token block."""
        device = torch.device("cuda")
        seq_len = self.seq_len + 37
        q, k, v = _structured_qkv(seq_len, device)
        impl = self._impl(sparsity=1e-6)
        with _patch_step(20):
            out = impl.forward(q, k, v, None)
        ref = _dense_reference(q, k, v, HEAD_DIM**-0.5)
        self.assertGreater(_cosine(out, ref), 0.999)

    def test_presorted_ragged_tail_oversubscribes_sms(self):
        """Make tail-mask ordering observable across multiple SM waves."""
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("the reverse-consumption constraint is specific to SM90")

        device = torch.device("cuda")
        seq_len = self.seq_len + 37
        shape = (1, seq_len, NUM_HEADS, HEAD_DIM)
        q = torch.zeros(shape, device=device, dtype=torch.bfloat16)
        k = torch.zeros_like(q)
        v = torch.ones_like(q)

        num_blocks = (seq_len + 63) // 64
        num_tiles = NUM_HEADS * num_blocks
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.assertGreater(num_tiles, 2 * num_sms)

        # The SM90 consumer visits slots from high to low and applies the tail
        # mask to the first block, so the caller places the ragged block last.
        topk = 8
        tail_block = num_blocks - 1
        sorted_blocks = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, tail_block],
            device=device,
            dtype=torch.int32,
        )
        sorted_index = (
            sorted_blocks.view(1, 1, 1, topk)
            .expand(1, NUM_HEADS, num_blocks, topk)
            .clone()
        )
        out = _run_subblock_sparse_attention(
            q, k, v, sorted_index, topk, HEAD_DIM**-0.5
        )

        torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=2e-3)

    def test_routing_finds_the_blocks_that_carry_the_mass(self):
        """At 0.75 sparsity the router must keep the blocks that matter.

        The random-budget control is the point of this test: with the same
        number of blocks but chosen at random the output collapses, so a high
        cosine here measures the routing, not a forgiving fixture.
        """
        device = torch.device("cuda")
        q, k, v = _structured_qkv(self.seq_len, device)
        ref = _dense_reference(q, k, v, HEAD_DIM**-0.5)

        impl = self._impl(sparsity=0.75)
        with _patch_step(20):
            routed = impl.forward(q, k, v, None)
        self.assertTrue(torch.isfinite(routed.float()).all())
        self.assertGreater(_cosine(routed, ref), 0.99)

        num_blocks = (self.seq_len + 63) // 64
        topk = impl.router.route(q, k, sparsity=0.75, softmax_scale=HEAD_DIM**-0.5).topk
        # Draw a random subset per row, not `randint`: sampling with replacement
        # would attend fewer distinct blocks and distort the softmax mass. Sort
        # the selected subset to honor the SM90 kernel's direct-call contract.
        random_index = (
            torch.rand(1, NUM_HEADS, num_blocks, num_blocks, device=device)
            .argsort(dim=-1)[..., :topk]
            .sort(dim=-1)
            .values.to(torch.int32)
        )
        random_out = _run_subblock_sparse_attention(
            q,
            k,
            v,
            random_index,
            topk,
            HEAD_DIM**-0.5,
        )
        self.assertLess(_cosine(random_out, ref), 0.9)

    def _assert_kernel_backed_mixed_query_mask(self):
        device = torch.device("cuda")
        scale = HEAD_DIM**-0.5
        generator = torch.Generator(device=device).manual_seed(7)
        q = torch.randn(
            1,
            3 * 64,
            NUM_HEADS,
            HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        k = torch.randn(
            1,
            16 * 64,
            NUM_HEADS,
            HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        v = torch.randn(
            k.shape,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )

        topk = 8
        sparse_blocks = torch.tensor(
            [
                [0, 2, 4, 6, 8, 10, 12, 14],
                [1, 3, 5, 7, 9, 11, 13, 15],
                [0, 1, 4, 5, 8, 9, 12, 13],
            ],
            device=device,
            dtype=torch.int32,
        )
        routed_index = (
            sparse_blocks.view(1, 1, 3, topk).expand(1, NUM_HEADS, 3, topk).clone()
        )
        plan = Mock(
            index=routed_index,
            topk=topk,
            num_blocks=16,
            density=0.5,
        )
        impl = self._impl(sparsity=0.5)
        impl.router = Mock(route=Mock(return_value=plan))
        sparse_query_block_mask = torch.tensor([True, False, True], device=device)

        out = impl._sparse_attention(
            q,
            k,
            v,
            sparse_query_block_mask=sparse_query_block_mask,
        )

        reference = torch.empty_like(q)
        all_blocks = torch.arange(16, device=device)
        token_offsets = torch.arange(64, device=device)
        for query_block in range(3):
            selected_blocks = (
                sparse_blocks[query_block].long()
                if sparse_query_block_mask[query_block]
                else all_blocks
            )
            selected_tokens = (
                selected_blocks[:, None] * 64 + token_offsets[None, :]
            ).reshape(-1)
            query_slice = slice(query_block * 64, (query_block + 1) * 64)
            reference[:, query_slice] = _dense_reference(
                q[:, query_slice],
                k.index_select(1, selected_tokens),
                v.index_select(1, selected_tokens),
                scale,
            )

        self.assertGreater(_cosine(out, reference), 0.999)

    def test_sm90_kernel_backed_mixed_query_mask(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("requires the SM90 SubBlock kernel")
        self._assert_kernel_backed_mixed_query_mask()

    def test_sm100_kernel_backed_mixed_query_mask(self):
        if torch.cuda.get_device_capability() != (10, 0):
            self.skipTest("requires the SM100 SubBlock kernel")
        self._assert_kernel_backed_mixed_query_mask()

    def test_sm120_kernel_backed_mixed_query_mask(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("requires the SM120 SubBlock kernel")
        self._assert_kernel_backed_mixed_query_mask()

    def test_skipped_step_is_bitwise_dense(self):
        device = torch.device("cuda")
        q, k, v = _structured_qkv(self.seq_len, device)
        impl = self._impl(sparsity=0.75, skip_first_steps=10)
        with _patch_step(3):
            skipped = impl.forward(q, k, v, None)
        dense = impl.dense_impl.forward(q, k, v, None)
        torch.testing.assert_close(skipped, dense, rtol=0, atol=0)

    def test_varlen_routes_each_document(self):
        """Packed [real | padding] layout, exactly MiniMax H3's cu_seqlens."""
        device = torch.device("cuda")
        used, total = self.seq_len, self.seq_len + 512
        q, k, v = _structured_qkv(total, device)
        q, k, v = q[0], k[0], v[0]
        cu_host = (0, used, total)
        cu = torch.tensor(cu_host, dtype=torch.int32, device=device)
        impl = self._impl(sparsity=1e-6)
        with _patch_step(20):
            out = impl.forward_varlen(
                q, k, v, cu_seqlens=cu, max_seqlen=used, cu_seqlens_host=cu_host
            )
        for start, stop in ((0, used), (used, total)):
            ref = _dense_reference(
                q[start:stop][None],
                k[start:stop][None],
                v[start:stop][None],
                HEAD_DIM**-0.5,
            )[0]
            self.assertGreater(_cosine(out[start:stop], ref), 0.999)


if __name__ == "__main__":
    unittest.main()
