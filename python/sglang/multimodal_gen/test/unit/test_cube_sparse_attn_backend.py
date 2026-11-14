# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for the cube sparse attention backend.

Mask semantics are locked against a token-level masked-SDPA oracle and the
reference policy: only 3D visual streams participate in global top-k.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.layers.attention.backends.cube_sparse_attn import (
    backend as cube_backend,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.cube_sparse_attn.backend import (
    CubeSparseAttentionImpl,
    CubeSparseAttentionMetadataBuilder,
    cube_sparse_attention,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.cube_sparse_attn.mask import (
    _cube_topk_selection,
    cube_topk_block_indices,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)

_CUBE_SIZE = (2, 2, 2)
_NUM_HEADS = 3
_HEAD_DIM = 8


def _build_t2va_packed(*, include_keyframe_cond=False):
    return minimax_h3_packed_sequence(
        # Span multiple cube labels so an accidental sparse text policy is
        # observable even when top-k clamps each non-empty group to one label.
        text_len=17,
        latent_t=2,
        latent_h=8,
        latent_w=12,
        audio_t=5,
        include_keyframe_cond=include_keyframe_cond,
        keyframe_frame_indices=[0] if include_keyframe_cond else None,
        frame_count=5 if include_keyframe_cond else None,
    )


def _build_metadata(packed, topk_ratio_list):
    return CubeSparseAttentionMetadataBuilder().build(
        packed=packed,
        local_cube_size=_CUBE_SIZE,
        topk_ratio_list=topk_ratio_list,
        num_steps=len(topk_ratio_list),
        device=torch.device("cpu"),
    )


def _random_qkv(seq_len):
    generator = torch.Generator().manual_seed(0)
    return tuple(
        torch.randn(
            seq_len, _NUM_HEADS, _HEAD_DIM, dtype=torch.float32, generator=generator
        )
        for _ in range(3)
    )


def _token_labels(precomputed):
    lengths = precomputed.layout.label_lengths
    device = lengths.device
    labels = torch.empty(
        precomputed.layout.real_total_len, dtype=torch.long, device=device
    )
    labels[precomputed.layout.sorted_real_indices] = torch.repeat_interleave(
        torch.arange(precomputed.layout.num_labels, device=device), lengths
    )
    return labels


def _reference_policy_block_mask(q_real, k_real, precomputed, topk_ratio):
    """Hermetic label-mask oracle for the reference sparse policy."""
    dim = q_real.shape[-1]
    qk_sorted = torch.cat((q_real, k_real), dim=-1)[
        precomputed.layout.sorted_real_indices
    ]
    label_lengths = precomputed.layout.label_lengths
    qk_pool = torch.segment_reduce(
        qk_sorted, "sum", lengths=label_lengths, axis=0, unsafe=True
    )
    qk_pool /= label_lengths.float().view(-1, 1, 1)

    q_pool, k_pool = torch.split(qk_pool, dim, dim=-1)
    scores = torch.einsum("lhd,mhd->hlm", q_pool, k_pool) * (dim**-0.5)
    candidate_mask = precomputed.layout.topk_mask
    sparse_sizes = candidate_mask.sum(dim=-1)
    sparse_labels = sparse_sizes > 0
    scores.masked_fill_(~candidate_mask.unsqueeze(0), float("-inf"))

    top_k_per_label = (sparse_sizes.to(torch.float32) * topk_ratio).to(torch.long)
    top_k_per_label.clamp_(min=1)
    top_k_per_label = torch.minimum(top_k_per_label, sparse_sizes)
    top_k_per_label = torch.where(
        sparse_labels, top_k_per_label, torch.zeros_like(top_k_per_label)
    )

    sorted_idx = torch.argsort(scores, dim=-1, descending=True, stable=True)
    selected_rank = torch.arange(scores.shape[-1], device=scores.device).view(
        1, 1, -1
    ) < top_k_per_label.view(1, -1, 1)
    block_mask = torch.zeros_like(scores, dtype=torch.bool)
    block_mask.scatter_(-1, sorted_idx, selected_rank.expand_as(sorted_idx))
    block_mask &= candidate_mask.unsqueeze(0)
    block_mask |= precomputed.layout.base_block_mask.unsqueeze(0)
    return block_mask


def _assert_tokens_are_dense(test_case, allowed, token_indices):
    token_indices = token_indices.to(torch.long)
    test_case.assertGreater(token_indices.numel(), 0)
    test_case.assertTrue(allowed.index_select(1, token_indices).all())
    test_case.assertTrue(allowed.index_select(2, token_indices).all())


def _naive_masked_attention(q, k, v, allowed, scale):
    scores = torch.einsum("ihd,jhd->hij", q, k) * scale
    scores = scores.masked_fill(~allowed, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hij,jhd->ihd", probs, v)


def cube_topk_block_mask(q_real, k_real, precomputed, topk_ratio):
    """Semantic ``[H, num_labels, num_labels]`` TopK mask -- test oracle.

    Production runs ``cube_topk_block_indices``, which expands
    ``_cube_topk_selection`` straight into physical KV rows.  This function
    expands the same selection into a semantic boolean mask instead: it is the
    readable form the tests assert against, and it lives here rather than in
    the production module because nothing outside the test suite reads it.

    ``test_compact_indices_match_semantic_mask`` locks the two expansions
    together, so this oracle drifting away from production is itself a test
    failure rather than a silent numeric shift.
    """
    layout = precomputed.layout
    selected_order, selected_counts = _cube_topk_selection(
        q_real, k_real, precomputed, topk_ratio
    )
    candidate_mask = layout.topk_mask
    selected_rank = torch.arange(
        selected_order.shape[-1], device=selected_order.device
    ).view(1, 1, -1) < selected_counts.view(1, -1, 1)
    block_mask = torch.zeros_like(selected_order, dtype=torch.bool)
    block_mask.scatter_(-1, selected_order, selected_rank.expand_as(selected_order))
    block_mask &= candidate_mask.unsqueeze(0)
    block_mask |= layout.base_block_mask.unsqueeze(0)
    return block_mask


def _physical_mask_from_block_layout(block_layout, layout, num_heads):
    """Expand compact KV rows back into a dense ``[H, B, B]`` physical mask.

    This is the inverse of what ``cube_topk_block_indices`` packs, so it lets a
    test compare the production compact path against the semantic mask that
    the ``cube_topk_block_mask`` oracle above returns.
    """
    num_blocks = layout.num_blocks
    physical_allowed = torch.zeros(num_heads, num_blocks, num_blocks, dtype=torch.bool)
    for count_key, index_key in (
        ("kv_num_blocks", "kv_indices"),
        ("full_kv_num_blocks", "full_kv_indices"),
    ):
        if block_layout[count_key] is None:
            continue
        counts = block_layout[count_key].expand(1, num_heads, -1)[0]
        indices = block_layout[index_key].expand(1, num_heads, -1, -1)[0]
        ranks = torch.arange(indices.shape[-1]).view(1, 1, -1)
        valid = ranks < counts.unsqueeze(-1)
        head_ids, query_ids, slot_ids = valid.nonzero(as_tuple=True)
        physical_allowed[
            head_ids, query_ids, indices[head_ids, query_ids, slot_ids].long()
        ] = True
    return physical_allowed


def _reference_block_sparse_attention(
    padded_q, padded_k, padded_v, block_layout, precomputed, softmax_scale
):
    """Dense reference honoring the _run_block_sparse_attention contract.

    Uncompiled flex_attention ignores BlockMask block structure (mask_mod
    only), so CPU tests substitute this oracle; the real compiled-flex kernel
    is exercised by the CUDA test class.
    """
    cube_token_size = precomputed.layout.cube_token_size
    is_real = precomputed.layout.is_real
    num_heads = padded_q.shape[1]
    physical_allowed = _physical_mask_from_block_layout(
        block_layout, precomputed.layout, num_heads
    )
    real = is_real.to(torch.bool)
    allowed = (
        physical_allowed.repeat_interleave(cube_token_size, dim=1).repeat_interleave(
            cube_token_size, dim=2
        )
        & real[None, :, None]
        & real[None, None, :]
    )
    scores = torch.einsum("ihd,jhd->hij", padded_q, padded_k) * softmax_scale
    scores = scores.masked_fill(~allowed, float("-inf"))
    probs = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    return torch.einsum("hij,jhd->ihd", probs, padded_v)


class TestCubeSparsePrecompute(unittest.TestCase):
    def test_layout_invariants(self):
        packed = _build_t2va_packed()
        metadata = _build_metadata(packed, [0.5])
        pre = metadata.precomputed
        used = int(packed["cu_seqlens"].view(-1)[1])
        cube_token_size = pre.layout.cube_token_size

        self.assertEqual(pre.layout.real_total_len, used)
        self.assertEqual(cube_token_size, 8)
        self.assertEqual(
            pre.layout.padded_seqlen, pre.layout.num_labels * cube_token_size
        )
        self.assertEqual(int(pre.layout.label_lengths.sum()), used)
        self.assertTrue((pre.layout.label_lengths <= cube_token_size).all())
        # expand/gather round-trip: every real token has a unique padded slot.
        expand = pre.layout.expand_indices
        self.assertEqual(expand.unique().numel(), used)
        self.assertTrue(
            torch.equal(pre.layout.gather_indices[expand], torch.arange(used))
        )
        self.assertEqual(int(pre.layout.is_real.sum()), used)

    def test_full_ratio_mask_is_dense_within_sample(self):
        packed = _build_t2va_packed()
        metadata = _build_metadata(packed, [1.0])
        pre = metadata.precomputed
        q, k, _ = _random_qkv(int(packed["seq_len"]))
        mask = cube_topk_block_mask(
            q[: pre.layout.real_total_len], k[: pre.layout.real_total_len], pre, 1.0
        )
        self.assertTrue(mask.all())

    def test_builder_rejects_bad_topk_list(self):
        packed = _build_t2va_packed()
        with self.assertRaisesRegex(ValueError, "denoise steps"):
            CubeSparseAttentionMetadataBuilder().build(
                packed=packed,
                local_cube_size=_CUBE_SIZE,
                topk_ratio_list=[0.5, 0.5],
                num_steps=3,
                device=torch.device("cpu"),
            )
        with self.assertRaisesRegex(ValueError, r"\(0, 1\]"):
            CubeSparseAttentionMetadataBuilder().build(
                packed=packed,
                local_cube_size=_CUBE_SIZE,
                topk_ratio_list=[0.0],
                num_steps=1,
                device=torch.device("cpu"),
            )

    def test_builder_requires_stream_layout(self):
        packed = _build_t2va_packed()
        del packed["stream_layout"]
        with self.assertRaisesRegex(ValueError, "stream_layout"):
            _build_metadata(packed, [1.0])

    def test_builder_requires_a_3d_cube_size(self):
        packed = _build_t2va_packed()
        with self.assertRaisesRegex(ValueError, "positive 3D size"):
            CubeSparseAttentionMetadataBuilder().build(
                packed=packed,
                local_cube_size=(8, 8),
                topk_ratio_list=[0.5],
                num_steps=1,
                device=torch.device("cpu"),
            )
        anisotropic = CubeSparseAttentionMetadataBuilder().build(
            packed=packed,
            local_cube_size=(2, 4, 8),
            topk_ratio_list=[0.5],
            num_steps=1,
            device=torch.device("cpu"),
        )
        self.assertEqual(anisotropic.precomputed.layout.cube_token_size, 64)


class TestCubeSparseModalityPolicy(unittest.TestCase):
    def _allowed_mask(self, packed, topk_ratio):
        metadata = _build_metadata(packed, [topk_ratio])
        pre = metadata.precomputed
        q, k, _ = _random_qkv(int(packed["seq_len"]))
        block_mask = cube_topk_block_mask(
            q[: pre.layout.real_total_len],
            k[: pre.layout.real_total_len],
            pre,
            topk_ratio,
        )
        labels = _token_labels(pre)
        allowed = block_mask[:, labels[:, None], labels[None, :]]
        return pre, q, k, block_mask, allowed

    def _assert_has_3d_sparse_labels(self, pre):
        self.assertGreater(int(pre.layout.sparse_label_mask.sum()), 0)

    def test_t2va_text_and_audio_are_dense(self):
        packed = _build_t2va_packed()
        pre, q, k, block_mask, allowed = self._allowed_mask(packed, 0.25)
        self._assert_has_3d_sparse_labels(pre)

        _assert_tokens_are_dense(self, allowed, packed["text_pos"].view(-1))
        _assert_tokens_are_dense(self, allowed, packed["audio_pos"].view(-1))

        img_pos = packed["img_pos"].view(-1).to(torch.long)
        target_video = img_pos[packed["update_mask"].view(-1).to(torch.bool)]
        target_allowed = allowed.index_select(1, target_video).index_select(
            2, target_video
        )
        self.assertFalse(target_allowed.all())
        self.assertTrue(
            torch.equal(
                block_mask,
                _reference_policy_block_mask(
                    q[: pre.layout.real_total_len],
                    k[: pre.layout.real_total_len],
                    pre,
                    0.25,
                ),
            )
        )

    def test_keyframe_and_target_share_joint_cube_labels(self):
        packed = _build_t2va_packed(include_keyframe_cond=True)
        pre, _, _, _, allowed = self._allowed_mask(packed, 0.25)
        self._assert_has_3d_sparse_labels(pre)

        img_pos = packed["img_pos"].view(-1).to(torch.long)
        update_mask = packed["update_mask"].view(-1).to(torch.bool)
        keyframe = img_pos[~update_mask]
        target = img_pos[update_mask]
        frame_rows = 4 * 6
        labels = _token_labels(pre)
        self.assertTrue(
            torch.equal(
                labels.index_select(0, keyframe),
                labels.index_select(0, target[:frame_rows]),
            )
        )
        # A keyframe duplicates the first temporal plane inside each joint
        # cube, so a semantic cube can span more than one physical block.
        self.assertGreater(
            int(pre.layout.label_lengths.max()), pre.layout.cube_token_size
        )
        self.assertFalse(allowed.index_select(1, keyframe).all())
        _assert_tokens_are_dense(self, allowed, packed["text_pos"].view(-1))
        _assert_tokens_are_dense(self, allowed, packed["audio_pos"].view(-1))

    def test_ref_image_and_audio_are_dense_but_all_videos_share_topk(self):
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=17,
            latent_t=2,
            latent_h=8,
            latent_w=12,
            audio_t=4,
            ref_blocks=[
                {"kind": "image", "latent_h": 8, "latent_w": 12},
                {
                    "kind": "video_audio",
                    "ref_audio_t": 3,
                    "latent_t": 2,
                    "latent_h": 8,
                    "latent_w": 12,
                },
            ],
        )
        pre, q, k, block_mask, allowed = self._allowed_mask(packed, 0.4)
        self._assert_has_3d_sparse_labels(pre)

        img_pos = packed["img_pos"].view(-1).to(torch.long)
        update_mask = packed["update_mask"].view(-1).to(torch.bool)
        condition_visual = img_pos[~update_mask]
        ref_image_tokens = 1 * 4 * 6
        ref_image = condition_visual[:ref_image_tokens]
        _assert_tokens_are_dense(self, allowed, ref_image)
        _assert_tokens_are_dense(self, allowed, packed["text_pos"].view(-1))
        _assert_tokens_are_dense(self, allowed, packed["audio_pos"].view(-1))

        # Ref video and target video each contain six 2x2x2 cubes. They must
        # occupy one shared global candidate pool rather than separate quotas.
        self.assertEqual(int(pre.layout.sparse_label_mask.sum()), 12)
        self.assertTrue(
            torch.equal(
                block_mask,
                _reference_policy_block_mask(
                    q[: pre.layout.real_total_len],
                    k[: pre.layout.real_total_len],
                    pre,
                    0.4,
                ),
            )
        )


class TestCubeSparseAttentionOutput(unittest.TestCase):
    def test_compact_layout_uses_fewer_active_indices_than_full_mask(self):
        packed = _build_t2va_packed(include_keyframe_cond=True)
        metadata = _build_metadata(packed, [0.4])
        pre = metadata.precomputed
        q, k, _ = _random_qkv(int(packed["seq_len"]))
        layout = cube_topk_block_indices(
            q[: pre.layout.real_total_len],
            k[: pre.layout.real_total_len],
            pre,
            0.4,
        )

        active_indices = int(layout["kv_num_blocks"].sum())
        full_boolean_cells = _NUM_HEADS * pre.layout.num_blocks**2
        self.assertLess(active_indices, full_boolean_cells)
        self.assertGreater(int(layout["kv_num_blocks"].sum()), 0)
        self.assertIsNone(layout["full_kv_num_blocks"])
        self.assertIsNone(layout["full_kv_indices"])

    def test_compact_indices_match_semantic_mask(self):
        """Lock the two expansions of ``_cube_topk_selection`` together.

        Production attention consumes ``cube_topk_block_indices`` (compact KV
        rows) while the parity and topology tests assert against
        ``cube_topk_block_mask`` (a semantic ``[H, L, L]`` mask). They share a
        selection helper but expand it independently, so without this test a
        divergence in either expansion would only surface as a small numeric
        drift inside an attention tolerance -- or not at all, since no
        production code path reads the semantic mask.
        """
        for include_keyframe_cond in (False, True):
            packed = _build_t2va_packed(include_keyframe_cond=include_keyframe_cond)
            for topk_ratio in (0.25, 0.4, 1.0):
                with self.subTest(keyframe=include_keyframe_cond, ratio=topk_ratio):
                    metadata = _build_metadata(packed, [topk_ratio])
                    pre = metadata.precomputed
                    real_total_len = pre.layout.real_total_len
                    q, k, _ = _random_qkv(int(packed["seq_len"]))
                    q_real = q[:real_total_len]
                    k_real = k[:real_total_len]

                    semantic = cube_topk_block_mask(q_real, k_real, pre, topk_ratio)
                    compact = cube_topk_block_indices(q_real, k_real, pre, topk_ratio)

                    block_labels = pre.layout.block_labels
                    expected = semantic[:, block_labels[:, None], block_labels[None, :]]
                    actual = _physical_mask_from_block_layout(
                        compact, pre.layout, _NUM_HEADS
                    )
                    self.assertTrue(
                        torch.equal(expected, actual),
                        f"{int((expected ^ actual).sum())} physical block "
                        "cells differ between the semantic mask and the "
                        "compact KV rows",
                    )

    def test_compact_kv_rows_never_exceed_block_count(self):
        """Pin the invariant that makes the KV truncation lossless.

        ``cube_topk_block_indices`` builds a candidate row of width
        ``base_capacity + topk_semantic_capacity * max_label_block_count``,
        which can exceed ``num_blocks``, then sorts and truncates to
        ``num_blocks``.  That is only lossless because base ids and selected
        ids are disjoint and ``label -> physical blocks`` is a partition, so no
        row can hold more than ``num_blocks`` distinct valid ids.  If the
        deduplication against ``base_block_mask`` were dropped, the truncation
        would silently discard real KV blocks -- visible only as slight output
        drift, never as an error.  Assert the count directly, and cross-check
        it against the semantic mask so an over-count cannot hide behind an
        equally wrong row width.
        """
        for include_keyframe_cond in (False, True):
            packed = _build_t2va_packed(include_keyframe_cond=include_keyframe_cond)
            for topk_ratio in (0.25, 0.4):
                with self.subTest(keyframe=include_keyframe_cond, ratio=topk_ratio):
                    metadata = _build_metadata(packed, [topk_ratio])
                    pre = metadata.precomputed
                    real_total_len = pre.layout.real_total_len
                    q, k, _ = _random_qkv(int(packed["seq_len"]))
                    q_real = q[:real_total_len]
                    k_real = k[:real_total_len]

                    compact = cube_topk_block_indices(q_real, k_real, pre, topk_ratio)
                    num_blocks = pre.layout.num_blocks
                    counts = compact["kv_num_blocks"]
                    self.assertLessEqual(
                        int(counts.max()),
                        num_blocks,
                        "a compact KV row claims more blocks than exist, so "
                        "the sort-and-truncate step dropped valid ids",
                    )

                    # The row width really is over-provisioned relative to
                    # num_blocks; without that the assertion above is vacuous.
                    self.assertGreaterEqual(
                        pre.layout.base_physical_layout["full_kv_indices"].shape[-1]
                        + pre.layout.topk_semantic_capacity
                        * pre.layout.max_label_block_count,
                        num_blocks,
                    )

                    semantic = cube_topk_block_mask(q_real, k_real, pre, topk_ratio)
                    block_labels = pre.layout.block_labels
                    expected_counts = semantic[
                        :, block_labels[:, None], block_labels[None, :]
                    ].sum(dim=-1, dtype=torch.int32)
                    self.assertTrue(
                        torch.equal(counts[0], expected_counts),
                        "compact KV counts disagree with the semantic mask",
                    )

    def _run_and_compare(self, packed, topk_ratio):
        metadata = _build_metadata(packed, [topk_ratio])
        pre = metadata.precomputed
        seq_len = int(packed["seq_len"])
        real_total_len = pre.layout.real_total_len
        q, k, v = _random_qkv(seq_len)
        scale = _HEAD_DIM**-0.5

        with mock.patch.object(
            cube_backend,
            "_run_block_sparse_attention",
            _reference_block_sparse_attention,
        ):
            out = cube_sparse_attention(q, k, v, metadata, softmax_scale=scale)

        label_mask = cube_topk_block_mask(
            q[:real_total_len], k[:real_total_len], pre, topk_ratio
        )
        labels = _token_labels(pre)
        allowed = label_mask[:, labels[:, None], labels[None, :]]
        expected = _naive_masked_attention(
            q[:real_total_len],
            k[:real_total_len],
            v[:real_total_len],
            allowed,
            scale,
        )

        torch.testing.assert_close(out[:real_total_len], expected, atol=2e-4, rtol=2e-4)
        self.assertEqual(out.shape, (seq_len, _NUM_HEADS, _HEAD_DIM))
        self.assertTrue((out[real_total_len:] == 0).all())

    def test_sparse_output_matches_masked_sdpa(self):
        self._run_and_compare(_build_t2va_packed(), topk_ratio=0.4)

    def test_keyframe_joint_layout_matches_masked_sdpa(self):
        self._run_and_compare(
            _build_t2va_packed(include_keyframe_cond=True), topk_ratio=0.4
        )

    def test_full_ratio_matches_dense_attention(self):
        packed = _build_t2va_packed()
        metadata = _build_metadata(packed, [1.0])
        real_total_len = metadata.precomputed.layout.real_total_len
        q, k, v = _random_qkv(int(packed["seq_len"]))
        scale = _HEAD_DIM**-0.5

        with mock.patch.object(
            cube_backend,
            "_run_block_sparse_attention",
            _reference_block_sparse_attention,
        ):
            out = cube_sparse_attention(q, k, v, metadata, softmax_scale=scale)

        dense = _naive_masked_attention(
            q[:real_total_len],
            k[:real_total_len],
            v[:real_total_len],
            torch.ones(_NUM_HEADS, real_total_len, real_total_len, dtype=torch.bool),
            scale,
        )
        torch.testing.assert_close(out[:real_total_len], dense, atol=2e-4, rtol=2e-4)

    def test_ref2va_layout_end_to_end(self):
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=5,
            latent_t=2,
            latent_h=8,
            latent_w=12,
            audio_t=4,
            ref_blocks=[
                {"kind": "image", "latent_h": 8, "latent_w": 12},
                {
                    "kind": "video_audio",
                    "ref_audio_t": 3,
                    "latent_t": 2,
                    "latent_h": 8,
                    "latent_w": 12,
                },
            ],
        )
        layout = packed["stream_layout"]
        self.assertEqual(layout["cond_image_shapes"], ((1, 4, 6), (2, 4, 6)))
        self.assertEqual(layout["cond_audio_stream_lens"], (6,))
        self.assertEqual(
            layout["cond_event_orders"],
            (("imgvid", 0), ("audio", 0), ("imgvid", 1)),
        )
        self._run_and_compare(packed, topk_ratio=0.4)


class TestCubeSparseRequestConfig(unittest.TestCase):
    def test_full_ratio_uses_native_dense_backend(self):
        packed = _build_t2va_packed()
        metadata = _build_metadata(packed, [1.0])
        q, k, v = _random_qkv(int(packed["seq_len"]))
        expected = torch.empty_like(q)
        impl = CubeSparseAttentionImpl(
            num_heads=_NUM_HEADS,
            head_size=_HEAD_DIM,
            causal=False,
            softmax_scale=_HEAD_DIM**-0.5,
        )
        impl._dense_impl = mock.Mock()
        impl._dense_impl.forward_varlen.return_value = expected

        with set_forward_context(0, metadata):
            actual = impl.forward_varlen(
                q,
                k,
                v,
                cu_seqlens=packed["cu_seqlens"].to(torch.int32),
                max_seqlen=int(packed["cu_seqlens"][1]),
            )

        self.assertIs(actual, expected)
        impl._dense_impl.forward_varlen.assert_called_once()

    def test_cube_config_defaults_unrelated_vsa_sparsity(self):
        server_args = SimpleNamespace(
            attention_backend_config={
                "local_cube_size": [4, 4, 4],
                "topk_ratio_list": [1.0],
            },
            enable_trace=False,
        )

        request = prepare_request(server_args, SamplingParams(prompt="test"))

        self.assertEqual(request.VSA_sparsity, 0.0)


class TestCubeSparseAttentionCuda(unittest.TestCase):
    """Compiled-flex path on CUDA with the production cube size (64 tokens)."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("requires CUDA")

    def _run(self, topk_ratio):
        device = torch.device("cuda")
        packed = minimax_h3_packed_sequence(
            text_len=33,
            latent_t=4,
            latent_h=32,
            latent_w=48,
            audio_t=17,
            include_keyframe_cond=False,
        )
        metadata = CubeSparseAttentionMetadataBuilder().build(
            packed=packed,
            local_cube_size=(4, 4, 4),
            topk_ratio_list=[topk_ratio],
            num_steps=1,
            device=device,
        )
        pre = metadata.precomputed
        real_total_len = pre.layout.real_total_len
        seq_len = int(packed["seq_len"])
        generator = torch.Generator(device="cpu").manual_seed(1)
        q, k, v = (
            torch.randn(seq_len, 4, 128, dtype=torch.float32, generator=generator).to(
                device=device, dtype=torch.bfloat16
            )
            for _ in range(3)
        )
        scale = 128**-0.5

        out = cube_sparse_attention(q, k, v, metadata, softmax_scale=scale)

        label_mask = cube_topk_block_mask(
            q[:real_total_len], k[:real_total_len], pre, topk_ratio
        )
        labels = _token_labels(pre)
        allowed = label_mask[:, labels[:, None], labels[None, :]]
        expected = _naive_masked_attention(
            q[:real_total_len].float(),
            k[:real_total_len].float(),
            v[:real_total_len].float(),
            allowed,
            scale,
        )
        torch.testing.assert_close(
            out[:real_total_len].float(), expected, atol=2.6e-2, rtol=2.6e-2
        )
        self.assertTrue((out[real_total_len:] == 0).all())

    def test_compiled_flex_sparse(self):
        self._run(topk_ratio=0.25)

    def test_compiled_flex_full_ratio(self):
        self._run(topk_ratio=1.0)


if __name__ == "__main__":
    unittest.main()
