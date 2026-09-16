# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Cosmos3 Multiview-AV port: visibility predicate against
the spec truth table, run-compressed block sparsity, padded flex attention
against a dense masked GQA oracle, camera-major helpers, wrapped temporal
positions, deployment-contract parsing, request validation, and registry
wiring."""

import copy
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview import (
    COSMOS3_MADS_CAMERAS,
    MULTIVIEW_BACKEND_ENV_VAR,
    Cosmos3MultiviewConfig,
    parse_multiview_deployment_config,
    validate_lidar_config,
)
from sglang.multimodal_gen.configs.sample.cosmos3_multiview import (
    Cosmos3MultiviewSamplingParams,
    clamp_multiview_guidance_scale,
    closest_multiview_aspect_ratio,
    normalize_multiview_aspect_ratio,
    parse_local_condition_indexes,
    validate_lidar_request,
    validate_multiview_request,
)
from sglang.multimodal_gen.registry import (
    _PIPELINE_REGISTRY,
    _discover_and_register_pipelines,
    _get_config_info,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview import (
    pack_state,
    unpack_state,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview_attention import (
    DEFAULT_MAX_UND_TOKENS,
    TRITON_SPARSE_BLOCK_SIZES,
    MaskItem,
    MultiviewAttentionContext,
    MultiviewBlockSparsity,
    MultiviewLayout,
    build_multiview_block_sparsity,
    build_multiview_flex_metadata,
    expand_multiview_condition_frame_indexes,
    fa4_sparse_block_sizes,
    get_multiview_attention_plan,
    multiview_pair_predicate,
    padded_multiview_flex_attention,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
    compute_mrope_position_ids_vision,
)
from sglang.multimodal_gen.runtime.models.vaes.cosmos3_lidar_decoder import (
    _DepthToPixels,
    crop_lidar_width,
    depth_to_space,
    lidar_network_to_metric,
)
from sglang.multimodal_gen.runtime.models.vaes.cosmos3_lidar_encoder import (
    Cosmos3LidarEncoder,
    _SpaceToDepth,
    pad_lidar_sweeps,
    required_lidar_sweeps,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_lidar_outputs import (
    lidar_output_payload,
    lidar_payload_for_response,
    render_lidar_range_frames,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_multiview import (
    COSMOS3_MULTIVIEW_EMPHASIS,
    Cosmos3MultiviewInputStage,
    fit_uint8_cthw,
    format_multiview_prompts,
    format_per_view_prompts,
    format_separate_view_captions,
    media_kind,
    pad_view_frames_uint8,
    synthetic_multiview_pixels,
)

# ``transformer/config.json["multiview"]`` of the Cosmos3-Nano-Transfer-Auto export.
DEPLOYMENT_BLOCK = {
    "align_temporal_positions_across_views": True,
    "attention_scope": "decomposed",
    "backend": "triton",
    "cameras": list(COSMOS3_MADS_CAMERAS),
    "causal_training_strategy": "none",
    "control_attends_sensor": True,
    "decomposed_temporal_window_seconds": None,
    "max_views": 11,
    "share_vision_temporal_positions": True,
}


# The V1.2 LiDAR contract of the joint export (transformer/config.json multiview.lidar).
LIDAR_BLOCK = {
    "apply_validity_mask": True,
    "dtype": "float32",
    "fps": 10.0,
    "latent_channels": 128,
    "network_config": {
        "base_channels": 128,
        "bottleneck_3d": True,
        "bottleneck_3d_causal_time": True,
        "bottleneck_3d_max_t": 32,
        "bottleneck_3d_rope": True,
        "depths": [3, 3, 3, 3],
        "dilation": [1, 1, 1, 1],
        "formulation": "VAE",
        "in_channels": 3,
        "mapping_depth": 2,
        "mask_as_input": False,
        "mlp_ratio": 3.0,
        "num_heads": [4, 4, 8, 8],
        "out_channels": 3,
        "patch_size": [2, 2],
        "positional_embedding": "learnable_embedding",
        "resolution": [128, 1808],
        "temporal_downsample": [False, False, False],
        "temporal_upsample": [False, False, False],
        "window_size": [5, 45],
        "z_dim": 128,
    },
    "range_projection": {
        "azimuth_end_degrees": -180.0,
        "azimuth_endpoint": False,
        "azimuth_start_degrees": 180.0,
        "coordinate_system": "x_forward_y_left_z_up",
        "intensity_encoding": "unit",
        "invalid_range_m": 0.0,
        "max_range_m": 100.0,
        "min_range_m": 5.0,
        "model_width": 1808,
        "model_width_transform": "circular_pad",
        "native_height": 128,
        "native_width": 3600,
        "return_selection": "nearest",
        "semantic_height": 128,
        "semantic_width": 1800,
        "sensor": "pandar128",
        "validity_threshold": 0.5,
    },
    "sample_posterior": False,
    "spatial_compression": [16, 16],
    "streaming_chunk_frames": 9,
    "streaming_context_frames": 9,
    "temporal_compression_factor": 1,
    "version": "1.2",
}
INFERENCE_DEFAULTS = {
    "control_guidance": 1.0,
    "control_guidance_interval": None,
    "emphasize_control_in_prompt": True,
    "fps": 30.0,
    "guidance": 6.0,
    "guidance_interval": None,
    "negative_metadata_mode": "none",
    "normalize_cfg": False,
    "num_steps": 35,
    "resolution": "480",
    "shift": 10.0,
    "sigma_max": 80.0,
}
SCHEMA2_BLOCK = {
    **DEPLOYMENT_BLOCK,
    "decomposed_temporal_window_seconds": 0.4,
    "schema_version": 2,
    "separate_view_text_tokenization": True,
    "variable_view_count": True,
    "inference_defaults": INFERENCE_DEFAULTS,
    "lidar": LIDAR_BLOCK,
}


def _adjust_multiview(params, deployment):
    """The multiview-specific half of ``_adjust``; the base adjustment needs live ServerArgs."""
    params._apply_deployment_defaults(deployment)
    params._resolve_canvas(deployment)
    params._apply_guidance_policy(deployment)
    return params


def _fa4_available() -> bool:
    """FA4 CuTe block-sparse kernels exist for SM90 and SM100 and need the package."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (
        9,
        10,
    ):
        return False
    try:
        import cutlass  # noqa: F401
        import flash_attn.cute  # noqa: F401
    except Exception:
        return False
    return True


def _transformer_config(multiview=None, backbone_type="cosmos3_multiview"):
    return {
        "backbone_type": backbone_type,
        "multiview": copy.deepcopy(
            DEPLOYMENT_BLOCK if multiview is None else multiview
        ),
    }


def _tiny_metadata(
    attention_scope="decomposed",
    *,
    decomposed_temporal_window_seconds=None,
    control_attends_sensor=False,
):
    """Two views x two frames x one patch per item, two real text tokens, no padding.

    Sequence: [T0 T1 | Wa0 Wa1 Wb0 Wb1 | Ra0 Ra1 Rb0 Rb1].
    """
    item = MaskItem(token_shape=(4, 1, 1), num_views=2, seconds_per_frame=0.5)
    control = MaskItem(
        token_shape=(4, 1, 1), num_views=2, is_control=True, seconds_per_frame=0.5
    )
    return build_multiview_flex_metadata(
        seq_len=10,
        full_q_offsets=(2, 6, 10),
        items_per_sample=(control, item),
        device="cpu",
        num_und=2,
        attention_scope=attention_scope,
        decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
        control_attends_sensor=control_attends_sensor,
    )


# Truth table transcribed from the Multiview-AV visibility spec, one row per
# (query role, key role). Deliberately not the boolean expression the
# implementation evaluates, so a sign error cannot cancel out.
_SPEC_VISIBILITY = {
    ("control", "und"): "always",
    ("control", "control"): "same_view",
    ("control", "sensor"): "configured_same_view",
    ("sensor", "und"): "always",
    ("sensor", "control"): "same_view",
    ("sensor", "sensor"): "in_scope",
}


def _token_role(vectors, index):
    if bool(vectors[4][index]):
        return "und"
    if bool(vectors[3][index]):
        return "control"
    return "sensor"


def _spec_visible(
    q_role,
    k_role,
    *,
    same_sample,
    same_view,
    same_frame,
    within_temporal_window,
    attention_scope,
    has_temporal_window,
    control_attends_sensor,
):
    if not same_sample:
        return False
    rule = _SPEC_VISIBILITY[(q_role, k_role)]
    if rule == "always":
        return True
    if rule == "same_view":
        return same_view
    if rule == "configured_same_view":
        return control_attends_sensor and same_view
    if attention_scope == "all_views":
        return True
    if attention_scope == "same_view":
        return same_view
    return same_view or (within_temporal_window if has_temporal_window else same_frame)


class TestVisibilityPredicate(unittest.TestCase):
    def test_metadata_is_camera_major_and_marks_padding(self):
        item = MaskItem(token_shape=(6, 1, 2), num_views=2, seconds_per_frame=0.4)
        metadata = build_multiview_flex_metadata(
            seq_len=18,
            full_q_offsets=(4, 16),
            items_per_sample=(item,),
            device="cpu",
            num_und=3,
            attention_scope="same_view",
        )
        self.assertEqual(
            metadata.sample_id.tolist(), [0, 0, 0, -1] + [0] * 12 + [-1, -1]
        )
        self.assertEqual(metadata.is_und.tolist(), [True] * 3 + [False] * 15)
        self.assertEqual(
            metadata.frame_id[4:16].tolist(), [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
        )
        self.assertEqual(metadata.view_id[4:16].tolist(), [0] * 6 + [1] * 6)
        torch.testing.assert_close(
            metadata.timestamp[4:16],
            torch.tensor([0.0, 0.0, 0.4, 0.4, 0.8, 0.8] * 2),
        )
        self.assertEqual(metadata.q_len, 14)
        self.assertEqual(metadata.kv_len, 18)

    def test_predicate_matches_spec_truth_table(self):
        for attention_scope in ("all_views", "same_view", "decomposed"):
            for control_attends_sensor in (False, True):
                for temporal_window in (None, 0.5):
                    with self.subTest(
                        scope=attention_scope,
                        control_attends_sensor=control_attends_sensor,
                        window=temporal_window,
                    ):
                        metadata = _tiny_metadata(
                            attention_scope,
                            decomposed_temporal_window_seconds=temporal_window,
                            control_attends_sensor=control_attends_sensor,
                        )
                        q_index = torch.arange(metadata.q_len)[:, None]
                        kv_index = torch.arange(metadata.kv_len)[None, :]
                        actual = multiview_pair_predicate(metadata, q_index, kv_index)
                        q_vectors = metadata.query_vectors()
                        k_vectors = metadata.key_vectors()
                        expected = torch.zeros_like(actual)
                        covered = set()
                        for q in range(metadata.q_len):
                            q_role = _token_role(q_vectors, q)
                            for k in range(metadata.kv_len):
                                k_role = _token_role(k_vectors, k)
                                covered.add((q_role, k_role))
                                gap = float(q_vectors[5][q]) - float(k_vectors[5][k])
                                expected[q, k] = _spec_visible(
                                    q_role,
                                    k_role,
                                    same_sample=int(q_vectors[0][q])
                                    == int(k_vectors[0][k]),
                                    same_view=int(q_vectors[2][q])
                                    == int(k_vectors[2][k]),
                                    same_frame=int(q_vectors[1][q])
                                    == int(k_vectors[1][k]),
                                    within_temporal_window=(
                                        -1e-4 <= gap <= (temporal_window or 0.0) + 1e-4
                                    ),
                                    attention_scope=attention_scope,
                                    has_temporal_window=temporal_window is not None,
                                    control_attends_sensor=control_attends_sensor,
                                )
                        self.assertEqual(covered, set(_SPEC_VISIBILITY))
                        self.assertTrue(actual[:, :2].all())
                        torch.testing.assert_close(actual, expected)

    def test_visibility_rule_examples(self):
        metadata = _tiny_metadata()

        def visible(q, keys):
            return multiview_pair_predicate(
                metadata, torch.tensor(q), torch.tensor(keys)
            ).tolist()

        # Control token camera A frame 0: text and own-camera control only.
        self.assertEqual(
            visible(0, [0, 1, 2, 3, 4, 6]), [True, True, True, True, False, False]
        )
        # RGB camera A frame 0: own-camera RGB at every frame, camera B RGB at
        # frame 0 only, own-camera control at every frame.
        self.assertEqual(
            visible(4, [2, 3, 6, 7, 8, 9]), [True, True, True, True, True, False]
        )
        self.assertEqual(
            visible(5, [2, 3, 6, 7, 8, 9]), [True, True, True, True, False, True]
        )
        # control_attends_sensor opens the own-camera RGB keys to control queries.
        enabled = _tiny_metadata(control_attends_sensor=True)
        self.assertEqual(
            multiview_pair_predicate(
                enabled, torch.tensor(0), torch.tensor([6, 7, 8])
            ).tolist(),
            [True, True, False],
        )

    def test_padding_queries_attend_only_padding(self):
        for attention_scope in ("all_views", "same_view", "decomposed"):
            with self.subTest(scope=attention_scope):
                item = MaskItem(token_shape=(2, 1, 2), num_views=1)
                metadata = build_multiview_flex_metadata(
                    seq_len=9,
                    full_q_offsets=(3, 7),
                    items_per_sample=(item,),
                    device="cpu",
                    num_und=2,
                    attention_scope=attention_scope,
                )
                allowed = multiview_pair_predicate(
                    metadata,
                    torch.arange(metadata.q_len)[:, None],
                    torch.arange(metadata.kv_len)[None, :],
                )
                q_padding = metadata.query_vectors()[0] == -1
                kv_padding = metadata.sample_id == -1
                self.assertTrue(allowed[q_padding][:, kv_padding].all())
                self.assertFalse(allowed[q_padding][:, ~kv_padding].any())
                self.assertFalse(allowed[~q_padding][:, kv_padding].any())
                self.assertTrue(allowed[q_padding].any(dim=-1).all())

    def test_decomposed_temporal_window_boundaries(self):
        cases = [
            (1.0, 1.0, 0.5, True),
            (1.0, 0.5, 0.5, True),
            (1.0, 0.4998, 0.5, False),
            (1.0, 1.00009, 0.5, True),
            (1.0, 1.0002, 0.5, False),
        ]
        for q_timestamp, k_timestamp, window, expected in cases:
            with self.subTest(q=q_timestamp, k=k_timestamp):
                metadata = _tiny_metadata(decomposed_temporal_window_seconds=window)
                # Two RGB tokens of different cameras and frames: only the
                # temporal window decides.
                metadata.timestamp[6] = q_timestamp
                metadata.timestamp[9] = k_timestamp
                allowed = multiview_pair_predicate(
                    metadata, torch.tensor(4), torch.tensor([9])
                )
                self.assertEqual(bool(allowed[0]), expected)

    def test_rejects_mixed_view_offsets_without_window(self):
        first = MaskItem(token_shape=(2, 1, 1), num_views=1)
        second = MaskItem(token_shape=(2, 1, 1), num_views=1, view_offset=1)
        with self.assertRaisesRegex(ValueError, "mixed view offsets"):
            build_multiview_flex_metadata(
                seq_len=6,
                full_q_offsets=(2, 4, 6),
                items_per_sample=(first, second),
                device="cpu",
                num_und=2,
            )


class TestBlockSparsity(unittest.TestCase):
    def _metadata(self, control_attends_sensor=True):
        control = MaskItem(token_shape=(4, 1, 2), num_views=2, is_control=True)
        target = MaskItem(token_shape=(4, 1, 2), num_views=2)
        # 8 padded text slots (3 real) + 8 control + 8 target tokens.
        return build_multiview_flex_metadata(
            seq_len=24,
            full_q_offsets=(8, 16, 24),
            items_per_sample=(control, target),
            device="cpu",
            num_und=3,
            control_attends_sensor=control_attends_sensor,
        )

    def test_block_classification_matches_dense_projection(self):
        metadata = self._metadata()
        q_block, kv_block = 4, 4
        sparsity = build_multiview_block_sparsity(
            metadata, q_block_size=q_block, kv_block_size=kv_block
        )
        dense = multiview_pair_predicate(
            metadata,
            torch.arange(metadata.q_len)[:, None],
            torch.arange(metadata.kv_len)[None, :],
        )
        num_q_blocks = metadata.q_len // q_block
        num_kv_blocks = metadata.kv_len // kv_block
        for qb in range(num_q_blocks):
            full = set(
                sparsity.full_indices[qb, : int(sparsity.full_counts[qb])].tolist()
            )
            partial = set(
                sparsity.partial_indices[
                    qb, : int(sparsity.partial_counts[qb])
                ].tolist()
            )
            self.assertFalse(full & partial)
            for kb in range(num_kv_blocks):
                tile = dense[
                    qb * q_block : (qb + 1) * q_block,
                    kb * kv_block : (kb + 1) * kv_block,
                ]
                if tile.all():
                    self.assertIn(kb, full, (qb, kb))
                elif tile.any():
                    self.assertIn(kb, partial, (qb, kb))
                else:
                    self.assertNotIn(kb, full | partial, (qb, kb))
        # Full-width contiguous index layout, as create_block_mask produces.
        self.assertEqual(sparsity.full_indices.shape, (num_q_blocks, num_kv_blocks))
        self.assertEqual(sparsity.partial_indices.shape, (num_q_blocks, num_kv_blocks))

    def test_run_table_reproduces_the_pair_predicate(self):
        metadata = self._metadata()
        sparsity = build_multiview_block_sparsity(
            metadata, q_block_size=4, kv_block_size=4
        )
        dense = multiview_pair_predicate(
            metadata,
            torch.arange(metadata.q_len)[:, None],
            torch.arange(metadata.kv_len)[None, :],
        )
        q_groups = (sparsity.q_word_base // sparsity.words_per_row).long()
        k_groups = sparsity.k_group_ids.long()
        via_table = sparsity.group_allowed[q_groups[:, None], k_groups[None, :]]
        torch.testing.assert_close(via_table, dense)
        # Packed words round-trip bit by bit.
        words = (
            sparsity.allowed_words.view(-1, sparsity.words_per_row).to(torch.int64)
            & 0xFFFFFFFF
        )
        bits = (words[:, :, None] >> torch.arange(32)) & 1
        unpacked = bits.reshape(words.shape[0], -1)[
            :, : sparsity.group_allowed.shape[1]
        ]
        torch.testing.assert_close(unpacked.bool(), sparsity.group_allowed)

    def test_block_mask_carries_block_visibility_and_exact_mask_mod(self):
        metadata = self._metadata()
        q_block, kv_block = 4, 4
        sparsity = build_multiview_block_sparsity(
            metadata, q_block_size=q_block, kv_block_size=kv_block
        )
        block_mask = sparsity.to_block_mask()
        self.assertEqual(block_mask.seq_lengths, (metadata.q_len, metadata.kv_len))
        dense = multiview_pair_predicate(
            metadata,
            torch.arange(metadata.q_len)[:, None],
            torch.arange(metadata.kv_len)[None, :],
        )
        # ``to_dense`` is block-granular: a block is visible when any pair is.
        visible_blocks = dense.view(
            metadata.q_len // q_block, q_block, metadata.kv_len // kv_block, kv_block
        ).any(dim=(1, 3))
        torch.testing.assert_close(block_mask.to_dense()[0, 0].bool(), visible_blocks)
        # Inside partial blocks the kernel consults mask_mod, which must be the
        # exact predicate.
        q_idx = torch.arange(metadata.q_len)[:, None]
        kv_idx = torch.arange(metadata.kv_len)[None, :]
        torch.testing.assert_close(
            block_mask.mask_mod(torch.tensor(0), torch.tensor(0), q_idx, kv_idx), dense
        )


class TestPaddedFlexAttention(unittest.TestCase):
    def _oracle(self, q, k, v, k_und, v_und, layout, real_und_len):
        """Dense masked GQA attention over [text | GEN] in float32."""
        padded_und = ((layout.max_und_tokens + 63) // 64) * 64
        padded_q = ((layout.gen_tokens + 63) // 64) * 64
        items = layout.mask_items()
        offsets = tuple(padded_und + i * layout.item_tokens for i in range(3))
        metadata = build_multiview_flex_metadata(
            seq_len=padded_und + padded_q,
            full_q_offsets=offsets,
            items_per_sample=items,
            device="cpu",
            num_und=real_und_len,
            attention_scope=layout.attention_scope,
            control_attends_sensor=layout.control_attends_sensor,
        )
        gen = layout.gen_tokens
        q_index = torch.arange(gen)[:, None]
        key_index = torch.cat(
            [torch.arange(real_und_len), padded_und + torch.arange(gen)]
        )[None, :]
        mask = multiview_pair_predicate(metadata, q_index, key_index)
        keys = torch.cat([k_und, k], dim=1).float()
        values = torch.cat([v_und, v], dim=1).float()
        heads, kv_heads = q.shape[2], k.shape[2]
        group = heads // kv_heads
        keys = keys.repeat_interleave(group, dim=2)
        values = values.repeat_interleave(group, dim=2)
        scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), keys) / (q.shape[-1] ** 0.5)
        scores = scores.masked_fill(~mask[None, None], float("-inf"))
        probs = scores.softmax(dim=-1)
        return torch.einsum("bhqk,bkhd->bqhd", probs, values)

    def _run(self, device, dtype, atol, rtol, backend="triton"):
        torch.manual_seed(0)
        layout = MultiviewLayout(
            num_views=2,
            latent_frames=6,
            patch_height=2,
            patch_width=3,
            control_attends_sensor=True,
            seconds_per_frame=0.2,
            backend=backend,
            max_und_tokens=70,
        )
        heads, kv_heads, head_dim, real_und_len = 4, 2, 16, 7
        gen = layout.gen_tokens
        q = torch.randn(1, gen, heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(1, gen, kv_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(1, gen, kv_heads, head_dim, device=device, dtype=dtype)
        k_und = torch.randn(
            1, real_und_len, kv_heads, head_dim, device=device, dtype=dtype
        )
        v_und = torch.randn(
            1, real_und_len, kv_heads, head_dim, device=device, dtype=dtype
        )
        context = MultiviewAttentionContext(layout, {}, {})
        out = padded_multiview_flex_attention(q, k, v, k_und, v_und, context)
        self.assertEqual(tuple(out.shape), (1, gen, heads, head_dim))
        expected = self._oracle(
            q.cpu(), k.cpu(), v.cpu(), k_und.cpu(), v_und.cpu(), layout, real_und_len
        )
        torch.testing.assert_close(out.float().cpu(), expected, atol=atol, rtol=rtol)
        return context

    def test_cpu_matches_dense_masked_gqa_oracle(self):
        context = self._run(torch.device("cpu"), torch.float32, atol=1e-4, rtol=1e-4)
        # One plan per (layout, text length) and reusable packing buffers.
        self.assertEqual(len(context.mask_cache), 1)
        self.assertEqual(
            {key[0].split(":")[1] for key in context.buffer_cache}, {"q", "k", "v"}
        )

    def test_padding_capacity_does_not_change_the_output(self):
        torch.manual_seed(1)
        base = dict(
            num_views=2,
            latent_frames=4,
            patch_height=1,
            patch_width=2,
            seconds_per_frame=0.5,
        )
        small = MultiviewLayout(max_und_tokens=10, **base)
        large = MultiviewLayout(max_und_tokens=200, **base)
        gen = small.gen_tokens
        q = torch.randn(1, gen, 2, 8)
        k = torch.randn(1, gen, 1, 8)
        v = torch.randn(1, gen, 1, 8)
        k_und = torch.randn(1, 5, 1, 8)
        v_und = torch.randn(1, 5, 1, 8)
        out_small = padded_multiview_flex_attention(
            q, k, v, k_und, v_und, MultiviewAttentionContext(small, {}, {})
        )
        out_large = padded_multiview_flex_attention(
            q, k, v, k_und, v_und, MultiviewAttentionContext(large, {}, {})
        )
        torch.testing.assert_close(out_small, out_large, atol=1e-5, rtol=1e-5)

    def test_plan_geometry_is_independent_of_prompt_length(self):
        layout = MultiviewLayout(
            num_views=2,
            latent_frames=4,
            patch_height=1,
            patch_width=2,
            max_und_tokens=100,
        )
        context = MultiviewAttentionContext(layout, {}, {})
        _, short = get_multiview_attention_plan(
            context,
            real_und_len=5,
            real_q_len=layout.gen_tokens,
            device=torch.device("cpu"),
        )
        _, long = get_multiview_attention_plan(
            context,
            real_und_len=90,
            real_q_len=layout.gen_tokens,
            device=torch.device("cpu"),
        )
        self.assertEqual(short.padded_und_len, 128)
        self.assertEqual(short.padded_und_len, long.padded_und_len)
        self.assertEqual(short.padded_q_len, long.padded_q_len)
        self.assertEqual(len(context.mask_cache), 2)
        with self.assertRaisesRegex(ValueError, "exceeds the layout capacity"):
            get_multiview_attention_plan(
                context,
                real_und_len=101,
                real_q_len=layout.gen_tokens,
                device=torch.device("cpu"),
            )
        with self.assertRaisesRegex(ValueError, "does not match the request layout"):
            get_multiview_attention_plan(
                context,
                real_und_len=5,
                real_q_len=layout.gen_tokens + 1,
                device=torch.device("cpu"),
            )

    @unittest.skipUnless(
        torch.cuda.is_available(), "needs a CUDA device for the Triton kernel"
    )
    def test_cuda_triton_kernel_matches_dense_oracle(self):
        self._run(torch.device("cuda"), torch.bfloat16, atol=3e-2, rtol=3e-2)

    @unittest.skipUnless(
        _fa4_available(),
        "needs an SM90 or SM100 device with the flash-attn-4 CuTe package",
    )
    def test_cuda_fa4_kernel_matches_dense_oracle(self):
        context = self._run(
            torch.device("cuda"), torch.bfloat16, atol=3e-2, rtol=3e-2, backend="fa4"
        )
        plan = next(iter(context.mask_cache.values()))
        self.assertIsInstance(plan, MultiviewBlockSparsity)
        self.assertEqual(
            (plan.q_block_size, plan.kv_block_size),
            fa4_sparse_block_sizes(torch.device("cuda")),
        )

    def test_fa4_block_map_matches_dense_projection(self):
        """Both FA4 geometries must classify tiles exactly like the 64x64 map does."""
        for block_sizes in ((256, 128), (128, 128)):
            with self.subTest(block_sizes=block_sizes):
                layout = MultiviewLayout(
                    num_views=2,
                    latent_frames=4,
                    patch_height=4,
                    patch_width=16,
                    control_attends_sensor=True,
                    backend="fa4",
                    max_und_tokens=100,
                    fa4_block_sizes=block_sizes,
                )
                context = MultiviewAttentionContext(layout, {}, {})
                plan, geometry = get_multiview_attention_plan(
                    context,
                    real_und_len=9,
                    real_q_len=layout.gen_tokens,
                    device=torch.device("cpu"),
                )
                self.assertIsInstance(plan, MultiviewBlockSparsity)
                self.assertEqual((plan.q_block_size, plan.kv_block_size), block_sizes)
                self.assertEqual(
                    (geometry.padded_q_len, geometry.padded_und_len), (512, 128)
                )
                self.assertEqual(plan.q_word_base.numel(), geometry.padded_q_len)
                self.assertEqual(
                    plan.k_group_ids.numel(),
                    geometry.padded_und_len + geometry.padded_q_len,
                )
                metadata = plan.metadata
                dense = multiview_pair_predicate(
                    metadata,
                    torch.arange(metadata.q_len)[:, None],
                    torch.arange(metadata.kv_len)[None, :],
                )
                q_block, kv_block = block_sizes
                for qb in range(metadata.q_len // q_block):
                    full = set(
                        plan.full_indices[qb, : int(plan.full_counts[qb])].tolist()
                    )
                    partial = set(
                        plan.partial_indices[
                            qb, : int(plan.partial_counts[qb])
                        ].tolist()
                    )
                    for kb in range(metadata.kv_len // kv_block):
                        tile = dense[
                            qb * q_block : (qb + 1) * q_block,
                            kb * kv_block : (kb + 1) * kv_block,
                        ]
                        if tile.all():
                            self.assertIn(kb, full, (qb, kb))
                        elif tile.any():
                            self.assertIn(kb, partial, (qb, kb))
                        else:
                            self.assertNotIn(kb, full | partial, (qb, kb))

    def test_fa4_block_geometry_follows_compute_capability(self):
        self.assertEqual(fa4_sparse_block_sizes(capability_major=9), (128, 128))
        self.assertEqual(fa4_sparse_block_sizes(capability_major=10), (256, 128))
        self.assertEqual(fa4_sparse_block_sizes(capability_major=11), (256, 128))
        for unsupported in (8, 12):
            with self.subTest(capability=unsupported):
                with self.assertRaisesRegex(ValueError, "not available"):
                    fa4_sparse_block_sizes(capability_major=unsupported)
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            fa4_sparse_block_sizes(torch.device("cpu"))
        common = dict(num_views=1, latent_frames=1, patch_height=1, patch_width=1)
        self.assertEqual(
            MultiviewLayout(**common).sparse_block_sizes(torch.device("cpu")),
            TRITON_SPARSE_BLOCK_SIZES,
        )
        pinned = MultiviewLayout(backend="fa4", fa4_block_sizes=(128, 128), **common)
        self.assertEqual(pinned.sparse_block_sizes(torch.device("cpu")), (128, 128))
        self.assertIn((128, 128), pinned.cache_key())
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            MultiviewLayout(backend="fa4", **common).sparse_block_sizes(
                torch.device("cpu")
            )
        with self.assertRaisesRegex(ValueError, "fa4_block_sizes"):
            MultiviewLayout(backend="fa4", fa4_block_sizes=(0, 128), **common)


class TestLayoutHelpers(unittest.TestCase):
    def test_expand_condition_indexes_camera_major(self):
        self.assertEqual(
            expand_multiview_condition_frame_indexes([0], 11, 11 * 24),
            [view * 24 for view in range(11)],
        )
        self.assertEqual(
            expand_multiview_condition_frame_indexes([1, 0, 99], 2, 6), [0, 1, 3, 4]
        )
        self.assertEqual(expand_multiview_condition_frame_indexes(None, 2, 6), [])
        with self.assertRaises(ValueError):
            expand_multiview_condition_frame_indexes([0], 2, 5)

    def test_layout_geometry_and_validation(self):
        layout = MultiviewLayout(
            num_views=11, latent_frames=264, patch_height=15, patch_width=26
        )
        self.assertEqual(layout.frames_per_view, 24)
        self.assertEqual(layout.item_tokens, 102_960)
        self.assertEqual(layout.gen_tokens, 205_920)
        self.assertEqual(layout.sparse_block_sizes(torch.device("cpu")), (64, 64))
        self.assertEqual(layout.max_und_tokens, DEFAULT_MAX_UND_TOKENS)
        with self.assertRaisesRegex(ValueError, "divisible by num_views"):
            MultiviewLayout(
                num_views=11, latent_frames=263, patch_height=15, patch_width=26
            )
        with self.assertRaisesRegex(ValueError, "backend"):
            MultiviewLayout(
                num_views=1,
                latent_frames=1,
                patch_height=1,
                patch_width=1,
                backend="cuda",
            )
        with self.assertRaisesRegex(ValueError, "attention_scope"):
            MultiviewLayout(
                num_views=1,
                latent_frames=1,
                patch_height=1,
                patch_width=1,
                attention_scope="all",
            )

    def test_temporal_position_period_wraps_camera_major_frames(self):
        ids, _ = compute_mrope_position_ids_vision(
            6,
            1,
            1,
            temporal_offset=0,
            device=torch.device("cpu"),
            temporal_position_period=3,
        )
        self.assertEqual(ids[0].tolist(), [0, 1, 2, 0, 1, 2])
        ids_fps, _ = compute_mrope_position_ids_vision(
            6,
            1,
            1,
            temporal_offset=0,
            device=torch.device("cpu"),
            fps=30.0,
            base_fps=24.0,
            temporal_compression_factor=4,
            temporal_position_period=3,
        )
        torch.testing.assert_close(
            ids_fps[0], torch.tensor([0.0, 0.8, 1.6, 0.0, 0.8, 1.6])
        )
        plain, _ = compute_mrope_position_ids_vision(
            6, 1, 1, temporal_offset=0, device=torch.device("cpu")
        )
        self.assertEqual(plain[0].tolist(), [0, 1, 2, 3, 4, 5])
        with self.assertRaisesRegex(ValueError, "positive"):
            compute_mrope_position_ids_vision(
                6,
                1,
                1,
                temporal_offset=0,
                device=torch.device("cpu"),
                temporal_position_period=0,
            )


class TestPixelHelpersAndPrompt(unittest.TestCase):
    def test_fisheye_canvas_crop_rounds_half_to_even(self):
        """1720x1080 fisheye sources resize to 523 rows for a 480x832 canvas; the
        21.5-row crop offset must round to 22 like imaginaire4. A floor offset
        shifted every fisheye anchor by one row (28 dB instead of 45 dB parity)."""
        ramp = (
            torch.arange(1080, dtype=torch.int64)
            .remainder(256)
            .to(torch.uint8)
            .view(1, 1, 1080, 1)
            .expand(3, 1, 1080, 1720)
        )

        actual = fit_uint8_cthw(ramp, height=480, width=832)

        resized = torch.nn.functional.interpolate(
            ramp.permute(1, 0, 2, 3).float(),
            size=(523, 832),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        expected = resized[0, :, 22:502].round().clamp(0, 255).to(torch.uint8)
        self.assertEqual(tuple(actual.shape), (3, 1, 480, 832))
        self.assertTrue(torch.equal(actual[:, 0], expected))
        self.assertFalse(
            torch.equal(actual[:, 0], resized[0, :, 21:501].round().to(torch.uint8))
        )

    def test_pad_view_frames_truncates_or_repeats_last_frame(self):
        frames = torch.stack(
            [torch.full((3, 2, 3), value, dtype=torch.uint8) for value in (10, 20)],
            dim=1,
        )
        padded = pad_view_frames_uint8(frames, num_frames=5)
        self.assertEqual(tuple(padded.shape), (3, 5, 2, 3))
        self.assertEqual(padded[:, 0].unique().tolist(), [10])
        self.assertEqual(padded[:, 1:].unique().tolist(), [20])
        truncated = pad_view_frames_uint8(frames, num_frames=1)
        self.assertEqual(truncated.unique().tolist(), [10])
        with self.assertRaisesRegex(ValueError, "zero frames"):
            pad_view_frames_uint8(frames[:, :0], num_frames=3)

    def test_synthetic_pixels_and_media_kind(self):
        pixels = synthetic_multiview_pixels(
            num_views=3, num_frames=5, height=4, width=6
        )
        self.assertEqual(tuple(pixels.shape), (1, 3, 15, 4, 6))
        self.assertEqual(pixels.dtype, torch.uint8)
        self.assertEqual(media_kind("a/front.PNG"), "image")
        self.assertEqual(media_kind("a/front.mp4"), "video")

    def test_prose_prompt_gets_metadata_and_emphasis(self):
        positive, negative = format_multiview_prompts(
            "A rainy street.",
            "",
            num_frames=93,
            fps=30.0,
            height=480,
            width=832,
        )
        self.assertEqual(
            positive,
            "A rainy street. The video is 3.1 seconds long and is of 30 FPS. "
            "This video is of 480x832 resolution. " + COSMOS3_MULTIVIEW_EMPHASIS,
        )
        self.assertEqual(
            negative,
            "The video is 3.1 seconds long and is of 30 FPS. This video is of 480x832 resolution.",
        )
        _, inverse = format_multiview_prompts(
            "x",
            "bad",
            num_frames=93,
            fps=30.0,
            height=480,
            width=832,
            negative_metadata_mode="inverse",
        )
        self.assertIn("is not 3.1 seconds long", inverse)
        _, none = format_multiview_prompts(
            "x",
            "bad",
            num_frames=93,
            fps=30.0,
            height=480,
            width=832,
            negative_metadata_mode="none",
        )
        self.assertEqual(none, "bad")

    def test_json_prompt_gets_metadata_fields(self):
        import json

        positive, _ = format_multiview_prompts(
            json.dumps({"description": "a bus"}),
            "",
            num_frames=93,
            fps=30.0,
            height=480,
            width=832,
        )
        json_part, _, suffix = positive.partition("} ")
        caption = json.loads(json_part + "}")
        self.assertEqual(caption["duration"], "3s")
        self.assertEqual(caption["fps"], 30.0)
        self.assertEqual(caption["resolution"], {"H": 480, "W": 832})
        self.assertEqual(caption["aspect_ratio"], "16,9")
        self.assertEqual(suffix, COSMOS3_MULTIVIEW_EMPHASIS)


class TestDeploymentConfig(unittest.TestCase):
    def test_accepts_the_exported_contract(self):
        deployment = parse_multiview_deployment_config(_transformer_config())
        self.assertEqual(deployment.cameras, COSMOS3_MADS_CAMERAS)
        self.assertEqual(deployment.num_views, 11)
        self.assertTrue(deployment.control_attends_sensor)
        self.assertTrue(deployment.align_temporal_positions_across_views)
        self.assertIsNone(deployment.decomposed_temporal_window_seconds)
        self.assertEqual(deployment.backend, "triton")

    def test_accepts_the_schema2_joint_contract(self):
        deployment = parse_multiview_deployment_config(
            _transformer_config(SCHEMA2_BLOCK)
        )
        self.assertEqual(deployment.schema_version, 2)
        self.assertFalse(deployment.is_legacy)
        self.assertTrue(deployment.separate_view_text_tokenization)
        self.assertTrue(deployment.variable_view_count)
        self.assertTrue(deployment.supports_lidar)
        self.assertEqual(deployment.decomposed_temporal_window_seconds, 0.4)
        self.assertEqual(deployment.inference_default("num_steps", 1), 35)
        self.assertEqual(deployment.inference_default("resolution", "720"), "480")
        self.assertEqual(deployment.inference_default("guidance_interval", "x"), "x")
        self.assertEqual(deployment.lidar["fps"], 10.0)
        # Schema 2 may reorder or subset the rig; legacy exports may not.
        block = copy.deepcopy(SCHEMA2_BLOCK)
        block["cameras"] = list(reversed(COSMOS3_MADS_CAMERAS))
        self.assertEqual(
            parse_multiview_deployment_config(_transformer_config(block)).cameras,
            tuple(reversed(COSMOS3_MADS_CAMERAS)),
        )
        for field, value, message in (
            ("schema_version", 3, "schema_version"),
            ("inference_defaults", {"resolution": "480"}, "inference_defaults"),
            ("separate_view_text_tokenization", "yes", "boolean"),
        ):
            with self.subTest(field=field):
                block = copy.deepcopy(SCHEMA2_BLOCK)
                block[field] = value
                with self.assertRaisesRegex((ValueError, TypeError), message):
                    parse_multiview_deployment_config(_transformer_config(block))
        block = copy.deepcopy(DEPLOYMENT_BLOCK)
        block["lidar"] = LIDAR_BLOCK
        with self.assertRaisesRegex(ValueError, "schema_version=2"):
            parse_multiview_deployment_config(_transformer_config(block))

    def test_lidar_contract_validation(self):
        self.assertEqual(validate_lidar_config(LIDAR_BLOCK)["version"], "1.2")
        for path, value, message in (
            (("version",), "1.1", "V1.2"),
            (("range_projection", "model_width"), 1800, "circularly padded"),
            (("network_config", "patch_size"), [2, 4], "disagree"),
            (("streaming_context_frames",), 4, "chunk length"),
        ):
            with self.subTest(path=path):
                block = copy.deepcopy(LIDAR_BLOCK)
                target = block
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_lidar_config(block)

    def test_rejects_wrong_backbone_before_multiview_fields(self):
        with self.assertRaisesRegex(ValueError, "backbone_type"):
            parse_multiview_deployment_config(
                {"backbone_type": None, "multiview": None}
            )

    def test_rejects_malformed_fields(self):
        bad_cases = {
            "causal_training_strategy": ("teacher_forcing", ValueError),
            "attention_scope": ("everything", ValueError),
            "decomposed_temporal_window_seconds": (-1.0, ValueError),
            "control_attends_sensor": ("yes", TypeError),
            "share_vision_temporal_positions": (False, ValueError),
            "cameras": (list(reversed(COSMOS3_MADS_CAMERAS)), ValueError),
            "max_views": (10, ValueError),
            "backend": ("cuda", ValueError),
        }
        for field, (value, error) in bad_cases.items():
            with self.subTest(field=field):
                block = copy.deepcopy(DEPLOYMENT_BLOCK)
                block[field] = value
                with self.assertRaises(error):
                    parse_multiview_deployment_config(_transformer_config(block))
        for field in DEPLOYMENT_BLOCK:
            with self.subTest(missing=field):
                block = copy.deepcopy(DEPLOYMENT_BLOCK)
                del block[field]
                with self.assertRaises(ValueError):
                    parse_multiview_deployment_config(_transformer_config(block))

    def test_config_flags_and_backend_resolution(self):
        config = Cosmos3MultiviewConfig()
        self.assertEqual(
            config.transformer_class_override, "Cosmos3MultiviewTransformer"
        )
        self.assertTrue(config.use_system_prompt)
        self.assertFalse(config.supports_action_endpoint())
        self.assertFalse(config.supports_dynamic_batching())
        with (
            mock.patch(
                "sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview._transformer_config",
                return_value=_transformer_config(),
            ),
            mock.patch(
                "sglang.multimodal_gen.configs.pipeline_configs.cosmos3.get_distilled_sigmas",
                return_value=None,
            ),
            mock.patch(
                "sglang.multimodal_gen.configs.pipeline_configs.cosmos3.is_edge_checkpoint",
                return_value=False,
            ),
            mock.patch(
                "sglang.multimodal_gen.configs.pipeline_configs.cosmos3.is_nano_checkpoint",
                return_value=True,
            ),
        ):
            config.update_config_from_dict({"model_path": "/models/mv"})
        self.assertEqual(config.multiview_deployment.num_views, 11)
        self.assertEqual(config.resolved_multiview_backend(), "triton")
        config.multiview_attention_backend = "fa4"
        self.assertEqual(config.resolved_multiview_backend(), "fa4")
        config.multiview_attention_backend = "sdpa"
        with self.assertRaises(ValueError):
            config.resolved_multiview_backend()
        # Environment override sits between the explicit field and the checkpoint.
        config.multiview_attention_backend = None
        with mock.patch.dict(os.environ, {MULTIVIEW_BACKEND_ENV_VAR: "fa4"}):
            self.assertEqual(config.resolved_multiview_backend(), "fa4")
            config.multiview_attention_backend = "triton"
            self.assertEqual(config.resolved_multiview_backend(), "triton")
            config.multiview_attention_backend = None
        with mock.patch.dict(os.environ, {MULTIVIEW_BACKEND_ENV_VAR: "sdpa"}):
            with self.assertRaisesRegex(ValueError, MULTIVIEW_BACKEND_ENV_VAR):
                config.resolved_multiview_backend()
        with mock.patch.dict(os.environ, {MULTIVIEW_BACKEND_ENV_VAR: ""}):
            self.assertEqual(config.resolved_multiview_backend(), "triton")
        for name in ("tp_size", "sp_degree", "ulysses_degree", "ring_degree"):
            with self.subTest(arg=name):
                server_args = SimpleNamespace(
                    tp_size=1,
                    sp_degree=1,
                    ulysses_degree=1,
                    ring_degree=1,
                    enable_cfg_parallel=False,
                )
                setattr(server_args, name, 2)
                with self.assertRaises(ValueError):
                    config.validate_server_args(server_args)
        # CFG parallel only splits the two branches across ranks; it is allowed.
        config.validate_server_args(
            SimpleNamespace(
                tp_size=1,
                sp_degree=1,
                ulysses_degree=1,
                ring_degree=1,
                enable_cfg_parallel=True,
            )
        )


def _views(cameras, *, vision=False):
    views = []
    for index, camera in enumerate(cameras):
        view = {"camera_key": camera, "control_path": f"control_{index}.mp4"}
        if vision:
            view["vision_path"] = f"vision_{index}.png"
        views.append(view)
    return views


class TestSamplingParamsAndInputStage(unittest.TestCase):
    def test_defaults_follow_the_reference(self):
        params = Cosmos3MultiviewSamplingParams()
        # The canvas is a bucket choice made at adjustment, not a field default.
        self.assertEqual((params.width, params.height), (None, None))
        self.assertEqual(params.num_frames, 93)
        self.assertEqual(params.fps, 30)
        self.assertEqual(params.guidance_scale, 6.0)
        self.assertEqual(params.num_inference_steps, 35)
        self.assertEqual(params.negative_metadata_mode, "same")
        self.assertEqual(params.resolve_views(), [])
        for name in ("multiview", "lidar", "resolution", "aspect_ratio"):
            self.assertIn(
                name, Cosmos3MultiviewSamplingParams.video_request_extra_fields()
            )

    def test_adjust_resolves_canvas_and_checkpoint_defaults(self):
        legacy = parse_multiview_deployment_config(_transformer_config())
        params = _adjust_multiview(
            Cosmos3MultiviewSamplingParams(guidance_scale=9.0), legacy
        )
        self.assertEqual((params.width, params.height), (832, 480))
        self.assertEqual(params.num_frames, 93)
        self.assertEqual(params.guidance_scale, 7.0)
        with self.assertRaisesRegex(ValueError, "fixed at 480p 16:9"):
            _adjust_multiview(Cosmos3MultiviewSamplingParams(resolution="720"), legacy)

        schema2 = parse_multiview_deployment_config(_transformer_config(SCHEMA2_BLOCK))
        params = _adjust_multiview(
            Cosmos3MultiviewSamplingParams(guidance_scale=9.0), schema2
        )
        self.assertEqual((params.width, params.height), (832, 480))
        self.assertEqual(params.num_frames, 201)
        self.assertEqual(params.guidance_scale, 9.0)
        self.assertEqual(params.flow_shift, 10.0)
        self.assertEqual(params.aspect_ratio, "auto")
        portrait = _adjust_multiview(
            Cosmos3MultiviewSamplingParams(resolution=720, aspect_ratio="9:16"), schema2
        )
        self.assertEqual((portrait.width, portrait.height), (720, 1280))
        explicit = _adjust_multiview(
            Cosmos3MultiviewSamplingParams(width=1104, height=832, resolution="720"),
            schema2,
        )
        self.assertEqual(explicit.aspect_ratio, "4,3")
        with self.assertRaisesRegex(ValueError, "requires width=1280"):
            _adjust_multiview(
                Cosmos3MultiviewSamplingParams(
                    width=832, height=480, resolution="720", aspect_ratio="16:9"
                ),
                schema2,
            )
        self.assertEqual(normalize_multiview_aspect_ratio("1920:1080"), "16,9")
        self.assertEqual(closest_multiview_aspect_ratio(1084, 1924, "480"), "16,9")
        self.assertEqual(closest_multiview_aspect_ratio(1000, 1000, "720"), "1,1")

    def test_views_from_multiview_object_and_control_path_list(self):
        params = Cosmos3MultiviewSamplingParams(
            multiview={
                "views": _views(("front", "left"), vision=True),
                "condition_video_as_image": True,
            }
        )
        views = params.resolve_views()
        self.assertEqual([view.camera_key for view in views], ["front", "left"])
        self.assertEqual(
            [view.vision for view in views], ["vision_0.png", "vision_1.png"]
        )
        self.assertTrue(params.resolved_condition_video_as_image())
        self.assertIsNone(params.resolved_local_condition_indexes())
        t2v = Cosmos3MultiviewSamplingParams(control_path=["a.mp4", "b.mp4"])
        self.assertEqual(
            [view.control for view in t2v.resolve_views()], ["a.mp4", "b.mp4"]
        )
        self.assertTrue(all(view.camera_key is None for view in t2v.resolve_views()))

    def test_request_validation(self):
        # Partial vision is view completion, admitted here and checked per checkpoint later.
        views = _views(("front", "left"))
        views[0]["vision_path"] = "front.mp4"
        self.assertEqual(len(validate_multiview_request({"views": views})), 2)
        with self.assertRaisesRegex(ValueError, "every view or none"):
            views = _views(("front", "left"))
            views[0]["prompt"] = "A car."
            validate_multiview_request({"views": views})
        with self.assertRaisesRegex(ValueError, "control_path"):
            validate_lidar_request({"control_path": "sweeps.tar"})
        self.assertEqual(
            validate_lidar_request({"control_path": "sweeps.safetensors"}),
            {"control_path": "sweeps.safetensors", "decode": True},
        )
        with self.assertRaisesRegex(ValueError, "control_path"):
            validate_multiview_request({"views": [{"camera_key": "front"}]})
        with self.assertRaisesRegex(ValueError, "Unsupported Cosmos3 multiview fields"):
            validate_multiview_request({"views": _views(("front",)), "lidar": {}})
        with self.assertRaisesRegex(ValueError, "resolution"):
            validate_multiview_request({"views": _views(("front",)), "resolution": 704})
        with self.assertRaisesRegex(ValueError, "aspect_ratio"):
            validate_multiview_request(
                {"views": _views(("front",)), "aspect_ratio": "2:1"}
            )
        with self.assertRaisesRegex(ValueError, "wsm"):
            Cosmos3MultiviewSamplingParams(wsm={"strength": 1.0})
        Cosmos3MultiviewSamplingParams(wsm={"weight": 1.0})
        with self.assertRaisesRegex(ValueError, "negative_metadata_mode"):
            Cosmos3MultiviewSamplingParams(negative_metadata_mode="sometimes")
        with self.assertRaisesRegex(ValueError, "max_sequence_length"):
            Cosmos3MultiviewSamplingParams(max_sequence_length=5000)
        self.assertEqual(parse_local_condition_indexes("1, 0"), [0, 1])
        self.assertEqual(clamp_multiview_guidance_scale(9.0), 7.0)
        self.assertEqual(clamp_multiview_guidance_scale(-1.0), 0.0)

    def test_lower_video_request_kwargs_parses_json_strings(self):
        request = SimpleNamespace(model_extra={}, num_frames=None, fps=None)
        kwargs = Cosmos3MultiviewSamplingParams.lower_video_request_kwargs(
            request,
            {
                "num_frames": 93,
                "fps": 30,
                "multiview": '{"views": [{"camera_key": "front", "control_path": "c.mp4"}]}',
                "wsm": "{}",
                "condition_video_as_image": "true",
                "negative_metadata_mode": "Same",
            },
        )
        self.assertEqual(kwargs["multiview"]["views"][0]["camera_key"], "front")
        self.assertEqual(kwargs["wsm"], {})
        self.assertIs(kwargs["condition_video_as_image"], True)
        self.assertEqual(kwargs["negative_metadata_mode"], "same")

    def test_input_stage_pins_camera_order_and_fills_implied_keys(self):
        deployment = parse_multiview_deployment_config(_transformer_config())
        stage = Cosmos3MultiviewInputStage(deployment)

        def batch_for(params, is_warmup=False):
            return SimpleNamespace(sampling_params=params, is_warmup=is_warmup)

        ordered = stage._resolve_views(
            batch_for(
                Cosmos3MultiviewSamplingParams(
                    multiview={"views": _views(COSMOS3_MADS_CAMERAS)}
                )
            )
        )
        self.assertEqual(
            [view.camera_key for view in ordered], list(COSMOS3_MADS_CAMERAS)
        )
        with self.assertRaisesRegex(ValueError, "camera order"):
            stage._resolve_views(
                batch_for(
                    Cosmos3MultiviewSamplingParams(
                        multiview={
                            "views": _views(tuple(reversed(COSMOS3_MADS_CAMERAS)))
                        }
                    )
                )
            )
        with self.assertRaisesRegex(ValueError, "exported checkpoint cameras"):
            stage._resolve_views(
                batch_for(
                    Cosmos3MultiviewSamplingParams(
                        multiview={"views": _views(("front",))}
                    )
                )
            )
        with self.assertRaisesRegex(ValueError, "exactly 11"):
            stage._resolve_views(
                batch_for(Cosmos3MultiviewSamplingParams(control_path=["c0.mp4"]))
            )
        # Schema 2 with variable_view_count admits a reordered subset but
        # insists on a caption per camera.
        subset_stage = Cosmos3MultiviewInputStage(
            parse_multiview_deployment_config(_transformer_config(SCHEMA2_BLOCK))
        )
        subset = _views((COSMOS3_MADS_CAMERAS[3], COSMOS3_MADS_CAMERAS[0]))
        with self.assertRaisesRegex(ValueError, "prompt"):
            subset_stage._resolve_views(
                batch_for(Cosmos3MultiviewSamplingParams(multiview={"views": subset}))
            )
        for view in subset:
            view["prompt"] = "A car."
        resolved = subset_stage._resolve_views(
            batch_for(Cosmos3MultiviewSamplingParams(multiview={"views": subset}))
        )
        self.assertEqual(
            [view.camera_key for view in resolved],
            [COSMOS3_MADS_CAMERAS[3], COSMOS3_MADS_CAMERAS[0]],
        )
        implied = stage._resolve_views(
            batch_for(
                Cosmos3MultiviewSamplingParams(
                    control_path=[f"c{i}.mp4" for i in range(11)]
                )
            )
        )
        self.assertEqual(
            [view.camera_key for view in implied], list(COSMOS3_MADS_CAMERAS)
        )
        with self.assertRaisesRegex(ValueError, "all images or all videos"):
            views = _views(COSMOS3_MADS_CAMERAS)
            views[3]["control_path"] = "c.png"
            stage._resolve_views(
                batch_for(Cosmos3MultiviewSamplingParams(multiview={"views": views}))
            )
        self.assertIsNone(
            stage._resolve_views(
                batch_for(Cosmos3MultiviewSamplingParams(), is_warmup=True)
            )
        )
        with self.assertRaisesRegex(ValueError, "multiview.views"):
            stage._resolve_views(batch_for(Cosmos3MultiviewSamplingParams()))


class TestPerViewCaptions(unittest.TestCase):
    CAMERAS = ("camera_front_wide_120fov", "camera_rear_tele_30fov")
    RIG = (
        "This multiview driving sequence contains time-aligned recordings from 2 "
        "vehicle-mounted cameras: front wide-angle camera (forward-facing, 120\u00b0 FOV); "
        "rear telephoto camera (backward-facing, 30\u00b0 FOV)."
    )

    def test_rig_header_matches_the_training_formatter(self):
        # Golden strings from imaginaire4 caption_format.format_separate_view_captions.
        separate = format_separate_view_captions(["A car.", "Trees."], self.CAMERAS)
        self.assertEqual(
            separate[0],
            f"{self.RIG}\n\nThe description below is for the front wide-angle camera mounted "
            "on the vehicle. This camera is facing forward and has a 120\u00b0 field of "
            "view:\n\nA car.",
        )
        self.assertEqual(
            separate[1],
            f"{self.RIG}\n\nThe description below is for the rear telephoto camera mounted "
            "on the vehicle. This camera is facing backward and has a 30\u00b0 field of "
            "view:\n\nTrees.",
        )
        prompts = format_per_view_prompts(
            ["A car.", "Trees."],
            self.CAMERAS,
            num_frames=17,
            fps=30.0,
            height=480,
            width=832,
            emphasis=COSMOS3_MULTIVIEW_EMPHASIS,
        )
        self.assertTrue(
            prompts[0].endswith(
                "A car. The video is 0.6 seconds long and is of 30 FPS. This video is of "
                f"480x832 resolution. {COSMOS3_MULTIVIEW_EMPHASIS}"
            )
        )
        with self.assertRaisesRegex(ValueError, "match the selected cameras"):
            format_separate_view_captions(["A car."], self.CAMERAS)

    def test_caption_scoping_in_the_predicate(self):
        # Two cameras, one frame, one patch: [C0 C0 | C1 | W0 W1 | R0 R1], caption lengths (2, 1).
        control = MaskItem(token_shape=(2, 1, 1), num_views=2, is_control=True)
        target = MaskItem(token_shape=(2, 1, 1), num_views=2)
        metadata = build_multiview_flex_metadata(
            seq_len=7,
            full_q_offsets=(3, 5, 7),
            items_per_sample=(control, target),
            device="cpu",
            num_und=3,
            caption_lengths=(2, 1),
        )
        self.assertEqual(metadata.view_id[:3].tolist(), [0, 0, 1])
        allowed = multiview_pair_predicate(
            metadata, torch.arange(metadata.q_len)[:, None], torch.arange(3)[None, :]
        )
        # Rows: W0 W1 R0 R1; camera 0 reads tokens 0-1, camera 1 reads token 2.
        self.assertEqual(
            allowed.tolist(), [[True, True, False], [False, False, True]] * 2
        )
        with self.assertRaisesRegex(ValueError, "partition"):
            build_multiview_flex_metadata(
                seq_len=7,
                full_q_offsets=(3, 5, 7),
                items_per_sample=(control, target),
                device="cpu",
                num_und=3,
                caption_lengths=(2, 2),
            )


class TestLidarItems(unittest.TestCase):
    def _metadata(self, window=0.4):
        # Camera items at 0.4 s per latent frame, LiDAR items at 0.1 s per sweep.
        camera = (2, 1, 1)
        lidar = (4, 1, 1)
        items = (
            MaskItem(camera, 1, is_control=True, seconds_per_frame=0.4),
            MaskItem(camera, 1, seconds_per_frame=0.4),
            MaskItem(
                lidar,
                1,
                view_offset=1,
                is_control=True,
                seconds_per_frame=0.1,
                is_lidar=True,
            ),
            MaskItem(lidar, 1, view_offset=1, seconds_per_frame=0.1, is_lidar=True),
        )
        return build_multiview_flex_metadata(
            seq_len=16,
            full_q_offsets=(4, 6, 8, 12, 16),
            items_per_sample=items,
            device="cpu",
            num_und=3,
            decomposed_temporal_window_seconds=window,
            control_attends_sensor=True,
            caption_lengths=(3,),
        )

    def test_lidar_reads_every_caption_and_registers_by_capture_time(self):
        metadata = self._metadata()
        self.assertEqual(metadata.view_id[8:16].tolist(), [-2] * 8)
        allowed = multiview_pair_predicate(
            metadata, torch.arange(metadata.q_len)[:, None], torch.arange(16)[None, :]
        )
        # Camera target frame 0 (row 2) and LiDAR target sweeps (rows 8-11).
        camera_frame0 = allowed[2]
        lidar_sweep0, lidar_sweep3 = allowed[8], allowed[11]
        self.assertTrue(camera_frame0[:3].all())  # the single caption
        self.assertTrue(lidar_sweep0[:3].all())  # LiDAR reads every caption
        # LiDAR target sees its own control at every sweep, never the camera control.
        self.assertEqual(lidar_sweep0[8:12].tolist(), [True] * 4)
        self.assertEqual(lidar_sweep0[4:6].tolist(), [False, False])
        # Sweep 3 (0.3 s) sees camera frame 0 (0.0 s) within the 0.4 s window
        # but camera frame 1 (0.4 s) lies ahead of it.
        self.assertEqual(lidar_sweep3[6:8].tolist(), [True, False])
        # Camera frame 0 (0.0 s) sees only LiDAR sweep 0 (0.0 s) among the targets.
        self.assertEqual(camera_frame0[12:16].tolist(), [True, False, False, False])
        # HD-map control attends its LiDAR target (control_attends_sensor) but not cameras.
        hdmap = allowed[4]
        self.assertEqual(hdmap[12:16].tolist(), [True] * 4)
        self.assertEqual(hdmap[6:8].tolist(), [False, False])

    def test_layout_items_drive_plan_offsets_and_tokens(self):
        camera = MaskItem((2, 1, 2), 2, seconds_per_frame=0.5)
        control = MaskItem((2, 1, 2), 2, is_control=True, seconds_per_frame=0.5)
        lidar = MaskItem(
            (3, 1, 3), 1, view_offset=2, seconds_per_frame=0.1, is_lidar=True
        )
        layout = MultiviewLayout(
            num_views=2,
            latent_frames=2,
            patch_height=1,
            patch_width=2,
            seconds_per_frame=0.5,
            decomposed_temporal_window_seconds=0.4,
            items=(control, camera, lidar),
            caption_lengths=(2, 2),
            max_und_tokens=64,
        )
        self.assertEqual(layout.gen_tokens, 4 + 4 + 9)
        default = MultiviewLayout(
            num_views=2, latent_frames=2, patch_height=1, patch_width=2
        )
        self.assertEqual([item.is_control for item in default.items], [True, False])
        self.assertNotEqual(layout.cache_key(), default.cache_key())
        context = MultiviewAttentionContext(layout=layout, mask_cache={})
        plan, geometry = get_multiview_attention_plan(
            context,
            real_und_len=4,
            real_q_len=layout.gen_tokens,
            device=torch.device("cpu"),
        )
        self.assertEqual(geometry.real_q_len, 17)
        self.assertEqual(
            plan.seq_lengths,
            (geometry.padded_q_len, geometry.padded_und_len + geometry.padded_q_len),
        )
        metadata = build_multiview_flex_metadata(
            seq_len=geometry.padded_und_len + geometry.padded_q_len,
            full_q_offsets=(
                geometry.padded_und_len,
                geometry.padded_und_len + 4,
                geometry.padded_und_len + 8,
                geometry.padded_und_len + 17,
            ),
            items_per_sample=layout.items,
            device="cpu",
            num_und=4,
            decomposed_temporal_window_seconds=0.4,
            caption_lengths=layout.caption_lengths,
        )
        self.assertEqual(
            metadata.view_id[geometry.padded_und_len + 8 :][:9].tolist(), [-2] * 9
        )
        self.assertEqual(metadata.view_id[:4].tolist(), [0, 0, 1, 1])
        with self.assertRaisesRegex(ValueError, "positive integers"):
            MultiviewLayout(
                num_views=2,
                latent_frames=2,
                patch_height=1,
                patch_width=2,
                caption_lengths=(0,),
            )

    def test_pack_unpack_roundtrip(self):
        camera = torch.arange(24.0).view(1, 2, 3, 2, 2)
        lidar = torch.arange(100.0, 118.0).view(1, 2, 3, 1, 3)
        packed = pack_state([camera, lidar])
        self.assertEqual(tuple(packed.shape), (1, 42))
        back = unpack_state(packed, (tuple(camera.shape[1:]), tuple(lidar.shape[1:])))
        torch.testing.assert_close(back[0], camera)
        torch.testing.assert_close(back[1], lidar)
        with self.assertRaisesRegex(ValueError, "declared geometries"):
            unpack_state(packed, ((2, 3, 2, 2),))

    def test_sweep_helpers_and_input_normalization(self):
        self.assertEqual(required_lidar_sweeps(17, 30.0, 10.0), 6)
        self.assertEqual(required_lidar_sweeps(201, 30.0, 10.0), 67)
        frames = torch.arange(3.0).view(1, 3, 1, 1).expand(3, 3, 1, 1).clone()
        padded = pad_lidar_sweeps(frames, 6)
        # Reflection pads [0, 1, 2] to [0, 1, 2, 2, 1], then one more reflected sweep.
        self.assertEqual(padded[0, :, 0, 0].tolist(), [0.0, 1.0, 2.0, 2.0, 1.0, 1.0])
        self.assertEqual(pad_lidar_sweeps(frames, 2).shape[1], 2)
        encoder = Cosmos3LidarEncoder(LIDAR_BLOCK)
        # Semantic-width input is circularly padded to the model width, ranges
        # normalized into [-1, 1], invalid rays filled with -1.
        clip = torch.zeros(3, 1, 128, 1800)
        clip[0, 0, 0, 0] = 52.5
        clip[1, 0, 0, 0] = 1.0
        clip[2, 0, 0, 0] = 1.0
        clip[0, 0, 1, 5] = 3.0  # below the 5 m minimum: invalid
        clip[2, 0, 1, 5] = 1.0
        prepared = encoder.prepare_input(clip)
        self.assertEqual(tuple(prepared.shape), (1, 3, 1, 128, 1808))
        self.assertAlmostEqual(prepared[0, 0, 0, 0, 4].item(), 0.0, places=5)
        self.assertEqual(prepared[0, 1, 0, 0, 4].item(), 1.0)
        self.assertEqual(prepared[0, 2, 0, 0, 4].item(), 1.0)
        self.assertEqual(prepared[0, 0, 0, 1, 9].item(), -1.0)
        self.assertEqual(prepared[0, 2, 0, 1, 9].item(), 0.0)
        # The circular pad wraps the last semantic columns in front of column 0.
        clip2 = torch.zeros(3, 1, 128, 1800)
        clip2[0, 0, 7, 1799] = 20.0
        clip2[2, 0, 7, 1799] = 1.0
        prepared2 = encoder.prepare_input(clip2)
        self.assertGreater(prepared2[0, 2, 0, 7, 3].item(), 0.5)


class TestLidarDecoder(unittest.TestCase):
    def test_network_to_metric_inverts_the_encoder_normalization(self):
        """Range is the inverse affine of the encoder's [5, 100] m -> [-1, 1] map,
        intensity the inverse of [0, 1] -> [-1, 1], and a ray whose mask logit
        falls below the 0.5 sigmoid cut reads zero range, zero intensity, validity 0."""
        torch.manual_seed(0)
        range_m = torch.rand(1, 1, 2, 4, 6) * 95.0 + 5.0
        intensity = torch.rand(1, 1, 2, 4, 6)
        valid = torch.rand(1, 1, 2, 4, 6) > 0.4
        network = torch.cat(
            (
                (range_m - 5.0) / 95.0 * 2.0 - 1.0,
                intensity * 2.0 - 1.0,
                torch.where(
                    valid, torch.full_like(range_m, 8.0), torch.full_like(range_m, -8.0)
                ),
            ),
            dim=1,
        )

        metric = lidar_network_to_metric(network, min_range_m=5.0, max_range_m=100.0)

        self.assertTrue(
            torch.allclose(
                metric[:, 0][valid[:, 0]], range_m[:, 0][valid[:, 0]], atol=1e-4
            )
        )
        self.assertTrue(
            torch.allclose(
                metric[:, 1][valid[:, 0]], intensity[:, 0][valid[:, 0]], atol=1e-6
            )
        )
        self.assertTrue(torch.equal(metric[:, 2], valid.float()[:, 0]))
        self.assertEqual(
            metric[:, :2][~valid.expand(-1, 2, -1, -1, -1)].abs().sum().item(), 0.0
        )
        probability = lidar_network_to_metric(
            network, min_range_m=5.0, max_range_m=100.0, apply_validity_mask=False
        )
        self.assertTrue(torch.allclose(probability[:, 2], torch.sigmoid(network[:, 2])))

    def test_pixel_shuffles_follow_the_reference_patch_order(self):
        """The decoder's expand and detokenizer lay a (p1, p2, c) channel group out as
        rows then columns, the inverse of the encoder's space-to-depth merge, so a
        checkpoint trained with einops' "(P1 P2 C)" ordering decodes on the same grid."""
        grouped = torch.arange(2 * 2 * 3 * 12, dtype=torch.float32).view(2, 2, 3, 12)

        self.assertTrue(torch.equal(_SpaceToDepth()(depth_to_space(grouped)), grouped))
        pixels = _DepthToPixels((2, 2), channels=3)(grouped)
        self.assertEqual(tuple(pixels.shape), (2, 3, 4, 6))
        # group index = p1 * 6 + p2 * 3 + c for the (row, column, channel) grouping
        for p1 in range(2):
            for p2 in range(2):
                for c in range(3):
                    self.assertEqual(
                        pixels[1, c, 1 * 2 + p1, 2 * 2 + p2].item(),
                        grouped[1, 1, 2, p1 * 6 + p2 * 3 + c].item(),
                    )
        self.assertTrue(
            torch.equal(_SpaceToDepth()(pixels.permute(0, 2, 3, 1)), grouped)
        )

    def test_crop_lidar_width_centers_the_azimuth_padding(self):
        clip = torch.arange(1808, dtype=torch.float32).view(1, 1, 1, 1, 1808)
        cropped = crop_lidar_width(clip, 1800)
        self.assertEqual(cropped.shape[-1], 1800)
        self.assertEqual(cropped[..., 0].item(), 4.0)
        with self.assertRaises(ValueError):
            crop_lidar_width(clip, 1801)

    def test_payload_and_preview_drop_invalid_rays(self):
        clip = torch.zeros(3, 2, 4, 8)
        clip[0] = 50.0
        clip[1] = 0.5
        clip[2] = 1.0
        clip[2, :, 0] = 0.0  # first beam dropped by the mask
        clip[0, :, 1] = 0.0  # second beam dropped by zero range
        frames = render_lidar_range_frames(clip, min_range_m=5.0, max_range_m=100.0)
        self.assertEqual(frames.shape, (2, 4, 8, 3))
        self.assertEqual(frames[:, :2].max(), 0)
        self.assertGreater(frames[:, 2:].max(), 0)
        payload = lidar_output_payload(
            clip,
            fps=10.0,
            min_range_m=5.0,
            max_range_m=100.0,
            files={"rangemap": "x"},
            include_arrays=True,
        )
        self.assertEqual(payload["sweeps"], 2)
        self.assertAlmostEqual(payload["valid_fraction"], 0.5)
        self.assertEqual(payload["range_m"][:, :2].max(), 0.0)
        self.assertEqual(
            set(lidar_payload_for_response(payload))
            & {"range_m", "intensity", "validity"},
            set(),
        )

    def test_video_api_accepts_an_empty_top_level_prompt(self):
        """Per-camera captions live in multiview.views[].prompt, so the multipart video
        endpoint must not reject the empty top-level prompt schema-2 requests send."""
        self.assertTrue(Cosmos3MultiviewSamplingParams.video_prompt_optional())

    def test_lidar_request_decode_flag(self):
        params = validate_lidar_request({"control_path": "/x/hdmap.safetensors"})
        self.assertTrue(params["decode"])
        params = validate_lidar_request(
            {"control_path": "/x/hdmap.safetensors", "decode": False}
        )
        self.assertFalse(params["decode"])
        with self.assertRaises(ValueError):
            validate_lidar_request(
                {"control_path": "/x/hdmap.safetensors", "decode": "no"}
            )
        with self.assertRaises(ValueError):
            validate_lidar_request(
                {"control_path": "/x/hdmap.safetensors", "outputs": []}
            )


class TestRegistry(unittest.TestCase):
    def test_pipeline_is_discoverable_and_paths_resolve(self):
        _discover_and_register_pipelines()
        self.assertIn("Cosmos3MultiviewPipeline", _PIPELINE_REGISTRY)
        for model_path in (
            "nvidia/Cosmos3-Nano-Transfer-Auto",
            "/mnt/models/Cosmos3-Nano-Transfer-Auto",
        ):
            with self.subTest(model_path=model_path):
                config_info = _get_config_info(model_path)
                self.assertIs(config_info.pipeline_config_cls, Cosmos3MultiviewConfig)
                self.assertIs(
                    config_info.sampling_param_cls, Cosmos3MultiviewSamplingParams
                )
        # The plain Nano release must keep resolving to the regular pipeline.
        self.assertIs(
            _get_config_info("nvidia/Cosmos3-Nano").pipeline_config_cls, Cosmos3Config
        )

    def test_class_name_detectors_stay_disjoint(self):
        expectations = {
            "Cosmos3MultiviewPipeline": Cosmos3MultiviewConfig,
            "Cosmos3OmniPipeline": Cosmos3Config,
        }
        for class_name, config_cls in expectations.items():
            with self.subTest(class_name=class_name):
                with mock.patch(
                    "sglang.multimodal_gen.registry.maybe_download_model_index",
                    return_value={"_class_name": class_name},
                ):
                    config_info = _get_config_info(f"acme/renamed-{class_name.lower()}")
                self.assertIs(config_info.pipeline_config_cls, config_cls)

    def test_transformer_class_is_resolvable(self):
        from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

        model_cls, _ = ModelRegistry.resolve_model_cls("Cosmos3MultiviewTransformer")
        self.assertEqual(model_cls.__name__, "Cosmos3MultiviewTransformer")
        self.assertEqual(
            model_cls._cross_attention_cls.__name__, "Cosmos3MultiviewCrossAttention"
        )


if __name__ == "__main__":
    unittest.main()
