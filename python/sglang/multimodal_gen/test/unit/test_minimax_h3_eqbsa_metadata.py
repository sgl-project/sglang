# SPDX-License-Identifier: Apache-2.0
"""EQBSA uses exact H3 video grids while text, audio, and images remain dense."""

import math
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.eqbsa_attn import (
    EQBSAAttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    MiniMaxH3DenoisingStage,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_BRANCH_MODULE = MiniMaxH3DenoiseBranch.__module__


def _layout(text_len=3, *, keyframes=False):
    return minimax_h3_packed_sequence(
        text_len=text_len,
        latent_t=2,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        include_keyframe_cond=keyframes,
        keyframe_frame_indices=[0, -1] if keyframes else None,
        frame_count=5 if keyframes else None,
    )


def _sparse_positions(packed):
    positions = set()
    for span in packed["video_spans"]:
        start = span["start"]
        positions.update(range(start, start + math.prod(span["latent_shape"])))
    return positions


class TestMiniMaxH3EQBSAMetadata(unittest.TestCase):
    def test_text_audio_prefix_and_padding_are_not_video(self):
        packed = _layout()
        self.assertEqual(packed["txt_len"], 3)
        self.assertEqual(
            packed["video_spans"], [{"start": 9, "latent_shape": [2, 2, 2]}]
        )
        self.assertEqual(_sparse_positions(packed), set(packed["img_pos"].tolist()))
        used = int(packed["cu_seqlens"][1])
        dense_rows = set(packed["text_pos"].tolist() + packed["audio_pos"].tolist())
        dense_rows.update(range(used, packed["seq_len"]))
        self.assertTrue(dense_rows.isdisjoint(_sparse_positions(packed)))

    def test_keyframes_remain_dense_before_audio_and_target_video(self):
        packed = _layout(keyframes=True)
        self.assertEqual(
            packed["video_spans"], [{"start": 17, "latent_shape": [2, 2, 2]}]
        )
        target_positions = packed["img_pos"][packed["update_mask"]].tolist()
        condition_positions = packed["img_pos"][~packed["update_mask"]].tolist()
        self.assertEqual(_sparse_positions(packed), set(target_positions))
        self.assertTrue(set(condition_positions).isdisjoint(_sparse_positions(packed)))

    def test_ordered_references_keep_each_video_grid_and_dense_gaps(self):
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=3,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=3,
            ref_blocks=[
                {"kind": "image", "latent_h": 4, "latent_w": 4},
                {"kind": "audio", "ref_audio_t": 2},
                {
                    "kind": "video",
                    "ref_audio_t": 1,
                    "latent_t": 2,
                    "latent_h": 4,
                    "latent_w": 4,
                },
                {
                    "kind": "video_audio",
                    "ref_audio_t": 2,
                    "latent_t": 1,
                    "latent_h": 4,
                    "latent_w": 6,
                },
            ],
        )
        self.assertEqual(
            packed["video_spans"],
            [
                {"start": 13, "latent_shape": [2, 2, 2]},
                {"start": 25, "latent_shape": [1, 2, 3]},
                {"start": 37, "latent_shape": [2, 2, 2]},
            ],
        )
        sparse_rows = _sparse_positions(packed)
        image_rows = set(range(3, 7))
        self.assertTrue(image_rows.isdisjoint(sparse_rows))
        self.assertTrue(set(packed["audio_pos"].tolist()).isdisjoint(sparse_rows))
        self.assertTrue(set(packed["text_pos"].tolist()).isdisjoint(sparse_rows))
        self.assertEqual(sparse_rows | image_rows, set(packed["img_pos"].tolist()))
        self.assertLess(max(sparse_rows), int(packed["cu_seqlens"][1]))

    def test_branch_text_length_changes_absolute_video_offsets(self):
        short = _layout(text_len=3)
        long = _layout(text_len=9)
        self.assertEqual(
            long["video_spans"][0]["start"] - short["video_spans"][0]["start"], 6
        )
        self.assertEqual(
            long["video_spans"][0]["latent_shape"],
            short["video_spans"][0]["latent_shape"],
        )

    def test_ulysses_branch_retains_global_spans_and_cumulative_lengths(self):
        packed = _layout(keyframes=True)
        with (
            patch(f"{_BRANCH_MODULE}.get_ulysses_ctx", return_value=(2, 1)),
            patch(f"{_BRANCH_MODULE}.get_ring_ctx", return_value=(1, 0)),
        ):
            branch = MiniMaxH3DenoiseBranch(
                packed=packed,
                text_embeddings=torch.zeros(3, 8),
                token_tags=packed["token_tags"],
                device=torch.device("cpu"),
            )
        params = branch.static_kwargs["packed_seq_params"]
        self.assertEqual(branch.local_row_slice.start, packed["seq_len"] // 2)
        self.assertEqual(params["txt_len"], 3)
        self.assertEqual(params["video_spans"], packed["video_spans"])
        self.assertEqual(params["cu_seqlens_q_host"], (0, 25, 64))

    def _forward_stage(
        self, backend_enum, spans, precision="bf16", *, model_backend=None
    ):
        metadata = object()
        builder = Mock()
        builder.build.return_value = metadata
        stage = SimpleNamespace(
            attn_backend=SimpleNamespace(
                get_enum=lambda: backend_enum,
                get_builder_cls=lambda: lambda: builder,
            ),
            server_args=SimpleNamespace(
                attention_backend_config={
                    "skip_first_steps": 2,
                    "sparsity": 0.6,
                    "precision": precision,
                }
            ),
            _maybe_get_bcg_runner=lambda model: None,
        )
        contexts = []

        @contextmanager
        def record_context(**kwargs):
            contexts.append(kwargs)
            yield

        call_kwargs = {"packed_seq_params": {"txt_len": 3, "video_spans": spans}}
        model = Mock(return_value=("video", "audio"))
        if model_backend is not None:
            model._resolved_attention_backend = model_backend
        context_module = "sglang.multimodal_gen.runtime.managers.forward_context"
        with (
            patch(f"{context_module}.set_forward_context", record_context),
            patch.object(EQBSAAttentionMetadataBuilder, "build", builder.build),
        ):
            result = MiniMaxH3DenoisingStage._forward_dit(
                stage, model, call_kwargs, 4, batch=object()
            )
        self.assertEqual(result, ("video", "audio"))
        return metadata, builder, contexts

    def test_stage_routes_spans_without_mutually_exclusive_text_length(self):
        spans = _layout()["video_spans"]
        metadata, builder, contexts = self._forward_stage(
            AttentionBackendEnum.EQBSA_ATTN, spans
        )
        builder.build.assert_called_once_with(
            current_timestep=4,
            skip_first_steps=2,
            sparsity=0.6,
            precision="bf16",
            video_spans=spans,
        )
        self.assertIs(contexts[0]["attn_metadata"], metadata)

    def test_precision_reaches_metadata_builder(self):
        spans = [{"start": 9, "latent_shape": [2, 2, 2]}]
        for precision in ("bf16", "mix", "fp8", "mxfp4"):
            with self.subTest(precision=precision):
                _, builder, _ = self._forward_stage(
                    AttentionBackendEnum.EQBSA_ATTN, spans, precision=precision
                )
                builder.build.assert_called_once_with(
                    current_timestep=4,
                    skip_first_steps=2,
                    sparsity=0.6,
                    precision=precision,
                    video_spans=spans,
                )

    def test_other_backends_do_not_require_eqbsa_layout(self):
        _, builder, contexts = self._forward_stage(
            AttentionBackendEnum.TORCH_SDPA, None
        )
        builder.build.assert_not_called()
        self.assertIsNone(contexts[0]["attn_metadata"])

    def test_resolved_transformer_override_builds_eqbsa_metadata(self):
        spans = _layout()["video_spans"]
        metadata, builder, contexts = self._forward_stage(
            AttentionBackendEnum.FA,
            spans,
            model_backend=AttentionBackendEnum.EQBSA_ATTN,
        )
        builder.build.assert_called_once()
        self.assertEqual(builder.build.call_args.kwargs["video_spans"], spans)
        self.assertIs(contexts[0]["attn_metadata"], metadata)

    def test_resolved_dense_override_skips_eqbsa_metadata(self):
        _, builder, contexts = self._forward_stage(
            AttentionBackendEnum.EQBSA_ATTN,
            None,
            model_backend=AttentionBackendEnum.FA,
        )
        builder.build.assert_not_called()
        self.assertIsNone(contexts[0]["attn_metadata"])

    def test_eqbsa_rejects_missing_layout_but_preserves_explicit_empty_spans(self):
        with self.assertRaisesRegex(ValueError, "requires video_spans"):
            self._forward_stage(AttentionBackendEnum.EQBSA_ATTN, None)
        _, builder, _ = self._forward_stage(AttentionBackendEnum.EQBSA_ATTN, [])
        self.assertEqual(builder.build.call_args.kwargs["video_spans"], [])


if __name__ == "__main__":
    unittest.main()
