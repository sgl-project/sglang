# SPDX-License-Identifier: Apache-2.0
"""MindIE-SD options and exact MiniMax-H3 packed layouts reach Rain Fusion V2."""

import math
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.rain_fusion_attn_v2 import (
    RainFusionAttentionV2MetadataBuilder,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3Attention
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
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
_MODEL_MODULE = MiniMaxH3Attention.__module__
_UNSET_PRECISION = object()


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


class TestMiniMaxH3RainFusionV2Metadata(unittest.TestCase):
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
        self,
        backend_enum,
        spans,
        precision=_UNSET_PRECISION,
        *,
        model_backend=None,
        use_real_builder=False,
        dense_last_steps=0,
        num_steps=None,
        options=None,
    ):
        metadata = object()
        if use_real_builder:
            builder = Mock(wraps=RainFusionAttentionV2MetadataBuilder())
        else:
            builder = Mock()
            builder.build.return_value = metadata
        sparse_config = {
            "skip_first_steps": 2,
            "sparsity": 0.6,
            "dense_last_steps": dense_last_steps,
        }
        if precision is not _UNSET_PRECISION:
            sparse_config["precision"] = precision
        sparse_config.update(options or {})
        requested_config = dict(sparse_config)
        stage = SimpleNamespace(
            attn_backend=SimpleNamespace(
                get_enum=lambda: backend_enum,
                get_builder_cls=lambda: lambda: builder,
            ),
            server_args=SimpleNamespace(attention_backend_config=sparse_config),
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
            patch.object(RainFusionAttentionV2MetadataBuilder, "build", builder.build),
        ):
            result = MiniMaxH3DenoisingStage._forward_dit(
                stage, model, call_kwargs, 4, batch=object(), num_steps=num_steps
            )
        self.assertEqual(result, ("video", "audio"))
        self.assertEqual(stage.server_args.attention_backend_config, requested_config)
        return contexts[0]["attn_metadata"], builder, contexts

    def test_stage_routes_spans_without_mutually_exclusive_text_length(self):
        spans = _layout()["video_spans"]
        metadata, builder, contexts = self._forward_stage(
            AttentionBackendEnum.RAIN_FUSION_ATTN_V2, spans
        )
        builder.build.assert_called_once_with(
            current_timestep=4,
            skip_first_steps=2,
            sparsity=0.6,
            dense_last_steps=0,
            num_steps=None,
            video_spans=spans,
        )
        self.assertIs(contexts[0]["attn_metadata"], metadata)

    def test_missing_precision_defaults_to_mix(self):
        spans = [{"start": 9, "latent_shape": [2, 2, 2]}]
        _, builder, contexts = self._forward_stage(
            AttentionBackendEnum.RAIN_FUSION_ATTN_V2, spans, use_real_builder=True
        )
        self.assertNotIn("precision", builder.build.call_args.kwargs)
        self.assertEqual(contexts[0]["attn_metadata"].precision, "mix")

    def test_explicit_mix_precision_reaches_metadata_builder(self):
        spans = [{"start": 9, "latent_shape": [2, 2, 2]}]
        _, builder, contexts = self._forward_stage(
            AttentionBackendEnum.RAIN_FUSION_ATTN_V2,
            spans,
            precision="mix",
            use_real_builder=True,
        )
        builder.build.assert_called_once_with(
            current_timestep=4,
            skip_first_steps=2,
            sparsity=0.6,
            precision="mix",
            dense_last_steps=0,
            num_steps=None,
            video_spans=spans,
        )
        self.assertEqual(contexts[0]["attn_metadata"].precision, "mix")

    def test_native_sparse_precision_modes_reach_metadata_builder(self):
        spans = [{"start": 9, "latent_shape": [2, 2, 2]}]
        for precision in ("bf16", "mix", "fp8", "mxfp4"):
            with self.subTest(precision=precision):
                _, builder, contexts = self._forward_stage(
                    AttentionBackendEnum.RAIN_FUSION_ATTN_V2,
                    spans,
                    precision=precision,
                    use_real_builder=True,
                )
                self.assertEqual(builder.build.call_args.kwargs["precision"], precision)
                self.assertEqual(contexts[0]["attn_metadata"].precision, precision)

    def test_h3_forwards_native_sparse_and_dense_options(self):
        options = {
            "sparse_type": "rf_v2",
            "precision": "fp8",
            "block_size": 256,
            "block_size_kv": 512,
            "inner_precise": 2,
            "dense_precision": "mxfp8",
            "dense_kwargs": {"pre_tokens": 2147483647},
        }
        _, builder, _ = self._forward_stage(
            AttentionBackendEnum.RAIN_FUSION_ATTN_V2,
            _layout()["video_spans"],
            options=options,
        )
        for name, value in options.items():
            self.assertEqual(builder.build.call_args.kwargs[name], value)

    def test_dense_tail_receives_the_actual_h3_step_count(self):
        spans = _layout()["video_spans"]
        _, builder, contexts = self._forward_stage(
            AttentionBackendEnum.RAIN_FUSION_ATTN_V2,
            spans,
            use_real_builder=True,
            dense_last_steps=2,
            num_steps=7,
        )
        self.assertEqual(builder.build.call_args.kwargs["current_timestep"], 4)
        self.assertEqual(builder.build.call_args.kwargs["num_steps"], 7)
        self.assertEqual(contexts[0]["attn_metadata"].dense_last_steps, 2)
        self.assertEqual(contexts[0]["attn_metadata"].num_steps, 7)

    def test_h3_computed_steps_and_layout_override_config_without_mutating_it(self):
        spans = _layout()["video_spans"]
        options = {"current_timestep": 99, "num_steps": 999, "video_spans": []}
        _, builder, _ = self._forward_stage(
            AttentionBackendEnum.RAIN_FUSION_ATTN_V2,
            spans,
            num_steps=7,
            options=options,
        )
        self.assertEqual(builder.build.call_args.kwargs["current_timestep"], 4)
        self.assertEqual(builder.build.call_args.kwargs["num_steps"], 7)
        self.assertEqual(builder.build.call_args.kwargs["video_spans"], spans)
        self.assertEqual(
            options, {"current_timestep": 99, "num_steps": 999, "video_spans": []}
        )

    def test_other_backends_do_not_require_rain_fusion_v2_layout(self):
        _, builder, contexts = self._forward_stage(
            AttentionBackendEnum.TORCH_SDPA, None
        )
        builder.build.assert_not_called()
        self.assertIsNone(contexts[0]["attn_metadata"])

    def test_resolved_transformer_override_builds_rain_fusion_v2_metadata(self):
        spans = _layout()["video_spans"]
        metadata, builder, contexts = self._forward_stage(
            AttentionBackendEnum.FA,
            spans,
            model_backend=AttentionBackendEnum.RAIN_FUSION_ATTN_V2,
        )
        builder.build.assert_called_once()
        self.assertEqual(builder.build.call_args.kwargs["video_spans"], spans)
        self.assertIs(contexts[0]["attn_metadata"], metadata)

    def test_resolved_dense_override_skips_rain_fusion_v2_metadata(self):
        _, builder, contexts = self._forward_stage(
            AttentionBackendEnum.RAIN_FUSION_ATTN_V2,
            None,
            model_backend=AttentionBackendEnum.FA,
        )
        builder.build.assert_not_called()
        self.assertIsNone(contexts[0]["attn_metadata"])

    def test_h3_forwards_missing_and_empty_spans(self):
        for spans in (None, []):
            with self.subTest(spans=spans):
                _, builder, _ = self._forward_stage(
                    AttentionBackendEnum.RAIN_FUSION_ATTN_V2, spans
                )
                self.assertEqual(builder.build.call_args.kwargs["video_spans"], spans)


class TestRainFusionV2ScheduleMetadata(unittest.TestCase):
    def test_generic_stage_forwards_full_config_and_normalizes_computed_layout(self):
        builder = Mock()
        stage = SimpleNamespace(
            attn_backend=SimpleNamespace(
                get_enum=lambda: AttentionBackendEnum.RAIN_FUSION_ATTN_V2,
                get_builder_cls=lambda: lambda: builder,
            )
        )
        config = {
            "sparse_type": "rf_v1",
            "precision": "bf16",
            "block_size": 256,
            "block_size_kv": 512,
            "dense_precision": "mxfp8",
            "dense_kwargs": {"pre_tokens": 2147483647},
            "current_timestep": 999,
            "num_steps": 999,
            "raw_latent_shape": [999, 999, 999],
            "patch_size": [999, 999, 999],
        }
        server_args = SimpleNamespace(
            attention_backend_config=config,
            pipeline_config=SimpleNamespace(
                dit_config=SimpleNamespace(patch_size=2, patch_size_t=1)
            ),
        )
        DenoisingStage._build_attn_metadata(
            stage,
            7,
            SimpleNamespace(raw_latent_shape=[1, 24, 2, 16, 16]),
            server_args,
            timesteps=torch.arange(9),
        )
        forwarded = builder.build.call_args.kwargs
        for name in (
            "sparse_type",
            "precision",
            "block_size",
            "block_size_kv",
            "dense_precision",
            "dense_kwargs",
        ):
            self.assertEqual(forwarded[name], config[name])
        self.assertEqual(forwarded["current_timestep"], 7)
        self.assertEqual(forwarded["num_steps"], 9)
        self.assertEqual(forwarded["raw_latent_shape"], [1, 24, 2, 16, 16])
        self.assertEqual(forwarded["patch_size"], (1, 2, 2))
        self.assertEqual(config["num_steps"], 999)
        self.assertEqual(config["raw_latent_shape"], [999, 999, 999])

    def test_generic_schedule_uses_timesteps_length_instead_of_request_steps(self):
        stage = SimpleNamespace(
            attn_backend=SimpleNamespace(
                get_enum=lambda: AttentionBackendEnum.RAIN_FUSION_ATTN_V2,
                get_builder_cls=lambda: RainFusionAttentionV2MetadataBuilder,
            )
        )
        server_args = SimpleNamespace(
            attention_backend_config={"dense_last_steps": 2},
            pipeline_config=SimpleNamespace(
                dit_config=SimpleNamespace(patch_size=(1, 1, 1))
            ),
        )
        metadata = DenoisingStage._build_attn_metadata(
            stage,
            7,
            SimpleNamespace(
                raw_latent_shape=[1, 24, 2, 16, 16], num_inference_steps=999
            ),
            server_args,
            timestep_value=500,
            timesteps=torch.arange(9),
        )
        self.assertEqual(metadata.current_timestep, 7)
        self.assertEqual(metadata.num_steps, 9)
        self.assertEqual(metadata.dense_last_steps, 2)

    def test_other_sparse_backends_do_not_receive_rain_fusion_v2_schedule_fields(self):
        for backend in (
            AttentionBackendEnum.RAIN_FUSION_ATTN,
            AttentionBackendEnum.BLOCK_SPARSE_ATTN,
        ):
            with self.subTest(backend=backend):
                builder = Mock()
                stage = SimpleNamespace(
                    attn_backend=SimpleNamespace(
                        get_enum=lambda: backend,
                        get_builder_cls=lambda: lambda: builder,
                    )
                )
                server_args = SimpleNamespace(
                    attention_backend_config={
                        "dense_last_steps": 2,
                        "sparse_type": "rf_v3",
                        "precision": "fp8",
                        "dense_precision": "mxfp8",
                        "dense_kwargs": {"pre_tokens": 2147483647},
                    },
                    pipeline_config=SimpleNamespace(
                        dit_config=SimpleNamespace(patch_size=(1, 1, 1))
                    ),
                )
                DenoisingStage._build_attn_metadata(
                    stage,
                    3,
                    SimpleNamespace(raw_latent_shape=[1, 24, 2, 16, 16]),
                    server_args,
                    timesteps=torch.arange(9),
                )
                self.assertNotIn("dense_last_steps", builder.build.call_args.kwargs)
                self.assertNotIn("num_steps", builder.build.call_args.kwargs)
                self.assertNotIn("sparse_type", builder.build.call_args.kwargs)
                self.assertNotIn("precision", builder.build.call_args.kwargs)
                self.assertNotIn("dense_precision", builder.build.call_args.kwargs)
                self.assertNotIn("dense_kwargs", builder.build.call_args.kwargs)


class TestMiniMaxH3RainFusionV2Binding(unittest.TestCase):
    def _bind(self, prefix):
        dense_impl = object()
        sparse_impl = object()
        dense_ctor = Mock(return_value=dense_impl)
        sparse_ctor = Mock(return_value=sparse_impl)
        dense_backend = SimpleNamespace(
            get_enum=lambda: AttentionBackendEnum.FA,
            get_impl_cls=lambda: dense_ctor,
        )
        sparse_backend = SimpleNamespace(
            get_enum=lambda: AttentionBackendEnum.RAIN_FUSION_ATTN_V2,
            get_impl_cls=lambda: sparse_ctor,
        )
        attention = SimpleNamespace(
            head_dim=128,
            num_heads=2,
            softmax_scale=128**-0.5,
            prefix=prefix,
            _cube_sparse_capable=True,
        )
        with patch(
            f"{_MODEL_MODULE}.get_attn_backend", return_value=dense_backend
        ) as select:
            MiniMaxH3Attention._set_attention_backend(attention, sparse_backend)
        return attention, dense_ctor, sparse_ctor, select

    def test_refiner_policy_is_applied_before_constructing_the_rain_fusion_v2_backend(
        self,
    ):
        attention, dense_ctor, sparse_ctor, select = self._bind(
            "token_refiner.blocks.0.attn"
        )
        select.assert_called_once_with(
            128, torch.bfloat16, selected_attention_backend=AttentionBackendEnum.FA
        )
        dense_ctor.assert_called_once()
        sparse_ctor.assert_not_called()
        self.assertIs(attention._attention_backend_enum, AttentionBackendEnum.FA)
        self.assertTrue(dense_ctor.call_args.kwargs["packed_trailing_padding"])

    def test_multimodal_dit_keeps_the_selected_rain_fusion_v2_backend(self):
        attention, dense_ctor, sparse_ctor, select = self._bind("blocks.0.attn")
        select.assert_not_called()
        dense_ctor.assert_not_called()
        sparse_ctor.assert_called_once()
        self.assertIs(
            attention._attention_backend_enum, AttentionBackendEnum.RAIN_FUSION_ATTN_V2
        )


if __name__ == "__main__":
    unittest.main()
