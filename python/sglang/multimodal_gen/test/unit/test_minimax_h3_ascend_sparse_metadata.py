# SPDX-License-Identifier: Apache-2.0
"""MindIE-SD options and exact MiniMax-H3 packed layouts reach Ascend sparse attention."""

import importlib
import math
import sys
import unittest
from contextlib import contextmanager, nullcontext
from functools import partial
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
import torch

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

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.layers.attention.backends.ascend_sparse_attn import (
        AscendSparseAttentionMetadataBuilder,
    )

_BRANCH_MODULE = MiniMaxH3DenoiseBranch.__module__
_MODEL_MODULE = MiniMaxH3Attention.__module__
_STAGE_MODULE = MiniMaxH3DenoisingStage.__module__
_UNSET_PRECISION = object()


class _MockedMindIETestCase(unittest.TestCase):
    def setUp(self):
        monkeypatch = pytest.MonkeyPatch()
        self.addCleanup(monkeypatch.undo)
        _mock_mindie_metadata_builder(monkeypatch)


def _mock_mindie_metadata_builder(monkeypatch):
    from sglang.multimodal_gen.runtime.layers.attention import backends

    module_name = f"{backends.__name__}.ascend_sparse_attn"
    # Register restoration even when the module was absent before this test.
    monkeypatch.setitem(sys.modules, module_name, None)
    monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setattr(backends, "ascend_sparse_attn", None, raising=False)
    vendor = ModuleType("mindiesd")
    vendor.sparse_attention = Mock(
        side_effect=AssertionError("Metadata tests must not execute MindIE operators")
    )
    with pytest.MonkeyPatch.context() as sdk_patch:
        sdk_patch.setitem(sys.modules, "mindiesd", vendor)
        backend = importlib.import_module(module_name)
    monkeypatch.setitem(
        globals(),
        "AscendSparseAttentionMetadataBuilder",
        backend.AscendSparseAttentionMetadataBuilder,
    )


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


class TestMiniMaxH3AscendSparseMetadata(_MockedMindIETestCase):
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
        skip_last_steps=0,
        num_steps=None,
        options=None,
    ):
        metadata = object()
        if use_real_builder:
            builder = Mock(wraps=AscendSparseAttentionMetadataBuilder())
        else:
            builder = Mock()
            builder.build.return_value = metadata
        sparse_config = {
            "skip_first_steps": 2,
            "sparsity": 0.6,
            "skip_last_steps": skip_last_steps,
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
            patch.object(AscendSparseAttentionMetadataBuilder, "build", builder.build),
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
            AttentionBackendEnum.ASCEND_SPARSE_ATTN, spans
        )
        builder.build.assert_called_once_with(
            current_timestep=4,
            skip_first_steps=2,
            sparsity=0.6,
            skip_last_steps=0,
            num_steps=None,
            video_spans=spans,
        )
        self.assertIs(contexts[0]["attn_metadata"], metadata)

    def test_missing_precision_defaults_to_bf16(self):
        spans = [{"start": 9, "latent_shape": [2, 2, 2]}]
        _, builder, contexts = self._forward_stage(
            AttentionBackendEnum.ASCEND_SPARSE_ATTN, spans, use_real_builder=True
        )
        self.assertNotIn("precision", builder.build.call_args.kwargs)
        self.assertEqual(contexts[0]["attn_metadata"].precision, "bf16")
        self.assertEqual(contexts[0]["attn_metadata"].sparsity, 0.6)
        self.assertEqual(
            contexts[0]["attn_metadata"].dense_kwargs, {"precision": "bf16"}
        )

    def test_explicit_mix_precision_reaches_metadata_builder(self):
        spans = [{"start": 9, "latent_shape": [2, 2, 2]}]
        _, builder, contexts = self._forward_stage(
            AttentionBackendEnum.ASCEND_SPARSE_ATTN,
            spans,
            precision="mix",
            use_real_builder=True,
        )
        builder.build.assert_called_once_with(
            current_timestep=4,
            skip_first_steps=2,
            sparsity=0.6,
            precision="mix",
            skip_last_steps=0,
            num_steps=None,
            video_spans=spans,
        )
        self.assertEqual(contexts[0]["attn_metadata"].precision, "mix")

    def test_native_sparse_precision_modes_reach_metadata_builder(self):
        spans = [{"start": 9, "latent_shape": [2, 2, 2]}]
        for precision in ("bf16", "mix", "fp8", "mxfp4"):
            with self.subTest(precision=precision):
                _, builder, contexts = self._forward_stage(
                    AttentionBackendEnum.ASCEND_SPARSE_ATTN,
                    spans,
                    precision=precision,
                    use_real_builder=True,
                )
                self.assertEqual(builder.build.call_args.kwargs["precision"], precision)
                self.assertEqual(contexts[0]["attn_metadata"].precision, precision)

    def test_h3_forwards_native_sparse_and_dense_options(self):
        options = {
            "sparse_type": "rf_v3",
            "precision": "bf16",
            "block_size": 256,
            "keep_sink": False,
            "keep_recent": True,
            "cdf_threshold": 0.9,
            "mxfp4_dst_type_max": 0.0,
            "mxfp4_scale_alg": None,
            "dense_kwargs": {
                "precision": "mxfp8",
                "head_chunk_size": 8,
                "qk_rotation": True,
            },
        }
        _, builder, _ = self._forward_stage(
            AttentionBackendEnum.ASCEND_SPARSE_ATTN,
            _layout()["video_spans"],
            options=options,
        )
        for name, value in options.items():
            self.assertEqual(builder.build.call_args.kwargs[name], value)

    def test_h3_rejects_unknown_sparse_configuration(self):
        for option in (
            "inner_precise",
            "block_size_kv",
            "keep_snik",
            "dense_last_steps",
            "dense_precision",
        ):
            with self.subTest(option=option):
                with self.assertRaisesRegex(ValueError, "Unknown|unknown"):
                    self._forward_stage(
                        AttentionBackendEnum.ASCEND_SPARSE_ATTN,
                        _layout()["video_spans"],
                        use_real_builder=True,
                        options={option: 4},
                    )

    def test_dense_tail_receives_the_actual_h3_step_count(self):
        spans = _layout()["video_spans"]
        _, builder, contexts = self._forward_stage(
            AttentionBackendEnum.ASCEND_SPARSE_ATTN,
            spans,
            use_real_builder=True,
            skip_last_steps=2,
            num_steps=7,
        )
        self.assertEqual(builder.build.call_args.kwargs["current_timestep"], 4)
        self.assertEqual(builder.build.call_args.kwargs["num_steps"], 7)
        self.assertEqual(contexts[0]["attn_metadata"].skip_last_steps, 2)
        self.assertEqual(contexts[0]["attn_metadata"].num_steps, 7)

    def test_image_loop_dense_tail_uses_the_actual_sigma_schedule(self):
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
            MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
            MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY,
        )

        packed = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=3,
            latent_t=1,
            latent_h=4,
            latent_w=4,
            audio_t=0,
            ref_blocks=[],
        )
        rows = torch.zeros(4, 96)
        positive = SimpleNamespace(packed=packed, attn_metadata=None)
        batch = SimpleNamespace(
            sigmas=[1.0, 0.7, 0.3, 0.0],
            num_inference_steps=50,
            is_warmup=False,
            extra={
                MINIMAX_H3_DENOISE_STATE_EXTRA_KEY: {
                    "latent_h": 4,
                    "latent_w": 4,
                    "initial_video_rows": rows,
                },
                MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY: {"positive": object()},
            },
        )
        metadata = []

        def record_model(**kwargs):
            metadata.append(get_forward_context().attn_metadata)
            return rows, torch.empty(0)

        model = Mock(side_effect=record_model)
        model._resolved_attention_backend = AttentionBackendEnum.ASCEND_SPARSE_ATTN
        model.final_layer = SimpleNamespace(_pdd_heads=object())
        server_args = SimpleNamespace(
            num_gpus=1,
            attention_backend_config={"skip_first_steps": 0, "skip_last_steps": 1},
        )
        stage = SimpleNamespace(
            transformer=model,
            server_args=server_args,
            device=torch.device("cpu"),
            _maybe_enable_cache_dit_and_torch_compile=Mock(),
            _maybe_get_bcg_runner=lambda model: None,
            _profile_denoising_step=Mock(),
            use_declared_component=lambda **kwargs: nullcontext(model),
            progress_bar=lambda **kwargs: nullcontext(Mock()),
        )
        stage._forward_dit = partial(MiniMaxH3DenoisingStage._forward_dit, stage)

        def run_image_loop(*, model, model_forward, sigmas, **kwargs):
            call_kwargs = {"packed_seq_params": {"video_spans": packed["video_spans"]}}
            for step in range(len(sigmas) - 1):
                model_forward(model, call_kwargs, step)
            return rows

        with (
            patch(f"{_STAGE_MODULE}._image_branch_inputs", return_value=positive),
            patch(
                f"{_BRANCH_MODULE}.minimax_h3_image_denoise_loop",
                side_effect=run_image_loop,
            ),
        ):
            MiniMaxH3DenoisingStage._run_image_loop(stage, batch, None, server_args)

        self.assertEqual([item.current_timestep for item in metadata], [0, 1, 2])
        self.assertEqual([item.num_steps for item in metadata], [3, 3, 3])
        self.assertTrue(all(item.skip_last_steps == 1 for item in metadata))
        self.assertTrue(
            all(item.video_spans == packed["video_spans"] for item in metadata)
        )
        self.assertEqual(batch.latents.shape, (1, 24, 1, 4, 4))

    def test_h3_computed_steps_and_layout_override_config_without_mutating_it(self):
        spans = _layout()["video_spans"]
        options = {"current_timestep": 99, "num_steps": 999, "video_spans": []}
        _, builder, _ = self._forward_stage(
            AttentionBackendEnum.ASCEND_SPARSE_ATTN,
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

    def test_other_backends_do_not_require_ascend_sparse_layout(self):
        _, builder, contexts = self._forward_stage(
            AttentionBackendEnum.TORCH_SDPA, None
        )
        builder.build.assert_not_called()
        self.assertIsNone(contexts[0]["attn_metadata"])

    def test_resolved_transformer_override_builds_ascend_sparse_metadata(self):
        spans = _layout()["video_spans"]
        metadata, builder, contexts = self._forward_stage(
            AttentionBackendEnum.FA,
            spans,
            model_backend=AttentionBackendEnum.ASCEND_SPARSE_ATTN,
        )
        builder.build.assert_called_once()
        self.assertEqual(builder.build.call_args.kwargs["video_spans"], spans)
        self.assertIs(contexts[0]["attn_metadata"], metadata)

    def test_resolved_dense_override_skips_ascend_sparse_metadata(self):
        _, builder, contexts = self._forward_stage(
            AttentionBackendEnum.ASCEND_SPARSE_ATTN,
            None,
            model_backend=AttentionBackendEnum.FA,
        )
        builder.build.assert_not_called()
        self.assertIsNone(contexts[0]["attn_metadata"])

    def test_h3_forwards_missing_and_empty_spans(self):
        for spans in (None, []):
            with self.subTest(spans=spans):
                _, builder, _ = self._forward_stage(
                    AttentionBackendEnum.ASCEND_SPARSE_ATTN, spans
                )
                self.assertEqual(builder.build.call_args.kwargs["video_spans"], spans)


class TestAscendSparseScheduleMetadata(_MockedMindIETestCase):
    def test_generic_stage_forwards_full_config_and_normalizes_computed_layout(self):
        builder = Mock()
        stage = SimpleNamespace(
            attn_backend=SimpleNamespace(
                get_enum=lambda: AttentionBackendEnum.ASCEND_SPARSE_ATTN,
                get_builder_cls=lambda: lambda: builder,
            )
        )
        config = {
            "sparse_type": "rf_v3",
            "precision": "bf16",
            "block_size": 256,
            "keep_sink": False,
            "keep_recent": True,
            "cdf_threshold": 0.9,
            "mxfp4_dst_type_max": 0.0,
            "mxfp4_scale_alg": None,
            "dense_kwargs": {
                "precision": "mxfp8",
                "head_chunk_size": 8,
                "qk_rotation": True,
            },
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
            "keep_sink",
            "keep_recent",
            "cdf_threshold",
            "mxfp4_dst_type_max",
            "mxfp4_scale_alg",
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
                get_enum=lambda: AttentionBackendEnum.ASCEND_SPARSE_ATTN,
                get_builder_cls=lambda: AscendSparseAttentionMetadataBuilder,
            )
        )
        server_args = SimpleNamespace(
            attention_backend_config={"skip_last_steps": 2},
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
        self.assertEqual(metadata.skip_last_steps, 2)

    def test_generic_stage_rejects_removed_dense_aliases(self):
        stage = SimpleNamespace(
            attn_backend=SimpleNamespace(
                get_enum=lambda: AttentionBackendEnum.ASCEND_SPARSE_ATTN,
                get_builder_cls=lambda: AscendSparseAttentionMetadataBuilder,
            )
        )
        for config in ({"dense_last_steps": 2}, {"dense_precision": "mxfp8"}):
            with self.subTest(config=config):
                original = dict(config)
                server_args = SimpleNamespace(
                    attention_backend_config=config,
                    pipeline_config=SimpleNamespace(
                        dit_config=SimpleNamespace(patch_size=(1, 1, 1))
                    ),
                )
                with self.assertRaisesRegex(ValueError, "Unknown|unknown"):
                    DenoisingStage._build_attn_metadata(
                        stage,
                        7,
                        SimpleNamespace(raw_latent_shape=[1, 24, 2, 16, 16]),
                        server_args,
                        timesteps=torch.arange(9),
                    )
                self.assertEqual(config, original)

    def test_generic_stage_rejects_unknown_sparse_configuration(self):
        stage = SimpleNamespace(
            attn_backend=SimpleNamespace(
                get_enum=lambda: AttentionBackendEnum.ASCEND_SPARSE_ATTN,
                get_builder_cls=lambda: AscendSparseAttentionMetadataBuilder,
            )
        )
        server_args = SimpleNamespace(
            attention_backend_config={"keep_snik": True},
            pipeline_config=SimpleNamespace(
                dit_config=SimpleNamespace(patch_size=(1, 1, 1))
            ),
        )
        with self.assertRaisesRegex(ValueError, "Unknown|unknown"):
            DenoisingStage._build_attn_metadata(
                stage,
                1,
                SimpleNamespace(raw_latent_shape=[1, 24, 2, 16, 16]),
                server_args,
                timesteps=torch.arange(9),
            )

    def test_other_sparse_backends_do_not_receive_ascend_sparse_schedule_fields(self):
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
                        "skip_last_steps": 2,
                        "sparse_type": "rf_v3",
                        "precision": "fp8",
                        "dense_kwargs": {
                            "precision": "mxfp8",
                            "head_chunk_size": 8,
                        },
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
                self.assertNotIn("skip_last_steps", builder.build.call_args.kwargs)
                self.assertNotIn("num_steps", builder.build.call_args.kwargs)
                self.assertNotIn("sparse_type", builder.build.call_args.kwargs)
                self.assertNotIn("precision", builder.build.call_args.kwargs)
                self.assertNotIn("dense_kwargs", builder.build.call_args.kwargs)


class TestMiniMaxH3AscendSparseBinding(_MockedMindIETestCase):
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
            get_enum=lambda: AttentionBackendEnum.ASCEND_SPARSE_ATTN,
            get_impl_cls=lambda: sparse_ctor,
        )
        attention = SimpleNamespace(
            head_dim=128,
            num_heads=2,
            softmax_scale=128**-0.5,
            prefix=prefix,
            _cube_sparse_capable=True,
            _quant_config=object(),
        )
        with patch(
            f"{_MODEL_MODULE}.get_attn_backend", return_value=dense_backend
        ) as select:
            MiniMaxH3Attention._set_attention_backend(attention, sparse_backend)
        return attention, dense_ctor, sparse_ctor, select

    def test_refiner_policy_is_applied_before_constructing_the_ascend_sparse_backend(
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
        self.assertIs(
            dense_ctor.call_args.kwargs["quant_config"], attention._quant_config
        )

    def test_multimodal_dit_keeps_the_selected_ascend_sparse_backend(self):
        attention, dense_ctor, sparse_ctor, select = self._bind("blocks.0.attn")
        select.assert_not_called()
        dense_ctor.assert_not_called()
        sparse_ctor.assert_called_once()
        self.assertIs(
            sparse_ctor.call_args.kwargs["quant_config"], attention._quant_config
        )
        self.assertIs(
            attention._attention_backend_enum, AttentionBackendEnum.ASCEND_SPARSE_ATTN
        )


if __name__ == "__main__":
    unittest.main()
