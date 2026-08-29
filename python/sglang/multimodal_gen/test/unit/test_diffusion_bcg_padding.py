import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import (
    DiffusionBreakableCudaGraphRunner,
    _CaptureEntry,
    _signature_kwargs,
)
from sglang.multimodal_gen.runtime.layers.attention import (
    DynamicVarlenMaskMeta,
    build_varlen_mask_meta,
)
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    _attn_mask_meta_local_pad,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    denoising as denoising_module,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import (
    BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS,
    BREAKABLE_CUDA_GRAPH_SUPPORTED_PIPELINE_CONFIGS,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import _print_warning_once


class QwenImageTransformer2DModel(torch.nn.Module):
    pass


class OtherTransformer2DModel(torch.nn.Module):
    pass


class Ideogram4Transformer2DModel(torch.nn.Module):
    pass


class MiniMaxH3DiTModel(torch.nn.Module):
    pass


class LongCatImageTransformer2DModel(torch.nn.Module):
    pass


class ZImageTransformer2DModel(torch.nn.Module):
    def rotary_emb(self, pos_ids):
        return torch.zeros(pos_ids.shape[0], 8, device=pos_ids.device)


class SanaVideoTransformer3DModel(torch.nn.Module):
    pass


class TestQualityFusionBCGCompatibility(unittest.TestCase):
    def setUp(self):
        self.stage = DenoisingStage.__new__(DenoisingStage)
        self.stage.server_args = SimpleNamespace(enable_breakable_cuda_graph=True)
        self.stage.transformer = OtherTransformer2DModel()
        self.stage.transformer_2 = None
        self.stage._quality_fusions_mounted = False

    @staticmethod
    def _batch(quality: str):
        return SimpleNamespace(sampling_params=SimpleNamespace(quality=quality))

    def test_rejects_high_when_dit_fusion_would_replace_captured_graph(self):
        unmounted = []
        handlers = (
            (
                "test fusion",
                lambda _: True,
                lambda transformer: unmounted.append(transformer),
            ),
        )

        with patch.object(denoising_module, "_QUALITY_FUSION_HANDLERS", handlers):
            with self.assertRaisesRegex(ValueError, "lossless warmup graphs"):
                self.stage._maybe_toggle_quality_fusions(self._batch("high"))

        self.assertEqual(unmounted, [self.stage.transformer])
        self.assertFalse(self.stage._quality_fusions_mounted)

    def test_allows_high_when_model_has_no_dit_quality_fusions(self):
        handlers = (("test fusion", lambda _: False, lambda _: None),)

        with patch.object(denoising_module, "_QUALITY_FUSION_HANDLERS", handlers):
            self.stage._maybe_toggle_quality_fusions(self._batch("high"))

        self.assertTrue(self.stage._quality_fusions_mounted)


def _fake_cache_dit_batch(*, is_warmup: bool) -> SimpleNamespace:
    return SimpleNamespace(
        is_warmup=is_warmup,
        sampling_params=SimpleNamespace(enable_cache_dit=None, cache_dit_params=None),
    )


class TestDiffusionBCGPadding(unittest.TestCase):
    def setUp(self):
        self.stage = DenoisingStage.__new__(DenoisingStage)
        self.qwen_model = QwenImageTransformer2DModel()
        self.ideogram_model = Ideogram4Transformer2DModel()
        self.longcat_model = LongCatImageTransformer2DModel()
        self.minimax_h3_model = MiniMaxH3DiTModel()
        self.zimage_model = ZImageTransformer2DModel()
        self.sana_video_model = SanaVideoTransformer3DModel()
        self.other_model = OtherTransformer2DModel()

    def _patch_buckets(self, *buckets: int):
        resolved = tuple(sorted({b for b in buckets if b > 0}))
        return patch.object(
            DenoisingStage,
            "_bcg_text_buckets",
            staticmethod(lambda: resolved),
        )

    def _qwen_kwargs(self, seq_len: int, *, fill: float = 1.0):
        return {
            "hidden_states": torch.zeros(1, 4096, 64),
            "timestep": torch.zeros(1),
            "encoder_hidden_states": [
                torch.full((1, seq_len, 3584), fill, dtype=torch.float32)
            ],
            "encoder_hidden_states_mask": None,
            "txt_seq_lens": [seq_len],
            "freqs_cis": (
                torch.zeros(4096, 128, dtype=torch.float32),
                torch.ones(seq_len, 128, dtype=torch.float32),
            ),
            "img_shapes": [[(1, 64, 64)]],
        }

    def test_qwen_prompt_lengths_share_bucket_signature(self):
        with self._patch_buckets(256, 512, 1024):
            short = self.stage._bcg_pad_prompt_kwargs(
                self._qwen_kwargs(19), current_model=self.qwen_model
            )
            longer = self.stage._bcg_pad_prompt_kwargs(
                self._qwen_kwargs(47), current_model=self.qwen_model
            )

        self.assertEqual(short["encoder_hidden_states"][0].shape, (1, 256, 3584))
        self.assertEqual(longer["encoder_hidden_states"][0].shape, (1, 256, 3584))
        self.assertEqual(short["encoder_hidden_states_mask"].shape, (1, 256))
        self.assertTrue(short["encoder_hidden_states_mask"][0, :19].all())
        self.assertFalse(short["encoder_hidden_states_mask"][0, 19:].any())
        self.assertTrue(longer["encoder_hidden_states_mask"][0, :47].all())
        self.assertFalse(longer["encoder_hidden_states_mask"][0, 47:].any())
        self.assertEqual(short["freqs_cis"][1].shape, (256, 128))
        self.assertEqual(short["txt_seq_lens"], [256])
        self.assertEqual(longer["txt_seq_lens"], [256])
        self.assertEqual(_signature_kwargs(short), _signature_kwargs(longer))

    def test_qwen_prompt_content_changes_do_not_change_signature(self):
        with self._patch_buckets(256, 512, 1024):
            first = self.stage._bcg_pad_prompt_kwargs(
                self._qwen_kwargs(47, fill=1.0), current_model=self.qwen_model
            )
            second = self.stage._bcg_pad_prompt_kwargs(
                self._qwen_kwargs(47, fill=2.0), current_model=self.qwen_model
            )

        self.assertFalse(
            torch.equal(
                first["encoder_hidden_states"][0],
                second["encoder_hidden_states"][0],
            )
        )
        self.assertEqual(_signature_kwargs(first), _signature_kwargs(second))

    def test_qwen_dynamic_varlen_meta_is_not_tail_pad_meta(self):
        self.assertEqual(_attn_mask_meta_local_pad(None), 0)
        self.assertEqual(_attn_mask_meta_local_pad({"local_pad": 7}), 7)
        self.assertEqual(_attn_mask_meta_local_pad(DynamicVarlenMaskMeta()), 0)

    def test_longcat_keeps_its_fixed_512_token_prompt_shape(self):
        kwargs = {
            "hidden_states": torch.zeros(1, 4096, 64),
            "timestep": torch.zeros(1),
            "encoder_hidden_states": torch.zeros(1, 512, 3584),
            "encoder_attention_mask": [torch.ones(1, 512, dtype=torch.long)],
            "encoder_hidden_states_mask": [torch.ones(1, 512, dtype=torch.bool)],
            "txt_ids": torch.zeros(512, 3),
            "img_ids": torch.zeros(4096, 3),
        }

        with self._patch_buckets(64, 128, 256, 512, 1024):
            out = self.stage._bcg_pad_prompt_kwargs(
                kwargs, current_model=self.longcat_model
            )

        self.assertIs(out, kwargs)
        self.assertEqual(out["encoder_hidden_states"].shape, (1, 512, 3584))
        self.assertEqual(out["txt_ids"].shape, (512, 3))

    def test_qwen_default_bucket_preserves_mask(self):
        def kwargs(valid_len: int):
            mask = torch.zeros(1, 64, dtype=torch.bool)
            mask[:, :valid_len] = True
            out = self._qwen_kwargs(64)
            out["encoder_hidden_states_mask"] = mask
            out["txt_seq_lens"] = [valid_len]
            return out

        first = self.stage._bcg_pad_prompt_kwargs(
            kwargs(19), current_model=self.qwen_model
        )
        second = self.stage._bcg_pad_prompt_kwargs(
            kwargs(47), current_model=self.qwen_model
        )

        self.assertEqual(first["encoder_hidden_states"][0].shape[1], 64)
        self.assertEqual(first["txt_seq_lens"], [64])
        self.assertEqual(second["txt_seq_lens"], [64])
        self.assertTrue(first["encoder_hidden_states_mask"][0, :19].all())
        self.assertFalse(first["encoder_hidden_states_mask"][0, 19:].any())
        self.assertTrue(second["encoder_hidden_states_mask"][0, :47].all())
        self.assertFalse(second["encoder_hidden_states_mask"][0, 47:].any())
        self.assertEqual(_signature_kwargs(first), _signature_kwargs(second))

    def test_non_qwen_kwargs_do_not_take_qwen_padding_path(self):
        kwargs = self._qwen_kwargs(47)
        with self._patch_buckets(256, 512, 1024):
            out = self.stage._bcg_pad_prompt_kwargs(
                kwargs, current_model=self.other_model
            )

        self.assertIs(out, kwargs)
        self.assertIsNone(out["encoder_hidden_states_mask"])
        self.assertEqual(out["encoder_hidden_states"][0].shape[1], 47)
        self.assertEqual(out["txt_seq_lens"], [47])

    def _zimage_kwargs(self, seq_len: int, *, fill: float = 1.0):
        return {
            "hidden_states": [torch.zeros(16, 1, 4, 4)],
            "timestep": torch.zeros(1),
            "guidance": torch.zeros(1),
            "encoder_hidden_states": [
                torch.full((seq_len, 16), fill, dtype=torch.float32)
            ],
            "encoder_hidden_states_mask": torch.ones(1, seq_len, dtype=torch.bool),
            "freqs_cis": (
                torch.zeros(seq_len, 8, dtype=torch.float32),
                torch.zeros(256, 8, dtype=torch.float32),
            ),
            "image_seq_len_target": 256,
        }

    def test_zimage_captures_at_native_caption_length(self):
        """Z-Image attends its caption slots unmasked (learned pad tokens act
        as attended registers), so padding to a shared text bucket changes how
        many registers every token attends and drifts the output
        (sgl-project/sglang#34183). The padder therefore captures at the
        incoming native length: lengths are never extended, and distinct
        native lengths intentionally do NOT share a graph signature (unseen
        lengths fall back to eager at serving time)."""
        with self._patch_buckets(64, 128):
            short = self.stage._bcg_pad_prompt_kwargs(
                self._zimage_kwargs(19), current_model=self.zimage_model
            )
            short_again = self.stage._bcg_pad_prompt_kwargs(
                self._zimage_kwargs(19), current_model=self.zimage_model
            )
            longer = self.stage._bcg_pad_prompt_kwargs(
                self._zimage_kwargs(47), current_model=self.zimage_model
            )

        # Captions keep their native length -- no bucket extension.
        self.assertEqual(short["encoder_hidden_states"][0].shape, (19, 16))
        self.assertEqual(longer["encoder_hidden_states"][0].shape, (47, 16))
        self.assertEqual(short["encoder_hidden_states_mask"].shape, (1, 19))
        self.assertEqual(short["caption_valid_lens"].item(), 19)
        self.assertEqual(longer["caption_valid_lens"].item(), 47)
        self.assertTrue(short["_use_caption_valid_mask"])
        self.assertTrue(short["encoder_hidden_states_mask"].all())
        self.assertEqual(short["freqs_cis"][0].shape, (19, 8))
        # Same native length -> same signature (graph reuse works); different
        # native lengths -> different signatures, by design.
        self.assertEqual(_signature_kwargs(short), _signature_kwargs(short_again))
        self.assertNotEqual(_signature_kwargs(short), _signature_kwargs(longer))

    def _minimax_h3_kwargs(self, text_seq: int):
        image_seq = 4
        audio_seq = 2
        media_seq = image_seq + audio_seq
        used = text_seq + media_seq
        packed_seq = 64
        audio_pos = torch.arange(text_seq, text_seq + audio_seq)
        img_pos = torch.arange(text_seq + audio_seq, used)
        return {
            "x": torch.zeros(1, packed_seq, 96),
            "audio_x": torch.zeros(1, packed_seq, 32),
            "img_position_ids": torch.zeros(1, packed_seq, 3),
            "unique_timesteps": torch.zeros(1),
            "inverse_indices": torch.zeros(packed_seq, dtype=torch.long),
            "update_mask": torch.ones(image_seq, dtype=torch.bool),
            "token_tags": torch.full((packed_seq,), -1, dtype=torch.long),
            "skip_mask_out_condition": False,
            "prompt_embeds": torch.ones(text_seq, 8),
            "refined_prompt_embeds_length": text_seq,
            "img_pos_info": {"position_ids": img_pos},
            "audio_pos_info": {"position_ids": audio_pos},
            "text_pos_info": {"position_ids": torch.arange(text_seq, dtype=torch.long)},
            "img_pos_for_infer_output_info": {"position_ids": img_pos},
            "local_embedding_layout": {
                "text_source_ids": torch.arange(text_seq),
                "text_row_ids": torch.arange(text_seq),
            },
            "packed_seq_params": {
                "cu_seqlens_q": torch.tensor([0, used, packed_seq], dtype=torch.int32),
                "max_seqlen_q": used,
            },
            "refiner_packed_seq_params": {
                "cu_seqlens_q": torch.tensor(
                    [0, text_seq, text_seq], dtype=torch.int32
                ),
                "max_seqlen_q": text_seq,
            },
        }

    def test_minimax_h3_prompt_lengths_share_bucket_signature(self):
        with self._patch_buckets(64, 128):
            short = self.stage._bcg_pad_prompt_kwargs(
                self._minimax_h3_kwargs(19), current_model=self.minimax_h3_model
            )
            longer = self.stage._bcg_pad_prompt_kwargs(
                self._minimax_h3_kwargs(47), current_model=self.minimax_h3_model
            )

        # H3 keeps the original aligned main sequence: growing it changes SP
        # row partitions and loses bitwise eager equivalence.
        self.assertEqual(short["x"].shape, (1, 64, 96))
        self.assertEqual(longer["x"].shape, (1, 64, 96))
        self.assertEqual(short["audio_x"].shape, (1, 64, 32))
        self.assertEqual(short["prompt_embeds"].shape, (64, 8))
        self.assertEqual(longer["prompt_embeds"].shape, (64, 8))
        self.assertNotIn("local_embedding_layout", short)
        self.assertNotIn("local_embedding_layout", longer)
        self.assertEqual(
            short["refined_prompt_embeds_length"].shape,
            torch.Size([]),
        )
        self.assertEqual(short["refined_prompt_embeds_length"].item(), 19)
        self.assertEqual(longer["refined_prompt_embeds_length"].item(), 47)
        self.assertEqual(
            short["packed_seq_params"]["cu_seqlens_q"].tolist(), [0, 25, 64]
        )
        self.assertEqual(
            longer["packed_seq_params"]["cu_seqlens_q"].tolist(), [0, 53, 64]
        )
        self.assertEqual(short["packed_seq_params"]["max_seqlen_q"], 64)
        self.assertEqual(short["refiner_packed_seq_params"]["max_seqlen_q"], 64)
        self.assertEqual(
            short["refiner_packed_seq_params"]["cu_seqlens_q"].tolist(),
            [0, 19, 64],
        )
        # Dummy prompt rows occupy the independent packed-attention segment;
        # real text/media remain in the first segment.
        self.assertEqual(
            short["text_pos_info"]["position_ids"][19:].tolist(),
            list(range(25, 70)),
        )
        self.assertEqual(
            longer["text_pos_info"]["position_ids"][47:].tolist(),
            list(range(53, 70)),
        )
        self.assertEqual(_signature_kwargs(short), _signature_kwargs(longer))

    def _ideogram_kwargs(self, text_seq: int, *, image_seq: int = 4):
        total_seq = text_seq + image_seq
        indicator = torch.zeros(1, total_seq, dtype=torch.long)
        if text_seq:
            indicator[:, :text_seq] = 3
        indicator[:, text_seq:] = 2
        segment_ids = torch.ones(1, total_seq, dtype=torch.long)
        if text_seq:
            segment_ids[:, :text_seq] = 1
        return {
            "llm_features": torch.ones(1, total_seq, 8),
            "x": torch.zeros(1, total_seq, 16),
            "t": torch.zeros(1),
            "position_ids": torch.zeros(1, total_seq, 3, dtype=torch.long),
            "segment_ids": segment_ids,
            "indicator": indicator,
            "attn_mask": segment_ids > 0,
            "attn_mask_meta": build_varlen_mask_meta(segment_ids > 0),
        }

    def test_ideogram_prompt_lengths_share_bucket_signature(self):
        with self._patch_buckets(64, 128):
            short = self.stage._bcg_pad_prompt_kwargs(
                self._ideogram_kwargs(19), current_model=self.ideogram_model
            )
            longer = self.stage._bcg_pad_prompt_kwargs(
                self._ideogram_kwargs(47), current_model=self.ideogram_model
            )

        self.assertEqual(short["llm_features"].shape, (1, 68, 8))
        self.assertEqual(longer["llm_features"].shape, (1, 68, 8))
        self.assertEqual(short["x"].shape, (1, 68, 16))
        self.assertEqual(short["position_ids"].shape, (1, 68, 3))
        self.assertEqual(short["segment_ids"][0, 23:].tolist(), [-1] * 45)
        self.assertFalse(short["attn_mask"][0, 23:].any())
        self.assertIsInstance(short["attn_mask_meta"], DynamicVarlenMaskMeta)
        self.assertIs(short["attn_mask_meta"], longer["attn_mask_meta"])
        self.assertEqual(_signature_kwargs(short), _signature_kwargs(longer))

    def test_ideogram_image_only_kwargs_are_not_prompt_padded(self):
        kwargs = self._ideogram_kwargs(0)
        with self._patch_buckets(64, 128):
            out = self.stage._bcg_pad_prompt_kwargs(
                kwargs, current_model=self.ideogram_model
            )

        self.assertIs(out, kwargs)
        self.assertEqual(out["x"].shape, (1, 4, 16))
        self.assertIsInstance(out["attn_mask_meta"], dict)

    def test_ideogram_is_registered_as_bcg_supported(self):
        self.assertIn(
            "ideogram-ai/ideogram-4-fp8",
            BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS,
        )
        self.assertIn(
            "comfy-org/ideogram-4",
            BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS,
        )
        self.assertIn(
            "fal/ideogram-v4-fast",
            BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS,
        )
        self.assertIn(
            "fal/ideogram-v4-instant",
            BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS,
        )
        self.assertIn(
            "Ideogram4PipelineConfig",
            BREAKABLE_CUDA_GRAPH_SUPPORTED_PIPELINE_CONFIGS,
        )

    def test_image_generation_models_are_registered_as_bcg_supported(self):
        for model_id in (
            "meituan-longcat/longcat-image",
            "qwen/qwen-image",
            "qwen/qwen-image-2512",
            "tongyi-mai/z-image",
            "tongyi-mai/z-image-turbo",
            "zai-org/glm-image",
        ):
            self.assertIn(model_id, BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS)

        for config_name in (
            "GlmImagePipelineConfig",
            "LongCatImagePipelineConfig",
            "QwenImagePipelineConfig",
            "ZImagePipelineConfig",
        ):
            self.assertIn(config_name, BREAKABLE_CUDA_GRAPH_SUPPORTED_PIPELINE_CONFIGS)

    def test_ltx_models_are_registered_as_bcg_supported(self):
        for model_id in (
            "lightricks/ltx-2",
            "lightricks/ltx-2.3",
        ):
            self.assertIn(model_id, BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS)

        self.assertIn(
            "LTX2PipelineConfig", BREAKABLE_CUDA_GRAPH_SUPPORTED_PIPELINE_CONFIGS
        )
        self.assertIn(
            "LTX23PipelineConfig", BREAKABLE_CUDA_GRAPH_SUPPORTED_PIPELINE_CONFIGS
        )

    def test_sana_video_keeps_its_fixed_prompt_shape(self):
        kwargs = {
            "encoder_hidden_states": torch.zeros(1, 300, 2304),
            "encoder_attention_mask": torch.ones(1, 300, dtype=torch.long),
        }
        with self._patch_buckets(64, 128, 256, 512, 1024):
            out = self.stage._bcg_pad_prompt_kwargs(
                kwargs, current_model=self.sana_video_model
            )

        self.assertIs(out, kwargs)
        self.assertEqual(out["encoder_hidden_states"].shape, (1, 300, 2304))
        self.assertEqual(out["encoder_attention_mask"].shape, (1, 300))
        self.assertIn(
            "efficient-large-model/sana-video_2b_480p_diffusers",
            BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS,
        )
        self.assertIn(
            "SanaVideoPipelineConfig",
            BREAKABLE_CUDA_GRAPH_SUPPORTED_PIPELINE_CONFIGS,
        )

    def test_dynamic_varlen_mask_meta_rebuilds_once_per_replay_token(self):
        builder = DynamicVarlenMaskMeta()
        mask = torch.tensor([[True, True, False, False]])
        calls = []

        def fake_build(current_mask):
            calls.append(current_mask.clone())
            return {"valid": int(current_mask.sum().item())}

        with (
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.layer."
                "build_varlen_mask_meta",
                side_effect=fake_build,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.layer."
                "get_current_replay_token",
                side_effect=[1, 1, 2],
            ),
        ):
            first = builder.resolve(mask)
            mask[0, 2] = True
            second = builder.resolve(mask)
            third = builder.resolve(mask)

        self.assertEqual(first, {"valid": 2})
        self.assertIs(second, first)
        self.assertEqual(third, {"valid": 3})
        self.assertEqual(len(calls), 2)

    def test_disabled_bcg_flag_skips_runner(self):
        self.stage.server_args = SimpleNamespace(
            enable_breakable_cuda_graph=False,
            enable_torch_compile=False,
        )
        self.stage._bcg_runners = {}
        self.stage._cache_dit_enabled = False

        self.assertIsNone(self.stage._maybe_get_bcg_runner(self.qwen_model))
        self.stage._maybe_torch_compile(self.qwen_model)
        self.stage._maybe_enable_cache_dit(1, _fake_cache_dit_batch(is_warmup=True))
        self.assertEqual(self.stage._bcg_runners, {})

    def test_bcg_warns_when_cache_dit_is_requested(self):
        self.stage.server_args = SimpleNamespace(enable_breakable_cuda_graph=True)
        _print_warning_once.cache_clear()
        self.addCleanup(_print_warning_once.cache_clear)

        with (
            patch.object(self.stage, "_cache_dit_requested", return_value=True),
            patch.object(denoising_module.logger, "warning") as warning,
        ):
            self.stage._maybe_enable_cache_dit(1, _fake_cache_dit_batch(is_warmup=True))
            self.stage._maybe_enable_cache_dit(
                1, _fake_cache_dit_batch(is_warmup=False)
            )

        warning.assert_called_once_with(
            "Cache-DiT was requested but is disabled because breakable CUDA "
            "graphs are enabled.",
            stacklevel=2,
        )

    def test_bcg_does_not_warn_when_cache_dit_is_not_requested(self):
        self.stage.server_args = SimpleNamespace(enable_breakable_cuda_graph=True)

        with (
            patch.object(self.stage, "_cache_dit_requested", return_value=False),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising."
                "logger.warning_once"
            ) as warning_once,
        ):
            self.stage._maybe_enable_cache_dit(1, _fake_cache_dit_batch(is_warmup=True))

        warning_once.assert_not_called()

    def test_bcg_runner_cache_is_per_model_module(self):
        self.stage.server_args = SimpleNamespace(enable_breakable_cuda_graph=True)
        self.stage._bcg_runners = {}

        def fake_runner(model, device):
            return SimpleNamespace(model=model, device=device)

        with (
            patch(
                "sglang.multimodal_gen.runtime.breakable_cuda_graph.runner."
                "DiffusionBreakableCudaGraphRunner",
                side_effect=fake_runner,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising."
                "get_local_torch_device",
                return_value=torch.device("cpu"),
            ),
        ):
            first = self.stage._maybe_get_bcg_runner(self.qwen_model)
            second = self.stage._maybe_get_bcg_runner(self.other_model)
            first_again = self.stage._maybe_get_bcg_runner(self.qwen_model)

        self.assertIs(first_again, first)
        self.assertIsNot(first, second)
        self.assertIs(first.model, self.qwen_model)
        self.assertIs(second.model, self.other_model)
        self.assertEqual(len(self.stage._bcg_runners), 2)

    def test_bcg_runner_rejects_too_many_segments(self):
        runner = object.__new__(DiffusionBreakableCudaGraphRunner)
        runner.max_segments = 2
        entry = _CaptureEntry(
            graph=SimpleNamespace(_break_fns=[], _segments=[object()] * 3),
            static_kwargs={},
            static_leaves=[],
            output=None,
            num_segments=3,
        )

        self.assertIn("captured 3 segments", runner._capture_limit_reason(entry))

    def test_bcg_runner_lazy_capture_only_during_warmup(self):
        runner = object.__new__(DiffusionBreakableCudaGraphRunner)

        with patch(
            "sglang.multimodal_gen.runtime.managers.forward_context.get_forward_context",
            return_value=SimpleNamespace(forward_batch=SimpleNamespace(is_warmup=True)),
        ):
            self.assertTrue(runner._should_capture_on_call(("sig",)))

        with patch(
            "sglang.multimodal_gen.runtime.managers.forward_context.get_forward_context",
            return_value=SimpleNamespace(
                forward_batch=SimpleNamespace(is_warmup=False)
            ),
        ):
            self.assertFalse(runner._should_capture_on_call(("sig",)))

    def test_bcg_runner_reset_drops_entries_and_marks_disabled(self):
        runner = object.__new__(DiffusionBreakableCudaGraphRunner)
        runner.device_module = SimpleNamespace(empty_cache=lambda: None)
        entry = _CaptureEntry(
            graph=SimpleNamespace(_break_fns=[lambda: None], _segments=[object()]),
            static_kwargs={"x": torch.zeros(1)},
            static_leaves=[torch.zeros(1)],
            output=torch.zeros(1),
            num_segments=1,
        )
        runner.entries = {("sig",): entry}
        runner._blocked = {("sig",)}

        runner.reset(disabled_reason="too much memory")

        self.assertEqual(runner.entries, {})
        self.assertEqual(runner._blocked, set())
        self.assertEqual(entry.graph._break_fns, [])
        self.assertEqual(entry.graph._segments, [])
        self.assertIsNone(entry.output)
        self.assertEqual(runner._disabled_reason, "too much memory")

    def test_bcg_runner_allows_unlimited_segments(self):
        runner = object.__new__(DiffusionBreakableCudaGraphRunner)
        runner.max_segments = 0
        entry = _CaptureEntry(
            graph=SimpleNamespace(_break_fns=[], _segments=[object()]),
            static_kwargs={},
            static_leaves=[],
            output=None,
            num_segments=1,
        )

        self.assertIsNone(runner._capture_limit_reason(entry))


if __name__ == "__main__":
    unittest.main()
