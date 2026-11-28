"""Unit tests for Qwen3-ASR native Hugging Face flat config and weight loading."""

import copy
import json
import os
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import torch
from transformers import AutoConfig
from transformers.models.qwen3_asr.configuration_qwen3_asr import (
    Qwen3ASRConfig as UpstreamHFQwen3ASRConfig,
)

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.configs.qwen3_asr import Qwen3ASRConfig, Qwen3ASRProcessor
from sglang.srt.hf_transformers.common import get_hf_text_config
from sglang.srt.hf_transformers.config import get_config
from sglang.srt.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.parallel_groups import parallel_scope, publish

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _qwen3_asr_1_7b_hf_config_dict() -> dict:
    """Real config.json shape from Qwen/Qwen3-ASR-1.7B-hf."""
    return {
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "audio_config": {
            "activation_dropout": 0,
            "activation_function": "gelu",
            "attention_dropout": 0,
            "conv_chunksize": 500,
            "d_model": 1024,
            "downsample_hidden_size": 480,
            "dropout": 0,
            "encoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "encoder_layers": 24,
            "initializer_range": 0.02,
            "max_position_embeddings": 13,
            "model_type": "qwen3_asr_encoder",
            "n_window": 50,
            "n_window_infer": 800,
            "num_key_value_heads": 16,
            "num_mel_bins": 128,
            "output_dim": 2048,
            "scale_embedding": False,
        },
        "audio_token_id": 151676,
        "dtype": "bfloat16",
        "eos_token_id": [151643, 151645],
        "initializer_range": 0.02,
        "model_type": "qwen3_asr",
        "pad_token_id": 151645,
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": None,
            "eos_token_id": None,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "intermediate_size": 6144,
            "layer_types": ["full_attention"] * 28,
            "max_position_embeddings": 65536,
            "max_window_layers": 28,
            "model_type": "qwen3",
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "pad_token_id": None,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
            "sliding_window": None,
            "tie_word_embeddings": True,
            "use_cache": True,
            "use_sliding_window": False,
            "vocab_size": 151936,
        },
        "tie_word_embeddings": True,
        "timestamp_token_id": 151705,
        "token_classification_bias": False,
        "transformers_version": "5.13.0.dev0",
    }


def _oruk_fourier_2b_config_dict() -> dict:
    """Real config.json shape from oruk/oruk-fourier-2b."""
    cfg = _qwen3_asr_1_7b_hf_config_dict()
    cfg["auto_map"] = {"AutoConfig": "configuration_qwen3_asr.Qwen3ASRConfig"}
    cfg["transformers_version"] = "5.21.0.dev0"
    return cfg


def _legacy_thinker_1_7b_config_dict() -> dict:
    """Real config.json shape from legacy Qwen/Qwen3-ASR-1.7B."""
    return {
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "model_type": "qwen3_asr",
        "support_languages": ["Chinese", "English"],
        "thinker_config": {
            "architectures": ["Qwen3ASRThinkerForConditionalGeneration"],
            "audio_config": {
                "activation_dropout": 0,
                "activation_function": "gelu",
                "attention_dropout": 0,
                "conv_chunksize": 500,
                "d_model": 1024,
                "downsample_hidden_size": 480,
                "dropout": 0,
                "encoder_attention_heads": 16,
                "encoder_ffn_dim": 4096,
                "encoder_layers": 24,
                "initializer_range": 0.02,
                "max_source_positions": 1500,
                "model_type": "qwen3_asr_audio_encoder",
                "n_window": 50,
                "n_window_infer": 800,
                "num_mel_bins": 128,
                "output_dim": 2048,
                "scale_embedding": False,
            },
            "audio_end_token_id": 151670,
            "audio_start_token_id": 151669,
            "audio_token_id": 151676,
            "dtype": "bfloat16",
            "initializer_range": 0.02,
            "model_type": "qwen3_asr_thinker",
            "text_config": {
                "attention_bias": False,
                "attention_dropout": 0.0,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 2048,
                "initializer_range": 0.02,
                "intermediate_size": 6144,
                "max_position_embeddings": 65536,
                "model_type": "qwen3_asr_text",
                "num_attention_heads": 16,
                "num_hidden_layers": 28,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-06,
                "rope_scaling": {
                    "interleaved": True,
                    "mrope_interleaved": True,
                    "mrope_section": [24, 20, 20],
                    "rope_type": "default",
                    "type": "default",
                },
                "rope_theta": 1000000,
                "tie_word_embeddings": True,
                "use_cache": True,
                "vocab_size": 151936,
            },
        },
        "transformers_version": "4.57.6",
    }


def _tiny_audio_config_dict() -> dict:
    return {
        "model_type": "qwen3_asr_encoder",
        "d_model": 32,
        "encoder_layers": 1,
        "encoder_attention_heads": 4,
        "encoder_ffn_dim": 64,
        "downsample_hidden_size": 8,
        "num_mel_bins": 16,
        "output_dim": 64,
        "n_window": 50,
        "n_window_infer": 100,
        "conv_chunksize": 500,
        "max_position_embeddings": 13,
    }


def _tiny_text_config_dict(tie_word_embeddings: bool = True) -> dict:
    return {
        "model_type": "qwen3",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 128,
        "max_position_embeddings": 1024,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        "tie_word_embeddings": tie_word_embeddings,
    }


def _build_checkpoint_tensors(
    *,
    format_kind: str,
    tie_word_embeddings: bool,
    include_lm_head: bool,
    seed: int = 42,
):
    """Build deterministic full-coverage checkpoint tensors for legacy or native HF layout."""
    gen = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    audio_prefix = (
        "thinker.audio_tower." if format_kind == "legacy" else "model.audio_tower."
    )
    llm_prefix = (
        "thinker.model." if format_kind == "legacy" else "model.language_model."
    )
    proj1_prefix = (
        "thinker.audio_tower.proj1."
        if format_kind == "legacy"
        else "model.multi_modal_projector.linear_1."
    )
    proj2_prefix = (
        "thinker.audio_tower.proj2."
        if format_kind == "legacy"
        else "model.multi_modal_projector.linear_2."
    )
    lm_head_name = (
        "thinker.lm_head.weight" if format_kind == "legacy" else "lm_head.weight"
    )

    weights = [
        # Audio CNN frontend + conv_out
        (f"{audio_prefix}conv2d1.weight", rand(8, 1, 3, 3)),
        (f"{audio_prefix}conv2d1.bias", rand(8)),
        (f"{audio_prefix}conv2d2.weight", rand(8, 8, 3, 3)),
        (f"{audio_prefix}conv2d2.bias", rand(8)),
        (f"{audio_prefix}conv2d3.weight", rand(8, 8, 3, 3)),
        (f"{audio_prefix}conv2d3.bias", rand(8)),
        (f"{audio_prefix}conv_out.weight", rand(32, 16)),
        # Audio encoder layer 0
        (f"{audio_prefix}layers.0.self_attn.q_proj.weight", rand(32, 32)),
        (f"{audio_prefix}layers.0.self_attn.q_proj.bias", rand(32)),
        (f"{audio_prefix}layers.0.self_attn.k_proj.weight", rand(32, 32)),
        (f"{audio_prefix}layers.0.self_attn.k_proj.bias", rand(32)),
        (f"{audio_prefix}layers.0.self_attn.v_proj.weight", rand(32, 32)),
        (f"{audio_prefix}layers.0.self_attn.v_proj.bias", rand(32)),
        (f"{audio_prefix}layers.0.self_attn.out_proj.weight", rand(32, 32)),
        (f"{audio_prefix}layers.0.self_attn.out_proj.bias", rand(32)),
        (f"{audio_prefix}layers.0.self_attn_layer_norm.weight", rand(32)),
        (f"{audio_prefix}layers.0.self_attn_layer_norm.bias", rand(32)),
        (f"{audio_prefix}layers.0.fc1.weight", rand(64, 32)),
        (f"{audio_prefix}layers.0.fc1.bias", rand(64)),
        (f"{audio_prefix}layers.0.fc2.weight", rand(32, 64)),
        (f"{audio_prefix}layers.0.fc2.bias", rand(32)),
        (f"{audio_prefix}layers.0.final_layer_norm.weight", rand(32)),
        (f"{audio_prefix}layers.0.final_layer_norm.bias", rand(32)),
        (f"{audio_prefix}ln_post.weight", rand(32)),
        (f"{audio_prefix}ln_post.bias", rand(32)),
        # Audio multimodal projector (proj1/proj2 vs multi_modal_projector.linear_1/2)
        (f"{proj1_prefix}weight", rand(32, 32)),
        (f"{proj1_prefix}bias", rand(32)),
        (f"{proj2_prefix}weight", rand(64, 32)),
        (f"{proj2_prefix}bias", rand(64)),
        # Language model
        (f"{llm_prefix}embed_tokens.weight", rand(128, 64)),
        (f"{llm_prefix}layers.0.self_attn.q_norm.weight", rand(16)),
        (f"{llm_prefix}layers.0.self_attn.k_norm.weight", rand(16)),
        (f"{llm_prefix}layers.0.self_attn.q_proj.weight", rand(64, 64)),
        (f"{llm_prefix}layers.0.self_attn.k_proj.weight", rand(32, 64)),
        (f"{llm_prefix}layers.0.self_attn.v_proj.weight", rand(32, 64)),
        (f"{llm_prefix}layers.0.self_attn.o_proj.weight", rand(64, 64)),
        (f"{llm_prefix}layers.0.mlp.gate_proj.weight", rand(128, 64)),
        (f"{llm_prefix}layers.0.mlp.up_proj.weight", rand(128, 64)),
        (f"{llm_prefix}layers.0.mlp.down_proj.weight", rand(64, 128)),
        (f"{llm_prefix}layers.0.input_layernorm.weight", rand(64)),
        (f"{llm_prefix}layers.0.post_attention_layernorm.weight", rand(64)),
        (f"{llm_prefix}norm.weight", rand(64)),
        # Ignored auxiliary / cached rotary tensors
        (f"{llm_prefix}rotary_emb.inv_freq", rand(8)),
        ("talker.unused.weight", rand(4, 4)),
        ("code2wav.unused.weight", rand(4, 4)),
    ]

    if include_lm_head:
        lm_head_weight = rand(128, 64)
        if tie_word_embeddings:
            # Distinct sentinel value to prove tied models skip lm_head.weight.
            lm_head_weight = torch.full((128, 64), -999.0, dtype=torch.float32)
        weights.append((lm_head_name, lm_head_weight))

    return weights


class TestQwen3ASRConfig(CustomTestCase):
    def _write_and_load(self, config_dict: dict, trust_remote_code: bool = False):
        with TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config_dict, f)
            cfg = get_config(tmpdir, trust_remote_code=trust_remote_code)
            mc = ModelConfig(model_path=tmpdir, trust_remote_code=trust_remote_code)
        return cfg, mc

    def test_native_hf_config_qwen3_asr_1_7b_hf(self):
        cfg, mc = self._write_and_load(_qwen3_asr_1_7b_hf_config_dict())
        self.assertIsInstance(cfg, Qwen3ASRConfig)
        self.assertEqual(cfg.audio_config.encoder_layers, 24)
        self.assertEqual(cfg.audio_config.d_model, 1024)
        self.assertEqual(cfg.audio_config.output_dim, 2048)
        self.assertEqual(cfg.audio_config.encoder_attention_heads, 16)
        self.assertEqual(cfg.audio_config.max_source_positions, 1500)
        self.assertIs(cfg.audio_config, cfg.thinker_config.audio_config)

        text_cfg = get_hf_text_config(cfg)
        self.assertIs(text_cfg, cfg.text_config)
        self.assertEqual(text_cfg.num_hidden_layers, 28)
        self.assertEqual(text_cfg.hidden_size, 2048)
        self.assertEqual(text_cfg.num_attention_heads, 16)
        self.assertEqual(text_cfg.num_key_value_heads, 8)
        self.assertEqual(text_cfg.head_dim, 128)
        self.assertEqual(text_cfg.vocab_size, 151936)
        self.assertTrue(text_cfg.tie_word_embeddings)
        self.assertEqual(text_cfg.dtype, torch.bfloat16)
        self.assertEqual(text_cfg.eos_token_id, [151643, 151645])
        self.assertEqual(text_cfg.pad_token_id, 151645)
        self.assertEqual(cfg.audio_token_id, 151676)

        self.assertEqual(mc.hidden_size, 2048)
        self.assertEqual(mc.num_hidden_layers, 28)
        self.assertTrue(mc.is_multimodal)
        self.assertTrue(mc.is_audio_model)
        self.assertTrue(mc.is_audio_understandable_model)
        self.assertFalse(mc.model_is_mrope)
        self.assertEqual(mc.dtype, torch.bfloat16)
        self.assertEqual(mc.hf_eos_token_id, {151643, 151645})

        copied = copy.deepcopy(cfg)
        self.assertEqual(copied.audio_config.d_model, 1024)
        self.assertEqual(copied.text_config.hidden_size, 2048)
        roundtripped = Qwen3ASRConfig.from_dict(cfg.to_dict())
        self.assertEqual(roundtripped.audio_config.encoder_layers, 24)
        self.assertEqual(roundtripped.text_config.num_hidden_layers, 28)
        self.assertTrue(roundtripped.text_config.tie_word_embeddings)

    def test_native_hf_config_oruk_fourier_2b(self):
        for trust_remote_code in (False, True):
            cfg, mc = self._write_and_load(
                _oruk_fourier_2b_config_dict(), trust_remote_code=trust_remote_code
            )
            self.assertIsInstance(cfg, Qwen3ASRConfig)
            self.assertEqual(cfg.audio_config.encoder_layers, 24)
            self.assertEqual(cfg.audio_config.d_model, 1024)
            self.assertEqual(cfg.audio_config.output_dim, 2048)
            self.assertEqual(cfg.text_config.num_hidden_layers, 28)
            self.assertEqual(cfg.text_config.hidden_size, 2048)
            self.assertTrue(cfg.text_config.tie_word_embeddings)
            self.assertEqual(mc.dtype, torch.bfloat16)
            self.assertEqual(mc.hf_eos_token_id, {151643, 151645})
            self.assertTrue(mc.is_audio_understandable_model)

    def test_legacy_thinker_config_backwards_compat(self):
        cfg, mc = self._write_and_load(_legacy_thinker_1_7b_config_dict())
        self.assertIsInstance(cfg, Qwen3ASRConfig)
        self.assertEqual(cfg.thinker_config.audio_config.encoder_layers, 24)
        self.assertEqual(cfg.thinker_config.audio_config.d_model, 1024)
        self.assertEqual(cfg.thinker_config.audio_config.output_dim, 2048)
        self.assertEqual(cfg.thinker_config.audio_start_token_id, 151669)
        self.assertEqual(cfg.thinker_config.audio_end_token_id, 151670)
        self.assertEqual(cfg.audio_token_id, 151676)
        self.assertEqual(cfg.thinker_config.text_config.hidden_size, 2048)
        self.assertTrue(mc.model_is_mrope)
        self.assertEqual(mc.dtype, torch.bfloat16)
        self.assertTrue(mc.is_audio_model)

    def test_top_level_tie_word_embeddings_propagation_does_not_mutate_input(self):
        for tie in (True, False):
            text_dict = _tiny_text_config_dict()
            text_dict.pop("tie_word_embeddings")
            snapshot = dict(text_dict)
            cfg = Qwen3ASRConfig(
                audio_config=_tiny_audio_config_dict(),
                text_config=text_dict,
                tie_word_embeddings=tie,
            )
            self.assertEqual(cfg.text_config.tie_word_embeddings, tie)
            self.assertEqual(text_dict, snapshot)

    def test_default_empty_config_and_autoconfig_registration(self):
        cfg = Qwen3ASRConfig()
        self.assertIsNotNone(cfg.thinker_config)
        self.assertIsNotNone(cfg.audio_config)
        self.assertIsNotNone(cfg.text_config)
        with TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(_oruk_fourier_2b_config_dict(), f)
            loaded = AutoConfig.from_pretrained(tmpdir, trust_remote_code=False)
        self.assertIsInstance(loaded, Qwen3ASRConfig)
        self.assertEqual(loaded.audio_config.d_model, 1024)


class TestQwen3ASRWeightLoading(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cpu"), role="test")

    def _build_model(self, config, *, pp_size: int = 1, pp_rank: int = 0):
        pp_group = SimpleNamespace(
            rank_in_group=pp_rank,
            world_size=pp_size,
            is_first_rank=(pp_rank == 0),
            is_last_rank=(pp_rank == pp_size - 1),
        )
        with parallel_scope(
            pp_size=pp_size,
            pp_rank=pp_rank,
            pp_group=pp_group,
        ):
            model = Qwen3ASRForConditionalGeneration(config)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        return model

    def test_native_hf_tied_weights_match_legacy_thinker_bit_for_bit(self):
        native_cfg = Qwen3ASRConfig(
            audio_config=_tiny_audio_config_dict(),
            text_config=_tiny_text_config_dict(tie_word_embeddings=True),
            tie_word_embeddings=True,
            dtype="float32",
        )
        legacy_cfg = Qwen3ASRConfig(
            thinker_config={
                "audio_config": _tiny_audio_config_dict(),
                "text_config": _tiny_text_config_dict(tie_word_embeddings=True),
                "dtype": "float32",
            }
        )

        native_model = self._build_model(native_cfg)
        legacy_model = self._build_model(legacy_cfg)

        # Native HF tied checkpoints (Qwen3-ASR-1.7B-hf, oruk/oruk-fourier-2b)
        # omit lm_head.weight completely; legacy checkpoints include it.
        native_weights = _build_checkpoint_tensors(
            format_kind="native_hf",
            tie_word_embeddings=True,
            include_lm_head=False,
            seed=123,
        )
        legacy_weights = _build_checkpoint_tensors(
            format_kind="legacy",
            tie_word_embeddings=True,
            include_lm_head=True,
            seed=123,
        )

        native_model.load_weights(native_weights)
        legacy_model.load_weights(legacy_weights)

        native_params = dict(native_model.named_parameters(remove_duplicate=False))
        legacy_params = dict(legacy_model.named_parameters(remove_duplicate=False))
        self.assertEqual(set(native_params.keys()), set(legacy_params.keys()))

        for name, native_param in native_params.items():
            self.assertGreater(
                native_param.abs().sum().item(),
                0.0,
                f"Parameter {name} was not populated by native HF weight loading",
            )
            torch.testing.assert_close(
                native_param, legacy_params[name], rtol=0.0, atol=0.0
            )

        # Verify projector weights landed on audio_tower.proj1 / proj2.
        weights_by_name = dict(native_weights)
        torch.testing.assert_close(
            native_params["audio_tower.proj1.weight"],
            weights_by_name["model.multi_modal_projector.linear_1.weight"],
        )
        torch.testing.assert_close(
            native_params["audio_tower.proj2.bias"],
            weights_by_name["model.multi_modal_projector.linear_2.bias"],
        )
        # Verify tied lm_head shares embed_tokens and was not overwritten by sentinel.
        torch.testing.assert_close(
            native_model.language_model.lm_head.weight,
            weights_by_name["model.language_model.embed_tokens.weight"],
        )
        torch.testing.assert_close(
            legacy_model.language_model.lm_head.weight,
            weights_by_name["model.language_model.embed_tokens.weight"],
        )

    def test_native_hf_and_legacy_untied_lm_head(self):
        for format_kind in ("native_hf", "legacy"):
            if format_kind == "native_hf":
                cfg = Qwen3ASRConfig(
                    audio_config=_tiny_audio_config_dict(),
                    text_config=_tiny_text_config_dict(tie_word_embeddings=False),
                    tie_word_embeddings=False,
                    dtype="float32",
                )
            else:
                cfg = Qwen3ASRConfig(
                    thinker_config={
                        "audio_config": _tiny_audio_config_dict(),
                        "text_config": _tiny_text_config_dict(
                            tie_word_embeddings=False
                        ),
                        "dtype": "float32",
                    }
                )
            model = self._build_model(cfg)
            weights = _build_checkpoint_tensors(
                format_kind=format_kind,
                tie_word_embeddings=False,
                include_lm_head=True,
                seed=77,
            )
            weights_by_name = dict(weights)
            model.load_weights(weights)

            embed_key = (
                "model.language_model.embed_tokens.weight"
                if format_kind == "native_hf"
                else "thinker.model.embed_tokens.weight"
            )
            lm_head_key = (
                "lm_head.weight"
                if format_kind == "native_hf"
                else "thinker.lm_head.weight"
            )
            self.assertIsNot(
                model.language_model.lm_head,
                model.language_model.model.embed_tokens,
            )
            torch.testing.assert_close(
                model.language_model.model.embed_tokens.weight,
                weights_by_name[embed_key],
            )
            torch.testing.assert_close(
                model.language_model.lm_head.weight,
                weights_by_name[lm_head_key],
            )

    def test_native_hf_tied_embeddings_on_pp_last_rank(self):
        text_cfg = _tiny_text_config_dict(tie_word_embeddings=True)
        text_cfg["num_hidden_layers"] = 2
        cfg = Qwen3ASRConfig(
            audio_config=_tiny_audio_config_dict(),
            text_config=text_cfg,
            tie_word_embeddings=True,
            dtype="float32",
        )
        model = self._build_model(cfg, pp_size=2, pp_rank=1)
        weights = _build_checkpoint_tensors(
            format_kind="native_hf",
            tie_word_embeddings=True,
            include_lm_head=False,
            seed=99,
        )
        weights_by_name = dict(weights)
        model.load_weights(weights)

        torch.testing.assert_close(
            model.language_model.lm_head.weight,
            weights_by_name["model.language_model.embed_tokens.weight"],
        )

    def test_upstream_hf_config_without_thinker_config_attribute(self):
        hf_cfg = UpstreamHFQwen3ASRConfig(
            audio_config=_tiny_audio_config_dict(),
            text_config=_tiny_text_config_dict(tie_word_embeddings=True),
            tie_word_embeddings=True,
        )
        self.assertFalse(hasattr(hf_cfg, "thinker_config"))
        self.assertFalse(hasattr(hf_cfg.audio_config, "max_source_positions"))

        # Verify audio encoder handles missing max_source_positions directly.
        encoder = Qwen3OmniMoeAudioEncoder(hf_cfg.audio_config)
        self.assertEqual(encoder.max_source_positions, 1500)

        model = self._build_model(hf_cfg)
        weights = _build_checkpoint_tensors(
            format_kind="native_hf",
            tie_word_embeddings=True,
            include_lm_head=False,
            seed=55,
        )
        model.load_weights(weights)
        for name, param in model.named_parameters(remove_duplicate=False):
            self.assertGreater(
                param.abs().sum().item(),
                0.0,
                f"Parameter {name} stayed zero for upstream HF Qwen3ASRConfig",
            )


class TestQwen3ASRProcessorFromPretrained(CustomTestCase):
    def test_forwards_revision_and_trust_remote_code(self):
        sentinel_fe = SimpleNamespace(sampling_rate=16000)
        sentinel_tok = SimpleNamespace(pad_token_id=151643)
        with (
            mock.patch(
                "sglang.srt.configs.qwen3_asr.AutoFeatureExtractor.from_pretrained",
                return_value=sentinel_fe,
            ) as mock_fe,
            mock.patch(
                "sglang.srt.configs.qwen3_asr.AutoTokenizer.from_pretrained",
                return_value=sentinel_tok,
            ) as mock_tok,
            mock.patch.object(Qwen3ASRProcessor, "__init__", return_value=None) as init,
        ):
            Qwen3ASRProcessor.from_pretrained(
                "oruk/oruk-fourier-2b",
                trust_remote_code=False,
                revision="8c2a8871f1304aa2f634da4fb95e24438c4945b9",
            )
            mock_fe.assert_called_once_with(
                "oruk/oruk-fourier-2b",
                trust_remote_code=False,
                revision="8c2a8871f1304aa2f634da4fb95e24438c4945b9",
            )
            mock_tok.assert_called_once_with(
                "oruk/oruk-fourier-2b",
                trust_remote_code=False,
                revision="8c2a8871f1304aa2f634da4fb95e24438c4945b9",
            )
            init.assert_called_once_with(
                feature_extractor=sentinel_fe, tokenizer=sentinel_tok
            )


if __name__ == "__main__":
    unittest.main()
