"""Unit tests for the sglang.srt.utils.hf_transformers subpackage.

Tests cover the pure utility functions (compat patches, config helpers,
context length, GGUF detection, etc.) that don't require actual model files.
"""

import tempfile
import unittest
from types import SimpleNamespace

from transformers import PretrainedConfig

from sglang.srt.utils.hf_transformers.common import (
    _is_deepseek_ocr2_model,
    _is_deepseek_ocr_model,
    _override_v_head_dim_if_zero,
    _patch_text_config,
    check_gguf_file,
    get_context_length,
    get_hf_text_config,
    get_rope_config,
)
from sglang.srt.utils.hf_transformers.tokenizer import _fix_special_tokens_pattern
from sglang.srt.utils.hf_transformers_patches import normalize_rope_scaling_compat
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


# ---------------------------------------------------------------------------
# normalize_rope_scaling_compat
# ---------------------------------------------------------------------------


class TestNormalizeRopeScalingCompat(unittest.TestCase):
    def test_adds_type_from_rope_type(self):
        cfg = PretrainedConfig()
        cfg.rope_scaling = {"rope_type": "llama3", "factor": 8.0}
        normalize_rope_scaling_compat(cfg)
        self.assertEqual(cfg.rope_scaling["type"], "llama3")

    def test_preserves_existing_type(self):
        cfg = PretrainedConfig()
        cfg.rope_scaling = {"rope_type": "llama3", "type": "custom", "factor": 8.0}
        normalize_rope_scaling_compat(cfg)
        self.assertEqual(cfg.rope_scaling["type"], "custom")

    def test_no_op_when_no_rope_scaling(self):
        cfg = PretrainedConfig()
        normalize_rope_scaling_compat(cfg)
        self.assertIsNone(getattr(cfg, "rope_scaling", None))

    def test_no_op_when_rope_scaling_is_none(self):
        cfg = PretrainedConfig()
        cfg.rope_scaling = None
        normalize_rope_scaling_compat(cfg)
        self.assertIsNone(cfg.rope_scaling)

    def test_recurses_into_text_config(self):
        text_cfg = PretrainedConfig()
        text_cfg.rope_scaling = {"rope_type": "yarn", "factor": 4.0}
        cfg = PretrainedConfig()
        cfg.text_config = text_cfg
        normalize_rope_scaling_compat(cfg)
        self.assertEqual(text_cfg.rope_scaling["type"], "yarn")

    def test_recurses_into_llm_config(self):
        llm_cfg = PretrainedConfig()
        llm_cfg.rope_scaling = {"rope_type": "dynamic", "factor": 2.0}
        cfg = PretrainedConfig()
        cfg.llm_config = llm_cfg
        normalize_rope_scaling_compat(cfg)
        self.assertEqual(llm_cfg.rope_scaling["type"], "dynamic")

    def test_no_crash_on_non_dict_rope_scaling(self):
        cfg = PretrainedConfig()
        cfg.rope_scaling = "not_a_dict"
        normalize_rope_scaling_compat(cfg)
        self.assertEqual(cfg.rope_scaling, "not_a_dict")

    def test_no_crash_on_dict_without_rope_type(self):
        cfg = PretrainedConfig()
        cfg.rope_scaling = {"factor": 4.0}
        normalize_rope_scaling_compat(cfg)
        self.assertNotIn("type", cfg.rope_scaling)


# ---------------------------------------------------------------------------
# get_rope_config
# ---------------------------------------------------------------------------


class TestGetRopeConfig(unittest.TestCase):
    def test_v5_rope_parameters(self):
        cfg = PretrainedConfig()
        cfg.rope_parameters = {"rope_theta": 10000.0, "rope_type": "default"}
        theta, params = get_rope_config(cfg)
        self.assertEqual(theta, 10000.0)
        self.assertIs(params, cfg.rope_parameters)

    def test_v4_fallback_remote_code_config(self):
        # Remote-code configs (SimpleNamespace) lack the v5 rope_parameters property
        cfg = SimpleNamespace(
            rope_theta=500000.0,
            rope_scaling={"type": "llama3", "factor": 8.0},
        )
        theta, params = get_rope_config(cfg)
        self.assertEqual(theta, 500000.0)
        self.assertEqual(params, {"type": "llama3", "factor": 8.0})

    def test_v4_no_scaling(self):
        cfg = SimpleNamespace(rope_theta=10000.0)
        theta, params = get_rope_config(cfg)
        self.assertEqual(theta, 10000.0)
        self.assertIsNone(params)


# ---------------------------------------------------------------------------
# _patch_text_config
# ---------------------------------------------------------------------------


class TestPatchTextConfig(unittest.TestCase):
    def test_propagates_parent_to_text(self):
        parent = PretrainedConfig()
        parent.pad_token_id = 0
        parent.bos_token_id = 1
        parent.eos_token_id = 2
        parent.tie_word_embeddings = False

        text = PretrainedConfig()
        text.num_attention_heads = 32

        result = _patch_text_config(parent, text)
        self.assertEqual(result.pad_token_id, 0)
        self.assertEqual(result.bos_token_id, 1)
        self.assertEqual(result.eos_token_id, 2)
        self.assertIs(result, text)

    def test_propagates_text_to_parent(self):
        parent = PretrainedConfig()
        text = PretrainedConfig()
        text.pad_token_id = 42

        _patch_text_config(parent, text)
        self.assertEqual(parent.pad_token_id, 42)

    def test_no_overwrite_when_both_have_attr(self):
        parent = PretrainedConfig()
        parent.pad_token_id = 0
        text = PretrainedConfig()
        text.pad_token_id = 99

        _patch_text_config(parent, text)
        self.assertEqual(parent.pad_token_id, 0)
        self.assertEqual(text.pad_token_id, 99)


# ---------------------------------------------------------------------------
# get_context_length
# ---------------------------------------------------------------------------


class TestGetContextLength(unittest.TestCase):
    def test_max_position_embeddings(self):
        cfg = PretrainedConfig()
        cfg.max_position_embeddings = 4096
        self.assertEqual(get_context_length(cfg), 4096)

    def test_max_sequence_length_takes_priority(self):
        cfg = PretrainedConfig()
        cfg.max_sequence_length = 8192
        cfg.max_position_embeddings = 4096
        self.assertEqual(get_context_length(cfg), 8192)

    def test_rope_scaling_factor(self):
        cfg = PretrainedConfig()
        cfg.max_position_embeddings = 4096
        cfg.rope_scaling = {"factor": 4.0}
        self.assertEqual(get_context_length(cfg), 16384)

    def test_rope_scaling_llama3_ignores_factor(self):
        cfg = PretrainedConfig()
        cfg.max_position_embeddings = 131072
        cfg.rope_scaling = {"rope_type": "llama3", "factor": 8.0}
        self.assertEqual(get_context_length(cfg), 131072)

    def test_original_max_position_embeddings_ignores_factor(self):
        cfg = PretrainedConfig()
        cfg.max_position_embeddings = 131072
        cfg.rope_scaling = {
            "factor": 8.0,
            "original_max_position_embeddings": 8192,
        }
        self.assertEqual(get_context_length(cfg), 131072)

    def test_default_when_no_keys(self):
        cfg = PretrainedConfig()
        self.assertEqual(get_context_length(cfg), 2048)


# ---------------------------------------------------------------------------
# check_gguf_file
# ---------------------------------------------------------------------------


class TestCheckGgufFile(unittest.TestCase):
    def test_gguf_suffix(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as f:
            self.assertTrue(check_gguf_file(f.name))

    def test_gguf_magic_header(self):
        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            f.write(b"GGUF" + b"\x00" * 100)
            f.flush()
            self.assertTrue(check_gguf_file(f.name))

    def test_non_gguf_file(self):
        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            f.write(b"NOT_GGUF" + b"\x00" * 100)
            f.flush()
            self.assertFalse(check_gguf_file(f.name))

    def test_nonexistent_file(self):
        self.assertFalse(check_gguf_file("/nonexistent/path/model.bin"))

    def test_directory(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(check_gguf_file(d))


# ---------------------------------------------------------------------------
# _is_deepseek_ocr_model / _is_deepseek_ocr2_model
# ---------------------------------------------------------------------------


class TestDeepseekOcrDetection(unittest.TestCase):
    def test_ocr_model_detected(self):
        cfg = PretrainedConfig()
        cfg.auto_map = {"AutoModel": "modeling_deepseekocr.DeepseekOCRForCausalLM"}
        self.assertTrue(_is_deepseek_ocr_model(cfg))

    def test_ocr2_model_detected(self):
        cfg = PretrainedConfig()
        cfg.auto_map = {"AutoModel": "modeling_deepseekocr2.DeepseekOCR2ForCausalLM"}
        self.assertTrue(_is_deepseek_ocr2_model(cfg))

    def test_non_ocr_model(self):
        cfg = PretrainedConfig()
        cfg.auto_map = {"AutoModel": "modeling_llama.LlamaForCausalLM"}
        self.assertFalse(_is_deepseek_ocr_model(cfg))
        self.assertFalse(_is_deepseek_ocr2_model(cfg))

    def test_no_auto_map(self):
        cfg = PretrainedConfig()
        self.assertFalse(_is_deepseek_ocr_model(cfg))
        self.assertFalse(_is_deepseek_ocr2_model(cfg))

    def test_empty_auto_map(self):
        cfg = PretrainedConfig()
        cfg.auto_map = {}
        self.assertFalse(_is_deepseek_ocr_model(cfg))
        self.assertFalse(_is_deepseek_ocr2_model(cfg))


# ---------------------------------------------------------------------------
# _override_v_head_dim_if_zero
# ---------------------------------------------------------------------------


class TestOverrideVHeadDimIfZero(unittest.TestCase):
    def test_patches_zero_v_head_dim(self):
        text_cfg = SimpleNamespace(v_head_dim=0)
        cfg = PretrainedConfig()
        cfg.text_config = text_cfg
        _override_v_head_dim_if_zero(cfg)
        self.assertEqual(text_cfg.v_head_dim, 128)

    def test_custom_patch_value(self):
        text_cfg = SimpleNamespace(v_head_dim=0)
        cfg = PretrainedConfig()
        cfg.text_config = text_cfg
        _override_v_head_dim_if_zero(cfg, patch=64)
        self.assertEqual(text_cfg.v_head_dim, 64)

    def test_no_patch_when_nonzero(self):
        text_cfg = SimpleNamespace(v_head_dim=256)
        cfg = PretrainedConfig()
        cfg.text_config = text_cfg
        _override_v_head_dim_if_zero(cfg)
        self.assertEqual(text_cfg.v_head_dim, 256)

    def test_dict_sub_config(self):
        cfg = PretrainedConfig()
        cfg.text_config = {"v_head_dim": 0}
        _override_v_head_dim_if_zero(cfg)
        self.assertEqual(cfg.text_config["v_head_dim"], 128)

    def test_no_sub_config(self):
        cfg = PretrainedConfig()
        _override_v_head_dim_if_zero(cfg)  # should not raise


# ---------------------------------------------------------------------------
# get_hf_text_config
# ---------------------------------------------------------------------------


class TestGetHfTextConfig(unittest.TestCase):
    def test_returns_config_for_pure_text_model(self):
        cfg = PretrainedConfig()
        cfg.architectures = ["LlamaForCausalLM"]
        result = get_hf_text_config(cfg)
        self.assertIs(result, cfg)

    def test_returns_text_config_for_multimodal(self):
        text_cfg = PretrainedConfig()
        text_cfg.num_attention_heads = 32
        cfg = PretrainedConfig()
        cfg.architectures = ["SomeVLMForCausalLM"]
        cfg.text_config = text_cfg
        result = get_hf_text_config(cfg)
        self.assertIs(result, text_cfg)

    def test_llm_config_priority_over_text_config(self):
        llm_cfg = PretrainedConfig()
        llm_cfg.num_attention_heads = 16
        text_cfg = PretrainedConfig()
        text_cfg.num_attention_heads = 32
        cfg = PretrainedConfig()
        cfg.architectures = ["SomeModel"]
        cfg.llm_config = llm_cfg
        cfg.text_config = text_cfg
        result = get_hf_text_config(cfg)
        self.assertIs(result, llm_cfg)

    def test_thinker_config_highest_priority(self):
        thinker_cfg = PretrainedConfig()
        thinker_cfg.num_attention_heads = 8
        cfg = PretrainedConfig()
        cfg.architectures = ["SomeModel"]
        cfg.thinker_config = thinker_cfg
        result = get_hf_text_config(cfg)
        self.assertIs(result, thinker_cfg)

    def test_thinker_config_with_text_sub_config(self):
        inner_text = PretrainedConfig()
        inner_text.num_attention_heads = 8
        thinker_cfg = PretrainedConfig()
        thinker_cfg.text_config = inner_text
        thinker_cfg.torch_dtype = "float16"
        cfg = PretrainedConfig()
        cfg.architectures = ["Qwen2OmniModel"]
        cfg.thinker_config = thinker_cfg
        result = get_hf_text_config(cfg)
        self.assertIs(result, inner_text)
        self.assertEqual(inner_text.torch_dtype, "float16")

    def test_converts_dict_sub_config(self):
        cfg = PretrainedConfig()
        cfg.architectures = ["SomeModel"]
        cfg.text_config = {
            "num_attention_heads": 32,
            "hidden_size": 4096,
        }
        result = get_hf_text_config(cfg)
        self.assertIsInstance(cfg.text_config, PretrainedConfig)
        self.assertEqual(result.num_attention_heads, 32)

    def test_llava_returns_parent_config(self):
        cfg = PretrainedConfig()
        cfg.architectures = ["LlavaForCausalLM"]
        text_cfg = PretrainedConfig()
        text_cfg.num_attention_heads = 32
        cfg.text_config = text_cfg
        result = get_hf_text_config(cfg)
        self.assertIs(result, cfg)

    def test_calls_normalize_rope_scaling(self):
        cfg = PretrainedConfig()
        cfg.architectures = ["LlamaForCausalLM"]
        cfg.rope_scaling = {"rope_type": "llama3", "factor": 8.0}
        get_hf_text_config(cfg)
        self.assertIn("type", cfg.rope_scaling)
        self.assertEqual(cfg.rope_scaling["type"], "llama3")


# ---------------------------------------------------------------------------
# _fix_special_tokens_pattern
# ---------------------------------------------------------------------------


class TestFixSpecialTokensPattern(unittest.TestCase):
    def test_fixes_cls_sep_with_missing_tokens(self):
        tok = SimpleNamespace(
            special_tokens_pattern="cls_sep",
            cls_token_id=None,
            sep_token_id=None,
        )
        _fix_special_tokens_pattern(tok)
        self.assertEqual(tok.special_tokens_pattern, "none")

    def test_no_change_when_tokens_present(self):
        tok = SimpleNamespace(
            special_tokens_pattern="cls_sep",
            cls_token_id=101,
            sep_token_id=102,
        )
        _fix_special_tokens_pattern(tok)
        self.assertEqual(tok.special_tokens_pattern, "cls_sep")

    def test_no_change_for_other_patterns(self):
        tok = SimpleNamespace(
            special_tokens_pattern="none",
            cls_token_id=None,
            sep_token_id=None,
        )
        _fix_special_tokens_pattern(tok)
        self.assertEqual(tok.special_tokens_pattern, "none")

    def test_no_change_when_no_pattern(self):
        tok = SimpleNamespace(cls_token_id=None, sep_token_id=None)
        _fix_special_tokens_pattern(tok)
        self.assertFalse(hasattr(tok, "special_tokens_pattern"))


# ---------------------------------------------------------------------------
# __init__.py re-exports
# ---------------------------------------------------------------------------


class TestModuleReExports(unittest.TestCase):
    def test_all_public_symbols_importable(self):
        import sglang.srt.utils.hf_transformers as pkg

        for name in pkg.__all__:
            self.assertTrue(
                hasattr(pkg, name),
                f"{name} listed in __all__ but not importable from package",
            )

    def test_shim_module_exports_match(self):
        import sglang.srt.utils.hf_transformers as pkg
        import sglang.srt.utils.hf_transformers_utils as shim

        for name in pkg.__all__:
            self.assertTrue(
                hasattr(shim, name),
                f"{name} not available through shim module hf_transformers_utils",
            )


# ---------------------------------------------------------------------------
# compat: _patch_removed_symbols
# ---------------------------------------------------------------------------


class TestPatchRemovedSymbols(unittest.TestCase):
    def test_llama_flash_attention2_exists(self):
        from transformers.models.llama import modeling_llama

        self.assertTrue(
            hasattr(modeling_llama, "LlamaFlashAttention2"),
            "LlamaFlashAttention2 should be patched onto modeling_llama",
        )

    def test_is_flash_attn_greater_or_equal_2_10_callable(self):
        import transformers.utils as _u

        self.assertTrue(
            hasattr(_u, "is_flash_attn_greater_or_equal_2_10"),
            "is_flash_attn_greater_or_equal_2_10 should be patched onto transformers.utils",
        )
        self.assertIsInstance(_u.is_flash_attn_greater_or_equal_2_10(), bool)


# ---------------------------------------------------------------------------
# compat: _patch_rope_parameters_validation
# ---------------------------------------------------------------------------


class TestPatchRopeParametersValidation(unittest.TestCase):
    def test_injects_rope_theta_into_rope_scaling(self):
        config_dict = {
            "model_type": "llama",
            "rope_theta": 500000.0,
            "max_position_embeddings": 131072,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        }
        config = PretrainedConfig.from_dict(config_dict)
        rope_params = getattr(config, "rope_parameters", None)
        if rope_params is not None:
            self.assertIn("rope_theta", rope_params)

    def test_no_injection_when_rope_theta_already_in_scaling(self):
        config_dict = {
            "model_type": "llama",
            "rope_theta": 500000.0,
            "max_position_embeddings": 131072,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 8.0,
                "rope_theta": 999.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        }
        config = PretrainedConfig.from_dict(config_dict)
        rope_params = getattr(config, "rope_parameters", None)
        if rope_params is not None:
            self.assertEqual(rope_params["rope_theta"], 999.0)

    def test_no_crash_without_rope_scaling(self):
        config_dict = {"model_type": "llama", "rope_theta": 10000.0}
        config = PretrainedConfig.from_dict(config_dict)
        self.assertIsNotNone(config)


# ---------------------------------------------------------------------------
# compat: _ensure_clean_up_tokenization_compat
# ---------------------------------------------------------------------------


class TestCleanUpTokenizationCompat(unittest.TestCase):
    def test_clean_up_tokenization_exists(self):
        from transformers import PreTrainedTokenizerBase

        self.assertTrue(hasattr(PreTrainedTokenizerBase, "clean_up_tokenization"))

    def test_clean_up_tokenization_callable(self):
        from transformers import PreTrainedTokenizerBase

        self.assertTrue(callable(PreTrainedTokenizerBase.clean_up_tokenization))


# ---------------------------------------------------------------------------
# compat: _ensure_is_torch_fx_available_compat
# ---------------------------------------------------------------------------


class TestIsTorchFxAvailableCompat(unittest.TestCase):
    def test_is_torch_fx_available_exists(self):
        import transformers.utils.import_utils as _iu

        self.assertTrue(hasattr(_iu, "is_torch_fx_available"))
        self.assertTrue(_iu.is_torch_fx_available())


# ---------------------------------------------------------------------------
# compat: _patch_nemotron_h_pattern
# ---------------------------------------------------------------------------


class TestPatchNemotronHPattern(unittest.TestCase):
    def test_pattern_to_list_skips_mlp_dash(self):
        try:
            from transformers.models.nemotron_h.configuration_nemotron_h import (
                NemotronHConfig,
            )

            result = NemotronHConfig._pattern_to_list("M-*-")
            self.assertEqual(result, ["mamba", "attention"])
        except ImportError:
            self.skipTest("NemotronHConfig not available in this transformers version")

    def test_pattern_to_list_standard_chars(self):
        try:
            from transformers.models.nemotron_h.configuration_nemotron_h import (
                NemotronHConfig,
            )

            result = NemotronHConfig._pattern_to_list("ME*")
            self.assertEqual(result, ["mamba", "moe", "attention"])
        except ImportError:
            self.skipTest("NemotronHConfig not available in this transformers version")


if __name__ == "__main__":
    unittest.main()
