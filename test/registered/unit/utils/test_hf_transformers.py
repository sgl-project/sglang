"""Unit tests for the sglang.srt.utils.hf_transformers subpackage.

Tests cover the pure utility functions (compat patches, config helpers,
context length, GGUF detection, etc.) that don't require actual model files.
"""

import inspect
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from transformers import PretrainedConfig
from transformers.image_processing_utils import BaseImageProcessor

import sglang.srt.utils.hf_transformers.processor as processor_utils
from sglang.srt.utils import hf_transformers_patches
from sglang.srt.utils.hf_transformers.common import (
    _is_deepseek_ocr2_model,
    _is_deepseek_ocr_model,
    _override_v_head_dim_if_zero,
    _patch_text_config,
    attach_additional_stop_token_ids,
    check_gguf_file,
    get_context_length,
    get_hf_text_config,
    get_rope_config,
    resolve_hf_gguf_reference,
)
from sglang.srt.utils.hf_transformers.tokenizer import _fix_special_tokens_pattern
from sglang.srt.utils.hf_transformers_patches import normalize_rope_scaling_compat
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# get_processor
# ---------------------------------------------------------------------------


class TestGetProcessor(unittest.TestCase):
    def test_does_not_forward_backend_to_auto_processor(self):
        config = SimpleNamespace(model_type="test_vlm", auto_map={})
        loaded_processor = MagicMock()
        loaded_processor.image_processor.backend = "torchvision"
        loaded_processor.tokenizer.chat_template = "template"
        auto_config = MagicMock()
        auto_config.from_pretrained.return_value = config
        auto_processor = MagicMock()
        auto_processor.from_pretrained.return_value = loaded_processor
        auto_image_processor = MagicMock()

        with patch.multiple(
            processor_utils,
            AutoConfig=auto_config,
            AutoProcessor=auto_processor,
            AutoImageProcessor=auto_image_processor,
        ):
            processor_utils.get_processor(
                "test-model", image_processor_backend="torchvision"
            )

        call_kwargs = auto_processor.from_pretrained.call_args.kwargs
        self.assertNotIn("backend", call_kwargs)
        self.assertNotIn("use_fast", call_kwargs)
        auto_image_processor.from_pretrained.assert_not_called()

    def test_applies_pil_backend_only_to_image_processor(self):
        config = SimpleNamespace(model_type="test_vlm", auto_map={})

        for processor_kwargs in (
            {"image_processor_backend": "pil"},
            {"use_fast": False},
        ):
            with self.subTest(processor_kwargs=processor_kwargs):
                loaded_processor = MagicMock()
                loaded_processor.image_processor.backend = "torchvision"
                loaded_processor.tokenizer.chat_template = "template"
                pil_processor = MagicMock(backend="pil")
                auto_config = MagicMock()
                auto_config.from_pretrained.return_value = config
                auto_processor = MagicMock()
                auto_processor.from_pretrained.return_value = loaded_processor
                auto_image_processor = MagicMock()
                auto_image_processor.from_pretrained.return_value = pil_processor

                with patch.multiple(
                    processor_utils,
                    AutoConfig=auto_config,
                    AutoProcessor=auto_processor,
                    AutoImageProcessor=auto_image_processor,
                ):
                    processor = processor_utils.get_processor(
                        "test-model", **processor_kwargs
                    )

                call_kwargs = auto_processor.from_pretrained.call_args.kwargs
                self.assertNotIn("backend", call_kwargs)
                self.assertNotIn("use_fast", call_kwargs)
                auto_image_processor.from_pretrained.assert_called_once_with(
                    "test-model",
                    trust_remote_code=False,
                    revision=None,
                    backend="pil",
                )
                self.assertIs(processor.image_processor, pil_processor)

    def test_resolves_model_name_before_loading_config(self):
        remote_model = "s3://bucket/model"
        local_model = "/cache/model"
        config = SimpleNamespace(model_type="clip", auto_map={})
        loaded_processor = MagicMock()
        loaded_processor.tokenizer.chat_template = "template"
        auto_config = MagicMock()
        auto_config.from_pretrained.return_value = config
        auto_processor = MagicMock()
        auto_processor.from_pretrained.return_value = loaded_processor

        def resolve_uri(path):
            return local_model if path == remote_model else path

        with patch.multiple(
            processor_utils,
            resolve_runai_obj_uri=MagicMock(side_effect=resolve_uri),
            AutoConfig=auto_config,
            AutoProcessor=auto_processor,
        ):
            processor = processor_utils.get_processor(
                "local-tokenizer",
                model_name=remote_model,
            )

        self.assertIs(processor, loaded_processor)
        auto_config.from_pretrained.assert_called_once_with(
            local_model,
            trust_remote_code=False,
            revision=None,
        )


# ---------------------------------------------------------------------------
# _patch_image_processor_kwargs
# ---------------------------------------------------------------------------


class TestImageProcessorKwargsPatch(unittest.TestCase):
    def test_filters_unsupported_kwargs_and_caches_signature(self):
        class StrictImageProcessor(BaseImageProcessor):
            model_input_names = ["pixel_values"]

            def preprocess(self, images, accepted=None):
                return {"images": images, "accepted": accepted}

        processor = StrictImageProcessor()
        with patch.object(
            hf_transformers_patches.inspect,
            "signature",
            wraps=inspect.signature,
        ) as signature:
            first = processor("first", accepted=True, device="cuda")
            second = processor("second", accepted=False, device="cuda")

        self.assertEqual(first, {"images": "first", "accepted": True})
        self.assertEqual(second, {"images": "second", "accepted": False})
        self.assertEqual(signature.call_count, 1)


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


class TestResolveHfGgufReference(unittest.TestCase):
    @patch("huggingface_hub.hf_hub_download", return_value="/cache/model-Q4_K.gguf")
    @patch("huggingface_hub.HfApi")
    def test_resolves_quant_type(self, api_cls, download):
        api_cls.return_value.repo_info.return_value.siblings = [
            SimpleNamespace(rfilename="model-Q4_K.gguf"),
            SimpleNamespace(rfilename="model-Q8_0.gguf"),
        ]

        resolved = resolve_hf_gguf_reference("owner/repo:Q4_K", revision="revision")

        self.assertEqual(resolved, "/cache/model-Q4_K.gguf")
        download.assert_called_once_with(
            "owner/repo", "model-Q4_K.gguf", revision="revision"
        )

    @patch("huggingface_hub.HfApi")
    def test_rejects_ambiguous_quant_type(self, api_cls):
        api_cls.return_value.repo_info.return_value.siblings = [
            SimpleNamespace(rfilename="fl2va-Q4_K.gguf"),
            SimpleNamespace(rfilename="ref2va-Q4_K.gguf"),
        ]

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            resolve_hf_gguf_reference("owner/repo:Q4_K")

    @patch("huggingface_hub.HfApi")
    def test_reports_available_files_when_quant_type_is_missing(self, api_cls):
        api_cls.return_value.repo_info.return_value.siblings = [
            SimpleNamespace(rfilename="model-Q4_K.gguf"),
            SimpleNamespace(rfilename="README.md"),
        ]

        with self.assertRaisesRegex(ValueError, "model-Q4_K.gguf"):
            resolve_hf_gguf_reference("owner/repo:Q8_0")


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
# attach_additional_stop_token_ids
# ---------------------------------------------------------------------------


class TestAttachAdditionalStopTokenIds(unittest.TestCase):
    """Bug regression: the Inkling bundle ships eos metadata unset while its
    turn-final marker <|content_model_end_sampling|> sits in added_tokens; the
    old detector only recognized <|eom_id|>, so generation ran to max length
    (documented by the Inkling GSM8K test)."""

    @staticmethod
    def _tokenizer(added):
        return SimpleNamespace(get_added_vocab=lambda: added)

    def test_inkling_end_sampling_registers_as_stop(self):
        tok = self._tokenizer({"<|content_model_end_sampling|>": 200006})
        attach_additional_stop_token_ids(tok)
        self.assertEqual(tok.additional_stop_token_ids, {200006})

    def test_eom_id_still_registers_as_stop(self):
        tok = self._tokenizer({"<|eom_id|>": 128008})
        attach_additional_stop_token_ids(tok)
        self.assertEqual(tok.additional_stop_token_ids, {128008})

    def test_k2_horizon_im_end_registers_as_stop(self):
        tok = self._tokenizer({"<|ifm|im_end|>": 64019})
        attach_additional_stop_token_ids(tok)
        self.assertEqual(tok.additional_stop_token_ids, {64019})

    def test_no_known_marker_yields_none(self):
        tok = self._tokenizer({"<|other|>": 7})
        attach_additional_stop_token_ids(tok)
        self.assertIsNone(tok.additional_stop_token_ids)


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


# ---------------------------------------------------------------------------
# compat: _patch_rope_parameters_validation
# ---------------------------------------------------------------------------


class TestRopeParametersValidationPatch(unittest.TestCase):
    """A config without `max_position_embeddings` must still get its
    `default_rope_type` resolved, and must still not raise."""

    def test_default_rope_type_survives_a_missing_max_position_embeddings(self):
        class _AxialConfig(PretrainedConfig):
            default_rope_type = "axial"

        config = _AxialConfig(rope_theta=10000.0)
        self.assertFalse(hasattr(config, "max_position_embeddings"))
        config.standardize_rope_params()

        self.assertEqual(config.rope_parameters["rope_type"], "axial")

    def test_scaling_rope_type_without_max_position_embeddings_does_not_raise(self):
        """The original guard's purpose: no `AttributeError` out of __post_init__."""

        class _ScalingConfig(PretrainedConfig):
            pass

        config = _ScalingConfig(
            rope_parameters={"rope_type": "yarn", "rope_theta": 10000.0}
        )
        self.assertFalse(hasattr(config, "max_position_embeddings"))
        config.standardize_rope_params()  # must not raise


# ---------------------------------------------------------------------------
# compat: _ensure_is_torch_fx_available_compat
# ---------------------------------------------------------------------------


class TestIsTorchFxAvailableCompat(unittest.TestCase):
    def test_is_torch_fx_available_exists(self):
        import transformers.utils.import_utils as _iu

        self.assertTrue(hasattr(_iu, "is_torch_fx_available"))
        self.assertTrue(_iu.is_torch_fx_available())


# ---------------------------------------------------------------------------
# AutoConfig registration
# ---------------------------------------------------------------------------


# `inkling_mm_model` predates the invariant: `sglang.srt.configs.inkling` writes
# `InklingMMConfig` straight into `_extra_content`, over a native `InklingConfig`.
_KNOWN_NAME_MISMATCHES = {"inkling_mm_model"}


class TestAutoConfigRegistration(unittest.TestCase):
    """`AutoConfig` must keep resolving to a class the Auto* mappings can key on.

    `_LazyAutoMapping` looks its entries up by config class `__name__`, so a
    SGLang class that shadows a native one under a different name silently
    disappears from PROCESSOR_MAPPING / TOKENIZER_MAPPING / MODEL_MAPPING.
    """

    def test_shadowing_entries_keep_the_native_class_name(self):
        from transformers.models.auto.configuration_auto import (
            CONFIG_MAPPING,
            CONFIG_MAPPING_NAMES,
        )

        import sglang.srt.configs  # noqa: F401  (populates the registrations)

        # `_extra_content` is exactly the set of classes SGLang registered, so
        # this covers `_CONFIG_REGISTRY` and the standalone registrations alike.
        for model_type, cls in CONFIG_MAPPING._extra_content.items():
            native_name = CONFIG_MAPPING_NAMES.get(model_type)
            if native_name is None or model_type in _KNOWN_NAME_MISMATCHES:
                continue
            with self.subTest(model_type=model_type):
                self.assertEqual(
                    cls.__name__,
                    native_name,
                    f"{cls.__name__} shadows the native {native_name} for "
                    f"'{model_type}'; the Auto* mappings key on __name__ and "
                    f"would stop resolving this model type",
                )

    def test_autoconfig_only_registrations_win_over_native(self):
        """zaya / cosmos3-edge have no `_CONFIG_REGISTRY` re-parse to fall back on."""
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        from sglang.srt.configs.cosmos3 import Cosmos3EdgeConfig
        from sglang.srt.configs.zaya import ZayaConfig

        for model_type, expected in (
            ("zaya", ZayaConfig),
            ("cosmos3_edge", Cosmos3EdgeConfig),
        ):
            with self.subTest(model_type=model_type):
                self.assertIs(CONFIG_MAPPING[model_type], expected)


# ---------------------------------------------------------------------------
# Pixtral vision rope
# ---------------------------------------------------------------------------


class TestPixtralVisionRope(unittest.TestCase):
    """The Pixtral tower takes its rope table from transformers, so a change to
    the axial recomposition would rotate every patch by the wrong angle with
    the shapes and the import both still intact."""

    def test_rope_table_matches_the_axial_closed_form(self):
        import torch
        from transformers import PixtralVisionConfig
        from transformers.models.pixtral.modeling_pixtral import (
            PixtralVisionRotaryEmbedding,
        )

        from sglang.srt.models.pixtral import position_meshgrid

        config = PixtralVisionConfig(
            hidden_size=64, num_attention_heads=4, image_size=32, patch_size=8
        )
        dim = config.head_dim
        max_side = config.image_size // config.patch_size
        base = config.rope_parameters["rope_theta"]

        # Separate H and W frequency ladders over the full grid, indexed by the
        # flattened patch offset, then duplicated for rotate_half.
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        freqs_h = torch.outer(torch.arange(max_side), freqs[::2]).float()
        freqs_w = torch.outer(torch.arange(max_side), freqs[1::2]).float()
        table = torch.cat(
            [
                freqs_h[:, None, :].repeat(1, max_side, 1),
                freqs_w[None, :, :].repeat(max_side, 1, 1),
            ],
            dim=-1,
        ).reshape(-1, dim // 2)
        table = torch.cat((table, table), dim=-1)

        # Two images of different aspect ratios, as the tower batches them.
        grids = [torch.empty(1, 3, 4), torch.empty(1, 2, 2)]
        position_ids = position_meshgrid(grids)
        flat = position_ids[:, 0] * max_side + position_ids[:, 1]
        expected = table[flat]

        cos, sin = PixtralVisionRotaryEmbedding(config)(
            torch.zeros(position_ids.shape[0], config.hidden_size), position_ids
        )

        torch.testing.assert_close(cos, expected.cos(), rtol=0, atol=1e-6)
        torch.testing.assert_close(sin, expected.sin(), rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
