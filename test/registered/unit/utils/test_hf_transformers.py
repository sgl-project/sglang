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
)
from sglang.srt.utils.hf_transformers.tokenizer import _fix_special_tokens_pattern
from sglang.srt.utils.hf_transformers_patches import (
    normalize_deepseek_v4_compat,
    normalize_rope_scaling_compat,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# get_processor
# ---------------------------------------------------------------------------


class TestGetProcessor(unittest.TestCase):
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
# normalize_deepseek_v4_compat (issue #34092)
# ---------------------------------------------------------------------------


class TestNormalizeDeepseekV4Compat(unittest.TestCase):
    """Guard #34092: transformers >= 4.57 renamed the DeepSeek V4 field *and*
    reshaped it — ``compress_ratios: list[int]`` became
    ``compress_rates: dict[str, int]`` paired with
    ``layer_types: list[str]``. sglang readers still index ``compress_ratios``
    by layer id (e.g. ``config.compress_ratios[layer_id]`` in
    ``models/deepseek_v4.py``), so the loader must rebuild a per-layer list
    from the two new fields. Upstream encodes sliding-attention layers as
    ``0`` in the legacy list; ``compress_rates`` does not carry an entry for
    them, so the rebuild has to special-case that layer type instead of
    indexing the dict blindly."""

    _LT_CSA = "compressed_sparse_attention"
    _LT_HCA = "heavily_compressed_attention"
    _LT_SWA = "sliding_attention"

    def _make_new_transformers_config(self, layer_types):
        cfg = PretrainedConfig()
        cfg.model_type = "deepseek_v4"
        cfg.compress_rates = {self._LT_CSA: 4, self._LT_HCA: 128}
        cfg.layer_types = layer_types
        return cfg

    def test_expands_compress_rates_dict_by_layer_types(self):
        cfg = self._make_new_transformers_config([self._LT_HCA, self._LT_CSA, self._LT_HCA])
        normalize_deepseek_v4_compat(cfg)
        # Legacy shape: list[int] indexable by layer id, values resolved via
        # ``compress_rates`` for compressed types.
        self.assertEqual(cfg.compress_ratios, [128, 4, 128])
        self.assertEqual(cfg.compress_ratios[0], 128)  # ``config.compress_ratios[layer_id]`` works

    def test_sliding_attention_layers_emit_legacy_zero(self):
        # Real DeepSeek V4 schedules interleave sliding-window layers with
        # compressed ones. ``compress_rates`` has no entry for
        # ``sliding_attention`` — the rebuild must emit ``0`` for those
        # slots (upstream's legacy encoding) rather than raising.
        cfg = self._make_new_transformers_config(
            [self._LT_SWA, self._LT_HCA, self._LT_SWA, self._LT_CSA]
        )
        normalize_deepseek_v4_compat(cfg)
        self.assertEqual(cfg.compress_ratios, [0, 128, 0, 4])

    def test_preserves_existing_legacy_list(self):
        # Older-transformers config already carries the legacy list; the
        # helper must not clobber it.
        cfg = PretrainedConfig()
        cfg.model_type = "deepseek_v4"
        cfg.compress_ratios = [0, 128, 4]
        normalize_deepseek_v4_compat(cfg)
        self.assertEqual(cfg.compress_ratios, [0, 128, 4])

    def test_treats_none_valued_compress_ratios_as_absent(self):
        # A config may carry ``compress_ratios = None`` (attribute exists but
        # unset). A plain ``hasattr`` check would short-circuit and hand
        # downstream a ``None`` — ``for r in None`` / ``None[layer_id]`` both
        # explode later. The rebuild must fire in this case just like when
        # the attribute is missing entirely. Same class of bug that vLLM's
        # harden pass caught in vllm-project/vllm#43443.
        cfg = self._make_new_transformers_config([self._LT_HCA, self._LT_CSA])
        cfg.compress_ratios = None
        normalize_deepseek_v4_compat(cfg)
        self.assertEqual(cfg.compress_ratios, [128, 4])

    def test_no_op_for_non_deepseek_v4_model_type(self):
        # Negative-branch contract: unrelated model types with coincidentally
        # named ``compress_rates`` / ``layer_types`` attributes must not be
        # touched. A regression that drops the model_type gate would silently
        # affect every model whose HF config happens to carry them.
        cfg = PretrainedConfig()
        cfg.model_type = "llama"
        cfg.compress_rates = {self._LT_CSA: 4}
        cfg.layer_types = [self._LT_CSA]
        normalize_deepseek_v4_compat(cfg)
        self.assertFalse(hasattr(cfg, "compress_ratios"))

    def test_unknown_layer_type_silently_falls_back_to_zero(self):
        # Unknown ``layer_types`` entries fall back to ``0`` — the same
        # encoding used for ``sliding_attention`` layers and every
        # downstream reader already treats ``0`` as "not a compressed
        # layer". Silent degradation is preferable to raising here: the
        # crash would surface at config-load time on any upstream that
        # adds a new layer type sglang has not yet been rebuilt for,
        # even when the new type does not need any special handling.
        # Matches the community consensus reached in
        # vllm-project/vllm#43443 and sgl-project#34128.
        cfg = self._make_new_transformers_config([self._LT_CSA, "mystery_layer"])
        normalize_deepseek_v4_compat(cfg)
        self.assertEqual(cfg.compress_ratios, [4, 0])

    def test_no_op_when_only_one_of_the_new_fields_is_present(self):
        # When the loaded config is malformed or partially populated (e.g. a
        # future upstream rename), leave attributes alone rather than
        # guessing — the downstream AttributeError is the clearer signal.
        cfg = PretrainedConfig()
        cfg.model_type = "deepseek_v4"
        cfg.compress_rates = {self._LT_CSA: 4}
        # layer_types deliberately absent
        normalize_deepseek_v4_compat(cfg)
        self.assertFalse(hasattr(cfg, "compress_ratios"))

    def test_no_op_when_compress_rates_is_not_a_dict(self):
        # A misconfigured ``model_override_args={"compress_rates": [...]}``
        # can hand the loader a list where a dict is expected. Fail loudly
        # by falling through to a plain AttributeError downstream rather
        # than reshaping garbage silently — an isinstance guard bounds the
        # blast radius to config-load rather than the first attention
        # layer's ``.get(...)`` call.
        cfg = self._make_new_transformers_config([self._LT_CSA])
        cfg.compress_rates = [4, 128]  # wrong shape
        normalize_deepseek_v4_compat(cfg)
        self.assertFalse(hasattr(cfg, "compress_ratios"))

    def test_no_op_when_layer_types_is_a_string(self):
        # Another common malformed-input shape: someone passes a single
        # ``"compressed_sparse_attention"`` string instead of a list of
        # them. ``for lt in layer_types`` would iterate characters and
        # produce nonsense ratios. The isinstance guard keeps the helper
        # a no-op so the downstream AttributeError is preserved as the
        # diagnostic surface.
        cfg = self._make_new_transformers_config(self._LT_CSA)  # not a list
        normalize_deepseek_v4_compat(cfg)
        self.assertFalse(hasattr(cfg, "compress_ratios"))

    def test_length_matches_layer_types_not_num_hidden_layers(self):
        # Contract: the rebuilt list is exactly as long as ``layer_types``.
        # sglang downstream (``configs/model_config.py`` and
        # ``models/deepseek_v4.py``) indexes it by layer id, so a shorter
        # list surfaces as ``IndexError`` at the right site rather than
        # this helper silently padding. Documented as a pinned contract
        # so a future "pad to num_hidden_layers" refactor gets caught.
        cfg = self._make_new_transformers_config([self._LT_CSA, self._LT_HCA])
        normalize_deepseek_v4_compat(cfg)
        self.assertEqual(len(cfg.compress_ratios), 2)

    def test_empty_compress_rates_produces_zero_ratios(self):
        # Degenerate but plausible config
        # (``compress_rates={}``): every non-SWA layer degrades to 0.
        # Downstream still gets a well-formed ``list[int]`` and reads
        # ``4 in []`` / ``sum(r == 4)`` / ``[layer_id]`` without crashing.
        cfg = self._make_new_transformers_config([self._LT_CSA, self._LT_HCA])
        cfg.compress_rates = {}
        normalize_deepseek_v4_compat(cfg)
        self.assertEqual(cfg.compress_ratios, [0, 0])

    def test_idempotent_across_repeated_calls(self):
        # The loader may run through ``get_hf_text_config`` more than once
        # for shared configs (multi-model launches, override paths). The
        # second invocation must be a no-op on a config that already has
        # the legacy list, and must not double-rebuild.
        cfg = self._make_new_transformers_config([self._LT_HCA, self._LT_CSA])
        normalize_deepseek_v4_compat(cfg)
        first = cfg.compress_ratios
        normalize_deepseek_v4_compat(cfg)
        second = cfg.compress_ratios
        self.assertIs(first, second)


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
