import json
import os
import tempfile
import unittest

from sglang.srt.hf_transformers.common import HF_STRICT_CONFIG_ERRORS
from sglang.srt.hf_transformers.config import (
    _coerce_ints_to_floats,
    _float_declared_fields,
    _hf_config_class,
    _load_hf_auto_config,
    _nested_config_classes,
)
from sglang.srt.hf_transformers.tokenizer import (
    _declared_tokenizer_class_is_abstract,
    _load_tokenizers_backend_direct,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# Minimal BPE tokenizer: single-char tokens, no merges, no unk.
_TOKENIZER_JSON = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "BPE",
        "dropout": None,
        "unk_token": None,
        "continuing_subword_prefix": None,
        "end_of_word_suffix": None,
        "fuse_unk": False,
        "vocab": {"h": 0, "e": 1, "l": 2, "o": 3, "i": 5, "t": 4},
        "merges": [],
    },
}


class _FloatyConfig:
    """Stands in for a transformers v5 strict-dataclass config class."""

    routed_scaling_factor: float = 2.5
    norm_eps: float | None = 1e-5
    hidden_size: int = 1024
    n_heads: int = 16
    tie_weights: bool = True


class TestFloatDeclaredFields(CustomTestCase):
    def test_finds_float_and_optional_float_fields(self):
        self.assertEqual(
            _float_declared_fields(_FloatyConfig),
            {"routed_scaling_factor", "norm_eps"},
        )


class TestHfConfigClass(CustomTestCase):
    def test_resolves_through_lazy_mapping(self):
        # transformers v5's CONFIG_MAPPING is lazy: `.get()` returns None
        # for known model types, so resolution must go through `[]`.
        cls = _hf_config_class("llama")
        self.assertIsNotNone(cls)
        self.assertEqual(cls.model_type, "llama")

    def test_unknown_or_missing_type(self):
        self.assertIsNone(_hf_config_class("no_such_model_type"))
        self.assertIsNone(_hf_config_class(None))


class TestCoerceIntsToFloats(CustomTestCase):
    def test_promotes_ints_only_for_float_declared_fields(self):
        config_dict = {
            "model_type": "no_such_model",
            "routed_scaling_factor": 1,
            "norm_eps": 3,
            "hidden_size": 128,
            "n_heads": 8,
            "tie_weights": True,
        }
        changed = _coerce_ints_to_floats(config_dict, _FloatyConfig)
        self.assertTrue(changed)
        self.assertIsInstance(config_dict["routed_scaling_factor"], float)
        self.assertEqual(config_dict["routed_scaling_factor"], 1.0)
        self.assertEqual(config_dict["norm_eps"], 3.0)
        # int-declared and bool fields stay untouched
        self.assertEqual(config_dict["hidden_size"], 128)
        self.assertEqual(config_dict["n_heads"], 8)
        self.assertIs(config_dict["tie_weights"], True)
        # idempotent
        self.assertFalse(_coerce_ints_to_floats(config_dict, _FloatyConfig))

    def test_unknown_class_changes_nothing(self):
        config_dict = {"model_type": "no_such_model", "routed_scaling_factor": 1}
        self.assertFalse(_coerce_ints_to_floats(config_dict))
        self.assertEqual(config_dict["routed_scaling_factor"], 1)


class TestLoadTokenizersBackendDirect(CustomTestCase):
    def test_returns_none_without_tokenizer_json(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(_load_tokenizers_backend_direct(d))

    def test_loads_tokenizer_json_directly(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "tokenizer.json"), "w") as f:
                json.dump(_TOKENIZER_JSON, f)
            with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
                json.dump({"tokenizer_class": "PreTrainedTokenizer"}, f)
            tok = _load_tokenizers_backend_direct(d)
            if tok is None:
                self.skipTest("transformers without TokenizersBackend")
            self.assertTrue(tok.is_fast)
            self.assertEqual(tok.encode("hi"), [0, 5])


class _NestedTextConfig:
    """Stands in for a sub-config class (e.g. a VLM text_config)."""

    loss_scale: float = 1.0
    hidden_size: int = 8


class _NestedParentConfig:
    """Stands in for a multimodal config that declares `sub_configs`."""

    sub_configs = {"text_config": _NestedTextConfig}
    norm_eps: float = 1e-5


class TestNestedSubConfig(CustomTestCase):
    def test_reads_sub_configs_mapping(self):
        self.assertEqual(
            _nested_config_classes(_NestedParentConfig),
            {"text_config": _NestedTextConfig},
        )
        self.assertEqual(_nested_config_classes(_FloatyConfig), {})

    def test_coerces_sub_config_without_model_type(self):
        # Checkpoint sub-configs usually carry no model_type, so a
        # CONFIG_MAPPING lookup cannot find their class.
        config_dict = {
            "norm_eps": 1,
            "text_config": {"loss_scale": 2, "hidden_size": 4},
        }
        self.assertTrue(_coerce_ints_to_floats(config_dict, _NestedParentConfig))
        self.assertEqual(config_dict["norm_eps"], 1.0)
        self.assertEqual(config_dict["text_config"]["loss_scale"], 2.0)
        self.assertEqual(config_dict["text_config"]["hidden_size"], 4)

    def test_leaves_unknown_sub_config_untouched(self):
        config_dict = {"other_config": {"norm_eps": 1}}
        self.assertFalse(_coerce_ints_to_floats(config_dict, _FloatyConfig))
        self.assertEqual(config_dict["other_config"]["norm_eps"], 1)


@unittest.skipUnless(
    HF_STRICT_CONFIG_ERRORS,
    "needs huggingface_hub strict-dataclass config validation",
)
class TestLoadHfAutoConfig(CustomTestCase):
    """End-to-end: AutoConfig raises, the retry builds a usable config."""

    def _write(self, directory, config):
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(config, f)

    def test_retries_and_promotes_int_to_float(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(
                d,
                {
                    "model_type": "llama",
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "intermediate_size": 128,
                    "vocab_size": 100,
                    "rms_norm_eps": 1,
                },
            )
            config = _load_hf_auto_config(d, trust_remote_code=False, revision=None)
        self.assertIsInstance(config.rms_norm_eps, float)
        self.assertEqual(config.rms_norm_eps, 1.0)
        self.assertEqual(config.hidden_size, 64)
        self.assertEqual(config._name_or_path, d)

    def test_reraises_when_nothing_can_be_coerced(self):
        # float -> int is not promoted (it would truncate), so the original
        # error must surface instead of a silently mangled config.
        with tempfile.TemporaryDirectory() as d:
            self._write(
                d,
                {
                    "model_type": "llama",
                    "hidden_size": 64.0,
                    "num_hidden_layers": 2,
                    "vocab_size": 100,
                },
            )
            with self.assertRaises(HF_STRICT_CONFIG_ERRORS):
                _load_hf_auto_config(d, trust_remote_code=False, revision=None)


class TestDeclaredTokenizerClassIsAbstract(CustomTestCase):
    def test_abstract_base_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
                json.dump({"tokenizer_class": "PreTrainedTokenizer"}, f)
            self.assertTrue(_declared_tokenizer_class_is_abstract(d))

    def test_concrete_class_is_not_abstract(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
                json.dump({"tokenizer_class": "PreTrainedTokenizerFast"}, f)
            self.assertFalse(_declared_tokenizer_class_is_abstract(d))

    def test_missing_config_is_not_abstract(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(_declared_tokenizer_class_is_abstract(d))


if __name__ == "__main__":
    unittest.main()
