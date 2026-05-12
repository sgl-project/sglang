"""Unit tests for srt/configs/model_config_parser_registry.py"""

import unittest

from transformers import PretrainedConfig

from sglang.srt.configs.model_config_parser_registry import (
    _MODEL_CONFIG_PARSER_REGISTRY,
    ModelConfigParserBase,
    get_model_config_parser,
    register_model_config_parser,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class _FakeParser(ModelConfigParserBase):
    def parse(self, model, trust_remote_code, revision=None, **kwargs):
        return PretrainedConfig()


class _AnotherFakeParser(ModelConfigParserBase):
    def parse(self, model, trust_remote_code, revision=None, **kwargs):
        return PretrainedConfig()


class TestModelConfigParserRegistry(CustomTestCase):
    def setUp(self):
        self._saved_registry = dict(_MODEL_CONFIG_PARSER_REGISTRY)
        _MODEL_CONFIG_PARSER_REGISTRY.clear()

    def tearDown(self):
        _MODEL_CONFIG_PARSER_REGISTRY.clear()
        _MODEL_CONFIG_PARSER_REGISTRY.update(self._saved_registry)

    def test_register_then_get_roundtrip(self):
        register_model_config_parser("fake")(_FakeParser)
        self.assertIsInstance(get_model_config_parser("fake"), _FakeParser)

    def test_register_rejects_non_subclass(self):
        class NotAParser:
            pass

        with self.assertRaises(ValueError) as ctx:
            register_model_config_parser("bad")(NotAParser)
        self.assertIn("ModelConfigParserBase", str(ctx.exception))

    def test_unknown_name_raises_with_registered_list(self):
        register_model_config_parser("fake")(_FakeParser)
        register_model_config_parser("another")(_AnotherFakeParser)
        with self.assertRaises(ValueError) as ctx:
            get_model_config_parser("does-not-exist")
        msg = str(ctx.exception)
        self.assertIn("does-not-exist", msg)
        self.assertIn("another", msg)
        self.assertIn("fake", msg)


if __name__ == "__main__":
    unittest.main()
