"""
Unit tests for rejecting input_embeds requests on models that can't use them.

Gemma 3n/4 forward() requires exactly one of input_ids and input_embeds, but an
input_embeds request reaches it with placeholder input_ids as well, which used
to kill the scheduler (#44895). The tokenizer manager now rejects such requests
up front.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.model_config import ModelImpl  # noqa: E402
from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.model_loader import utils as model_loader_utils  # noqa: E402
from sglang.srt.model_loader.utils import supports_input_embeds  # noqa: E402
from sglang.srt.runtime_context import get_context  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

UNSUPPORTED_ARCHS = [
    "Gemma3nForCausalLM",
    "Gemma3nForConditionalGeneration",
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "Gemma4UnifiedForConditionalGeneration",
]
SUPPORTED_ARCHS = ["LlamaForCausalLM", "Gemma3ForConditionalGeneration"]


def _model_config(arch: str) -> SimpleNamespace:
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[arch]),
        quantization=None,
        model_impl=ModelImpl.AUTO,
        is_embedding_gemma=False,
    )


def _make_tokenizer_manager(arch: str) -> TokenizerManager:
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.model_config = _model_config(arch)
    tm.model_supports_input_embeds = None
    return tm


class TestSupportsInputEmbeds(CustomTestCase):
    def test_gemma3n_and_gemma4_opt_out(self):
        for arch in UNSUPPORTED_ARCHS:
            with self.subTest(arch=arch):
                self.assertFalse(supports_input_embeds(_model_config(arch)))

    def test_other_models_default_to_supported(self):
        for arch in SUPPORTED_ARCHS:
            with self.subTest(arch=arch):
                self.assertTrue(supports_input_embeds(_model_config(arch)))


class TestTokenizerManagerRejectsInputEmbeds(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(disable_radix_cache=True)
        override.install()
        self.addCleanup(override.restore)

    def test_input_embeds_request_is_rejected(self):
        for arch in UNSUPPORTED_ARCHS:
            with self.subTest(arch=arch):
                tm = _make_tokenizer_manager(arch)
                obj = GenerateReqInput(input_embeds=[[0.1, 0.2], [0.3, 0.4]])
                with self.assertRaisesRegex(
                    ValueError, f"input_embeds is not supported.*{arch}"
                ):
                    asyncio.run(tm._tokenize_one_request(obj))

    def test_supported_model_passes_the_check(self):
        for arch in SUPPORTED_ARCHS:
            with self.subTest(arch=arch):
                _make_tokenizer_manager(arch)._validate_input_embeds_support()

    def test_architecture_is_resolved_once(self):
        tm = _make_tokenizer_manager("Gemma4ForConditionalGeneration")
        with patch.object(
            model_loader_utils,
            "supports_input_embeds",
            wraps=model_loader_utils.supports_input_embeds,
        ) as resolve:
            for _ in range(2):
                with self.assertRaises(ValueError):
                    tm._validate_input_embeds_support()
        self.assertEqual(resolve.call_count, 1)


if __name__ == "__main__":
    unittest.main()
