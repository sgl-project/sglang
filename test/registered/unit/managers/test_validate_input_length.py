"""Unit tests for the prompt-length check (validate_input_length).

Turns red if the rejection loses OpenAI's context-length sentence, which
OpenAI-compatible clients use to tell a context overflow from other 400s, if
the auto-truncate and in-range paths change, if a multimodal prompt gets
truncated, or if the tokenizer-side check breaks on an embedding request.
"""

import asyncio
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import EmbeddingReqInput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.managers.utils import validate_input_length  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(num_tokens):
    return SimpleNamespace(origin_input_ids=array("q", range(num_tokens)))


class TestValidateInputLength(CustomTestCase):
    def test_rejection_uses_openai_context_length_wording(self):
        req = _req(1500)
        msg = validate_input_length(
            req, max_req_input_len=1018, allow_auto_truncate=False, is_multimodal=False
        )
        self.assertIn(
            "Input length (1500 tokens) exceeds the maximum allowed length (1018 tokens).",
            msg,
        )
        self.assertIn("This model's maximum context length is 1018 tokens", msg)
        self.assertIn("--allow-auto-truncate", msg)
        self.assertEqual(len(req.origin_input_ids), 1500)

    def test_auto_truncate_unchanged(self):
        req = _req(1500)
        self.assertIsNone(
            validate_input_length(
                req,
                max_req_input_len=1018,
                allow_auto_truncate=True,
                is_multimodal=False,
            )
        )
        self.assertEqual(list(req.origin_input_ids), list(range(1018)))

    def test_in_range_unchanged(self):
        req = _req(1017)
        for allow_auto_truncate in (False, True):
            self.assertIsNone(
                validate_input_length(
                    req,
                    max_req_input_len=1018,
                    allow_auto_truncate=allow_auto_truncate,
                    is_multimodal=False,
                )
            )
        self.assertEqual(len(req.origin_input_ids), 1017)

    def test_multimodal_prompt_is_rejected_never_truncated(self):
        for allow_auto_truncate in (False, True):
            req = _req(1019)
            msg = validate_input_length(
                req,
                max_req_input_len=1018,
                allow_auto_truncate=allow_auto_truncate,
                is_multimodal=True,
            )
            self.assertIn(
                "Multimodal prompt is too long after expanding multimodal tokens.", msg
            )
            self.assertIn("This model's maximum context length is 1018 tokens", msg)
            self.assertNotIn("--allow-auto-truncate", msg)
            self.assertEqual(len(req.origin_input_ids), 1019)
            self.assertIsNone(
                validate_input_length(
                    _req(1018),
                    max_req_input_len=1018,
                    allow_auto_truncate=allow_auto_truncate,
                    is_multimodal=True,
                )
            )

    def test_embedding_request_reaches_tokenizer_check(self):
        tm = TokenizerManager.__new__(TokenizerManager)
        tm.model_config = MagicMock()
        tm.mm_processor = None
        tm._validate_one_request = Mock()
        tm._create_tokenized_object = Mock()
        disagg = SimpleNamespace(language_model_only=False, language_only=False)
        with patch(
            "sglang.srt.managers.tokenizer_manager.get_disagg", return_value=disagg
        ):
            asyncio.run(
                tm._tokenize_one_request(EmbeddingReqInput(input_ids=[1, 2, 3]))
            )
        tm._validate_one_request.assert_called_once()
        self.assertFalse(tm._validate_one_request.call_args.kwargs["is_multimodal"])


if __name__ == "__main__":
    unittest.main()
