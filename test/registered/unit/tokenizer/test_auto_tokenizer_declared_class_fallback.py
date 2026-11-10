"""Unit tests for _auto_tokenizer_from_pretrained's declared-class fallback —
no server, no model loading, no network."""

import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.utils.hf_transformers import tokenizer as tokenizer_mod

# Mirrors the real error raised by transformers when a model_type maps to the
# generic TokenizersBackend but the hub repo only ships vocab.json/merges.txt
# (e.g. granitemoe).
COULDNT_INSTANTIATE_ERR = (
    "Couldn't instantiate the backend tokenizer from one of: "
    "(1) a `tokenizers` library serialization file, "
    "(2) a slow tokenizer instance to convert or "
    "(3) an equivalent slow tokenizer class to instantiate and convert."
)


class TestAutoTokenizerDeclaredClassFallback(CustomTestCase):
    def test_recovers_via_declared_class_on_instantiate_failure(self):
        """When AutoTokenizer can't build a backend tokenizer, the declared
        tokenizer_config.json class should be used if it loads successfully."""
        sentinel_tokenizer = object()
        with (
            patch.object(
                tokenizer_mod.AutoTokenizer,
                "from_pretrained",
                side_effect=ValueError(COULDNT_INSTANTIATE_ERR),
            ),
            patch.object(
                tokenizer_mod,
                "_load_tokenizer_by_declared_class",
                return_value=sentinel_tokenizer,
            ) as mock_declared,
        ):
            result = tokenizer_mod._auto_tokenizer_from_pretrained("some/model")

        self.assertIs(result, sentinel_tokenizer)
        mock_declared.assert_called_once()

    def test_reraises_original_error_when_declared_class_also_fails(self):
        """If the declared-class fallback can't improve on AutoTokenizer
        either (returns None), the original ValueError must propagate
        unchanged rather than being swallowed or replaced."""
        with (
            patch.object(
                tokenizer_mod.AutoTokenizer,
                "from_pretrained",
                side_effect=ValueError(COULDNT_INSTANTIATE_ERR),
            ),
            patch.object(
                tokenizer_mod,
                "_load_tokenizer_by_declared_class",
                return_value=None,
            ),
        ):
            with self.assertRaises(ValueError) as ctx:
                tokenizer_mod._auto_tokenizer_from_pretrained("some/model")

        self.assertEqual(str(ctx.exception), COULDNT_INSTANTIATE_ERR)


if __name__ == "__main__":
    unittest.main()
