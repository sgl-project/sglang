"""End-to-end verification that --tokenizer-backend=fastokens swaps the
backend of the loaded tokenizer with fastokens' _TokenizerShim.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
)

TOKENIZER_MODEL = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN

register_cpu_ci(est_time=30, suite="stage-a-test-cpu")


try:
    import fastokens  # noqa: F401

    HAS_FASTOKENS = True
except ImportError:
    HAS_FASTOKENS = False


@unittest.skipUnless(HAS_FASTOKENS, "fastokens package not installed")
class TestFastokensBackend(CustomTestCase):
    def test_shim_is_applied(self):
        # `_TokenizerShim` is fastokens' private compat shim. SGLang's
        # integration relies on `tokenizer._tokenizer` being an instance of
        # this class to confirm fastokens is wired up. If fastokens renames
        # or restructures it, update both this assertion and any code in
        # SGLang that depends on the same private name.
        from fastokens._compat import _TokenizerShim

        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(
            TOKENIZER_MODEL,
            tokenizer_backend="fastokens",
        )
        backend = getattr(tokenizer, "_tokenizer", None)
        self.assertIsInstance(
            backend,
            _TokenizerShim,
            f"Expected tokenizer._tokenizer to be _TokenizerShim, "
            f"got {type(backend).__name__}",
        )

    def test_encode_decode_roundtrip(self):
        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(
            TOKENIZER_MODEL,
            tokenizer_backend="fastokens",
        )
        text = "Hello, world!"
        ids = tokenizer.encode(text, add_special_tokens=False)
        self.assertGreater(len(ids), 0)
        self.assertEqual(tokenizer.decode(ids, skip_special_tokens=True), text)


if __name__ == "__main__":
    unittest.main()
