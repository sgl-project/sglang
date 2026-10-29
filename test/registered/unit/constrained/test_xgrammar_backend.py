import unittest

from xgrammar import TokenizerInfo

from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_VOCAB = [chr(c) for c in range(32, 127)]
_SCHEMA = (
    '{"type":"object","properties":{"a":{"type":"integer"},'
    '"b":{"type":"array","items":{"type":"integer"}}},"required":["a","b"]}'
)


class _CharTokenizer:
    def init_xgrammar(self):
        return TokenizerInfo(_VOCAB), None


class TestXGrammarJsonWhitespace(CustomTestCase):
    def _accepts(self, text: str, *, any_whitespace: bool) -> bool:
        backend = XGrammarGrammarBackend(
            _CharTokenizer(), vocab_size=len(_VOCAB), any_whitespace=any_whitespace
        )
        return backend.dispatch_json(_SCHEMA).matcher.accept_string(text)

    def test_disable_any_whitespace_constrains_to_compact_json(self):
        """--constrained-json-disable-any-whitespace must produce compact JSON."""
        self.assertTrue(self._accepts('{"a":1,"b":[2,3]}', any_whitespace=False))
        self.assertFalse(self._accepts('{"a": 1, "b": [2, 3]}', any_whitespace=False))


if __name__ == "__main__":
    unittest.main()
