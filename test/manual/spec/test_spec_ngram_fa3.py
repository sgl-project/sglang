"""NGRAM speculative-decoding test, FA3 attention-backend variant.

Backend: `--attention-backend fa3`.
Not registered in any CI suite -- runnable manually only.
"""

import unittest

from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.ngram_fixture import NgramServerBase


class TestNgramSpeculativeDecodingBase(NgramServerBase, GSM8KMixin):
    attention_backend = "fa3"


if __name__ == "__main__":
    unittest.main()
