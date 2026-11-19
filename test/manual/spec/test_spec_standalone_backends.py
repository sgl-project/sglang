"""V2 standalone speculative decoding on the triton and flashinfer backends.

CI runs only fa3 (test/registered/spec/test_spec_standalone.py).
Not registered in any CI suite -- runnable manually only.
"""

import unittest

from sglang.test.server_fixtures.standalone_fixture import StandaloneServerBase
from sglang.test.test_utils import CustomTestCase


class TestStandaloneV2SpeculativeDecodingTriton(StandaloneServerBase, CustomTestCase):
    attention_backend = "triton"


class TestStandaloneV2SpeculativeDecodingFlashinfer(
    StandaloneServerBase, CustomTestCase
):
    attention_backend = "flashinfer"


if __name__ == "__main__":
    unittest.main()
