# SPDX-License-Identifier: Apache-2.0
"""Regression tests for NunchakuConfig's skip-pattern handling.

Two bugs: (1) a skip match returned ``None`` for a ``LinearBase`` layer,
which trips ``assert self.quant_method is not None`` downstream instead of
leaving the layer unquantized; (2) skip matching lowercased the prefix
while the AWQ/SVDQ pattern matching right below it is case-sensitive, so a
skip pattern with any uppercase character could never match and the layer
fell through to the default SVDQ quantizer instead of being skipped.
"""

import unittest
from unittest import mock

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.nunchaku_config import (
    NunchakuConfig,
)


class _RulesWithSkip:
    @staticmethod
    def get_nunchaku_quant_rules():
        return {"skip": ["Emb"], "awq_w4a16": [], "svdq_w4a4": []}


class TestNunchakuSkipReturnsUnquantized(unittest.TestCase):
    def test_skip_match_returns_unquantized_method_not_none(self):
        config = NunchakuConfig(model_cls=_RulesWithSkip)
        layer = mock.Mock(spec=LinearBase)

        method = config.get_quant_method(layer, "block.Emb_table.weight")

        self.assertIsInstance(method, UnquantizedLinearMethod)

    def test_skip_matching_is_case_sensitive_like_awq_and_svdq(self):
        # Same pattern, different case in the prefix: must NOT skip, mirroring
        # the exact-case matching awq_w4a16/svdq_w4a4 already use.
        config = NunchakuConfig(model_cls=_RulesWithSkip)
        layer = mock.Mock(spec=LinearBase)

        method = config.get_quant_method(layer, "block.emb_table.weight")

        self.assertNotIsInstance(method, UnquantizedLinearMethod)


if __name__ == "__main__":
    unittest.main()
