"""CPU contracts for the fused suffix-attention merge dispatch guard."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.suffix_attention_merge import (
    can_use_fused_suffix_attention_merge,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSuffixAttentionMergeDispatch(CustomTestCase):
    def _inputs(self):
        layer = SimpleNamespace(
            head_dim=64,
            v_head_dim=64,
            is_cross_attention=False,
            logit_cap=0.0,
        )
        q = torch.empty((16, 8 * 64), dtype=torch.bfloat16)
        key_cache = torch.empty((8, 16, 2, 64), dtype=torch.bfloat16)
        value_cache = torch.empty_like(key_cache)
        return layer, q, key_cache, value_cache

    def _eligible(self, **overrides):
        layer, q, key_cache, value_cache = self._inputs()
        arguments = dict(
            layer=layer,
            q=q,
            key_cache=key_cache,
            value_cache=value_cache,
            extra_kwargs={},
        )
        arguments.update(overrides)
        return can_use_fused_suffix_attention_merge(**arguments)

    def test_standard_attention_is_eligible(self):
        self.assertTrue(self._eligible())

    def test_special_attention_features_fall_back(self):
        self.assertFalse(self._eligible(extra_kwargs={"sinks": object()}))

        layer, _, _, _ = self._inputs()
        layer.is_cross_attention = True
        self.assertFalse(self._eligible(layer=layer))

        layer, _, _, _ = self._inputs()
        layer.logit_cap = 20.0
        self.assertFalse(self._eligible(layer=layer))

    def test_unsupported_tensor_layout_falls_back(self):
        layer, _, _, _ = self._inputs()
        layer.v_head_dim = 32
        self.assertFalse(self._eligible(layer=layer))

        _, q, key_cache, value_cache = self._inputs()
        self.assertFalse(
            self._eligible(
                q=q.float(),
                key_cache=key_cache.float(),
                value_cache=value_cache.float(),
            )
        )


if __name__ == "__main__":
    unittest.main()
