"""Breakable prefill CUDA graphs capture the linear-attention (KDA) extend only
in buckets up to SGLANG_LINEAR_PREFILL_GRAPH_MAX_TOKENS; larger buckets keep it
as an eager break with eagerly planned metadata.

    python -m pytest test/registered/unit/layers/attention/test_linear_prefill_graph_max_tokens.py -v
"""

import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _backend(calls):
    linear = SimpleNamespace(
        init_forward_metadata=lambda fb: calls.append(("eager", fb.extend_num_tokens)),
        init_prefill_graph_metadata=lambda fb: (
            calls.append(("graph", fb.extend_num_tokens)) or "tables"
        ),
    )
    return SimpleNamespace(
        full_attn_backend=SimpleNamespace(init_forward_metadata=lambda fb: None),
        linear_attn_backend=linear,
        _is_plain_extend=lambda fb: True,
    )


def _capture(backend, num_tokens):
    return (
        HybridLinearAttnBackend.init_forward_metadata_for_breakable_cuda_graph_capture(
            backend, SimpleNamespace(extend_num_tokens=num_tokens)
        )
    )


class TestLinearPrefillGraphMaxTokens(unittest.TestCase):
    def test_default_captures_every_bucket(self):
        calls = []
        backend = _backend(calls)
        self.assertEqual(_capture(backend, 16384), "tables")
        self.assertEqual(calls, [("graph", 16384)])

    def test_buckets_above_the_limit_break_eagerly(self):
        calls = []
        backend = _backend(calls)
        with envs.SGLANG_LINEAR_PREFILL_GRAPH_MAX_TOKENS.override(8192):
            self.assertEqual(_capture(backend, 8192), "tables")
            self.assertIsNone(_capture(backend, 16384))
        self.assertEqual(calls, [("graph", 8192), ("eager", 16384)])


if __name__ == "__main__":
    unittest.main()
