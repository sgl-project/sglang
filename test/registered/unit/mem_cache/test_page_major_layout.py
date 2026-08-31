"""CPU correctness tests for the page-major envelope Mamba state views.

Covers the standalone ``build_page_major_mamba_views`` builder (no allocator /
shared pool): conv / temporal state views with correct shapes and no aliasing
across layers / slots. The unified pool stores its Mamba/KDA state through
these views.

Runs on CPU — pure-torch advanced indexing, no Triton.

    python -m pytest test/registered/unit/mem_cache/test_page_major_layout.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.layout.page_major import (
    build_page_major_mamba_views,
    mamba_entry_bytes,
)

_DEV = "cpu"


class TestMambaEnvelopeViews(unittest.TestCase):
    def test_conv_temporal_shapes_no_alias(self):
        layers, slots = 2, 4
        conv_shapes = [(2, 3)]
        temp_shape = (2, 2)
        conv_dt, temp_dt = torch.bfloat16, torch.float32
        entry = mamba_entry_bytes(
            layer_num=layers,
            conv_state_shapes=conv_shapes,
            conv_dtype=conv_dt,
            temporal_state_shape=temp_shape,
            temporal_dtype=temp_dt,
        )
        raw = torch.zeros(slots * entry, dtype=torch.uint8, device=_DEV)
        conv_views, temporal = build_page_major_mamba_views(
            raw,
            layer_num=layers,
            conv_state_shapes=conv_shapes,
            conv_dtype=conv_dt,
            temporal_state_shape=temp_shape,
            temporal_dtype=temp_dt,
            max_slots=slots,
        )
        self.assertEqual(tuple(conv_views[0].shape), (layers, slots, 2, 3))
        self.assertEqual(tuple(temporal.shape), (layers, slots, 2, 2))
        for L in range(layers):
            for s in range(slots):
                temporal[L, s] = float(s + L * 10 + 1)
        for L in range(layers):
            for s in range(slots):
                self.assertTrue(torch.all(temporal[L, s] == float(s + L * 10 + 1)))


if __name__ == "__main__":
    unittest.main()
