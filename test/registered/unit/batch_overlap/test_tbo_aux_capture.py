"""Eagle3/DFlash aux capture inside the TBO region.

Each sub-batch captures over its own contiguous slice of the parent's tokens, padded
to ``tbo_padded_len``; the merge scatters both back into full-batch tensors. CPU-only.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.batch_overlap.two_batch_overlap import (
    TboAuxCaptureSink,
    _merge_tbo_aux_captures,
    _model_forward_filter_inputs,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN = 3


def _fake_parent(ranges, padded_lens=None):
    padded_lens = padded_lens or [t - s for s, t in ranges]
    return SimpleNamespace(
        tbo_children=[
            SimpleNamespace(tbo_parent_token_range=r, tbo_padded_len=p)
            for r, p in zip(ranges, padded_lens)
        ]
    )


class TestMergeTboAuxCaptures(CustomTestCase):
    def test_scatters_both_sub_batches_by_token_range(self):
        original_len = 6
        parent = _fake_parent([(0, 4), (4, 6)])
        full = [torch.randn(original_len, HIDDEN) for _ in range(3)]
        sinks = [
            TboAuxCaptureSink(layer_ids={2}, captures=[t[0:4] for t in full]),
            TboAuxCaptureSink(layer_ids={2}, captures=[t[4:6] for t in full]),
        ]

        merged = _merge_tbo_aux_captures(sinks, parent, original_len=original_len)

        self.assertEqual(len(merged), 3)
        for got, want in zip(merged, full):
            self.assertEqual(got.shape, want.shape)
            torch.testing.assert_close(got, want)

    def test_trims_sub_batch_padding(self):
        # A sub-batch tensor is padded past its token range; the tail must be dropped.
        original_len = 5
        parent = _fake_parent([(0, 3), (3, 5)], padded_lens=[4, 4])
        a = torch.arange(4 * HIDDEN, dtype=torch.float32).reshape(4, HIDDEN)
        b = torch.arange(4 * HIDDEN, dtype=torch.float32).reshape(4, HIDDEN) + 100

        merged = _merge_tbo_aux_captures(
            [
                TboAuxCaptureSink(layer_ids={0}, captures=[a]),
                TboAuxCaptureSink(layer_ids={0}, captures=[b]),
            ],
            parent,
            original_len=original_len,
        )

        self.assertEqual(merged[0].shape, (original_len, HIDDEN))
        torch.testing.assert_close(merged[0][0:3], a[:3])
        torch.testing.assert_close(merged[0][3:5], b[:2])

    def test_rejects_mismatched_capture_counts(self):
        parent = _fake_parent([(0, 2), (2, 4)])
        with self.assertRaises(AssertionError):
            _merge_tbo_aux_captures(
                [
                    TboAuxCaptureSink(layer_ids={0}, captures=[torch.zeros(2, HIDDEN)]),
                    TboAuxCaptureSink(layer_ids={0}, captures=[]),
                ],
                parent,
                original_len=4,
            )


class TestFilterInputsSinkKey(CustomTestCase):
    """The ops chain forwards this dict as kwargs, so the key must be absent for
    models whose ``op_comm_prepare_attn`` has no ``aux_capture_sink`` parameter."""

    def _filter(self, sink):
        child = SimpleNamespace(tbo_parent_token_range=(0, 2), tbo_padded_len=2)
        return _model_forward_filter_inputs(
            hidden_states=torch.zeros(4, HIDDEN),
            residual=torch.zeros(4, HIDDEN),
            positions=torch.zeros(4, dtype=torch.int64),
            output_forward_batch=child,
            tbo_subbatch_index=0,
            aux_capture_sink=sink,
        )

    def test_key_omitted_without_sink(self):
        self.assertNotIn("aux_capture_sink", self._filter(None))

    def test_key_present_with_sink(self):
        sink = TboAuxCaptureSink(layer_ids={1})
        self.assertIs(self._filter(sink)["aux_capture_sink"], sink)


if __name__ == "__main__":
    unittest.main()
