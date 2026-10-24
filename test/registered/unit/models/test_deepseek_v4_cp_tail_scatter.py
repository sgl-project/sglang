import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import LateLayerTail
from sglang.srt.models import deepseek_v4
from sglang.srt.models.deepseek_v4 import _needs_cp_tail_scatter, _scatter_tail_rows
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _tail(**overrides):
    fields = dict(
        token_indices=torch.tensor([1, 3]),
        positions=torch.tensor([1, 3]),
        extend_seq_lens=torch.tensor([4]),
        extend_seq_lens_cpu=[4],
        swa_out_cache_loc=torch.tensor([1, 3]),
    )
    fields.update(overrides)
    return LateLayerTail(**fields)


class TestScatterTailRows(unittest.TestCase):
    def test_zero_fills_rows_outside_the_tail(self):
        out = _scatter_tail_rows(
            _tail(), torch.tensor([[10.0], [20.0]]), num_tokens=4, zero=True
        )
        self.assertEqual(tuple(out.shape), (4, 1))
        self.assertEqual(out[1, 0].item(), 10.0)
        self.assertEqual(out[3, 0].item(), 20.0)
        self.assertEqual(out[[0, 2]].abs().sum().item(), 0.0)

    def test_zero_fills_rows_before_a_contiguous_tail(self):
        out = _scatter_tail_rows(
            _tail(contiguous_start=2), torch.tensor([[10.0], [20.0]]), 4, zero=True
        )
        self.assertEqual(out[:2].abs().sum().item(), 0.0)
        self.assertTrue(torch.equal(out[2:], torch.tensor([[10.0], [20.0]])))

    def test_default_keeps_the_existing_path(self):
        out = _scatter_tail_rows(_tail(), torch.tensor([[10.0], [20.0]]), num_tokens=4)
        self.assertEqual(out[1, 0].item(), 10.0)
        self.assertEqual(out[3, 0].item(), 20.0)


class TestNeedsCpTailScatter(unittest.TestCase):
    def _check(self, *, cuda, cp, tail):
        with (
            mock.patch.object(deepseek_v4, "_is_cuda", cuda),
            mock.patch.object(deepseek_v4, "is_cp_active", return_value=cp),
        ):
            return _needs_cp_tail_scatter(tail, object())

    def test_requires_cuda_cp_and_a_tail(self):
        tail = _tail()
        self.assertTrue(self._check(cuda=True, cp=True, tail=tail))
        self.assertFalse(self._check(cuda=False, cp=True, tail=tail))
        self.assertFalse(self._check(cuda=True, cp=False, tail=tail))
        self.assertFalse(self._check(cuda=True, cp=True, tail=None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
