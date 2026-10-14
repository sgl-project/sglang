"""When the lm_head matmul goes through aiter tgemm on gfx1250.

tgemm.mm takes a 2D bf16 activation and a bf16 weight; any other shape or
dtype keeps torch.matmul.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.layers import logits_processor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_VOCAB = 32
_HIDDEN = 16


def _ok(hidden_shape=(4, _HIDDEN), hidden_dtype=torch.bfloat16, weight_dtype=None):
    hidden_states = torch.zeros(hidden_shape, dtype=hidden_dtype)
    weight = torch.zeros((_VOCAB, _HIDDEN), dtype=weight_dtype or hidden_dtype)
    return logits_processor._aiter_lm_head_ok(hidden_states, weight)


class TestAiterLmHeadGate(CustomTestCase):
    def setUp(self):
        patcher = mock.patch.object(logits_processor, "_use_aiter_lm_head", True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_2d_bf16_uses_tgemm(self):
        self.assertTrue(_ok())

    def test_off_gfx1250_keeps_torch_matmul(self):
        with mock.patch.object(logits_processor, "_use_aiter_lm_head", False):
            self.assertFalse(_ok())

    def test_other_shapes_and_dtypes_keep_torch_matmul(self):
        for kwargs in (
            dict(hidden_shape=(2, 4, _HIDDEN)),
            dict(hidden_dtype=torch.float16),
            dict(hidden_dtype=torch.float32),
            dict(weight_dtype=torch.float32),
        ):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                self.assertFalse(_ok(**kwargs))


if __name__ == "__main__":
    unittest.main()
