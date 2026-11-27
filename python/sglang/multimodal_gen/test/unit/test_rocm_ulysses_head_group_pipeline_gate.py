# SPDX-License-Identifier: Apache-2.0
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.multimodal_gen.runtime.layers.attention import layer

GROUPS_ENV = "SGLANG_DIFFUSION_ULYSSES_PIPELINE_GROUPS"


class _Dev:
    """Stands in for a device tensor: the gate reads shape, is_cuda and q[0]."""

    def __init__(self, shape, is_cuda=True):
        self.shape = shape
        self.is_cuda = is_cuda

    def __getitem__(self, idx):
        return _Dev(self.shape[1:], self.is_cuda)


def _call(shape, groups, rocm=True, world=2, is_cuda=True, helper=None, attend=None):
    q = _Dev(shape, is_cuda)
    helper = helper or mock.Mock(return_value=torch.zeros(shape[1], shape[2], shape[3]))
    with (
        mock.patch.dict(os.environ, {GROUPS_ENV: str(groups)}),
        mock.patch.object(
            layer, "current_platform", SimpleNamespace(is_rocm=lambda: rocm)
        ),
        mock.patch.object(layer, "get_ulysses_parallel_world_size", return_value=world),
        mock.patch.object(layer, "ulysses_head_group_pipelined_attention", helper),
        mock.patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ),
    ):
        out = layer.USPAttention._rocm_head_group_pipelined(
            SimpleNamespace(), q, q, q, attend or (lambda a, b, c: a)
        )
    return out, helper


class TestRocmUlyssesHeadGroupPipelineGate(unittest.TestCase):
    """The ROCm head-group pipeline only engages for an explicit group count that
    divides the heads per rank; every other call keeps the plain exchange."""

    SHAPE = (1, 16, 40, 128)  # [B, S_local, H, D]

    def test_applies_with_explicit_groups(self):
        out, helper = _call(self.SHAPE, groups=4)
        helper.assert_called_once()
        self.assertEqual(helper.call_args.args[-1], 4)
        self.assertEqual(tuple(out.shape), (1, 16, 40, 128))

    def test_auto_and_disabled_values_keep_the_plain_exchange(self):
        for groups in (-1, 0, 1):
            out, helper = _call(self.SHAPE, groups=groups)
            self.assertIsNone(out)
            helper.assert_not_called()

    def test_other_platforms_and_hosts_are_untouched(self):
        for kwargs in ({"rocm": False}, {"is_cuda": False}):
            out, helper = _call(self.SHAPE, groups=4, **kwargs)
            self.assertIsNone(out)
            helper.assert_not_called()

    def test_declines_when_heads_do_not_divide_or_batch_is_not_one(self):
        out, helper = _call((1, 16, 40, 128), groups=3)  # 40 % (2 * 3) != 0
        self.assertIsNone(out)
        out, helper = _call((2, 16, 40, 128), groups=4)
        self.assertIsNone(out)
        helper.assert_not_called()

    def test_attend_wrapper_adds_and_drops_the_batch_dim(self):
        seen = {}

        def attend(qg, kg, vg):
            seen["shape"] = tuple(qg.shape)
            return qg

        helper = mock.Mock(return_value=torch.zeros(16, 40, 128))
        _call(self.SHAPE, groups=4, helper=helper, attend=attend)
        out = helper.call_args.args[3](*(torch.zeros(32, 5, 128),) * 3)
        self.assertEqual(seen["shape"], (1, 32, 5, 128))
        self.assertEqual(tuple(out.shape), (32, 5, 128))
        self.assertTrue(out.is_contiguous())


if __name__ == "__main__":
    unittest.main()
