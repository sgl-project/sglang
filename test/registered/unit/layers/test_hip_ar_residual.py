import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.communication.hip_ar_residual import (
    enabled,
    try_all_reduce_add,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHipArResidual(CustomTestCase):
    def test_helper_stays_off_when_flag_disabled(self):
        x = torch.zeros(2, 7168, dtype=torch.bfloat16)
        residual = torch.ones_like(x)
        with (
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.envs.SGLANG_ROCM_AR_RESIDUAL.get",
                return_value=False,
            ),
        ):
            self.assertFalse(enabled())
            self.assertIsNone(try_all_reduce_add(x, residual))

    def test_helper_stays_off_when_not_hip(self):
        x = torch.zeros(2, 7168, dtype=torch.bfloat16)
        with (
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.is_hip",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.envs.SGLANG_ROCM_AR_RESIDUAL.get",
                return_value=True,
            ),
        ):
            self.assertFalse(enabled())
            self.assertIsNone(try_all_reduce_add(x, torch.ones_like(x)))

    def test_helper_fail_closed_without_communicator(self):
        x = torch.zeros(2, 7168, dtype=torch.bfloat16)
        residual = torch.ones_like(x)
        group = Mock()
        group.ca_comm = None
        with (
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.envs.SGLANG_ROCM_AR_RESIDUAL.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=group,
            ),
        ):
            self.assertTrue(enabled())
            self.assertIsNone(try_all_reduce_add(x, residual))

    def test_helper_dispatches_residual_kernel(self):
        x = torch.zeros(2, 7168, dtype=torch.bfloat16)
        residual = torch.ones_like(x)
        out = torch.full_like(x, 3)
        ca_comm = Mock()
        ca_comm.disabled = False
        ca_comm.custom_all_reduce_residual.return_value = out
        group = Mock()
        group.ca_comm = ca_comm
        with (
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.envs.SGLANG_ROCM_AR_RESIDUAL.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=group,
            ),
        ):
            got = try_all_reduce_add(x, residual)
        self.assertIs(got, out)
        ca_comm.custom_all_reduce_residual.assert_called_once_with(x, residual)

    def test_helper_skips_m8_when_capped_at_4(self):
        x = torch.zeros(8, 7168, dtype=torch.bfloat16)
        residual = torch.ones_like(x)
        ca_comm = Mock()
        ca_comm.disabled = False
        group = Mock()
        group.ca_comm = ca_comm
        with (
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.envs.SGLANG_ROCM_AR_RESIDUAL.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.envs.SGLANG_ROCM_AR_RESIDUAL_MAX_TOKENS.get",
                return_value=4,
            ),
            patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=group,
            ),
        ):
            self.assertIsNone(try_all_reduce_add(x, residual))
        ca_comm.custom_all_reduce_residual.assert_not_called()

    def test_helper_covers_m8(self):
        x = torch.zeros(8, 7168, dtype=torch.bfloat16)
        residual = torch.ones_like(x)
        out = torch.full_like(x, 3)
        ca_comm = Mock()
        ca_comm.disabled = False
        ca_comm.custom_all_reduce_residual.return_value = out
        group = Mock()
        group.ca_comm = ca_comm
        with (
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.envs.SGLANG_ROCM_AR_RESIDUAL.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.envs.SGLANG_ROCM_AR_RESIDUAL_MAX_TOKENS.get",
                return_value=8,
            ),
            patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=group,
            ),
        ):
            self.assertIs(try_all_reduce_add(x, residual), out)
        ca_comm.custom_all_reduce_residual.assert_called_once_with(x, residual)

    def test_helper_normalizes_kda_residual_view(self):
        x = torch.zeros(8, 7168, dtype=torch.bfloat16)
        residual = torch.ones(1, 8, 7168, dtype=torch.bfloat16)
        out = torch.full_like(x, 3)
        ca_comm = Mock()
        ca_comm.disabled = False
        ca_comm.custom_all_reduce_residual.return_value = out
        group = Mock()
        group.ca_comm = ca_comm
        with (
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.envs.SGLANG_ROCM_AR_RESIDUAL.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.envs.SGLANG_ROCM_AR_RESIDUAL_MAX_TOKENS.get",
                return_value=8,
            ),
            patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=group,
            ),
        ):
            self.assertIs(try_all_reduce_add(x, residual), out)
        actual_residual = ca_comm.custom_all_reduce_residual.call_args.args[1]
        self.assertEqual(actual_residual.shape, x.shape)
        self.assertEqual(actual_residual.data_ptr(), residual.data_ptr())

    def test_helper_covers_m4(self):
        x = torch.zeros(4, 7168, dtype=torch.bfloat16)
        residual = torch.ones_like(x)
        out = torch.full_like(x, 3)
        ca_comm = Mock()
        ca_comm.disabled = False
        ca_comm.custom_all_reduce_residual.return_value = out
        group = Mock()
        group.ca_comm = ca_comm
        with (
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.envs.SGLANG_ROCM_AR_RESIDUAL.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=group,
            ),
        ):
            self.assertIs(try_all_reduce_add(x, residual), out)

    def test_none_residual_uses_plain_custom_ar(self):
        x = torch.zeros(2, 7168, dtype=torch.bfloat16)
        out = torch.ones_like(x)
        ca_comm = Mock()
        ca_comm.disabled = False
        ca_comm.custom_all_reduce.return_value = out
        group = Mock()
        group.ca_comm = ca_comm
        with (
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.envs.SGLANG_ROCM_AR_RESIDUAL.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=group,
            ),
        ):
            got = try_all_reduce_add(x, None)
        self.assertIs(got, out)
        ca_comm.custom_all_reduce.assert_called_once_with(x)


if __name__ == "__main__":
    unittest.main()
