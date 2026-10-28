# SPDX-License-Identifier: Apache-2.0
import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.kernels.ops.mamba.clover_verify_conv import clover_parallel_causal_conv
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCloverParallelCausalConv(CustomTestCase):
    def test_parallel_conv_deduplicated_snapshots(self):
        torch.manual_seed(9)
        width = 32
        x = torch.randn(2, 256, width, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(3, 256, 3, device="cuda", dtype=x.dtype)
        before = state.clone()
        weight = torch.randn(256, 4, device="cuda", dtype=x.dtype)
        backing = torch.full((3, 256, width + 2), 123.0, device="cuda", dtype=x.dtype)
        windows = backing.as_strided(
            (3, width, 256, 3), (256 * (width + 2), 1, width + 2, 1)
        )
        slots = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
        cp_slots = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
        out = clover_parallel_causal_conv(
            x,
            state,
            weight,
            activation="silu",
            conv_state_indices=slots,
            intermediate_conv_window=windows,
            intermediate_state_indices=cp_slots,
        )
        serial_backing = torch.full_like(backing, 123.0)
        serial_windows = serial_backing.as_strided(windows.shape, windows.stride())
        serial_out = causal_conv1d_update(
            x,
            state.clone(),
            weight,
            activation="silu",
            conv_state_indices=slots,
            intermediate_conv_window=serial_windows,
            intermediate_state_indices=cp_slots,
        )
        torch.testing.assert_close(out, serial_out, rtol=0, atol=0)
        torch.testing.assert_close(backing, serial_backing, rtol=0, atol=0)
        history = torch.cat((state[slots.long()], x), dim=-1).float()
        for t in range(width):
            ref = F.silu((history[:, :, t : t + 4] * weight.float()).sum(-1))
            torch.testing.assert_close(out[:, :, t].float(), ref, atol=0.04, rtol=0.03)
            self.assertTrue(
                torch.equal(
                    windows[cp_slots.long(), t],
                    history[:, :, t + 1 : t + 4].to(x.dtype),
                )
            )
        self.assertTrue(torch.equal(state, before))
        self.assertTrue(torch.all(backing[0] == 123))


if __name__ == "__main__":
    unittest.main()
