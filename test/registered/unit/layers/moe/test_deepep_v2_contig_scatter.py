"""Tests for the DeepEP v2 contiguous-layout scatter kernel."""

import unittest

import torch

from sglang.kernels.ops.moe.ep_moe_kernels import ep_scatter_from_psum
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

DEVICE = "cuda"
HIDDEN = 256
SCALE_HIDDEN = HIDDEN // 128
# _fwd_kernel_ep_scatter_psum_init requires the row count to be BLOCK_E-aligned,
# which matches the 128-row expert alignment DeepEP v2 dispatches with.
ALIGN = 128


class TestDeepEPv2ContigScatter(CustomTestCase):
    """Guards `ep_scatter_from_psum`, the DeepEP v2 prefill permute.

    PR #35758 gave `_fwd_kernel_ep_scatter_2` two new positional arguments and
    updated only the DeepEP v1 caller, so every deepep_v2 prefill raised
    `TypeError: dynamic_func() missing 2 required positional arguments`.
    """

    # Local expert ids per (token, slot); -1 marks a route to a remote expert,
    # and 7 is out of this rank's expert range.
    RECV_TOPK = [
        [0, -1],
        [0, -1],
        [0, 1],
        [1, 7],
        [-1, -1],
    ]
    NUM_LOCAL_EXPERTS = 2

    def _run(self, dtype, with_scale):
        num_recv = len(self.RECV_TOPK)
        recv_x = (
            torch.arange(num_recv * HIDDEN, dtype=torch.float32, device=DEVICE).reshape(
                num_recv, HIDDEN
            )
            % 100
        ).to(dtype)
        recv_topk = torch.tensor(self.RECV_TOPK, dtype=torch.int64, device=DEVICE)
        psum = torch.tensor(
            [ALIGN * (e + 1) for e in range(self.NUM_LOCAL_EXPERTS)],
            dtype=torch.int32,
            device=DEVICE,
        )
        all_tokens = int(psum[-1].item())

        recv_x_scale = None
        output_tensor_scale = None
        if with_scale:
            recv_x_scale = torch.arange(
                num_recv * SCALE_HIDDEN, dtype=torch.float32, device=DEVICE
            ).reshape(num_recv, SCALE_HIDDEN)
            output_tensor_scale = torch.zeros(
                (all_tokens, SCALE_HIDDEN), dtype=torch.float32, device=DEVICE
            )

        output_tensor = torch.zeros((all_tokens, HIDDEN), device=DEVICE, dtype=dtype)
        m_indices = torch.empty(all_tokens, device=DEVICE, dtype=torch.int32)
        output_index = torch.empty_like(recv_topk)
        expert_start_loc = torch.empty_like(psum)

        ep_scatter_from_psum(
            recv_x,
            recv_x_scale,
            recv_topk,
            psum,
            expert_start_loc,
            output_tensor,
            output_tensor_scale,
            m_indices,
            output_index,
        )
        return recv_x, recv_x_scale, output_tensor, output_tensor_scale, output_index

    def _check(self, dtype, with_scale):
        recv_x, recv_x_scale, out, out_scale, output_index = self._run(
            dtype, with_scale
        )
        index = output_index.tolist()

        for token, slots in enumerate(self.RECV_TOPK):
            for slot, expert in enumerate(slots):
                dest = index[token][slot]
                if not 0 <= expert < self.NUM_LOCAL_EXPERTS:
                    # Remote and out-of-range routes must land on the sentinel;
                    # the post-permute gather reads this as "no contribution".
                    self.assertEqual(dest, -1, msg=f"{token=} {slot=} {expert=}")
                    continue
                # Each accepted route owns a distinct row inside its expert's slab.
                self.assertTrue(
                    ALIGN * expert <= dest < ALIGN * (expert + 1),
                    msg=f"{token=} {slot=} {expert=} {dest=}",
                )
                torch.testing.assert_close(out[dest].float(), recv_x[token].float())
                if with_scale:
                    torch.testing.assert_close(out_scale[dest], recv_x_scale[token])

        accepted = [
            index[t][s]
            for t, slots in enumerate(self.RECV_TOPK)
            for s, e in enumerate(slots)
            if 0 <= e < self.NUM_LOCAL_EXPERTS
        ]
        self.assertEqual(len(set(accepted)), len(accepted))

    def test_scatter_bf16_without_scales(self):
        self._check(torch.bfloat16, with_scale=False)

    def test_scatter_fp8_with_scales(self):
        self._check(torch.float8_e4m3fn, with_scale=True)


if __name__ == "__main__":
    unittest.main()
