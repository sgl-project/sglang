"""Unit tests for `_fwd_kernel_ep_scatter_1` (per-expert offset finalization and
the grouped `m_indices` layout)."""

import unittest

import torch
import triton

from sglang.kernels.ops.moe.ep_moe_kernels import _fwd_kernel_ep_scatter_1
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

BLOCK_E = 128  # must match the launcher's BLOCK_E


def run_scatter_1(num_recv_tokens_per_expert, num_valid_tokens_per_expert):
    """Launch `_fwd_kernel_ep_scatter_1` and return (expert_start_loc, m_indices)."""
    num_experts = num_recv_tokens_per_expert.shape[0]
    total_tokens = int(num_recv_tokens_per_expert.sum().item())
    # m_indices must be padded to a multiple of BLOCK_E.
    m_indices = torch.empty(
        ((total_tokens + BLOCK_E - 1) // BLOCK_E) * BLOCK_E,
        device=num_recv_tokens_per_expert.device,
        dtype=torch.int32,
    )
    expert_start_loc = torch.empty(
        num_experts, device=num_recv_tokens_per_expert.device, dtype=torch.int32
    )
    _fwd_kernel_ep_scatter_1[(num_experts,)](
        num_recv_tokens_per_expert,
        num_valid_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=8,
        BLOCK_E=BLOCK_E,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )
    return expert_start_loc, m_indices


class TestEpScatter1(CustomTestCase):
    def check(self, recv, valid):
        num_experts = len(recv)
        recv_t = torch.tensor(recv, dtype=torch.int32, device="cuda")
        valid_t = torch.tensor(valid, dtype=torch.int32, device="cuda")
        expert_start_loc, m_indices = run_scatter_1(recv_t, valid_t)

        # expert_start_loc: exclusive prefix sum of recv counts.
        expected_starts = []
        offset = 0
        for r in recv:
            expected_starts.append(offset)
            offset += r
        self.assertEqual(expert_start_loc.tolist(), expected_starts)

        # Per-expert segment: [start, start+valid) = expert id, then -1 padding.
        for expert_id in range(num_experts):
            start = expected_starts[expert_id]
            n_recv, n_valid = recv[expert_id], valid[expert_id]
            seg = m_indices[start : start + n_recv]
            self.assertEqual(
                seg.tolist(),
                [expert_id] * n_valid + [-1] * (n_recv - n_valid),
                f"m_indices segment mismatch for expert {expert_id}",
            )

    def test_power_of_two_experts(self):
        self.check(
            [128, 256, 0, 128],
            [100, 180, 0, 128],
        )

    def test_non_power_of_two_experts(self):
        # Exercises the masked/padded lanes of BLOCK_EXPERT_NUM.
        self.check(
            [128, 128, 64, 0, 128, 128],
            [128, 90, 64, 0, 1, 127],
        )

    def test_all_empty_experts(self):
        self.check([0] * 5, [0] * 5)

    def test_single_expert(self):
        self.check([256], [256])

    def test_fully_padded_expert(self):
        # recv > 0 but valid == 0: the whole segment is -1 padding.
        self.check([128, 128, 64], [128, 0, 64])

    def test_multi_block_expert_segment(self):
        # One expert spanning several BLOCK_E iterations (300 -> 3 iterations).
        self.check([300, 128, 265, 0], [299, 1, 200, 0])

    def test_many_experts(self):
        # num_experts > 32, so BLOCK_EXPERT_NUM jumps to 64.
        recv = [137, 0, 64, 128, 1, 255, 0, 128, 90, 33] * 4
        valid = [100, 0, 64, 7, 0, 255, 0, 128, 90, 1] * 4
        self.check(recv, valid)

    def test_tail_mask_repeat(self):
        # Non-BLOCK_E-aligned counts exercise the defensive tail mask; repeat,
        # since a single pass can pass by luck.
        recv = [128, 128, 64, 0, 128, 128]
        valid = [128, 90, 64, 0, 1, 127]
        num_experts = len(recv)
        for _ in range(50):
            recv_t = torch.tensor(recv, dtype=torch.int32, device="cuda")
            valid_t = torch.tensor(valid, dtype=torch.int32, device="cuda")
            total = sum(recv)
            m_indices = torch.full(
                (((total + BLOCK_E - 1) // BLOCK_E) * BLOCK_E,),
                -999,
                device="cuda",
                dtype=torch.int32,
            )
            expert_start_loc = torch.empty(
                (num_experts,), device="cuda", dtype=torch.int32
            )
            _fwd_kernel_ep_scatter_1[(num_experts,)](
                recv_t,
                valid_t,
                expert_start_loc,
                m_indices,
                num_experts=num_experts,
                num_warps=8,
                BLOCK_E=BLOCK_E,
                BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
            )
            # [0,128,256,320,320,448]; expert 4's segment used to be clobbered.
            self.assertEqual(m_indices[320].item(), 4)

    def test_trailing_padding_not_written(self):
        # The kernel must not store beyond sum(recv); sentinel-fill and check.
        recv = [128, 64]
        valid = [100, 64]
        num_experts = len(recv)
        recv_t = torch.tensor(recv, dtype=torch.int32, device="cuda")
        valid_t = torch.tensor(valid, dtype=torch.int32, device="cuda")
        total = sum(recv)
        padded = ((total + BLOCK_E - 1) // BLOCK_E) * BLOCK_E
        m_indices = torch.full((padded,), -999, device="cuda", dtype=torch.int32)
        expert_start_loc = torch.empty(num_experts, device="cuda", dtype=torch.int32)
        _fwd_kernel_ep_scatter_1[(num_experts,)](
            recv_t,
            valid_t,
            expert_start_loc,
            m_indices,
            num_experts=num_experts,
            num_warps=8,
            BLOCK_E=BLOCK_E,
            BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
        )
        self.assertEqual(m_indices[:total].tolist(), [0] * 100 + [-1] * 28 + [1] * 64)
        self.assertEqual(
            m_indices[total:].tolist(),
            [-999] * (padded - total),
            "kernel wrote into the trailing padding region",
        )

    def test_starts_are_writable_once_and_complete(self):
        """Every expert's slot must be finalized by its own program (no holes)."""
        recv = [128, 0, 128, 128]
        valid = [50, 0, 128, 100]
        num_experts = len(recv)
        recv_t = torch.tensor(recv, dtype=torch.int32, device="cuda")
        valid_t = torch.tensor(valid, dtype=torch.int32, device="cuda")
        total = sum(recv)
        m_indices = torch.empty(
            ((total + BLOCK_E - 1) // BLOCK_E) * BLOCK_E,
            device="cuda",
            dtype=torch.int32,
        )
        expert_start_loc = torch.full(
            (num_experts,), -12345, dtype=torch.int32, device="cuda"
        )
        _fwd_kernel_ep_scatter_1[(num_experts,)](
            recv_t,
            valid_t,
            expert_start_loc,
            m_indices,
            num_experts=num_experts,
            num_warps=8,
            BLOCK_E=BLOCK_E,
            BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
        )
        expected = []
        offset = 0
        for r in recv:
            expected.append(offset)
            offset += r
        self.assertEqual(expert_start_loc.tolist(), expected)
        self.assertNotIn(-12345, expert_start_loc.tolist())


if __name__ == "__main__":
    unittest.main()
