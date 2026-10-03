"""Unit tests for `_fwd_kernel_ep_scatter_1` in sglang.kernels.ops.moe.ep_moe_kernels.

Covers the per-expert offset finalization (`expert_start_loc`) and the
grouped `m_indices` layout, including non-power-of-two expert counts,
empty experts, and fully padded experts. The kernel now writes each
program's own offset once (selected from the in-register cumsum) instead
of every program storing the whole prefix-sum array.
"""

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import triton

from sglang.kernels.ops.moe.ep_moe_kernels import _fwd_kernel_ep_scatter_1
from sglang.test.test_utils import CustomTestCase

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

        # m_indices: per-expert segment [start, start+valid) = expert id,
        # [start+valid, start+recv) = -1; total buffer covered up to
        # sum(recv) (the launcher only guarantees that prefix; trailing
        # padding lanes may hold garbage from torch.empty).
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
        # recv > 0 but valid == 0: the whole segment is -1 padding. A program
        # with a nonzero loop trip count must still write only its own
        # segment (docstring claims this case is covered).
        self.check([128, 128, 64], [128, 0, 64])

    def test_multi_block_expert_segment(self):
        # A single expert spanning more than one BLOCK_E iteration (300 ->
        # 3 loop iterations), exercising the num_stages=4 pipelined loop
        # body, including the tail iteration with a partial block.
        self.check([300, 128, 265, 0], [299, 1, 200, 0])

    def test_many_experts(self):
        # num_experts > 32 so BLOCK_EXPERT_NUM jumps to 64: many masked
        # lanes in the cumsum and a larger grid of concurrent programs.
        recv = [137, 0, 64, 128, 1, 255, 0, 128, 90, 33] * 4
        valid = [100, 0, 64, 7, 0, 255, 0, 128, 90, 1] * 4
        self.check(recv, valid)

    def test_stress_race_repeat(self):
        # The store loop previously wrote past a non-BLOCK_E-aligned
        # segment and raced with the next expert's program (flaky, not
        # deterministic). A single pass can pass by luck; repeat to make
        # the race, if reintroduced, observable with high probability.
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
            # [0,128,256,320,320,448]; expert 4's segment is the one a
            # past overrun of expert 2's tail block used to clobber.
            self.assertEqual(m_indices[320].item(), 4)

    def test_trailing_padding_not_written(self):
        # m_indices is padded to a BLOCK_E multiple; the kernel must not
        # store beyond sum(recv). Fill everything with a sentinel and
        # check the trailing padding lanes survive untouched.
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
        """Every expert's slot must be finalized by its own program.

        Fills the buffer with a sentinel and checks that every position is
        overwritten with the correct offset (no holes left from the
        single-element-per-program store).
        """
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
