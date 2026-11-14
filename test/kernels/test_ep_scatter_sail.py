"""Index-level test for the SAIL variant of the EP-MoE scatter kernel.

`ep_scatter_sail` is the SAIL (PPU) counterpart of `ep_scatter`: it recomputes the
destination slot from the padded per-expert layout instead of taking an explicit
`num_valid_tokens_per_expert`, and it writes the packed expert id into `m_indices`.

This test checks the *contract* rather than the exact slot numbering, so it does not
duplicate the kernel's arithmetic:

1. `output_index` is a bijection onto the padded rows that the scatter touches.
2. Every gathered row maps back to the expert that produced it
   (`m_indices[output_index[t, k]] == recv_topk[t, k]`).
3. Every gathered row carries the source token's payload
   (`output_tensor[output_index[t, k]] == recv_x[t]`).

It runs on any CUDA-compatible device (including SAIL, where `torch.cuda` is the
device API), so it does not need a PPU-specific runner.
"""

import random
import unittest

import torch

from sglang.kernels.ops.moe.ep_moe_kernels import ep_scatter_sail
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30)


def _padded_layout(counts, block_e):
    """Exclusive prefix sums of the BLOCK_E-aligned per-expert counts."""
    starts = []
    total = 0
    for c in counts:
        starts.append(total)
        total += ((c + block_e - 1) // block_e) * block_e
    return torch.tensor(starts, dtype=torch.int32), total


@unittest.skipIf(not torch.cuda.is_available(), "requires a CUDA-compatible device")
class TestEpScatterSail(unittest.TestCase):
    def _run_case(self, num_experts, num_tokens, topk, hidden, block_e, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device("cuda")

        recv_topk = torch.stack(
            [
                torch.randperm(num_experts, device=device)[:topk]
                for _ in range(num_tokens)
            ]
        ).to(torch.int32)

        counts = torch.bincount(
            recv_topk.flatten().to(torch.int64), minlength=num_experts
        )
        expert_start_loc, padded_total = _padded_layout(counts.tolist(), block_e)
        expert_start_loc = expert_start_loc.to(device)

        recv_x = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
        output_tensor = torch.zeros(
            padded_total, hidden, device=device, dtype=torch.bfloat16
        )
        m_indices = torch.full((padded_total,), -1, device=device, dtype=torch.int32)
        output_index = torch.full(
            (num_tokens, topk), -1, device=device, dtype=torch.int32
        )

        ep_scatter_sail(
            recv_x=recv_x,
            recv_topk=recv_topk,
            num_recv_tokens_per_expert=counts.to(torch.int32),
            expert_start_loc=expert_start_loc,
            output_tensor=output_tensor,
            m_indices=m_indices,
            output_index=output_index,
            recv_x_scale=None,
            output_tensor_scale=None,
            BLOCK_E=block_e,
            BLOCK_D=min(128, hidden),
        )

        # 1. bijection onto the touched rows
        flat_index = output_index.flatten()
        self.assertTrue(
            bool((flat_index >= 0).all()), "every (token, topk) must be assigned a slot"
        )
        self.assertTrue(
            bool((flat_index < padded_total).all()),
            "assigned slot must be inside the padded buffer",
        )
        self.assertEqual(
            flat_index.unique().numel(),
            num_tokens * topk,
            "two (token, topk) pairs mapped onto the same slot",
        )

        # 2. slot -> expert agrees with recv_topk
        self.assertTrue(
            torch.equal(m_indices[flat_index].reshape(num_tokens, topk), recv_topk),
            "m_indices at the assigned slot does not carry the expected expert id",
        )

        # 3. payload copied to the assigned slot
        self.assertTrue(
            torch.equal(
                output_tensor[flat_index].reshape(num_tokens, topk, hidden),
                recv_x.unsqueeze(1).expand(num_tokens, topk, hidden),
            ),
            "token payload was not copied to the assigned slot",
        )

    def test_small(self):
        self._run_case(
            num_experts=8, num_tokens=17, topk=2, hidden=256, block_e=16, seed=0
        )

    def test_uneven_experts(self):
        # deliberately skewed load: some experts get many tokens, most get none
        self._run_case(
            num_experts=32, num_tokens=64, topk=4, hidden=512, block_e=32, seed=7
        )

    def test_single_token(self):
        self._run_case(
            num_experts=4, num_tokens=1, topk=1, hidden=128, block_e=8, seed=3
        )


if __name__ == "__main__":
    unittest.main()
