"""Malformed-tree handling in build_tree_kernel_efficient.

The reference walk mirrors csrc/cpu/spec.cpp, which stops both on an ancestor
that was never selected and on a chain that loops.
"""

import unittest

import torch

from sglang.srt.speculative.eagle_utils import TreeMaskMode
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=2, suite="stage-b-test-1-gpu-small-amd")

TOPK = 2
NUM_STEPS = 3
DRAFT_TOKEN_NUM = 4
BS = 2

# an entry below topk is the root and ends the walk; two requests, so the
# per-request offset in the ancestor search is exercised
SELECTED = torch.tensor([[4, 2, 0], [2, 4, 0]], dtype=torch.int64)
SEQ_LENS = torch.tensor([7, 11], dtype=torch.int64)

PARENT_VALID = torch.tensor(
    [[0, 0, 2, 0, 0], [0, 4, 0, 0, 0]], dtype=torch.int64
)  # chains that reach the root
# 99 is never selected
PARENT_ABSENT = torch.tensor(
    [[0, 99, 99, 99, 99], [0, 99, 99, 99, 99]], dtype=torch.int64
)
# node 1 is its own ancestor
PARENT_LOOPS = torch.tensor([[0, 2, 2, 0, 0], [0, 4, 4, 0, 0]], dtype=torch.int64)


def _ref(parent_list, selected_index, seq_lens):
    """positions[] and the QLEN_ONLY mask, as the CPU backend computes them."""
    positions = []
    mask = torch.zeros(
        BS * DRAFT_TOKEN_NUM * DRAFT_TOKEN_NUM, dtype=torch.bool
    ).reshape(BS, DRAFT_TOKEN_NUM, DRAFT_TOKEN_NUM)
    for bid in range(BS):
        sel = selected_index[bid].tolist()
        par = parent_list[bid].tolist()
        seq_len = int(seq_lens[bid])
        for tid in range(DRAFT_TOKEN_NUM):
            mask[bid][tid][0] = True  # every token attends to the root
            if tid == 0:
                positions.append(seq_len)
                continue
            position, cur = 0, tid - 1
            while position < NUM_STEPS:
                position += 1
                mask[bid][tid][cur + 1] = True
                if sel[cur] // TOPK == 0:
                    break
                token_idx = par[sel[cur] // TOPK]
                found = -1
                for p in range(DRAFT_TOKEN_NUM - 1):
                    if sel[p] == token_idx:
                        found = p
                        break
                if found < 0:
                    break
                cur = found
            positions.append(position + seq_len)
    return torch.tensor(positions, dtype=torch.int64), mask.flatten()


def _run_cuda(parent_list, selected_index, seq_lens):
    device = "cuda"
    tree_mask = torch.full(
        (BS * DRAFT_TOKEN_NUM * DRAFT_TOKEN_NUM,),
        True,
        dtype=torch.bool,
        device=device,
    )
    positions = torch.zeros(BS * DRAFT_TOKEN_NUM, dtype=torch.int64, device=device)
    retrieve_buf = torch.full(
        (3, BS, DRAFT_TOKEN_NUM), -1, dtype=torch.int64, device=device
    )
    retrieve_index, retrieve_next_token, retrieve_next_sibling = retrieve_buf
    torch.ops.sgl_kernel.build_tree_kernel_efficient(
        parent_list.to(device),
        selected_index.to(device),
        seq_lens.to(device),
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        TOPK,
        NUM_STEPS,
        DRAFT_TOKEN_NUM,
        TreeMaskMode.QLEN_ONLY,
    )
    torch.cuda.synchronize()
    return positions.cpu(), tree_mask.cpu()


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestBuildEagleTreeMalformed(unittest.TestCase):
    def _check(self, parent_list):
        got_pos, got_mask = _run_cuda(parent_list, SELECTED, SEQ_LENS)
        want_pos, want_mask = _ref(parent_list, SELECTED, SEQ_LENS)
        self.assertTrue(
            torch.equal(got_pos, want_pos),
            f"positions {got_pos.tolist()} != reference {want_pos.tolist()}",
        )
        # an unbounded walk writes past its own row, which positions alone cannot see
        self.assertTrue(
            torch.equal(got_mask, want_mask),
            "tree_mask differs from the reference at "
            f"{(got_mask != want_mask).nonzero().flatten().tolist()}",
        )

    def test_valid_chain(self):
        """The bound must not change a tree that was already well formed."""
        self._check(PARENT_VALID)

    def test_absent_ancestor(self):
        """An ancestor that was never selected stops the walk, as on CPU."""
        self._check(PARENT_ABSENT)

    def test_looping_ancestor_chain(self):
        """A chain that never reaches the root stops at the depth bound."""
        self._check(PARENT_LOOPS)


if __name__ == "__main__":
    unittest.main()
