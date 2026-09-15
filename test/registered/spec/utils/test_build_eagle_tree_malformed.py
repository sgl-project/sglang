"""build_tree_kernel_efficient on malformed trees, across all three mask modes.

The reference mirrors csrc/cpu/spec.cpp, which is the implementation this kernel is being
brought in line with: the parent lookup searches only the draft_token_num - 1 entries
selected_index holds per request, a token whose parent was not selected is skipped, and the
root-ward walk is bounded by depth.

All three mask layouts encode the same logical [bs][token][ancestor] matrix, so the reference
builds that once and encodes it per layout:

  QLEN_ONLY             row of draft_token_num bools per token
  FULL_MASK             row of verified_seq_len + draft_token_num bools; the leading
                        verified_seq_len entries belong to the caller and are not compared
  QLEN_ONLY_BITPACKING  num_bytes_per_item bytes per token, bit k of the item is column k

Every case compares positions, tree_mask, retrieve_index, retrieve_next_token and
retrieve_next_sibling. Checking positions and the mask alone leaves the retrieval tree, which
is what a missing parent actually corrupts, without an oracle.
"""

import itertools
import unittest

import torch

from sglang.srt.speculative.eagle_utils import TreeMaskMode
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=4, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=4, suite="stage-b-test-1-gpu-small-amd")

TOPK = 2
NUM_STEPS = 3
DRAFT_TOKEN_NUM = 4
BS = 2

# an entry below topk ends the walk at the root; two requests, so the per-request offset in the
# lookup is exercised rather than assumed
SELECTED = torch.tensor([[4, 2, 0], [2, 4, 0]], dtype=torch.int64)
SEQ_LENS = torch.tensor([7, 11], dtype=torch.int64)

CASES = {
    # chains that reach the root
    "valid_chain": torch.tensor([[0, 0, 2, 0, 0], [0, 4, 0, 0, 0]], dtype=torch.int64),
    # the direct parent of a token is a value selected_index never holds
    "missing_direct_parent": torch.tensor(
        [[0, 99, 2, 0, 0], [0, 99, 0, 0, 0]], dtype=torch.int64
    ),
    # every lookup misses, so the search runs to the end of the row on every step
    "missing_ancestor": torch.tensor(
        [[0, 99, 99, 99, 99], [0, 99, 99, 99, 99]], dtype=torch.int64
    ),
    # a node is its own ancestor, so an unbounded walk never reaches the root
    "cyclic_ancestor": torch.tensor(
        [[0, 2, 2, 0, 0], [0, 4, 4, 0, 0]], dtype=torch.int64
    ),
    # a chain exactly depth long: the bound must not cut a walk that is still valid
    "max_valid_depth": torch.tensor(
        [[0, 0, 4, 2, 0], [0, 0, 2, 4, 0]], dtype=torch.int64
    ),
}

# Request 0's parent is absent from its own row but present as request 1's first entry. A search
# that runs one element too far reads request 1's row and can match there. The outputs cannot see
# it -- the extra step makes a match indistinguishable from not-found -- so this case exists for
# a memory checker rather than for its assertions.
# 6 is absent from request 0's row, is request 1's first entry, and 6 // topk stays inside
# parent_list, so the case probes the row overrun and not the separate question of an
# out-of-range parent_tb_idx, which this kernel does not validate and this PR does not claim to.
CROSS_ROW_SELECTED = torch.tensor([[4, 2, 0], [6, 4, 0]], dtype=torch.int64)
CROSS_ROW_PARENT = torch.tensor([[0, 6, 4, 0, 0], [0, 0, 6, 4, 0]], dtype=torch.int64)

MODES = [TreeMaskMode.FULL_MASK, TreeMaskMode.QLEN_ONLY, TreeMaskMode.QLEN_ONLY_BITPACKING]
MODE_NAMES = {
    TreeMaskMode.FULL_MASK: "FULL_MASK",
    TreeMaskMode.QLEN_ONLY: "QLEN_ONLY",
    TreeMaskMode.QLEN_ONLY_BITPACKING: "QLEN_ONLY_BITPACKING",
}


def _bytes_per_item(draft_token_num):
    if draft_token_num > 16:
        return 4
    if draft_token_num > 8:
        return 2
    return 1


def _reference(parent_list, selected_index, seq_lens):
    """The logical outputs: positions, an [bs][token][column] matrix, and the retrieval tree."""
    sel_stride = DRAFT_TOKEN_NUM - 1
    positions = torch.zeros(BS * DRAFT_TOKEN_NUM, dtype=torch.int64)
    retrieve_index = torch.full((BS, DRAFT_TOKEN_NUM), -1, dtype=torch.int64)
    retrieve_next_token = torch.full((BS, DRAFT_TOKEN_NUM), -1, dtype=torch.int64)
    retrieve_next_sibling = torch.full((BS, DRAFT_TOKEN_NUM), -1, dtype=torch.int64)
    matrix = torch.zeros(BS, DRAFT_TOKEN_NUM, DRAFT_TOKEN_NUM, dtype=torch.bool)

    def find(sel, token_idx):
        for p in range(sel_stride):
            if sel[p] == token_idx:
                return p
        return -1

    for bid in range(BS):
        sel = selected_index[bid].tolist()
        par = parent_list[bid].tolist()
        seq_len = int(seq_lens[bid])

        # the retrieval tree, built by tid == 0 walking i downward
        for i in range(DRAFT_TOKEN_NUM - 1, 0, -1):
            retrieve_index[bid][i] = bid * DRAFT_TOKEN_NUM + i
            parent_tb_idx = sel[i - 1] // TOPK
            parent_position = 0
            if parent_tb_idx > 0:
                found = find(sel, par[parent_tb_idx])
                if found < 0:
                    continue  # the token is skipped, as on CPU
                parent_position = found + 1
            if retrieve_next_token[bid][parent_position] == -1:
                retrieve_next_token[bid][parent_position] = i
            else:
                origin = int(retrieve_next_token[bid][parent_position])
                retrieve_next_token[bid][parent_position] = i
                retrieve_next_sibling[bid][i] = origin
        retrieve_index[bid][0] = bid * DRAFT_TOKEN_NUM

        # one root-ward walk per token, which fills that token's mask row and its position
        for tid in range(DRAFT_TOKEN_NUM):
            matrix[bid][tid][0] = True
            if tid == 0:
                positions[bid * DRAFT_TOKEN_NUM] = seq_len
                continue
            position, cur = 0, tid - 1
            while position < NUM_STEPS:
                position += 1
                matrix[bid][tid][cur + 1] = True
                if sel[cur] // TOPK == 0:
                    break
                found = find(sel, par[sel[cur] // TOPK])
                if found < 0:
                    break
                cur = found
            positions[bid * DRAFT_TOKEN_NUM + tid] = position + seq_len

    return positions, matrix, retrieve_index, retrieve_next_token, retrieve_next_sibling


def _decode_mask(raw, matrix_shape, seq_lens, mode):
    """The kernel's mask buffer, decoded back to the logical [bs][token][column] matrix."""
    bs, n, _ = matrix_shape
    out = torch.zeros(bs, n, n, dtype=torch.bool)
    if mode == TreeMaskMode.QLEN_ONLY:
        return raw.reshape(bs, n, n).clone()
    if mode == TreeMaskMode.FULL_MASK:
        off = 0
        for bid in range(bs):
            seq_len = int(seq_lens[bid])
            row = seq_len + n
            for tid in range(n):
                base = off + row * tid + seq_len
                out[bid][tid] = raw[base : base + n]
            off += row * n
        return out
    nbytes = _bytes_per_item(n)
    for bid in range(bs):
        for tid in range(n):
            item = raw[(bid * n + tid) * nbytes : (bid * n + tid + 1) * nbytes]
            for col in range(n):
                out[bid][tid][col] = bool(int(item[col // 8]) >> (col % 8) & 1)
    return out


def _run(parent_list, selected_index, seq_lens, mode, op=None):
    op = op or torch.ops.sgl_kernel.build_tree_kernel_efficient
    device = "cuda"
    n = DRAFT_TOKEN_NUM
    if mode == TreeMaskMode.QLEN_ONLY:
        mask = torch.zeros(BS * n * n, dtype=torch.bool, device=device)
    elif mode == TreeMaskMode.FULL_MASK:
        total = sum((int(s) + n) * n for s in seq_lens)
        mask = torch.zeros(total, dtype=torch.bool, device=device)
    else:
        mask = torch.zeros(BS * n * _bytes_per_item(n), dtype=torch.uint8, device=device)

    positions = torch.zeros(BS * n, dtype=torch.int64, device=device)
    buf = torch.full((3, BS, n), -1, dtype=torch.int64, device=device)
    ri, rnt, rns = buf
    op(
        parent_list.to(device),
        selected_index.to(device),
        seq_lens.to(device),
        mask,
        positions,
        ri,
        rnt,
        rns,
        TOPK,
        NUM_STEPS,
        n,
        int(mode),
    )
    torch.cuda.synchronize()
    return positions.cpu(), mask.cpu(), ri.cpu(), rnt.cpu(), rns.cpu()


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestBuildEagleTreeMalformed(unittest.TestCase):
    def _check(self, name, parent_list, selected_index, mode, op=None):
        got_pos, got_raw, got_ri, got_rnt, got_rns = _run(
            parent_list, selected_index, SEQ_LENS, mode, op
        )
        want_pos, want_matrix, want_ri, want_rnt, want_rns = _reference(
            parent_list, selected_index, SEQ_LENS
        )
        got_matrix = _decode_mask(got_raw, want_matrix.shape, SEQ_LENS, mode)
        tag = f"{name}/{MODE_NAMES[mode]}"
        for label, got, want in (
            ("positions", got_pos, want_pos),
            ("tree_mask", got_matrix, want_matrix),
            ("retrieve_index", got_ri, want_ri),
            ("retrieve_next_token", got_rnt, want_rnt),
            ("retrieve_next_sibling", got_rns, want_rns),
        ):
            self.assertTrue(
                torch.equal(got, want),
                f"{tag}: {label}\n  got  {got.tolist()}\n  want {want.tolist()}",
            )

    def test_matrix(self):
        for (name, parents), mode in itertools.product(CASES.items(), MODES):
            with self.subTest(case=name, mode=MODE_NAMES[mode]):
                self._check(name, parents, SELECTED, mode)

    def test_cross_row_lookup(self):
        """A lookup that runs past its row can reach the next request's first entry.

        The assertions here cannot see that read; they exist so the case is covered by the
        oracle as well. Run this file under compute-sanitizer to see the read itself.
        """
        for mode in MODES:
            with self.subTest(mode=MODE_NAMES[mode]):
                self._check(
                    "cross_row", CROSS_ROW_PARENT, CROSS_ROW_SELECTED, mode
                )


if __name__ == "__main__":
    unittest.main()
