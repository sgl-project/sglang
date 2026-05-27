"""TP=2 multiprocess all-reduce harness for Double Sparsity.

Uses gloo backend (CPU) so no GPU is required. Three tests:

1. Positive: after all_reduce(SUM), both TP ranks produce bit-equal
   selected_token_indices — confirming cross-rank agreement.
2. Negative: without all_reduce, each rank independently selects from its
   own partial scores and produces divergent indices — confirming the
   all-reduce is load-bearing.
3. Permutation: two ranks hold different physical-slot orderings for the
   same logical sequence; after all-reduce via retrieve_topk_via_labels
   (logical-domain mode), both ranks agree on the same logical positions
   while their physical slot mappings differ.
"""

from __future__ import annotations

import os
import socket
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
    all_reduce_token_scores,
    retrieve_topk_via_labels,
    select_topk_sequence_order,
)

_WORLD_SIZE = 2


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _init_pg(rank: int, world_size: int, port: int) -> dist.ProcessGroup:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    return dist.new_group(list(range(world_size)))


# ─── worker: positive ────────────────────────────────────────────────────────


def _worker_positive(rank: int, world_size: int, tmpdir: str, port: int) -> None:
    pg = _init_pg(rank, world_size, port)
    if rank == 0:
        # Partial scores for heads owned by rank 0
        scores = torch.tensor([[1.0, 2.0, 10.0, 0.5, 3.0, 5.0, 0.1, 4.0]])
    else:
        # Partial scores for heads owned by rank 1
        scores = torch.tensor([[0.1, 0.2, 0.5, 8.0, 0.3, 0.4, 7.0, 6.0]])
    scores = all_reduce_token_scores(scores, process_group=pg)
    indices, valid_lengths = select_topk_sequence_order(scores, max_top_k=2)
    torch.save(
        {"indices": indices, "valid_lengths": valid_lengths},
        os.path.join(tmpdir, f"pos_{rank}.pt"),
    )
    dist.destroy_process_group()


# ─── worker: negative ────────────────────────────────────────────────────────


def _worker_negative(rank: int, world_size: int, tmpdir: str, port: int) -> None:
    _init_pg(rank, world_size, port)
    if rank == 0:
        scores = torch.tensor([[1.0, 2.0, 10.0, 0.5, 3.0, 5.0, 0.1, 4.0]])
    else:
        scores = torch.tensor([[0.1, 0.2, 0.5, 8.0, 0.3, 0.4, 7.0, 6.0]])
    # No all_reduce — each rank independently selects from its own partial scores
    indices, valid_lengths = select_topk_sequence_order(scores, max_top_k=2)
    torch.save(
        {"indices": indices, "valid_lengths": valid_lengths},
        os.path.join(tmpdir, f"neg_{rank}.pt"),
    )
    dist.destroy_process_group()


# ─── worker: permutation ─────────────────────────────────────────────────────


def _worker_permutation(rank: int, world_size: int, tmpdir: str, port: int) -> None:
    pg = _init_pg(rank, world_size, port)

    # 1 layer, 4 physical tokens, 1 head, label_dim=1, head_dim=1
    L, T, H, Ld = 1, 4, 1, 1

    channel_selection = torch.zeros(L, H, Ld, dtype=torch.int32)
    channel_weights = torch.ones(L, H, Ld, dtype=torch.float32)
    written = torch.ones(L, T, dtype=torch.bool)
    # query value = 1.0 so score[logical i] = 1.0 * sig[physical_slot_i]
    queries = torch.ones(1, H, 1, dtype=torch.float32)
    req_pool_indices = torch.zeros(1, dtype=torch.int32)
    seq_lens = torch.tensor([T], dtype=torch.int32)

    if rank == 0:
        # Physical slot sigs: [3.0, 4.0, 1.0, 2.0]; identity slot mapping.
        # Logical scores → [3.0, 4.0, 1.0, 2.0]
        sigs = torch.tensor([3.0, 4.0, 1.0, 2.0]).view(L, T, H, Ld)
        req_to_token = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    else:
        # Physical slot sigs: [0.1, 0.2, 8.0, 7.0]; reversed slot mapping.
        # Logical 0→phys 3→7.0, logical 1→phys 2→8.0, logical 2→phys 1→0.2, logical 3→phys 0→0.1
        # Logical scores → [7.0, 8.0, 0.2, 0.1]
        sigs = torch.tensor([0.1, 0.2, 8.0, 7.0]).view(L, T, H, Ld)
        req_to_token = torch.tensor([[3, 2, 1, 0]], dtype=torch.int32)

    indices, valid_lengths = retrieve_topk_via_labels(
        queries=queries,
        token_signatures=sigs,
        written=written,
        channel_selection=channel_selection,
        channel_weights=channel_weights,
        layer_id=0,
        max_top_k=2,
        process_group=pg,
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        seq_lens=seq_lens,
    )

    n = int(valid_lengths[0].item())
    logical_pos = indices[0, :n].long()
    physical_slots = req_to_token[0][logical_pos]

    torch.save(
        {
            "indices": indices,
            "valid_lengths": valid_lengths,
            "logical_pos": logical_pos,
            "physical_slots": physical_slots,
        },
        os.path.join(tmpdir, f"perm_{rank}.pt"),
    )
    dist.destroy_process_group()


# ─── test class ──────────────────────────────────────────────────────────────


class TestDoubleSparsityTPAllReduce(unittest.TestCase):
    """TP=2 all-reduce harness for Double Sparsity token selection."""

    def test_all_reduce_produces_bit_equal_indices(self):
        """Both TP ranks select the same logical token indices after all-reduce (SUM).

        Fixture arithmetic:
          Combined scores (SUM): [1.1, 2.2, 10.5, 8.5, 3.3, 5.4, 7.1, 10.0]
          Top-2 ascending:       tokens 2 (10.5) and 7 (10.0) → [2, 7]
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            port = _find_free_port()
            mp.spawn(_worker_positive, args=(_WORLD_SIZE, tmpdir, port), nprocs=_WORLD_SIZE)
            r0 = torch.load(os.path.join(tmpdir, "pos_0.pt"), weights_only=True)
            r1 = torch.load(os.path.join(tmpdir, "pos_1.pt"), weights_only=True)

        self.assertEqual(int(r0["valid_lengths"].item()), 2)
        self.assertEqual(int(r1["valid_lengths"].item()), 2)
        expected = torch.tensor([[2, 7]], dtype=torch.int32)
        self.assertTrue(torch.equal(r0["indices"], expected), f"rank0 got {r0['indices']}")
        self.assertTrue(torch.equal(r1["indices"], r0["indices"]),
                        f"ranks diverged: rank0={r0['indices']} rank1={r1['indices']}")

    def test_without_all_reduce_indices_diverge(self):
        """Without all-reduce each rank selects from partial scores and diverges.

        Fixture arithmetic:
          Rank 0 partial [1,2,10,.5,3,5,.1,4]  → top-2: tokens 2 (10.0) and 5 (5.0) → [2, 5]
          Rank 1 partial [.1,.2,.5,8,.3,.4,7,6] → top-2: tokens 3 (8.0) and 6 (7.0) → [3, 6]
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            port = _find_free_port()
            mp.spawn(_worker_negative, args=(_WORLD_SIZE, tmpdir, port), nprocs=_WORLD_SIZE)
            r0 = torch.load(os.path.join(tmpdir, "neg_0.pt"), weights_only=True)
            r1 = torch.load(os.path.join(tmpdir, "neg_1.pt"), weights_only=True)

        exp0 = torch.tensor([[2, 5]], dtype=torch.int32)
        exp1 = torch.tensor([[3, 6]], dtype=torch.int32)
        self.assertTrue(torch.equal(r0["indices"], exp0), f"rank0 got {r0['indices']}")
        self.assertTrue(torch.equal(r1["indices"], exp1), f"rank1 got {r1['indices']}")
        self.assertFalse(
            torch.equal(r0["indices"], r1["indices"]),
            "indices should diverge without all-reduce",
        )

    def test_physical_slot_permutation_logical_positions_agree(self):
        """Different physical-slot orderings per rank: same logical positions after all-reduce.

        Fixture arithmetic:
          Rank 0 logical scores (identity mapping):  [3.0, 4.0, 1.0, 2.0]
          Rank 1 logical scores (reversed mapping):  [7.0, 8.0, 0.2, 0.1]
          After SUM:                                 [10.0, 12.0, 1.2, 2.1]
          Top-2 ascending:                           logical positions [0, 1]

        Physical slots for logical [0, 1]:
          Rank 0 (req_to_token identity):  physical [0, 1]
          Rank 1 (req_to_token reversed):  physical [3, 2]
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            port = _find_free_port()
            mp.spawn(_worker_permutation, args=(_WORLD_SIZE, tmpdir, port), nprocs=_WORLD_SIZE)
            r0 = torch.load(os.path.join(tmpdir, "perm_0.pt"), weights_only=True)
            r1 = torch.load(os.path.join(tmpdir, "perm_1.pt"), weights_only=True)

        self.assertEqual(int(r0["valid_lengths"].item()), 2)
        self.assertEqual(int(r1["valid_lengths"].item()), 2)

        # Both ranks agree on logical positions
        self.assertTrue(
            torch.equal(r0["logical_pos"], r1["logical_pos"]),
            f"logical positions diverged: rank0={r0['logical_pos']} rank1={r1['logical_pos']}",
        )
        expected_logical = torch.tensor([0, 1])
        self.assertTrue(
            torch.equal(r0["logical_pos"], expected_logical),
            f"rank0 logical positions: got {r0['logical_pos']}, expected {expected_logical}",
        )

        # Physical slots are rank-specific
        self.assertFalse(
            torch.equal(r0["physical_slots"], r1["physical_slots"]),
            "physical slots should differ per rank",
        )
        self.assertTrue(
            torch.equal(r0["physical_slots"], torch.tensor([0, 1])),
            f"rank0 physical: got {r0['physical_slots']}",
        )
        self.assertTrue(
            torch.equal(r1["physical_slots"], torch.tensor([3, 2])),
            f"rank1 physical: got {r1['physical_slots']}",
        )


if __name__ == "__main__":
    unittest.main()
