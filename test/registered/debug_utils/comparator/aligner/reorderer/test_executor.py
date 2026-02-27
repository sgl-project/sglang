import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.reorderer.executor import (
    _reorder_zigzag_to_natural,
    _reorder_zigzag_to_natural_thd,
    execute_reorderer_plan,
)
from sglang.srt.debug_utils.comparator.aligner.reorderer.types import (
    ReordererPlan,
    ZigzagToNaturalThdParams,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.executor import (
    execute_unsharder_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import (
    CpThdConcatParams,
    UnsharderPlan,
)
from sglang.srt.debug_utils.comparator.dims import ParallelAxis
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


def _zigzag_order(cp_size: int) -> list[int]:
    """Build zigzag interleaving order for 2*cp_size chunks."""
    order: list[int] = []
    num_chunks: int = cp_size * 2
    for i in range(cp_size):
        order.append(i)
        order.append(num_chunks - 1 - i)
    return order


def _zigzag_split_seq(seq_natural: torch.Tensor, *, cp_size: int) -> list[torch.Tensor]:
    """Split a natural-order seq into per-rank zigzag segments.

    Returns: list of per-rank tensors, where rank_i holds chunks assigned by zigzag.
    """
    num_chunks: int = cp_size * 2
    chunks: list[torch.Tensor] = list(seq_natural.chunk(num_chunks, dim=0))
    order: list[int] = _zigzag_order(cp_size)
    zigzagged: torch.Tensor = torch.cat([chunks[i] for i in order], dim=0)
    return list(zigzagged.chunk(cp_size, dim=0))


class TestZigzagToNatural:
    def test_zigzag_to_natural_cp2(self) -> None:
        """cp_size=2: zigzag order [0,3,1,2] -> natural [0,1,2,3]."""
        natural = torch.arange(24).reshape(4, 6)
        chunks = list(natural.chunk(4, dim=0))

        zigzag_order: list[int] = [0, 3, 1, 2]
        zigzagged = torch.cat([chunks[i] for i in zigzag_order], dim=0)

        result = _reorder_zigzag_to_natural(zigzagged, dim=0, cp_size=2)
        assert torch.equal(result, natural)

    def test_zigzag_to_natural_cp3(self) -> None:
        """cp_size=3: zigzag 162534 -> natural 123456 (1-indexed)."""
        natural = torch.arange(60).reshape(6, 10)
        chunks = list(natural.chunk(6, dim=0))

        zigzag_order: list[int] = [0, 5, 1, 4, 2, 3]
        zigzagged = torch.cat([chunks[i] for i in zigzag_order], dim=0)

        result = _reorder_zigzag_to_natural(zigzagged, dim=0, cp_size=3)
        assert torch.equal(result, natural)

    def test_zigzag_to_natural_arbitrary_dim(self) -> None:
        """Reorder along dim=1 instead of dim=0."""
        natural = torch.arange(48).reshape(3, 4, 4)
        chunks = list(natural.chunk(4, dim=1))

        zigzag_order: list[int] = [0, 3, 1, 2]
        zigzagged = torch.cat([chunks[i] for i in zigzag_order], dim=1)

        result = _reorder_zigzag_to_natural(zigzagged, dim=1, cp_size=2)
        assert torch.equal(result, natural)


class TestZigzagToNaturalThd:
    def test_single_seq(self) -> None:
        """Single seq THD reorder: equivalent to whole-tensor reorder."""
        natural = torch.arange(100)
        zigzag_ranks: list[torch.Tensor] = _zigzag_split_seq(natural, cp_size=2)
        zigzagged: torch.Tensor = torch.cat(zigzag_ranks, dim=0)

        result = _reorder_zigzag_to_natural_thd(
            zigzagged, dim=0, cp_size=2, seq_lens=[100]
        )
        assert torch.equal(result, natural)

    def test_multi_seq(self) -> None:
        """Two seqs of different lengths, each independently reordered."""
        seq_a_natural = torch.arange(100)
        seq_b_natural = torch.arange(100, 164)

        seq_a_zigzag: torch.Tensor = torch.cat(
            _zigzag_split_seq(seq_a_natural, cp_size=2), dim=0
        )
        seq_b_zigzag: torch.Tensor = torch.cat(
            _zigzag_split_seq(seq_b_natural, cp_size=2), dim=0
        )

        combined_zigzag: torch.Tensor = torch.cat([seq_a_zigzag, seq_b_zigzag], dim=0)
        result = _reorder_zigzag_to_natural_thd(
            combined_zigzag, dim=0, cp_size=2, seq_lens=[100, 64]
        )

        expected: torch.Tensor = torch.cat([seq_a_natural, seq_b_natural], dim=0)
        assert torch.equal(result, expected)

    def test_with_tail_pad(self) -> None:
        """THD reorder with trailing global padding preserved unchanged."""
        seq_natural = torch.arange(100)
        pad: torch.Tensor = torch.full((56,), fill_value=-1)

        seq_zigzag: torch.Tensor = torch.cat(
            _zigzag_split_seq(seq_natural, cp_size=2), dim=0
        )
        combined: torch.Tensor = torch.cat([seq_zigzag, pad], dim=0)

        result = _reorder_zigzag_to_natural_thd(
            combined, dim=0, cp_size=2, seq_lens=[100]
        )

        assert torch.equal(result[:100], seq_natural)
        assert torch.equal(result[100:], pad)

    def test_with_hidden_dim(self) -> None:
        """THD reorder with trailing hidden dimension (shape [T, H])."""
        torch.manual_seed(42)
        hidden: int = 8
        seq_natural = torch.randn(100, hidden)

        seq_zigzag: torch.Tensor = torch.cat(
            _zigzag_split_seq(seq_natural, cp_size=2), dim=0
        )

        result = _reorder_zigzag_to_natural_thd(
            seq_zigzag, dim=0, cp_size=2, seq_lens=[100]
        )
        assert torch.equal(result, seq_natural)

    def test_with_leading_batch_dim(self) -> None:
        """THD reorder with leading batch dim: shape [B, T, H], t is dim=1."""
        torch.manual_seed(42)
        batch: int = 2
        hidden: int = 4
        seq_a_natural = torch.randn(batch, 100, hidden)
        seq_b_natural = torch.randn(batch, 64, hidden)
        full_natural: torch.Tensor = torch.cat([seq_a_natural, seq_b_natural], dim=1)

        # Zigzag each seq along dim=1
        def zigzag_along_dim1(t: torch.Tensor) -> torch.Tensor:
            num_chunks: int = 2 * 2  # cp_size=2
            chunks: list[torch.Tensor] = list(t.chunk(num_chunks, dim=1))
            order: list[int] = [0, 3, 1, 2]  # zigzag for cp_size=2
            return torch.cat([chunks[i] for i in order], dim=1)

        seq_a_zigzag: torch.Tensor = zigzag_along_dim1(seq_a_natural)
        seq_b_zigzag: torch.Tensor = zigzag_along_dim1(seq_b_natural)
        combined_zigzag: torch.Tensor = torch.cat([seq_a_zigzag, seq_b_zigzag], dim=1)

        result = _reorder_zigzag_to_natural_thd(
            combined_zigzag, dim=1, cp_size=2, seq_lens=[100, 64]
        )
        assert torch.equal(result, full_natural)


class TestThdCpZigzagE2E:
    """End-to-end unshard + reorder tests for THD CP zigzag format.

    Simulates Miles/Megatron forward data splitting:

    cp_size=2, batch with 2 seqs: seqA(100 tokens), seqB(61→pad to 64)

    Forward:
      seqA(100): chunk_size=25, 4 chunks → rank0=[chunk0+chunk3](50), rank1=[chunk1+chunk2](50)
      seqB(64):  chunk_size=16, 4 chunks → rank0=[chunk0+chunk3](32), rank1=[chunk1+chunk2](32)
      global pad → align to 128
      rank0: [seqA_r0(50) | seqB_r0(32) | pad(46)] = 128 tokens
      rank1: [seqA_r1(50) | seqB_r1(32) | pad(46)] = 128 tokens
      global cu_seqlens: [0, 100, 164, 256]

    Comparator undo:
      Step 1 THD unshard: per-seq cross-rank concat → [seqA_zigzag(100) | seqB_zigzag(64) | pad(92)]
      Step 2 THD reorder: per-seq zigzag→natural → [seqA_natural(100) | seqB_natural(64) | pad(92)]
    """

    def test_thd_cp2_two_seqs(self) -> None:
        """cp_size=2, 2 seqs (100, 61→64) + global pad."""
        torch.manual_seed(42)
        cp_size: int = 2
        total_per_rank: int = 128

        seq_a_natural = torch.randn(100)
        seq_b_natural_raw = torch.randn(61)
        seq_b_padded = torch.cat([seq_b_natural_raw, torch.zeros(3)])  # pad 61→64

        seq_a_ranks: list[torch.Tensor] = _zigzag_split_seq(
            seq_a_natural, cp_size=cp_size
        )
        seq_b_ranks: list[torch.Tensor] = _zigzag_split_seq(
            seq_b_padded, cp_size=cp_size
        )

        # Build per-rank tensors: [seqA_r | seqB_r | pad_r]
        rank_tensors: list[torch.Tensor] = []
        for rank in range(cp_size):
            used: int = seq_a_ranks[rank].shape[0] + seq_b_ranks[rank].shape[0]
            pad_len: int = total_per_rank - used
            rank_tensor: torch.Tensor = torch.cat(
                [seq_a_ranks[rank], seq_b_ranks[rank], torch.zeros(pad_len)]
            ).refine_names("t")
            rank_tensors.append(rank_tensor)

        # Step 1: THD unshard
        seq_lens_per_rank: list[int] = [50, 32, 46]
        unshard_plan = UnsharderPlan(
            axis=ParallelAxis.CP,
            params=CpThdConcatParams(dim_name="t", seq_lens_per_rank=seq_lens_per_rank),
            groups=[[0, 1]],
        )
        with warning_sink.context():
            unsharded: list[torch.Tensor] = execute_unsharder_plan(
                unshard_plan, rank_tensors
            )
        assert len(unsharded) == 1

        # Step 2: THD reorder
        reorder_seq_lens: list[int] = [s * cp_size for s in seq_lens_per_rank]
        reorder_plan = ReordererPlan(
            params=ZigzagToNaturalThdParams(
                dim_name="t", cp_size=cp_size, seq_lens=reorder_seq_lens
            )
        )
        reordered: list[torch.Tensor] = execute_reorderer_plan(reorder_plan, unsharded)
        assert len(reordered) == 1

        result: torch.Tensor = reordered[0].rename(None)
        assert torch.equal(result[:100], seq_a_natural)
        assert torch.equal(result[100:164], seq_b_padded)

    def test_thd_cp3_single_seq(self) -> None:
        """cp_size=3, single seq (120 tokens)."""
        torch.manual_seed(42)
        cp_size: int = 3
        seq_natural = torch.randn(120)

        seq_ranks: list[torch.Tensor] = _zigzag_split_seq(seq_natural, cp_size=cp_size)

        rank_tensors: list[torch.Tensor] = [t.refine_names("t") for t in seq_ranks]

        # Step 1: THD unshard
        seq_len_per_rank: int = 120 // cp_size  # 40
        unshard_plan = UnsharderPlan(
            axis=ParallelAxis.CP,
            params=CpThdConcatParams(
                dim_name="t", seq_lens_per_rank=[seq_len_per_rank]
            ),
            groups=[list(range(cp_size))],
        )
        with warning_sink.context():
            unsharded: list[torch.Tensor] = execute_unsharder_plan(
                unshard_plan, rank_tensors
            )
        assert len(unsharded) == 1

        # Step 2: THD reorder
        reorder_plan = ReordererPlan(
            params=ZigzagToNaturalThdParams(
                dim_name="t", cp_size=cp_size, seq_lens=[120]
            )
        )
        reordered: list[torch.Tensor] = execute_reorderer_plan(reorder_plan, unsharded)
        assert len(reordered) == 1

        result: torch.Tensor = reordered[0].rename(None)
        assert torch.equal(result, seq_natural)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
