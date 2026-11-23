"""Kimi-K3 DSPARK prefix capture over gfx950's row-sharded stream."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.layer_boundary.residual import attn_bank, attn_bank_hip
from sglang.srt.models import kimi_k3
from sglang.srt.models.kimi_k3_amd import kimi_k3_hip
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

WORLD = 8
# 9 tokens over 8 ranks: the row-sharded tail leaves 2 rows on ranks 0-3, 1 on rank 4
# and none on ranks 5-7.
TOTAL = 9
HIDDEN = 4


class _EqualSizeGroup:
    """Attention-TP group stub holding every rank's stream rows; like RCCL, its
    gathers require the same row count on every rank."""

    world_size = WORLD

    def __init__(self, full: torch.Tensor, rank: int):
        self.full = full
        self.rank = rank

    def _rank_rows(self, rank: int, width: int) -> torch.Tensor:
        rows = self.full[rank * width : (rank + 1) * width]
        return torch.nn.functional.pad(rows, (0, 0, 0, width - rows.shape[0]))

    def all_gather_into_tensor(self, out: torch.Tensor, inp: torch.Tensor) -> None:
        if out.shape[0] != inp.shape[0] * WORLD:
            raise RuntimeError("all-gather needs equal sizes on every rank")
        # This rank must send its own rows, padded; the others' come from `full`.
        if not torch.equal(inp, self._rank_rows(self.rank, inp.shape[0])):
            raise RuntimeError("this rank sent rows other than its own shard")
        out.copy_(torch.cat([self._rank_rows(r, inp.shape[0]) for r in range(WORLD)]))

    def all_gather(self, inp: torch.Tensor, dim: int = 0) -> torch.Tensor:
        width = -(-TOTAL // WORLD)
        if inp.shape[0] != width:
            raise RuntimeError("all-gather needs equal sizes on every rank")
        if not torch.equal(inp, self._rank_rows(self.rank, width)):
            raise RuntimeError("this rank sent rows other than its own shard")
        return torch.cat([self._rank_rows(r, width) for r in range(WORLD)], dim=dim)


class TestDSparkPrefixCaptureRowShards(CustomTestCase):
    def _capture(self, rank: int) -> torch.Tensor:
        full = torch.arange(TOTAL * HIDDEN, dtype=torch.float32).view(TOTAL, HIDDEN)
        start, valid = attn_bank_hip.row_shard(TOTAL, WORLD, rank)
        parallel = SimpleNamespace(
            attn_tp_size=WORLD,
            attn_tp_rank=rank,
            attn_tp_group=_EqualSizeGroup(full, rank),
        )
        bank = SimpleNamespace(block_residual=torch.zeros(TOTAL, 1, HIDDEN))
        model = SimpleNamespace(
            dspark_aux_hidden_stream="prefix",
            attn_bank=SimpleNamespace(require=lambda: bank),
        )
        with (
            envs.SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_STREAM.override(True),
            patch.object(attn_bank, "_is_gfx95", True),
            patch.object(attn_bank, "get_parallel", return_value=parallel),
            patch.object(attn_bank_hip, "get_parallel", return_value=parallel),
            patch.object(kimi_k3, "get_parallel", return_value=parallel),
            patch.object(kimi_k3, "_is_gfx95", True),
            patch.object(
                kimi_k3,
                "gather_stream_rows_hip",
                kimi_k3_hip.gather_stream_rows_hip,
                create=True,
            ),
            patch.object(kimi_k3.residual_batch, "snapshot", lambda h, fb: h),
        ):
            captured = kimi_k3.KimiK3LinearModel._dspark_capture_stream(
                model, 0, full[start : start + valid], forward_batch=None
            )
        return captured, full

    def test_short_and_empty_row_shards_capture_every_row(self):
        """A prefix capture of a row-sharded stream whose token count does not divide
        by attention TP must still hand the draft every row, in order, on every rank."""
        for rank in (0, 4, 5, 7):
            with self.subTest(rank=rank):
                captured, full = self._capture(rank)
                self.assertTrue(torch.equal(captured, full))


if __name__ == "__main__":
    unittest.main()
