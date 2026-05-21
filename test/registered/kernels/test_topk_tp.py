import os
import unittest
from typing import Callable, Dict

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

try:
    from sgl_kernel import (
        fast_topk_transform_fused,
        fast_topk_transform_ragged_fused,
        fast_topk_v2,
    )
except ImportError as e:
    fast_topk_transform_fused = None
    fast_topk_transform_ragged_fused = None
    fast_topk_v2 = None
    _SGL_KERNEL_IMPORT_ERROR = e
else:
    _SGL_KERNEL_IMPORT_ERROR = None


register_cuda_ci(est_time=60, suite="stage-b-kernel-unit-1-gpu-large")

BATCH_SIZE = 64
# NSA's fast top-k kernels are specialized to return 2048 sparse indices; this
# test ranks over 131072 synthetic candidates to stress the large-input path.
TOPK_INPUT_LEN = 131072
INDEX_TOPK = 2048
TIE_CANDIDATES = 8192
GUARD_CANDIDATES = 64
ROW_START_SPAN = 512
TP_SIZES = (2, 4, 8)


def _parse_env_bool(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "y")


def _is_dsa_topk_broadcast_enabled() -> bool:
    env_value = os.getenv("SGLANG_DSA_TOPK_BROADCAST")
    if env_value is not None:
        return _parse_env_bool(env_value)

    broadcast_env = getattr(envs, "SGLANG_DSA_TOPK_BROADCAST", None)
    return broadcast_env is not None and broadcast_env.get()


skip_if_dsa_topk_broadcast = unittest.skipIf(
    _is_dsa_topk_broadcast_enabled(),
    "Raw top-k TP stability diagnostic is skipped when RK0 broadcast is enabled.",
)


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestNSATopkTPStability(CustomTestCase):
    def test_stable_topk_reference_across_tp_partitions(self):
        score, lengths, row_starts = self._make_tie_heavy_score()

        outputs = self._run_for_tp_sizes(
            self._stable_topk_indices,
            score,
            lengths,
            row_starts,
        )

        self._assert_outputs_match_across_tps(outputs, "stable topk reference")

    def test_stable_paged_transform_reference_across_tp_partitions(self):
        score, lengths, row_starts = self._make_tie_heavy_score()
        page_table = self._make_page_table()

        def run_chunk(score_chunk, lengths_chunk, row_starts_chunk, chunk_start):
            indices = self._stable_topk_indices(
                score_chunk, lengths_chunk, row_starts_chunk
            )
            return torch.gather(
                page_table[chunk_start : chunk_start + score_chunk.shape[0]],
                dim=1,
                index=indices.long(),
            ).to(torch.int32)

        outputs = self._run_for_tp_sizes(
            run_chunk,
            score,
            lengths,
            row_starts,
            needs_chunk_start=True,
        )

        self._assert_outputs_match_across_tps(
            outputs, "stable paged transform reference"
        )

    def test_stable_ragged_transform_reference_across_tp_partitions(self):
        score, lengths, row_starts = self._make_tie_heavy_score()
        topk_indices_offset = (
            torch.arange(BATCH_SIZE, dtype=torch.int32, device="cuda") * TOPK_INPUT_LEN
        )

        def run_chunk(score_chunk, lengths_chunk, row_starts_chunk, chunk_start):
            indices = self._stable_topk_indices(
                score_chunk, lengths_chunk, row_starts_chunk
            )
            chunk_size = score_chunk.shape[0]
            offsets = topk_indices_offset[chunk_start : chunk_start + chunk_size]
            return indices + offsets.unsqueeze(1)

        outputs = self._run_for_tp_sizes(
            run_chunk,
            score,
            lengths,
            row_starts,
            needs_chunk_start=True,
        )

        self._assert_outputs_match_across_tps(
            outputs, "stable ragged transform reference"
        )

    @skip_if_dsa_topk_broadcast
    def test_fast_topk_stable_across_tp_partitions(self):
        self._skip_if_sgl_kernel_unavailable()
        score, lengths, row_starts = self._make_tie_heavy_score()

        outputs = self._run_for_tp_sizes(
            lambda s, l, r: fast_topk_v2(s, l, INDEX_TOPK, row_starts=r),
            score,
            lengths,
            row_starts,
        )

        self._assert_outputs_match_across_tps(outputs, "fast_topk_v2")

    @skip_if_dsa_topk_broadcast
    def test_paged_transform_stable_across_tp_partitions(self):
        self._skip_if_sgl_kernel_unavailable()
        score, lengths, row_starts = self._make_tie_heavy_score()
        page_table = self._make_page_table()

        def run_chunk(score_chunk, lengths_chunk, row_starts_chunk, chunk_start):
            chunk_size = score_chunk.shape[0]
            cu_seqlens_q = torch.arange(
                0, chunk_size + 1, dtype=torch.int32, device=score_chunk.device
            )
            return fast_topk_transform_fused(
                score=score_chunk,
                lengths=lengths_chunk,
                page_table_size_1=page_table[chunk_start : chunk_start + chunk_size],
                cu_seqlens_q=cu_seqlens_q,
                topk=INDEX_TOPK,
                row_starts=row_starts_chunk,
            )

        outputs = self._run_for_tp_sizes(
            run_chunk,
            score,
            lengths,
            row_starts,
            needs_chunk_start=True,
        )

        self._assert_outputs_match_across_tps(outputs, "paged topk transform")

    @skip_if_dsa_topk_broadcast
    def test_ragged_transform_stable_across_tp_partitions(self):
        self._skip_if_sgl_kernel_unavailable()
        score, lengths, row_starts = self._make_tie_heavy_score()
        topk_indices_offset = (
            torch.arange(BATCH_SIZE, dtype=torch.int32, device="cuda") * TOPK_INPUT_LEN
        )

        def run_chunk(score_chunk, lengths_chunk, row_starts_chunk, chunk_start):
            chunk_size = score_chunk.shape[0]
            return fast_topk_transform_ragged_fused(
                score=score_chunk,
                lengths=lengths_chunk,
                topk_indices_offset=topk_indices_offset[
                    chunk_start : chunk_start + chunk_size
                ],
                topk=INDEX_TOPK,
                row_starts=row_starts_chunk,
            )

        outputs = self._run_for_tp_sizes(
            run_chunk,
            score,
            lengths,
            row_starts,
            needs_chunk_start=True,
        )

        self._assert_outputs_match_across_tps(outputs, "ragged topk transform")

    def _skip_if_sgl_kernel_unavailable(self):
        if _SGL_KERNEL_IMPORT_ERROR is not None:
            self.skipTest(f"sgl_kernel is unavailable: {_SGL_KERNEL_IMPORT_ERROR}")

    def _make_tie_heavy_score(self):
        device = "cuda"
        score_width = TOPK_INPUT_LEN + ROW_START_SPAN
        score = torch.full(
            (BATCH_SIZE, score_width), -10000.0, dtype=torch.float32, device=device
        )
        lengths = torch.full(
            (BATCH_SIZE,), TOPK_INPUT_LEN, dtype=torch.int32, device=device
        )
        row_starts = (
            torch.arange(BATCH_SIZE, dtype=torch.int32, device=device) * 37
        ) % ROW_START_SPAN

        low_scores = -1000.0 - (
            torch.arange(TOPK_INPUT_LEN, dtype=torch.float32, device=device) % 997
        )
        guard_scores = 2.0 + torch.arange(
            GUARD_CANDIDATES, dtype=torch.float32, device=device
        )

        for row in range(BATCH_SIZE):
            start = int(row_starts[row].item())
            window = score[row, start : start + TOPK_INPUT_LEN]
            window.copy_(low_scores)

            tie_start = 4096 + (row * 257) % 4096
            guard_start = tie_start - GUARD_CANDIDATES
            window[guard_start:tie_start] = guard_scores
            window[tie_start : tie_start + TIE_CANDIDATES] = 1.0

        return score, lengths, row_starts

    def _make_page_table(self):
        page_table = torch.arange(TOPK_INPUT_LEN, dtype=torch.int32, device="cuda")
        page_table = page_table.unsqueeze(0).repeat(BATCH_SIZE, 1)
        row_offsets = (
            torch.arange(BATCH_SIZE, dtype=torch.int32, device="cuda").unsqueeze(1)
            * TOPK_INPUT_LEN
        )
        return page_table + row_offsets

    def _stable_topk_indices(
        self, score: torch.Tensor, lengths: torch.Tensor, row_starts: torch.Tensor
    ) -> torch.Tensor:
        self.assertTrue(torch.all(lengths == TOPK_INPUT_LEN))
        offsets = torch.arange(TOPK_INPUT_LEN, dtype=torch.long, device=score.device)
        gather_indices = row_starts.long().unsqueeze(1) + offsets.unsqueeze(0)
        score_windows = torch.gather(score, dim=1, index=gather_indices)
        return torch.argsort(score_windows, dim=-1, descending=True, stable=True)[
            :, :INDEX_TOPK
        ].to(torch.int32)

    def _run_for_tp_sizes(
        self,
        run_chunk: Callable,
        score: torch.Tensor,
        lengths: torch.Tensor,
        row_starts: torch.Tensor,
        needs_chunk_start: bool = False,
    ) -> Dict[int, torch.Tensor]:
        outputs = {}
        for tp_size in TP_SIZES:
            self.assertEqual(BATCH_SIZE % tp_size, 0)
            chunk_size = BATCH_SIZE // tp_size
            chunks = []
            for chunk_start in range(0, BATCH_SIZE, chunk_size):
                chunk_end = chunk_start + chunk_size
                args = (
                    score[chunk_start:chunk_end],
                    lengths[chunk_start:chunk_end],
                    row_starts[chunk_start:chunk_end],
                )
                if needs_chunk_start:
                    chunks.append(run_chunk(*args, chunk_start))
                else:
                    chunks.append(run_chunk(*args))
            outputs[tp_size] = torch.cat(chunks, dim=0)
            torch.cuda.synchronize()
        return outputs

    def _assert_outputs_match_across_tps(
        self, outputs: Dict[int, torch.Tensor], label: str
    ):
        baseline_tp = TP_SIZES[0]
        baseline = outputs[baseline_tp]
        self.assertEqual(baseline.shape, (BATCH_SIZE, INDEX_TOPK))
        self.assertEqual(baseline.dtype, torch.int32)

        for tp_size in TP_SIZES[1:]:
            candidate = outputs[tp_size]
            self.assertEqual(candidate.shape, baseline.shape)
            mismatches = torch.nonzero(candidate != baseline)
            if mismatches.numel() == 0:
                continue

            first = mismatches[0].tolist()
            row, col = first
            self.fail(
                f"{label} differs between TP={baseline_tp} and TP={tp_size}: "
                f"first mismatch at row={row}, col={col}, "
                f"TP={baseline_tp} value={baseline[row, col].item()}, "
                f"TP={tp_size} value={candidate[row, col].item()}, "
                f"total mismatches={mismatches.shape[0]}"
            )


if __name__ == "__main__":
    unittest.main()
