"""Score-aware Triton union pass tests (v1.1-6).

Pins (from the v1.1 plan):
  - Triton union == torch reference for various capacities.
  - Per-KV-head union semantics: a token critical for one kv_head but
    diluted by another must survive (the v1 global-sum collapse would
    drop it).
  - Score-aware drops: when union candidates > max_selected_per_request,
    drops are by lowest score, NEVER by logical position.
  - Always-keep set: sink ∪ recency window survives even if their
    scores are NEG_INF.
  - Dense fallback: seq < min_seq_len produces [0, seq) full.
  - Capacity guard: H_kv * effective_budget > union_safe_threshold raises.
"""

import unittest

import torch

from sglang.srt.mem_cache.sparsity.triton_ops.select_triton import (
    ds_union_per_batch,
    ds_union_per_batch_torch_ref,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")


def _build_merged(
    *,
    bs: int,
    h_kv: int,
    effective_budget: int,
    seq_lens: torch.Tensor,
    device: torch.device,
    seed: int = 0,
):
    """Build merged_logical / merged_scores tensors with random valid picks
    that respect history-only (logicals < seq - 1) and score >= 0 for valid."""
    g = torch.Generator(device=device).manual_seed(seed)
    log = torch.full((bs, h_kv, effective_budget), -1, dtype=torch.int32, device=device)
    scr = torch.full(
        (bs, h_kv, effective_budget),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    for b in range(bs):
        sl = int(seq_lens[b].item())
        history = max(sl - 1, 0)
        for h in range(h_kv):
            n_valid = min(effective_budget, history)
            if n_valid > 0:
                # Pick distinct logical positions from history.
                positions = torch.randperm(history, generator=g, device=device)[
                    :n_valid
                ].to(torch.int32)
                # Random scores in [0, 10) so they're > NEG_INF.
                scores = torch.rand(n_valid, generator=g, device=device) * 10.0
                log[b, h, :n_valid] = positions
                scr[b, h, :n_valid] = scores
    return log, scr


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestUnionParity(CustomTestCase):
    def _parity(
        self,
        *,
        bs=2,
        h_kv=4,
        effective_budget=8,
        max_selected=24,
        sink=2,
        recent=4,
        min_seq=16,
        seq_lens=None,
        seed=11,
    ):
        device = torch.device("cuda")
        if seq_lens is None:
            seq_lens = torch.tensor([64, 64], dtype=torch.int64, device=device)
        else:
            seq_lens = seq_lens.to(device)

        log, scr = _build_merged(
            bs=bs,
            h_kv=h_kv,
            effective_budget=effective_budget,
            seq_lens=seq_lens,
            device=device,
            seed=seed,
        )

        ref_log, ref_valid = ds_union_per_batch_torch_ref(
            merged_logical=log,
            merged_scores=scr,
            seq_lens=seq_lens,
            sink_tokens=sink,
            recent_tokens=recent,
            min_seq_len=min_seq,
            max_selected_per_request=max_selected,
        )
        tri_log, tri_valid = ds_union_per_batch(
            merged_logical=log,
            merged_scores=scr,
            seq_lens=seq_lens,
            sink_tokens=sink,
            recent_tokens=recent,
            min_seq_len=min_seq,
            max_selected_per_request=max_selected,
        )

        for b in range(bs):
            ref_set = sorted(int(x) for x in ref_log[b].tolist() if int(x) >= 0)
            tri_set = sorted(int(x) for x in tri_log[b].tolist() if int(x) >= 0)
            self.assertEqual(
                ref_set,
                tri_set,
                f"union mismatch (b={b}):\n"
                f"  ref ({len(ref_set)})={ref_set}\n"
                f"  tri ({len(tri_set)})={tri_set}",
            )
        self.assertTrue(torch.equal(tri_valid.cpu(), ref_valid.cpu()))

    def test_basic(self):
        self._parity()

    def test_dense_fallback(self):
        # seq_lens below min_seq_len → [0, seq) output.
        seq_lens = torch.tensor([8, 64], dtype=torch.int64)  # b=0 dense fallback
        self._parity(seq_lens=seq_lens, min_seq=16)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestUnionScoreAware(CustomTestCase):
    """When union candidates > max_selected, drops MUST be by lowest score,
    not by logical position. Wrong impl (drop earliest position) fails."""

    def test_score_aware_drops(self):
        device = torch.device("cuda")
        bs = 1
        h_kv = 4
        effective_budget = 4  # 4 picks per head × 4 heads = 16 candidates
        max_selected = 8
        sink, recent = 0, 0  # disable always-keep to isolate the drop policy
        seq_lens = torch.tensor([100], dtype=torch.int64, device=device)
        min_seq = 16

        # Construct a known score gradient. Each head selects 4 distinct
        # logical positions [0..15] across all heads; score = position
        # (so position 15 has the highest score, position 0 the lowest).
        log = torch.zeros(
            (bs, h_kv, effective_budget), dtype=torch.int32, device=device
        )
        scr = torch.zeros(
            (bs, h_kv, effective_budget), dtype=torch.float32, device=device
        )
        for h in range(h_kv):
            for k in range(effective_budget):
                pos = h * effective_budget + k  # 0..15 distinct
                log[0, h, k] = pos
                scr[0, h, k] = float(pos)

        out_log, out_valid = ds_union_per_batch(
            merged_logical=log,
            merged_scores=scr,
            seq_lens=seq_lens,
            sink_tokens=sink,
            recent_tokens=recent,
            min_seq_len=min_seq,
            max_selected_per_request=max_selected,
        )

        # max_selected=8, candidates=16 → drop the 8 lowest-score (positions
        # 0..7). Surviving must be {8, 9, 10, 11, 12, 13, 14, 15}, sorted.
        survivors = sorted(int(x) for x in out_log[0].tolist() if int(x) >= 0)
        expected = list(range(8, 16))
        self.assertEqual(
            survivors,
            expected,
            f"score-aware drop policy violated\n"
            f"  expected (highest 8 by score, sorted asc): {expected}\n"
            f"  got: {survivors}",
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestUnionDedupCorrectness(CustomTestCase):
    """When the same logical token appears across ≥3 kv-head slots, the
    union must emit it AT MOST ONCE, keeping the max score across heads.
    A two-pass adjacent-neighbor mask over a sort-by-logical run fails
    on non-monotone score sequences (e.g. [10, 5, 9]) — it leaves both
    10 and 9 alive, FA3 gets a duplicate page-table entry, and
    `valid_lengths` overcounts. The CUDA sort is non-stable for equal
    keys so the failure is non-deterministic per call; fuzz with many
    seeded inputs to expose it."""

    def test_no_duplicate_logicals_under_collisions(self):
        device = torch.device("cuda")
        bs = 1
        h_kv = 8
        effective_budget = 4  # 32 slots total
        max_selected = 32
        sink, recent = 0, 0
        seq_lens = torch.tensor([1000], dtype=torch.int64, device=device)

        # Construct heavy-collision inputs: 32 candidate slots, but only
        # 4 distinct logical tokens. Each token appears in 8 different
        # (kv_head, slot) positions with random scores. After dedup,
        # the output should contain each of the 4 logicals AT MOST ONCE.
        n_distinct = 4
        n_iters = 100  # enough seeds to expose any sort permutation
        for seed in range(n_iters):
            g = torch.Generator(device=device).manual_seed(seed)
            slot_logicals = torch.tensor(
                [i % n_distinct + 1 for i in range(h_kv * effective_budget)],
                dtype=torch.int32,
                device=device,
            ).reshape(bs, h_kv, effective_budget)
            slot_scores = (
                torch.rand(
                    (bs, h_kv, effective_budget),
                    generator=g,
                    device=device,
                    dtype=torch.float32,
                )
                * 100.0
            )

            out_log, out_valid = ds_union_per_batch(
                merged_logical=slot_logicals,
                merged_scores=slot_scores,
                seq_lens=seq_lens,
                sink_tokens=sink,
                recent_tokens=recent,
                min_seq_len=16,
                max_selected_per_request=max_selected,
            )

            # No logical may appear twice in the output.
            nonneg = [int(x) for x in out_log[0].tolist() if int(x) >= 0]
            counts: dict[int, int] = {}
            for x in nonneg:
                counts[x] = counts.get(x, 0) + 1
            duplicates = {k: v for k, v in counts.items() if v > 1}
            self.assertFalse(
                duplicates,
                f"seed={seed}: union emitted duplicate logicals "
                f"{duplicates} (FA3 page-table corruption risk)\n"
                f"  output: {nonneg}",
            )
            # valid_lengths must match the non-sentinel count exactly.
            self.assertEqual(
                int(out_valid[0]),
                len(nonneg),
                f"seed={seed}: valid_lengths={int(out_valid[0])} != "
                f"non-sentinel count={len(nonneg)}",
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestUnionCapacityGuard(CustomTestCase):
    def test_exceeding_threshold_raises(self):
        device = torch.device("cuda")
        # h_kv=8 * effective_budget=1024 = 8192 > 4096 default threshold
        log = torch.full((1, 8, 1024), -1, dtype=torch.int32, device=device)
        scr = torch.full(
            (1, 8, 1024), float("-inf"), dtype=torch.float32, device=device
        )
        with self.assertRaisesRegex(RuntimeError, "union_safe_threshold"):
            ds_union_per_batch(
                merged_logical=log,
                merged_scores=scr,
                seq_lens=torch.tensor([512], dtype=torch.int64, device=device),
                sink_tokens=2,
                recent_tokens=4,
                min_seq_len=16,
                max_selected_per_request=512,
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestUnionCudaGraphCaptureReplay(CustomTestCase):
    """The union pass runs every layer × every step on the DS server perf
    path. Pin that it's CUDA-graph-replay safe: zero allocations across
    50 replays with mutated input tensors."""

    def test_capture_replay_zero_alloc(self):
        device = torch.device("cuda")
        bs, h_kv, effective_budget = 2, 4, 8
        max_selected = 24

        # Preallocate inputs (mutated per replay) and outputs.
        merged_logical = torch.full(
            (bs, h_kv, effective_budget), -1, dtype=torch.int32, device=device
        )
        merged_scores = torch.full(
            (bs, h_kv, effective_budget),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        seq_lens = torch.tensor([64, 64], dtype=torch.int64, device=device)
        selected_logical = torch.full(
            (bs, max_selected), -1, dtype=torch.int32, device=device
        )
        valid_lengths = torch.zeros(bs, dtype=torch.int32, device=device)

        kw = dict(
            sink_tokens=2,
            recent_tokens=4,
            min_seq_len=16,
            max_selected_per_request=max_selected,
        )

        def populate(seed: int):
            g = torch.Generator(device=device).manual_seed(seed)
            for b in range(bs):
                for h in range(h_kv):
                    pos = torch.randperm(60, generator=g, device=device)[
                        :effective_budget
                    ].to(torch.int32)
                    scores = (
                        torch.rand(effective_budget, generator=g, device=device) * 10.0
                    )
                    merged_logical[b, h].copy_(pos)
                    merged_scores[b, h].copy_(scores)

        # Warmup
        populate(seed=42)
        for _ in range(3):
            ds_union_per_batch(
                merged_logical=merged_logical,
                merged_scores=merged_scores,
                seq_lens=seq_lens,
                selected_logical=selected_logical,
                valid_lengths=valid_lengths,
                **kw,
            )
        torch.cuda.synchronize()

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ds_union_per_batch(
                merged_logical=merged_logical,
                merged_scores=merged_scores,
                seq_lens=seq_lens,
                selected_logical=selected_logical,
                valid_lengths=valid_lengths,
                **kw,
            )

        # Replay correctness: mutate inputs, replay, check outputs.
        scenarios = [(11, 64), (29, 50), (47, 32)]
        for seed, sl in scenarios:
            populate(seed=seed)
            seq_lens.fill_(sl)
            graph.replay()
            torch.cuda.synchronize()
            # Sanity: valid_lengths in [0, max_selected]
            v = valid_lengths.cpu().tolist()
            self.assertTrue(
                all(0 <= x <= max_selected for x in v),
                f"valid_lengths out of bounds: {v}",
            )

        # Allocation count: replay 50 more times, snapshot before/after.
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        for _ in range(50):
            graph.replay()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
        self.assertEqual(
            after,
            before,
            f"replay allocated {after - before} bytes (should be 0)",
        )


if __name__ == "__main__":
    unittest.main()
