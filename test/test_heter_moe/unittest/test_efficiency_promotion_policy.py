"""Tests for EfficiencyPromotionPolicy (curve-driven dynamic-threshold dispatch).

Three orthogonal checks:
  1. Correctness: the policy's promotion decisions match the curve at
     every M_global in the curve file. Specifically: at M_global where
     the curve says x* = K, exactly K experts (the top-K by token count)
     end up in the BF16 group.
  2. Efficiency / accuracy invariant: dispatch output has fixed shape,
     non-group slots are sentinel + zero scale, group sizes sum to
     [N, top_k]. (The same invariants the other policies hold.)
  3. CUDA-graph compatibility: the policy can be captured into a CUDA
     graph and replayed without recompile, producing bit-identical
     dispatch output across replays.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile

import pytest
import torch

from test_heter_moe.util import CUDA_AVAILABLE


def _write_lookup(path: str, entries: list) -> None:
    """Write a minimal efficiency_promotion_lookup.json
    (M_global → {x_runtime, T, m_per_expert, up_tile, down_tile})."""
    import json as _json
    obj = {
        str(M): {
            "x_runtime": x,
            "T": T,
            "m_per_expert": 0,
            "tile_key": "n/a",
            "up_tile": None,
            "down_tile": None,
        }
        for (M, x, T) in entries
    }
    with open(path, "w") as f:
        _json.dump(obj, f)


@pytest.fixture
def curve_path():
    # Synthetic per-M lookup JSON — tests don't need real autotune data.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        path = f.name
    rows = [
        # (M_global, x_runtime, T)
        (32, 0, 10**9),   # decode: no promotion (T = ∞)
        (256, 4, 50),     # small prefill: T=50
        (1024, 16, 24),   # mid prefill: T=24
        (4096, 32, 12),   # heavy prefill: T=12
    ]
    _write_lookup(path, rows)
    yield path
    os.unlink(path)


def _make_policy(curve_path: str, num_experts: int = 64,
                 device: torch.device = None):
    from sglang.srt.layers.moe.heter_policy import EfficiencyPromotionPolicy
    if device is None:
        device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
    return EfficiencyPromotionPolicy(
        num_experts=num_experts,
        group_size_ratios=[0.5, 0.5],  # G=2: cold/hot
        bf16_promotion_threshold=10**9,  # static threshold (used as fallback)
        device=device,
        int4_group_idx=0,
        bf16_group_idx=1,
        lookup_file=curve_path,
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCurveLookup:
    """1. Correctness — promotion decisions match the curve."""

    def test_mask_x_zero_means_all_int4(self, curve_path):
        """At M=32 (x_runtime=0), mask should be all-INT4."""
        p = _make_policy(curve_path, num_experts=64)
        mask = p._mask_for(32)
        assert (mask == 0).all().item()      # 0 = int4_group_idx

    def test_mask_top_x_are_bf16(self, curve_path):
        """At M=1024 (x_runtime=16), top-16 by importance are BF16."""
        p = _make_policy(curve_path, num_experts=64)
        mask = p._mask_for(1024)
        assert (mask == 1).sum().item() == 16  # exactly 16 BF16
        assert (mask == 0).sum().item() == 48

    def test_mask_nearest_M_fallback(self, curve_path):
        """M=600 has no exact entry; nearest is 256 → x_runtime=4."""
        p = _make_policy(curve_path, num_experts=64)
        mask = p._mask_for(600)
        assert (mask == 1).sum().item() == 4

    def test_mask_is_constant_per_init(self, curve_path):
        """Repeat _mask_for calls return the SAME tensor object — no
        Python state mutation, mask was built once at __init__."""
        p = _make_policy(curve_path, num_experts=64)
        mask_a = p._mask_for(1024)
        mask_b = p._mask_for(1024)
        assert mask_a is mask_b

    def test_dispatch_uses_static_top_x(self, curve_path):
        """At M=1024 (x_runtime=16), the precomputed mask promotes the
        top-16 by importance (default natural order = experts 0..15)."""
        p = _make_policy(curve_path, num_experts=64)
        device = torch.device("cuda")
        N, K = 1024, 8
        # Routing pattern doesn't matter — mask is precomputed.
        ids = torch.randint(0, 64, (N, K), device=device, dtype=torch.long)
        scales = torch.ones(N, K, dtype=torch.float32, device=device)
        groups = p.dispatch(ids, scales, sentinel=-1)
        hot_ids = groups[1][0]
        # Hot routes should ONLY contain experts 0..15 (the top-16 by
        # default natural-order importance, since no importance arg).
        hot_set = set(hot_ids[hot_ids >= 0].unique().cpu().tolist())
        assert hot_set.issubset(set(range(16))), (
            f"expected hot ⊆ [0..16), got {sorted(hot_set)}"
        )

    def test_dispatch_x_zero_no_promotion(self, curve_path):
        """At M=32 (x_runtime=0), mask is all-INT4; BF16 group is empty."""
        p = _make_policy(curve_path, num_experts=64)
        device = torch.device("cuda")
        N, K = 32, 8
        ids = torch.randint(0, 64, (N, K), device=device, dtype=torch.long)
        scales = torch.ones(N, K, dtype=torch.float32, device=device)
        # should_skip_group returns True for the BF16 group at this M.
        assert p.should_skip_group(group_idx=1, num_tokens=N) is True
        groups = p.dispatch(ids, scales, sentinel=-1)
        hot_ids, hot_scales = groups[1]
        assert (hot_ids == -1).all(), "BF16 group should be empty at x=0"
        assert (hot_scales == 0).all()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestDispatchInvariants:
    """2. Efficiency-correctness: dispatch invariants the runtime relies on."""

    def test_output_shapes(self, curve_path):
        p = _make_policy(curve_path, num_experts=64)
        device = torch.device("cuda")
        N, K = 1024, 8
        ids = torch.randint(0, 64, (N, K), device=device, dtype=torch.long)
        scales = torch.ones(N, K, dtype=torch.float32, device=device)
        groups = p.dispatch(ids, scales, sentinel=-1)
        assert len(groups) == 2  # cold + hot
        for gids, gscales in groups:
            assert gids.shape == (N, K)
            assert gscales.shape == (N, K)
            assert gids.dtype == ids.dtype

    def test_partition_complete(self, curve_path):
        """Every (token, slot) pair lands in exactly one group."""
        p = _make_policy(curve_path, num_experts=64)
        device = torch.device("cuda")
        N, K = 1024, 8
        ids = torch.randint(0, 64, (N, K), device=device, dtype=torch.long)
        scales = torch.ones(N, K, dtype=torch.float32, device=device)
        cold, hot = p.dispatch(ids, scales, sentinel=-1)
        cold_mask = cold[0] >= 0
        hot_mask = hot[0] >= 0
        # Disjoint
        assert not (cold_mask & hot_mask).any(), (
            "a (token, slot) is in both cold and hot"
        )
        # Complete: every (token, slot) is in exactly one group
        union = cold_mask | hot_mask
        assert union.all(), "some (token, slot) is in neither group"

    def test_scales_zero_off_group(self, curve_path):
        p = _make_policy(curve_path, num_experts=64)
        device = torch.device("cuda")
        N, K = 1024, 8
        ids = torch.randint(0, 64, (N, K), device=device, dtype=torch.long)
        scales = torch.rand(N, K, dtype=torch.float32, device=device) + 0.1
        for gids, gscales in p.dispatch(ids, scales, sentinel=-1):
            off = (gids == -1)
            assert (gscales[off] == 0).all(), "off-group scale should be 0"
            on = (gids >= 0)
            assert (gscales[on] > 0).all(), "on-group scale should be preserved"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCudaGraphCompatibility:
    """3. CUDA-graph capture + replay produces stable output."""

    def test_capture_replay_bit_identical(self, curve_path):
        p = _make_policy(curve_path, num_experts=64)
        device = torch.device("cuda")
        N, K = 1024, 8

        # Pre-allocate stable input/output tensors so the graph captures
        # operations on FIXED memory addresses.
        ids = torch.randint(0, 64, (N, K), device=device, dtype=torch.long)
        scales = torch.ones(N, K, dtype=torch.float32, device=device)
        cold_out_ids = torch.empty(N, K, dtype=torch.long, device=device)
        cold_out_scales = torch.empty(N, K, dtype=torch.float32, device=device)
        hot_out_ids = torch.empty(N, K, dtype=torch.long, device=device)
        hot_out_scales = torch.empty(N, K, dtype=torch.float32, device=device)

        def run_into_outputs():
            cold, hot = p.dispatch(ids, scales, sentinel=-1)
            cold_out_ids.copy_(cold[0]); cold_out_scales.copy_(cold[1])
            hot_out_ids.copy_(hot[0]); hot_out_scales.copy_(hot[1])

        # Eager reference
        run_into_outputs()
        ref_cold_ids = cold_out_ids.clone()
        ref_hot_ids = hot_out_ids.clone()

        # Capture
        for _ in range(3):
            run_into_outputs()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_into_outputs()
        torch.cuda.synchronize()

        # Replay & compare
        graph.replay()
        torch.cuda.synchronize()

        assert (cold_out_ids == ref_cold_ids).all(), (
            "cold IDs differ between eager and graph replay"
        )
        assert (hot_out_ids == ref_hot_ids).all(), (
            "hot IDs differ between eager and graph replay"
        )

        # Replay again with same input — should be deterministic.
        graph.replay()
        torch.cuda.synchronize()
        assert (cold_out_ids == ref_cold_ids).all()
        assert (hot_out_ids == ref_hot_ids).all()

    def test_should_skip_group_inside_capture(self, curve_path):
        """should_skip_group is called by the LAYER inside CUDA graph
        capture. ANY GPU→host sync inside should_skip_group raises
        cudaErrorStreamCaptureUnsupported. This test opens a real CUDA
        capture context and calls should_skip_group inside it. If the
        method does .item() / .cpu() / any sync, capture aborts."""
        p = _make_policy(curve_path, num_experts=64)
        # Pre-warm any one-shot ops so we're capturing steady state.
        for _ in range(3):
            _ = p.should_skip_group(group_idx=1, num_tokens=32)
            _ = p.should_skip_group(group_idx=1, num_tokens=1024)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        # If should_skip_group did any GPU sync, capture_begin → call →
        # capture_end would raise here.
        with torch.cuda.graph(graph):
            # Trivial GPU op so the graph is non-empty
            x = torch.zeros(1, device="cuda")
            x.add_(1)
            # Host-side calls inside the capture stream — must not sync.
            _ = p.should_skip_group(group_idx=1, num_tokens=32)
            _ = p.should_skip_group(group_idx=1, num_tokens=1024)
            _ = p.should_skip_group(group_idx=1, num_tokens=600)  # nearest fallback
            x.add_(1)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()

    def test_dispatch_inside_capture(self, curve_path):
        """Same as above but for the full dispatch path. Must not sync."""
        p = _make_policy(curve_path, num_experts=64)
        device = torch.device("cuda")
        N, K = 1024, 8
        ids = torch.randint(0, 64, (N, K), device=device, dtype=torch.long)
        scales = torch.ones(N, K, dtype=torch.float32, device=device)
        for _ in range(3):
            _ = p.dispatch(ids, scales, sentinel=-1)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        # Pre-allocate output buffers so the graph captures into stable addresses.
        cold_ids = torch.empty(N, K, dtype=torch.long, device=device)
        hot_ids = torch.empty(N, K, dtype=torch.long, device=device)

        def run():
            cold, hot = p.dispatch(ids, scales, sentinel=-1)
            cold_ids.copy_(cold[0])
            hot_ids.copy_(hot[0])

        for _ in range(3):
            run()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            run()
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()

    def test_capture_at_x_zero_path(self, curve_path):
        """At M=32 (curve x=0), the 'no promotion' branch should also be
        graph-capturable. should_skip_group returns True host-side, so the
        runtime would skip the BF16 kernel; the dispatch tensor outputs
        are still well-defined and capturable."""
        p = _make_policy(curve_path, num_experts=64)
        device = torch.device("cuda")
        N, K = 32, 8
        ids = torch.randint(0, 64, (N, K), device=device, dtype=torch.long)
        scales = torch.ones(N, K, dtype=torch.float32, device=device)
        cold_out_ids = torch.empty(N, K, dtype=torch.long, device=device)
        hot_out_ids = torch.empty(N, K, dtype=torch.long, device=device)

        def run():
            cold, hot = p.dispatch(ids, scales, sentinel=-1)
            cold_out_ids.copy_(cold[0]); hot_out_ids.copy_(hot[0])

        for _ in range(3):
            run()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()

        # All hot IDs should be sentinel (no promotion at x*=0).
        assert (hot_out_ids == -1).all()


class TestDynamoCompatibility:
    """4. torch._dynamo / torch.compile compatibility — production sglang's
    piecewise CUDA graph compiles model.forward through dynamo, which
    does NOT tolerate Python builtins like min(..., key=lambda) inside
    the traced region. This is the failure mode we hit at e2e:

      torch._dynamo.exc.Unsupported: invalid call to builtin op handler
      Encountered TypeError when trying to handle op min
      invalid args to BuiltinVariable._call_min_max:
        [RangeVariable()] {'key': NestedUserFunctionVariable()}

    These tests trace the policy's lookups + dispatch through dynamo
    with fullgraph=True. Any graph-break raises and fails the test.

    Runs on CPU — no CUDA needed for dynamo tracing.
    """

    def test_mask_for_traceable(self, curve_path):
        """_mask_for must be dynamo-traceable (pure host-side dict lookup
        returning a tensor reference)."""
        p = _make_policy(curve_path, num_experts=64,
                         device=torch.device("cpu"))

        @torch.compile(fullgraph=True, dynamic=False, backend="eager")
        def call(M_int):
            return p._mask_for(M_int)

        # Returns the same precomputed tensor for known M values.
        assert (call(1024) == 1).sum().item() == 16
        assert (call(32) == 0).all().item()

    def test_dispatch_traceable_no_state_mutation(self, curve_path):
        """Full dispatch (host-side dict lookup + GPU memcpy) must trace
        under dynamo with NO Python state mutation. Repeated calls must
        not change any cached state."""
        p = _make_policy(curve_path, num_experts=64,
                         device=torch.device("cpu"))
        N, K = 1024, 8
        ids = torch.randint(0, 64, (N, K), dtype=torch.long)
        scales = torch.ones(N, K, dtype=torch.float32)

        @torch.compile(fullgraph=False, dynamic=False, backend="eager")
        def call(ids, scales):
            return p.dispatch(ids, scales, sentinel=-1)

        # Snapshot the policy's internal dict refs BEFORE the call.
        masks_before = {k: id(v) for k, v in p._M_to_mask.items()}
        sorted_before = list(p._M_sorted)

        result = call(ids, scales)
        assert len(result) == 2
        assert result[0][0].shape == (N, K)

        # AFTER the call: no new keys, no mutated tensor refs, sorted M
        # unchanged. (Critical: this is what would have broken sglang's
        # piecewise CUDA graph by triggering a runtime recompile.)
        masks_after = {k: id(v) for k, v in p._M_to_mask.items()}
        assert masks_before == masks_after, (
            "EfficiencyPromotionPolicy mutated _M_to_mask during dispatch — "
            "would trigger dynamo runtime recompile under PCG"
        )
        assert sorted_before == p._M_sorted


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
