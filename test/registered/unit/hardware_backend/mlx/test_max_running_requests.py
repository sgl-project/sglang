"""Regression: MlxModelRunnerStub must honor --max-running-requests instead of
hardcoding min(pool // 2, 4096). The stub previously ignored the flag, so MLX
concurrency silently diverged from what the user requested (and from CUDA).

On hybrid / linear-attention models the resolved concurrency must additionally
be backed by the auxiliary-state pool: each running request allocates one slot
out of max_mamba_cache_size, so an unbounded cap made the third request die
with "Not enough MLX auxiliary state slots" (AssertionError inside the
scheduler) when max_running_requests=4 was backed by max_mamba_cache_size=2.

Guards `_resolve_max_running_requests` (mirroring the base resolver's capacity
clamp, dp split, mamba bound, and zero-reject) plus the hybrid `initialize()`
and request-allocation path. MLX-gated because importing the stub pulls in
mlx.core.
"""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"

if _HAS_MLX:
    from sglang.srt.hardware_backend.mlx.model_runner_stub import (
        MLX_AUX_STATE_SIZE_MAX_RUNNING_REQUESTS_RATIO as RATIO,
    )
    from sglang.srt.hardware_backend.mlx.model_runner_stub import (
        MlxModelRunnerStub,
    )

    class _PlainStub(MlxModelRunnerStub):
        # Shadow the base property: a standard attention-only model.
        mambaish_config = None

    class _HybridStub(MlxModelRunnerStub):
        # Shadow the base property: a hybrid / linear-attention model.
        mambaish_config = object()


def _stub(
    max_running_requests,
    max_total_num_tokens,
    dp_size=1,
    hybrid=False,
    max_mamba_cache_size=None,
):
    """A stub carrying only what _resolve_max_running_requests reads."""
    cls = _HybridStub if hybrid else _PlainStub
    stub = cls.__new__(cls)
    stub.server_args = SimpleNamespace(
        max_running_requests=max_running_requests,
        max_mamba_cache_size=max_mamba_cache_size,
    )
    stub.max_total_num_tokens = max_total_num_tokens
    stub.dp_size = dp_size
    return stub


def _hybrid_stub_for_initialize(max_running_requests, max_mamba_cache_size, pool=100):
    """A stub carrying what the real initialize() reads (hybrid path)."""
    stub = _HybridStub.__new__(_HybridStub)
    stub._mlx_pool_size = pool
    stub.dp_size = 1
    stub.server_args = SimpleNamespace(
        enable_memory_saver=False,
        max_running_requests=max_running_requests,
        max_mamba_cache_size=max_mamba_cache_size,
    )
    stub.model_config = SimpleNamespace(
        is_hybrid_swa=False,
        sliding_window_size=None,
        attention_chunk_size=None,
        dtype="float16",
        num_hidden_layers=1,
        num_attention_layers=1,
        context_len=64,
    )
    return stub


def _fake_req():
    return SimpleNamespace(
        req_pool_idx=None,
        inflight_middle_chunks=0,
        kv_committed_len=0,
        mamba_pool_idx=None,
        mamba_ping_pong_track_buffer=None,
    )


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxMaxRunningRequests(CustomTestCase):
    def test_flag_unset_uses_capacity_default(self):
        # No flag -> min(pool // 2, 4096), the previous default.
        self.assertEqual(_stub(None, 1000)._resolve_max_running_requests(), 500)
        self.assertEqual(_stub(None, 100_000)._resolve_max_running_requests(), 4096)

    def test_flag_honored_within_capacity(self):
        # THE REGRESSION: an explicit flag must be honored, not ignored.
        self.assertEqual(_stub(1, 100_000)._resolve_max_running_requests(), 1)
        self.assertEqual(_stub(64, 100_000)._resolve_max_running_requests(), 64)

    def test_flag_split_per_dp_worker(self):
        # Mirrors the base clamp: the requested value is divided across dp workers.
        self.assertEqual(
            _stub(8, 100_000, dp_size=2)._resolve_max_running_requests(), 4
        )

    def test_flag_clamped_to_capacity(self):
        # A flag larger than the KV pool can hold is capped at pool // 2.
        self.assertEqual(_stub(100_000, 1000)._resolve_max_running_requests(), 500)


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxHybridAuxStateBound(CustomTestCase):
    """Hybrid models: concurrency must be backed by the auxiliary-state pool.

    Each running request allocates one auxiliary slot; the pool holds
    max_mamba_cache_size slots with RATIO slots reserved per concurrent
    request (headroom for radix-held snapshots). Without the bound, request
    slots outnumber auxiliary slots and allocation asserts mid-serving.
    """

    def test_requested_flag_bounded_by_aux_capacity(self):
        # requested=8 but the aux pool backs only 16 // RATIO = 4.
        self.assertEqual(
            _stub(
                8, 100_000, hybrid=True, max_mamba_cache_size=4 * RATIO
            )._resolve_max_running_requests(),
            4,
        )

    def test_default_path_also_bounded_by_aux_capacity(self):
        # No --max-running-requests: the capacity default must still be
        # bounded, else the 4096 default overruns a small aux pool the same way.
        self.assertEqual(
            _stub(
                None, 100_000, hybrid=True, max_mamba_cache_size=2 * RATIO
            )._resolve_max_running_requests(),
            2,
        )

    def test_infeasible_aux_capacity_raises_at_startup(self):
        # Reviewer repro: max_running_requests=4, max_mamba_cache_size=2.
        # Bound is 2 // RATIO = 0 -> fail fast instead of asserting mid-serving.
        with self.assertRaisesRegex(RuntimeError, "max_mamba_cache_size"):
            _stub(
                4, 100, hybrid=True, max_mamba_cache_size=2
            )._resolve_max_running_requests()

    def test_aux_flag_unset_is_not_bounded(self):
        # Negative branch: with max_mamba_cache_size unset the pool is sized
        # FROM the resolved cap (RATIO x), so no bound must be applied.
        self.assertEqual(
            _stub(8, 100_000, hybrid=True)._resolve_max_running_requests(), 8
        )

    def test_non_hybrid_ignores_aux_flag(self):
        # Negative branch: a plain attention model never applies the bound.
        self.assertEqual(
            _stub(
                8, 100_000, hybrid=False, max_mamba_cache_size=2
            )._resolve_max_running_requests(),
            8,
        )


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxHybridInitializeAllocation(CustomTestCase):
    """End-to-end hybrid path: real initialize(), real aux-pool allocation."""

    def test_every_request_slot_is_backed_by_an_aux_slot(self):
        # With the bound, every slot the scheduler may fill can allocate its
        # auxiliary slot. Pre-fix, request slot 3 of 4 raised
        # "Not enough MLX auxiliary state slots" (aux pool had 2 slots).
        stub = _hybrid_stub_for_initialize(
            max_running_requests=8, max_mamba_cache_size=2 * RATIO
        )
        stub.initialize()
        self.assertEqual(stub.max_running_requests, 2)
        pool = stub.req_to_token_pool
        for _ in range(stub.max_running_requests):
            self.assertIsNotNone(pool.alloc([_fake_req()]))

    def test_infeasible_config_fails_at_initialize(self):
        # Reviewer repro end-to-end: 4 request slots backed by 2 aux slots
        # must be rejected at startup, not crash on the third allocation.
        stub = _hybrid_stub_for_initialize(
            max_running_requests=4, max_mamba_cache_size=2
        )
        with self.assertRaisesRegex(RuntimeError, "max_mamba_cache_size"):
            stub.initialize()

    def test_default_aux_sizing_uses_shared_ratio(self):
        # Drift guard: with the flag unset, initialize() sizes the aux pool
        # from the resolved cap with the SAME ratio the bound uses, so the
        # sizing and the bound cannot diverge.
        stub = _hybrid_stub_for_initialize(
            max_running_requests=2, max_mamba_cache_size=None
        )
        stub.initialize()
        self.assertEqual(stub.max_running_requests, 2)
        self.assertEqual(stub.req_to_token_pool.auxiliary_state_pool.size, 2 * RATIO)


if __name__ == "__main__":
    unittest.main()
