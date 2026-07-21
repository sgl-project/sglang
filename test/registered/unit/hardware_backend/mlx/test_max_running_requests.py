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
from unittest import mock

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


def _arch(hybrid):
    """Make the stub see a hybrid / linear-attention (or plain) model.

    ``mambaish_config`` is a free function resolved from
    ``model_config.hf_config`` (the ModelRunner property was retired), so the
    fake model config cannot carry the answer -- patch the symbol the stub
    module imported instead.
    """
    return mock.patch(
        "sglang.srt.hardware_backend.mlx.model_runner_stub.mambaish_config",
        return_value=object() if hybrid else None,
    )


def _resolve(stub, hybrid=False):
    """Run the resolver with the model architecture patched (see _arch)."""
    with _arch(hybrid):
        return stub._resolve_max_running_requests()


def _stub(
    max_running_requests,
    max_total_num_tokens,
    dp_size=1,
    max_mamba_cache_size=None,
    disable_radix_cache=False,
):
    """A stub carrying only what _resolve_max_running_requests reads."""
    stub = MlxModelRunnerStub.__new__(MlxModelRunnerStub)
    stub.model_config = SimpleNamespace()  # only handed to the patched _arch fn
    stub.server_args = SimpleNamespace(
        max_running_requests=max_running_requests,
        max_mamba_cache_size=max_mamba_cache_size,
        disable_radix_cache=disable_radix_cache,
    )
    stub.max_total_num_tokens = max_total_num_tokens
    stub.dp_size = dp_size
    return stub


def _hybrid_stub_for_initialize(
    max_running_requests, max_mamba_cache_size, pool=100, disable_radix_cache=False
):
    """A stub carrying what the real initialize() reads (hybrid path)."""
    stub = MlxModelRunnerStub.__new__(MlxModelRunnerStub)
    stub._mlx_pool_size = pool
    stub.dp_size = 1
    stub.device = "cpu"  # read by init_ngram_embedding_manager
    stub.server_args = SimpleNamespace(
        enable_memory_saver=False,
        max_running_requests=max_running_requests,
        max_mamba_cache_size=max_mamba_cache_size,
        disable_radix_cache=disable_radix_cache,
    )
    stub.model_config = SimpleNamespace(
        is_hybrid_swa=False,
        sliding_window_size=None,
        attention_chunk_size=None,
        dtype="float16",
        num_hidden_layers=1,
        num_attention_layers=1,
        context_len=64,
        use_ngram_embedding=False,  # short-circuits NgramEmbeddingManager
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
        self.assertEqual(_resolve(_stub(None, 1000)), 500)
        self.assertEqual(_resolve(_stub(None, 100_000)), 4096)

    def test_flag_honored_within_capacity(self):
        # THE REGRESSION: an explicit flag must be honored, not ignored.
        self.assertEqual(_resolve(_stub(1, 100_000)), 1)
        self.assertEqual(_resolve(_stub(64, 100_000)), 64)

    def test_flag_split_per_dp_worker(self):
        # Mirrors the base clamp: the requested value is divided across dp workers.
        self.assertEqual(_resolve(_stub(8, 100_000, dp_size=2)), 4)

    def test_flag_clamped_to_capacity(self):
        # A flag larger than the KV pool can hold is capped at pool // 2.
        self.assertEqual(_resolve(_stub(100_000, 1000)), 500)


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
            _resolve(_stub(8, 100_000, max_mamba_cache_size=4 * RATIO), hybrid=True),
            4,
        )

    def test_default_path_also_bounded_by_aux_capacity(self):
        # No --max-running-requests: the capacity default must still be
        # bounded, else the 4096 default overruns a small aux pool the same way.
        self.assertEqual(
            _resolve(_stub(None, 100_000, max_mamba_cache_size=2 * RATIO), hybrid=True),
            2,
        )

    def test_infeasible_aux_capacity_raises_at_startup(self):
        # Reviewer repro: max_running_requests=4, max_mamba_cache_size=2.
        # Bound is 2 // RATIO = 0 -> fail fast instead of asserting mid-serving.
        with self.assertRaisesRegex(RuntimeError, "max_mamba_cache_size"):
            _resolve(_stub(4, 100, max_mamba_cache_size=2), hybrid=True)

    def test_aux_flag_unset_is_not_bounded(self):
        # Negative branch: with max_mamba_cache_size unset the pool is sized
        # FROM the resolved cap (RATIO x), so no bound must be applied.
        self.assertEqual(_resolve(_stub(8, 100_000), hybrid=True), 8)

    def test_non_hybrid_ignores_aux_flag(self):
        # Negative branch: a plain attention model never applies the bound.
        self.assertEqual(
            _resolve(_stub(8, 100_000, max_mamba_cache_size=2), hybrid=False), 8
        )

    def test_radix_disabled_uses_one_slot_per_request(self):
        # With --disable-radix-cache there are no radix-held snapshots to
        # reserve headroom for: each live request holds exactly one auxiliary
        # slot, so max_mamba_cache_size=8 backs all 8 requests (the fixed
        # RATIO would wrongly cut this to 8 // RATIO).
        self.assertEqual(
            _resolve(
                _stub(8, 100_000, max_mamba_cache_size=8, disable_radix_cache=True),
                hybrid=True,
            ),
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
        with _arch(hybrid=True):
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
        with _arch(hybrid=True), self.assertRaisesRegex(
            RuntimeError, "max_mamba_cache_size"
        ):
            stub.initialize()

    def test_radix_disabled_backs_every_request_slot_one_to_one(self):
        # Reviewer repro: with --disable-radix-cache, size=8 aux slots must
        # back all 8 request slots (one live slot each) instead of being cut
        # to 8 // RATIO by snapshot headroom that can never be used.
        stub = _hybrid_stub_for_initialize(
            max_running_requests=8,
            max_mamba_cache_size=8,
            disable_radix_cache=True,
        )
        with _arch(hybrid=True):
            stub.initialize()
        self.assertEqual(stub.max_running_requests, 8)
        pool = stub.req_to_token_pool
        for _ in range(stub.max_running_requests):
            self.assertIsNotNone(pool.alloc([_fake_req()]))

    def test_radix_disabled_default_sizing_is_one_to_one(self):
        # Drift guard for the no-radix path: with the flag unset the aux pool
        # is sized with the same 1x ratio the bound uses.
        stub = _hybrid_stub_for_initialize(
            max_running_requests=3,
            max_mamba_cache_size=None,
            disable_radix_cache=True,
        )
        with _arch(hybrid=True):
            stub.initialize()
        self.assertEqual(stub.max_running_requests, 3)
        self.assertEqual(stub.req_to_token_pool.auxiliary_state_pool.size, 3)

    def test_radix_disabled_sequential_requests_release_their_aux_slot(self):
        # THE LEAK: with radix disabled, release_kv_cache's free_mamba_cache
        # fallback never fires (the MLX pool is not a HybridReqToTokenPool)
        # and ChunkCache frees token KV only, so pool.free(req) was the only
        # release hook left -- and it freed just the request row. Every
        # finished request permanently consumed one auxiliary slot and the
        # (cap + 1)-th SEQUENTIAL request crashed with "Not enough MLX
        # auxiliary state slots" even at concurrency 1. The pool now owns
        # auxiliary release in this configuration: allocate/free/reallocate
        # far past the pool size must succeed, with every slot returned.
        stub = _hybrid_stub_for_initialize(
            max_running_requests=2,
            max_mamba_cache_size=2,
            disable_radix_cache=True,
        )
        with _arch(hybrid=True):
            stub.initialize()
        pool = stub.req_to_token_pool
        aux_capacity = pool.auxiliary_state_pool.available_size()
        for _ in range(3 * aux_capacity):
            req = _fake_req()
            self.assertIsNotNone(pool.alloc([req]))
            pool.free(req)  # as release_kv_cache does after ChunkCache
            self.assertIsNone(req.mamba_pool_idx)
            self.assertEqual(pool.auxiliary_state_pool.available_size(), aux_capacity)

    def test_radix_enabled_free_does_not_touch_aux_slot(self):
        # Retention contract: with the radix cache enabled the tree component
        # owns auxiliary release (it frees or adopts the slot and nulls
        # req.mamba_pool_idx BEFORE the row is freed). pool.free(req) must
        # therefore never release auxiliary slots itself -- even if called
        # while mamba_pool_idx is still set -- or a tree-owned snapshot slot
        # could be recycled under a live radix node.
        stub = _hybrid_stub_for_initialize(
            max_running_requests=2,
            max_mamba_cache_size=2 * RATIO,
            disable_radix_cache=False,
        )
        with _arch(hybrid=True):
            stub.initialize()
        pool = stub.req_to_token_pool
        free_before = pool.auxiliary_state_pool.available_size()
        req = _fake_req()
        pool.alloc([req])
        pool.free(req)
        self.assertIsNotNone(req.mamba_pool_idx)  # slot NOT released by free()
        self.assertEqual(pool.auxiliary_state_pool.available_size(), free_before - 1)

    def test_default_aux_sizing_uses_shared_ratio(self):
        # Drift guard: with the flag unset, initialize() sizes the aux pool
        # from the resolved cap with the SAME ratio the bound uses, so the
        # sizing and the bound cannot diverge.
        stub = _hybrid_stub_for_initialize(
            max_running_requests=2, max_mamba_cache_size=None
        )
        with _arch(hybrid=True):
            stub.initialize()
        self.assertEqual(stub.max_running_requests, 2)
        self.assertEqual(stub.req_to_token_pool.auxiliary_state_pool.size, 2 * RATIO)


if __name__ == "__main__":
    unittest.main()
