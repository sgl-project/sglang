"""Pin the memory-safe KV-pool / prefill-chunk sizing in ``_compute_pool_size``.

The MLX backend auto-sizes its attention KV pool against the Metal working set.
On Apple Silicon the working-set *limit* is only a no-pressure recommendation, so
budgeting the pool against it alone leaves a long-prompt prefill with no standing
headroom -- and the prefill overflows into an uncatchable command-buffer OOM that
aborts the scheduler. ``_compute_pool_size`` therefore (a) clamps the budget to
real free memory, (b) caps the pool at the running set's context need so the freed
memory stays available as prefill headroom, (c) models the real resident KV (with radix
off, each request's private cache pre-allocated to its full span -- not the token pool)
when deriving the chunk cap, and (d) fails loudly at startup when even a minimum-size
safe prefill cannot fit. Because that sizing snapshots memory before any forward has
run, ``_calibrate_prefill_chunk`` then replays a real prefill in-server to MEASURE the
resident baseline and per-token transient and refine the chunk from those -- the actual
fix for the baseline undercount that let the chunk grow too large.

These tests drive both with mocked device / memory readings (and a mocked probe forward)
-- no model load, no GPU -- so the behaviours above are deterministic and CI-enforceable.
"""

from __future__ import annotations

import contextlib
import importlib.util
import types
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"

if _HAS_MLX:
    from sglang.srt.hardware_backend.mlx import model_runner as mr

# Qwen3-30B-A3B-4bit-shaped attention: GQA (32 query heads, 4 kv heads), 48 layers,
# head_dim 128, KV stored fp32 (4 B) -> 196608 B/slot. Numbers below are chosen so
# the arithmetic lands cleanly on each branch.
_LIMIT = 19_000_000_000  # Metal recommended working-set limit (bytes)
_N_KV, _HEAD_DIM, _N_LAYERS, _N_Q = 4, 128, 48, 32
# Calibrated per-(token x ctx) prefill activation cost for the shape above.
_PER_TOK_CTX = (mr._MLX_PREFILL_ACT_BYTES_PER_QHEAD * _N_Q) if _HAS_MLX else 0


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxComputePoolSize(unittest.TestCase):
    def _runner(
        self,
        *,
        mem_fraction,
        context_length,
        disable_radix=True,
        max_running=1,
    ):
        """An MlxModelRunner with model-dependent hooks stubbed (no weights loaded)."""
        r = mr.MlxModelRunner.__new__(mr.MlxModelRunner)
        r._mem_fraction_static = mem_fraction
        r._context_length = context_length
        r.disable_radix_cache = disable_radix
        r._max_running_requests = max_running
        r._max_safe_prefill_chunk = None
        # Per-request ContiguousAttentionKVCache span (pre-allocated to max_seq_len);
        # _compute_pool_size uses it to model the real resident KV with radix off.
        r._max_seq_len = 4096
        r._get_attn_config = lambda: (_N_KV, _HEAD_DIM, mr.mx.float32)
        r._attention_module_for_layer = lambda _idx: types.SimpleNamespace(n_heads=_N_Q)
        r._cache_layout = types.SimpleNamespace(
            num_attention_layers=_N_LAYERS, first_attention_layer_index=0
        )
        return r

    @contextlib.contextmanager
    def _memory(self, *, used, free):
        """Mock the device working-set limit, resident bytes, and real free memory."""
        with mock.patch.object(
            mr.mx,
            "device_info",
            return_value={"max_recommended_working_set_size": _LIMIT},
        ), mock.patch.object(
            mr.mx, "get_active_memory", return_value=used
        ), mock.patch.object(
            mr.psutil,
            "virtual_memory",
            return_value=types.SimpleNamespace(available=free),
        ):
            yield

    def test_pool_capped_to_running_context_not_greedy_budget(self):
        # Plenty of budget for a huge pool, but with radix off the pool can only ever
        # use the running set's contexts -- it must cap there (2048 * 1 * 1.25 = 2560)
        # and leave the rest as prefill headroom, not grab the whole budget.
        r = self._runner(mem_fraction=0.9, context_length=2048)
        with self._memory(used=8_000_000_000, free=12_000_000_000):
            pool = r._compute_pool_size(None)
        self.assertEqual(pool, 2560)
        self.assertGreaterEqual(r._max_safe_prefill_chunk, mr._MLX_MIN_PREFILL_CHUNK)

    def test_chunk_shrinks_with_real_free_memory(self):
        # Identical configs differing only in real free memory: when the system is
        # tighter, dynamic_budget clamps lower, so the derived safe chunk must shrink.
        # This is the clamp that the (fixed) working-set limit alone would miss.
        r_tight = self._runner(mem_fraction=0.95, context_length=2048)
        with self._memory(used=8_000_000_000, free=2_000_000_000):
            r_tight._compute_pool_size(None)
        r_loose = self._runner(mem_fraction=0.95, context_length=2048)
        with self._memory(used=8_000_000_000, free=100_000_000_000):
            r_loose._compute_pool_size(None)
        self.assertLess(
            r_tight._max_safe_prefill_chunk, r_loose._max_safe_prefill_chunk
        )

    def test_more_running_requests_shrink_chunk(self):
        # More concurrent requests -> more resident per-request KV caches -> less
        # activation headroom -> a smaller safe chunk. Pins that the sizing tracks the
        # configured max_running_requests (radix off, where each request owns a cache).
        r1 = self._runner(mem_fraction=0.95, context_length=2048, max_running=1)
        with self._memory(used=8_000_000_000, free=2_800_000_000):
            r1._compute_pool_size(None)
        r2 = self._runner(mem_fraction=0.95, context_length=2048, max_running=2)
        with self._memory(used=8_000_000_000, free=2_800_000_000):
            r2._compute_pool_size(None)
        self.assertLess(r2._max_safe_prefill_chunk, r1._max_safe_prefill_chunk)

    def test_loud_fail_when_pool_below_context(self):
        # Weights nearly fill the mem-fraction budget -> pool can't hold one context
        # window. Must raise a clear error, not silently floor the pool.
        r = self._runner(mem_fraction=0.85, context_length=2048)
        with self._memory(used=16_000_000_000, free=5_000_000_000):
            with self.assertRaises(RuntimeError) as ctx:
                r._compute_pool_size(None)
        msg = str(ctx.exception)
        self.assertIn("cannot fit both the KV cache and a safe prefill", msg)
        self.assertIn("KV pool", msg)

    def test_loud_fail_when_safe_chunk_below_minimum(self):
        # Pool fits the context, but real free memory leaves too little activation
        # headroom for even a minimum-size safe prefill chunk -> must raise (rather
        # than start and crash later on a long prompt).
        r = self._runner(mem_fraction=0.95, context_length=2048)
        with self._memory(used=8_000_000_000, free=1_000_000_000):
            with self.assertRaises(RuntimeError) as ctx:
                r._compute_pool_size(None)
        self.assertIn("safe prefill chunk", str(ctx.exception))

    def test_explicit_pool_still_derives_chunk_cap(self):
        # An explicit --max-total-tokens pool must NOT bypass chunk-cap derivation:
        # the runner still has to compute max_safe_prefill_chunk so the worker caps
        # chunked prefill (the explicit path previously left it None == uncapped).
        r = self._runner(mem_fraction=0.9, context_length=2048)
        with self._memory(used=8_000_000_000, free=12_000_000_000):
            pool = r._compute_pool_size(4000)
        self.assertEqual(pool, 4000)
        self.assertIsNotNone(r._max_safe_prefill_chunk)
        self.assertGreaterEqual(r._max_safe_prefill_chunk, mr._MLX_MIN_PREFILL_CHUNK)

    def test_radix_on_keeps_prior_sizing_and_no_chunk_cap(self):
        # Radix on uses a single greedy, pre-allocated shared pool; the radix-off
        # private-cache chunk-sizing model does not apply. A tight config that would
        # loud-fail under radix off (see test_loud_fail_when_safe_chunk_below_minimum,
        # same memory) must NOT fail here: radix-on behaviour is unchanged by the fix.
        # The chunk cap stays None (worker leaves chunked prefill alone) and the
        # admission coefficient stays 0 (runtime gate inactive for radix on).
        r = self._runner(mem_fraction=0.95, context_length=2048, disable_radix=False)
        with self._memory(used=8_000_000_000, free=1_000_000_000):
            pool = r._compute_pool_size(None)
        self.assertGreater(pool, 0)
        self.assertIsNone(r._max_safe_prefill_chunk)
        self.assertEqual(r._prefill_act_per_tok_ctx, 0)

    def test_max_running_honors_explicit(self):
        # An explicit --max-running-requests is honored verbatim (floored at 1), so the
        # scheduler's running cap matches what the chunk sizing assumed.
        r = self._runner(mem_fraction=0.9, context_length=2048, max_running=3)
        self.assertEqual(
            r._memory_safe_max_running(11_000_000_000, 800_000_000, 1024, 2048), 3
        )

    def test_max_running_default_is_memory_safe(self):
        # No explicit cap: default to how many private caches fit while leaving a minimum
        # prefill's activation headroom, minus the overlap pipeline slack. A tight budget
        # must collapse to 1 (never the old pool//2 heuristic that ignored cache residency).
        r = self._runner(mem_fraction=0.9, context_length=2048, max_running=None)
        tight = r._memory_safe_max_running(2_000_000_000, 800_000_000, 1024, 2048)
        loose = r._memory_safe_max_running(20_000_000_000, 800_000_000, 1024, 2048)
        self.assertEqual(tight, 1)
        self.assertGreater(loose, 1)

    def test_effective_max_running_set_and_exposed(self):
        # _compute_pool_size resolves the cap and exposes it; the stub reads this so the
        # scheduler never admits more concurrent requests than the working set can hold.
        r = self._runner(mem_fraction=0.9, context_length=2048, max_running=1)
        with self._memory(used=8_000_000_000, free=12_000_000_000):
            r._compute_pool_size(None)
        self.assertEqual(r._effective_max_running, 1)
        self.assertEqual(r.max_running_requests, 1)

    def test_radix_on_defers_max_running_to_stub(self):
        # Radix on: the private-cache concurrency model does not apply, so the cap is None
        # and the stub falls back to its pool-derived heuristic (behaviour unchanged).
        r = self._runner(mem_fraction=0.95, context_length=2048, disable_radix=False)
        with self._memory(used=8_000_000_000, free=12_000_000_000):
            r._compute_pool_size(None)
        self.assertIsNone(r.max_running_requests)


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxPrefillAdmission(unittest.TestCase):
    """Pin the runtime prefill admission gate (Layer A): given LIVE memory, does a
    prefill chunk's estimated activation peak fit, or must it be rejected?

    Startup sizing (Layer B) leaves a standing margin; this gate re-checks against
    memory free *right now*, so external pressure that appeared after startup is
    caught and turned into a clean rejection instead of an uncatchable Metal OOM.
    """

    def _runner(self, *, per_tok_ctx=_PER_TOK_CTX, cache_bytes=0, cache_pool=None):
        r = mr.MlxModelRunner.__new__(mr.MlxModelRunner)
        r._prefill_act_per_tok_ctx = per_tok_ctx
        r._max_safe_prefill_chunk = 398
        # Radix-off per-request cache the gate also charges (0 = exercise activation only).
        r.disable_radix_cache = True
        r._radix_off_cache_bytes = cache_bytes
        r._cache_pool = [] if cache_pool is None else cache_pool
        return r

    @contextlib.contextmanager
    def _memory(self, *, used, free, limit=_LIMIT):
        with mock.patch.object(
            mr.mx,
            "device_info",
            return_value={"max_recommended_working_set_size": limit},
        ), mock.patch.object(
            mr.mx, "get_active_memory", return_value=used
        ), mock.patch.object(
            mr.psutil,
            "virtual_memory",
            return_value=types.SimpleNamespace(available=free),
        ):
            yield

    def test_admits_when_headroom_exceeds_estimate(self):
        # Plenty of free memory -> the chunk's activation peak fits -> admit.
        r = self._runner()
        with self._memory(used=8_000_000_000, free=12_000_000_000):
            fits, estimate, headroom = r.prefill_fits_live_memory(398, 1900)
        self.assertTrue(fits)
        self.assertGreater(estimate, 0)
        self.assertLessEqual(estimate, headroom)

    def test_rejects_when_headroom_below_estimate(self):
        # Same prompt, but live free memory has collapsed -> the peak no longer fits
        # -> reject (this is the case that would otherwise abort the scheduler).
        r = self._runner()
        with self._memory(used=8_000_000_000, free=500_000_000):
            fits, estimate, headroom = r.prefill_fits_live_memory(398, 1900)
        self.assertFalse(fits)
        self.assertGreater(estimate, headroom)

    def test_headroom_clamped_to_real_free_memory(self):
        # Even with a huge working-set limit, a tiny amount of real free memory must
        # bound the headroom -- budgeting against the limit alone is what lets a
        # prefill overflow under pressure. Big limit + small free -> reject.
        r = self._runner()
        with self._memory(used=8_000_000_000, free=400_000_000, limit=100_000_000_000):
            fits, _estimate, headroom = r.prefill_fits_live_memory(398, 1900)
        self.assertLessEqual(headroom, 400_000_000)
        self.assertFalse(fits)

    def test_estimate_uses_calibrated_cost(self):
        # The estimate is exactly the shared calibrated cost x chunk x context.
        r = self._runner()
        with self._memory(used=8_000_000_000, free=12_000_000_000):
            _fits, estimate, _headroom = r.prefill_fits_live_memory(300, 1500)
        self.assertEqual(estimate, r._prefill_act_per_tok_ctx * 300 * 1500)

    def test_admits_when_calibration_unknown(self):
        # No calibrated cost (explicit pool / no context) -> never block on a missing
        # estimate; the gate admits and leaves protection to startup sizing.
        r = self._runner(per_tok_ctx=0)
        with self._memory(used=16_000_000_000, free=0):
            fits, estimate, _headroom = r.prefill_fits_live_memory(398, 1900)
        self.assertTrue(fits)
        self.assertEqual(estimate, 0)

    def test_estimate_includes_new_cache_bytes(self):
        # Radix off with an empty cache pool: the prefill will allocate a fresh private
        # cache, so its fixed cost is added to the activation estimate. Without this, a
        # short prompt (tiny activation) sails through right up to the working-set edge and
        # then the cache allocation OOMs -- the concurrency hole this closes.
        r = self._runner(cache_bytes=750_000_000)
        with self._memory(used=8_000_000_000, free=12_000_000_000):
            _fits, estimate, _ = r.prefill_fits_live_memory(300, 1500)
        self.assertEqual(
            estimate, r._prefill_act_per_tok_ctx * 300 * 1500 + 750_000_000
        )

    def test_new_cache_bytes_skipped_when_pool_has_free_cache(self):
        # A finished cache can be reused (already resident, already counted in live active),
        # so no new allocation is charged on top.
        r = self._runner(cache_bytes=750_000_000, cache_pool=[object()])
        with self._memory(used=8_000_000_000, free=12_000_000_000):
            _fits, estimate, _ = r.prefill_fits_live_memory(300, 1500)
        self.assertEqual(estimate, r._prefill_act_per_tok_ctx * 300 * 1500)


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxSchedulerAdmissionGate(unittest.TestCase):
    """Pin the scheduler-side gate: a waiting prefill that does not fit live memory is
    aborted with set_finish_with_abort (the existing clean-rejection path); one that
    fits is left untouched."""

    def _scheduler(self, runner):
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        sched = types.SimpleNamespace()
        sched.tp_worker = types.SimpleNamespace(_mlx_runner=runner)
        self._reject = SchedulerMlxOverlapMixin._mlx_reject_unfittable_prefills
        return sched

    def _req(self, n_tokens, finished=False):
        req = types.SimpleNamespace()
        req.origin_input_ids = [1] * n_tokens
        req.rid = "rid-test"
        req.finished = lambda: finished
        req.set_finish_with_abort = mock.Mock()
        return req

    def _runner(self, *, chunk=398, per_tok_ctx=_PER_TOK_CTX):
        r = mr.MlxModelRunner.__new__(mr.MlxModelRunner)
        r._prefill_act_per_tok_ctx = per_tok_ctx
        r._max_safe_prefill_chunk = chunk
        r.disable_radix_cache = True
        r._radix_off_cache_bytes = 0
        r._cache_pool = []
        return r

    @contextlib.contextmanager
    def _memory(self, *, used, free):
        with mock.patch.object(
            mr.mx,
            "device_info",
            return_value={"max_recommended_working_set_size": _LIMIT},
        ), mock.patch.object(
            mr.mx, "get_active_memory", return_value=used
        ), mock.patch.object(
            mr.psutil,
            "virtual_memory",
            return_value=types.SimpleNamespace(available=free),
        ):
            yield

    def test_aborts_unfittable_waiting_prefill(self):
        runner = self._runner()
        sched = self._scheduler(runner)
        req = self._req(1900)
        sched.waiting_queue = [req]
        with self._memory(used=8_000_000_000, free=500_000_000):
            self._reject(sched)
        req.set_finish_with_abort.assert_called_once()
        # The error must name the GPU-memory cause so the client gets an actionable 400.
        self.assertIn("memory", req.set_finish_with_abort.call_args.args[0].lower())

    def test_admits_fittable_waiting_prefill(self):
        runner = self._runner()
        sched = self._scheduler(runner)
        req = self._req(1900)
        sched.waiting_queue = [req]
        with self._memory(used=8_000_000_000, free=12_000_000_000):
            self._reject(sched)
        req.set_finish_with_abort.assert_not_called()

    def test_skips_when_no_chunk_cap(self):
        # Radix on / no auto-derived cap -> max_safe_prefill_chunk is None -> the gate
        # is a no-op (startup sizing already governs); never touch the request.
        runner = self._runner(chunk=None)
        sched = self._scheduler(runner)
        req = self._req(1900)
        sched.waiting_queue = [req]
        with self._memory(used=8_000_000_000, free=10_000):
            self._reject(sched)
        req.set_finish_with_abort.assert_not_called()

    def test_skips_already_finished_req(self):
        # A request already aborted (e.g. by the prompt-length check) is left alone.
        runner = self._runner()
        sched = self._scheduler(runner)
        req = self._req(1900, finished=True)
        sched.waiting_queue = [req]
        with self._memory(used=8_000_000_000, free=500_000_000):
            self._reject(sched)
        req.set_finish_with_abort.assert_not_called()


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxProbeCalibration(unittest.TestCase):
    """Pin the in-server prefill probe's decision logic (_calibrate_prefill_chunk).

    The GPU forwards (_run_prefill_probe) are mocked so the test is deterministic and
    GPU-free; what is exercised is what the probe DOES with a measured
    (baseline, peak, ctx, caches_held): derive the chunk against the measured resident
    baseline using the conservative cost, refine the admission coefficient to the measured
    transient, fail loudly when even a floor-size prefill will not fit or the box cannot
    hold the steady-state cache set, and otherwise degrade safely (keep the startup
    estimate) on a failed or inconclusive measurement.

    target_caches = max_running + _MLX_PIPELINE_CACHE_SLACK; with max_running=1 that is 3.
    """

    def _runner(
        self,
        *,
        context_length=2048,
        startup_chunk=398,
        startup_per_tok_ctx=1024,
        max_running=1,
    ):
        r = mr.MlxModelRunner.__new__(mr.MlxModelRunner)
        r._context_length = context_length
        r._max_running_requests = max_running
        # The probe sizes its cache set from the resolved cap, not the raw server arg.
        r._effective_max_running = max_running
        r.disable_radix_cache = True  # default MLX config; no shared-pool adjustment
        r.model = object()  # non-None: the probe proceeds
        r._max_safe_prefill_chunk = startup_chunk
        r._prefill_act_per_tok_ctx = startup_per_tok_ctx
        r._radix_off_cache_bytes = 0
        return r

    @property
    def _target_caches(self):
        return 1 + mr._MLX_PIPELINE_CACHE_SLACK

    @contextlib.contextmanager
    def _env(self, *, limit=_LIMIT, free=50_000_000_000):
        with mock.patch.object(
            mr.mx,
            "device_info",
            return_value={"max_recommended_working_set_size": limit},
        ), mock.patch.object(
            mr.psutil,
            "virtual_memory",
            return_value=types.SimpleNamespace(available=free),
        ):
            yield

    def test_refines_chunk_down_from_measured_baseline(self):
        # Measured baseline near the limit (little headroom) -> the probe must lower the
        # optimistic startup chunk to a measured-safe value, and never below the floor.
        r = self._runner(startup_chunk=398)
        baseline, peak = int(17.00 * 1024**3), int(17.20 * 1024**3)
        with self._env(), mock.patch.object(
            r,
            "_run_prefill_probe",
            return_value=(baseline, peak, 2048, self._target_caches),
        ):
            r._calibrate_prefill_chunk()
        self.assertLess(r._max_safe_prefill_chunk, 398)
        self.assertGreaterEqual(r._max_safe_prefill_chunk, mr._MLX_PROBE_FLOOR_CHUNK)
        self.assertLessEqual(r._max_safe_prefill_chunk, 2048)

    def test_admission_coefficient_set_to_measured(self):
        # The runtime admission gate must reuse the probe's measured transient (accurate,
        # so it does not over-reject), distinct from the conservative sizing cost.
        r = self._runner(startup_per_tok_ctx=1024)
        baseline, peak = int(17.00 * 1024**3), int(17.20 * 1024**3)
        with self._env(), mock.patch.object(
            r,
            "_run_prefill_probe",
            return_value=(baseline, peak, 2048, self._target_caches),
        ):
            r._calibrate_prefill_chunk()
        transient = peak - baseline
        self.assertEqual(
            r._prefill_act_per_tok_ctx, -(-transient // (128 * 2048))  # ceil
        )

    def test_probe_holds_running_plus_slack_caches(self):
        # The probe must measure against the steady-state resident set for the configured
        # concurrency: running + _MLX_PIPELINE_CACHE_SLACK private caches.
        r = self._runner(max_running=2)
        baseline, peak = int(17.00 * 1024**3), int(17.20 * 1024**3)
        captured = {}

        def fake_probe(chunk_probe, ctx_target, target_caches, mlx_limit):
            captured["target_caches"] = target_caches
            return baseline, peak, ctx_target, target_caches

        with self._env(), mock.patch.object(
            r, "_run_prefill_probe", side_effect=fake_probe
        ):
            r._calibrate_prefill_chunk()
        self.assertEqual(captured["target_caches"], 2 + mr._MLX_PIPELINE_CACHE_SLACK)

    def test_loud_fail_when_even_floor_chunk_will_not_fit(self):
        # Baseline so close to the limit (and real free memory so low) that even a
        # floor-size prefill's transient does not fit -> fail loudly at startup rather
        # than crash later on a long prompt.
        r = self._runner()
        limit = _LIMIT
        baseline, peak = limit - 10_000_000, limit - 10_000_000 + 60_000_000
        with self._env(limit=limit, free=10_000_000):
            with mock.patch.object(
                r,
                "_run_prefill_probe",
                return_value=(baseline, peak, 2048, self._target_caches),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    r._calibrate_prefill_chunk()
        self.assertIn("cannot serve this context", str(ctx.exception))

    def test_loud_fail_when_cannot_hold_resident_cache_set(self):
        # The box could not materialise the full steady-state cache set (held < target),
        # so production would OOM once the pool fills -> fail loudly even if the partial
        # baseline leaves apparent headroom.
        r = self._runner()
        baseline, peak = int(16.5 * 1024**3), int(16.7 * 1024**3)
        with self._env(), mock.patch.object(
            r,
            "_run_prefill_probe",
            return_value=(baseline, peak, 2048, self._target_caches - 1),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                r._calibrate_prefill_chunk()
        self.assertIn("cannot serve this context", str(ctx.exception))

    def test_keeps_startup_estimate_when_probe_raises(self):
        # A GPU/model error in the replay must not wedge startup; keep the estimate.
        r = self._runner(startup_chunk=310)
        with self._env(), mock.patch.object(
            r, "_run_prefill_probe", side_effect=RuntimeError("metal boom")
        ):
            r._calibrate_prefill_chunk()
        self.assertEqual(r._max_safe_prefill_chunk, 310)

    def test_keeps_startup_estimate_when_transient_trivial(self):
        # Peak ~= baseline (no measurable transient) -> distrust it, keep the estimate.
        r = self._runner(startup_chunk=310)
        baseline = int(17.0 * 1024**3)
        with self._env(), mock.patch.object(
            r,
            "_run_prefill_probe",
            return_value=(baseline, baseline + 1024, 2048, self._target_caches),
        ):
            r._calibrate_prefill_chunk()
        self.assertEqual(r._max_safe_prefill_chunk, 310)

    def test_skipped_without_context_length(self):
        # No context_length -> no auto chunk cap to refine; do nothing.
        r = self._runner(context_length=None, startup_chunk=None)
        with self._env():
            r._calibrate_prefill_chunk()
        self.assertIsNone(r._max_safe_prefill_chunk)


if __name__ == "__main__":
    unittest.main()
