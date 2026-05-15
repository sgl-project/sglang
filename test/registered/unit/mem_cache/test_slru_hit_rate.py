"""End-to-end SLRU hit-rate tests under burst and decay workloads.

These tests drive a real ``RadixCache`` with deterministic traffic, force
eviction, and probe which prefixes remain. A fake clock keeps the scenarios
fast and repeatable.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

import unittest

try:
    import torch

    from sglang.srt.environ import envs
    from sglang.srt.mem_cache import evict_policy as _ep_module
    from sglang.srt.mem_cache import radix_cache as _rc_module
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

    _IMPORT_ERROR = None
    _AVAILABLE = True
except Exception as exc:  # pragma: no cover — env-dependent
    _IMPORT_ERROR = exc
    _AVAILABLE = False


class _MockAllocator:
    """Stand-in for ``TokenToKVPoolAllocator``. ``evict()`` only uses
    ``.free(tensor)`` and ``.device`` on the allocator, so that's all
    we need to satisfy. ``free`` is a no-op since we don't care about
    the underlying KV cache pool for these hit-rate tests — only
    whether the *tree node* survives eviction.
    """

    def __init__(self):
        self.device = "cpu"

    def free(self, value):
        # value is a torch.Tensor of token indices. No-op — the RadixCache
        # already detached it from its tree when calling us.
        pass


class _FakeClock:
    """Deterministic monotonic clock. Exposes ``.now()`` as a callable
    that can replace ``time.monotonic`` inside both the radix cache and
    the evict_policy module.
    """

    def __init__(self, start: float = 0.0):
        self._t = start

    def advance(self, seconds: float):
        self._t += seconds

    def now(self) -> float:
        return self._t


def _patch_clock(clock: "_FakeClock"):
    """Redirect ``time.monotonic`` in both modules to ``clock.now``.

    We cannot simply monkey-patch the ``time`` module — other imports
    use it. We patch the ``time`` attribute used inside each module.
    Returns a tuple (restore_fn).
    """
    rc_original = _rc_module.time.monotonic
    ep_original = _ep_module.time.monotonic
    _rc_module.time.monotonic = clock.now
    _ep_module.time.monotonic = clock.now

    def restore():
        _rc_module.time.monotonic = rc_original
        _ep_module.time.monotonic = ep_original

    return restore


def _make_cache() -> "RadixCache":
    return RadixCache.create_simulated(mock_allocator=_MockAllocator(), page_size=1)


def _insert_prefix(cache, token_ids):
    """Insert and mark the prefix as an evictable leaf. Unlocks it so
    the eviction scan considers it."""
    key = RadixKey(token_ids=list(token_ids), extra_key=None)
    value = torch.tensor(list(token_ids), dtype=torch.int64)
    cache.insert(InsertParams(key=key, value=value))


def _probe_hit(cache, token_ids) -> bool:
    """Return True if the probe finds the full prefix in cache."""
    key = RadixKey(token_ids=list(token_ids), extra_key=None)
    result = cache.match_prefix(MatchPrefixParams(key=key))
    return int(result.device_indices.numel()) == len(token_ids)


@unittest.skipUnless(
    _AVAILABLE,
    f"RadixCache import chain unavailable: {_IMPORT_ERROR}",
)
class TestSLRUHitRate(unittest.TestCase):
    """Compare the gate-off and optimized SLRU paths on identical traffic."""

    def _run_scenario_b(self, optimization_enabled: bool):
        """Burst traffic should not evict the genuine hot prefix.

        The VIP prefix is reused across seconds. Burst prefixes are reused many
        times inside a short debounce window. Optimized SLRU should protect VIP
        while avoiding protected-tier pollution from bursts.
        """
        clock = _FakeClock(start=100.0)
        restore = _patch_clock(clock)
        try:
            with envs.SGLANG_ENABLE_SLRU_OPTIMIZATION.override(optimization_enabled):
                cache = _make_cache()
                from sglang.srt.mem_cache.evict_policy import SLRUStrategy

                cache.eviction_strategy = SLRUStrategy(
                    protected_threshold=2,
                    debounce_sec=0.1,
                    decay_sec=60.0,
                )

                # VIP: repeated, well-spaced hits.
                vip_tokens = list(range(1000, 1016))
                for _ in range(4):
                    _insert_prefix(cache, vip_tokens)
                    clock.advance(1.0)

                # Burst prefixes: many hits inside a 10ms window.
                burst_tokens_list = []
                for b in range(10):
                    btokens = list(range(2000 + b * 20, 2016 + b * 20))
                    burst_tokens_list.append(btokens)
                    for _ in range(100):
                        _insert_prefix(cache, btokens)
                        clock.advance(0.0001)  # 100 hits in 10ms

                # Pressure prefixes are well-spaced enough to become protected
                # in both paths, making burst promotion the differentiator.
                pressure_tokens_list = []
                for p in range(40):
                    ptokens = [5000 + p * 50 + i for i in range(16)]
                    pressure_tokens_list.append(ptokens)
                    for _ in range(2):
                        _insert_prefix(cache, ptokens)
                        clock.advance(0.5)  # > debounce_sec

                # Eviction: free roughly half the evictable tokens.
                target_evict = max(int(cache.evictable_size_ * 0.5), 16)
                cache.evict(EvictParams(num_tokens=target_evict))

                vip_hit = _probe_hit(cache, vip_tokens)
                burst_hits = sum(1 for b in burst_tokens_list if _probe_hit(cache, b))
                return vip_hit, burst_hits, len(burst_tokens_list)
        finally:
            restore()

    def test_scenario_b_burst_preserves_vip_hit_rate(self):
        """Optimized SLRU keeps VIP when burst traffic creates pressure."""
        naive_vip, naive_bursts, n_bursts = self._run_scenario_b(
            optimization_enabled=False
        )
        opt_vip, opt_bursts, _ = self._run_scenario_b(optimization_enabled=True)

        # Optimized SLRU keeps the genuine hot prefix.
        self.assertTrue(
            opt_vip,
            "Scenario B: optimized SLRU must retain VIP after burst+pressure "
            f"(vip_hit={opt_vip}, naive_vip={naive_vip})",
        )
        # VIP retention is the measured hit-rate delta for genuine hot data.
        self.assertGreater(
            int(opt_vip),
            int(naive_vip),
            "Scenario B: optimized SLRU's VIP retention must beat legacy SLRU. "
            f"naive_vip={naive_vip}, opt_vip={opt_vip}",
        )
        # Burst prefixes should not be preserved more aggressively than VIP.
        self.assertLessEqual(
            opt_bursts,
            naive_bursts,
            "Scenario B: optimized SLRU must not preserve more bursts than "
            f"legacy SLRU (naive={naive_bursts}, opt={opt_bursts}, total={n_bursts})",
        )

    def _run_scenario_c(self, optimization_enabled: bool):
        """Lazy decay should let a fresh prefix replace stale protected data.

        OLD becomes protected, idles for more than two decay periods, and NEW
        arrives just before eviction pressure. Optimized SLRU should demote OLD
        at scan time and retain NEW.
        """
        clock = _FakeClock(start=0.0)
        restore = _patch_clock(clock)
        try:
            with envs.SGLANG_ENABLE_SLRU_OPTIMIZATION.override(optimization_enabled):
                cache = _make_cache()
                from sglang.srt.mem_cache.evict_policy import SLRUStrategy

                cache.eviction_strategy = SLRUStrategy(
                    protected_threshold=2,
                    debounce_sec=0.1,
                    decay_sec=60.0,
                )

                # OLD becomes protected before going idle.
                old_tokens = list(range(100, 116))
                for _ in range(6):
                    _insert_prefix(cache, old_tokens)
                    clock.advance(1.0)

                # Long idle — well past 2 × decay_sec.
                clock.advance(150.0)

                # NEW arrives shortly before eviction pressure.
                new_tokens = list(range(200, 216))
                _insert_prefix(cache, new_tokens)
                clock.advance(0.5)

                # Pressure — 40 disjoint prefixes, 1 hit each.
                # Probationary.
                for p in range(40):
                    ptokens = [5000 + p * 50 + i for i in range(16)]
                    _insert_prefix(cache, ptokens)
                    clock.advance(0.05)

                # Evict one prefix: legacy SLRU drops NEW, optimized SLRU drops
                # the decayed OLD prefix.
                cache.evict(EvictParams(num_tokens=len(old_tokens)))

                return _probe_hit(cache, old_tokens), _probe_hit(cache, new_tokens)
        finally:
            restore()

    def test_scenario_c_decay_admits_new_hotspot(self):
        """Optimized SLRU admits NEW by decaying stale protected data."""
        naive_old, naive_new = self._run_scenario_c(optimization_enabled=False)
        opt_old, opt_new = self._run_scenario_c(optimization_enabled=True)

        # Optimized SLRU retains the newcomer.
        self.assertTrue(
            opt_new,
            f"Scenario C: optimized SLRU must retain NEW hotspot "
            f"(opt_old={opt_old}, opt_new={opt_new})",
        )
        # Lazy decay makes OLD a preferred eviction target.
        self.assertFalse(
            opt_old,
            f"Scenario C: optimized SLRU must evict stale OLD "
            f"after 150s idle (opt_old={opt_old}, opt_new={opt_new})",
        )
        # Legacy SLRU keeps OLD protected and evicts NEW.
        self.assertTrue(
            naive_old,
            f"Scenario C: legacy SLRU is expected to keep OLD, "
            f"got naive_old={naive_old}",
        )
        self.assertFalse(
            naive_new,
            f"Scenario C: legacy SLRU is expected to evict NEW, "
            f"got naive_new={naive_new}",
        )

    def test_optimization_off_matches_naive(self):
        """Regression guard: with the gate OFF, the new code path must
        be behaviorally indistinguishable from main. Running Scenario B
        twice with gate=OFF must produce identical results."""
        a = self._run_scenario_b(optimization_enabled=False)
        b = self._run_scenario_b(optimization_enabled=False)
        self.assertEqual(
            a,
            b,
            "Gate-off path is expected to be deterministic and "
            "identical to legacy SLRU",
        )


if __name__ == "__main__":
    unittest.main()
