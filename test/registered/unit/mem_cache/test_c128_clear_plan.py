"""Unit tests for the templated C128 request-state reset.

``DeepSeekV4TokenToKVPool.clear_c128_req_state`` used to issue two kernels per
C128 pool (``fill_(-inf)`` + ``zero_`` on the two halves of the slot), i.e. ~40
launches per admitted request. The written constants never change, so
``_build_c128_clear_plan`` bakes them into one template and the reset becomes a
single ``torch._foreach_copy_``.

These tests pin the properties the optimization has to keep:
  - the bytes written are bit-for-bit what the per-pool loop wrote, for both
    C128 layouts (``ONLINE_C128`` on and off) and several dtypes;
  - only the addressed request slot is touched;
  - the plan is built once and reused;
  - a non-homogeneous pool set returns ``None`` and falls back to the loop.

    python -m pytest test/registered/unit/mem_cache/test_c128_clear_plan.py -v
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.mem_cache.deepseek_v4_memory_pool as pool_mod
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

_NUM_REQS = 4
_RING = 3
_HEAD_DIM = 8
_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _make_state(
    *, online: bool, dtype: torch.dtype, width_mult: int = 1
) -> torch.Tensor:
    """Allocate one pool's kv_score buffer, filled with a recognizable pattern.

    Online layout indexes one row per request slot and splits it in three;
    offline indexes ``ring_size`` rows per slot and splits each in two.
    """
    if online:
        rows, width = _NUM_REQS, 3 * _HEAD_DIM * width_mult
    else:
        rows, width = _NUM_REQS * _RING, 2 * _HEAD_DIM * width_mult
    return (
        torch.arange(rows * width, dtype=torch.float32).reshape(rows, width).to(dtype)
    )


def _make_pool(state: torch.Tensor, *, ratio: int = 128) -> SimpleNamespace:
    return SimpleNamespace(
        ratio=ratio,
        ring_size=_RING,
        kv_score_buffer=SimpleNamespace(kv_score=state),
    )


def _make_kv_pool(states, *, extra=()) -> DeepSeekV4TokenToKVPool:
    """A pool object carrying only what the two methods under test read."""
    obj = object.__new__(DeepSeekV4TokenToKVPool)
    obj.compress_state_pools = [_make_pool(s) for s in states] + list(extra)
    obj._c128_clear_plan_built = False
    obj._c128_clear_plan = None
    return obj


def _legacy_clear(kv_pool: DeepSeekV4TokenToKVPool, req_pool_idx: int) -> None:
    """The pre-optimization body, kept here as the reference implementation."""
    for pool in kv_pool.compress_state_pools:
        if pool is None or pool.ratio != 128:
            continue
        state = pool.kv_score_buffer.kv_score
        if pool_mod.ONLINE_C128:
            row = state[req_pool_idx]
            head_dim = row.shape[-1] // 3
            row[:head_dim].fill_(float("-inf"))
            row[head_dim:].zero_()
        else:
            start = req_pool_idx * pool.ring_size
            rows = state[start : start + pool.ring_size]
            half = rows.shape[-1] // 2
            rows[:, :half].zero_()
            rows[:, half:].fill_(float("-inf"))


class TestC128ClearPlan(CustomTestCase):
    def _assert_matches_legacy(self, *, online: bool, dtype: torch.dtype, slot: int):
        with mock.patch.object(pool_mod, "ONLINE_C128", online):
            fast_states = [_make_state(online=online, dtype=dtype) for _ in range(3)]
            ref_states = [s.clone() for s in fast_states]

            _make_kv_pool(fast_states).clear_c128_req_state(slot)
            _legacy_clear(_make_kv_pool(ref_states), slot)

            for i, (fast, ref) in enumerate(zip(fast_states, ref_states)):
                # view as bytes: -inf may saturate in the pool dtype, and the
                # claim is that both paths saturate identically.
                self.assertTrue(
                    torch.equal(fast.view(torch.uint8), ref.view(torch.uint8)),
                    f"pool {i} differs (online={online}, dtype={dtype}, slot={slot})",
                )

    def test_online_layout_matches_legacy_bytes(self):
        for dtype in _DTYPES:
            for slot in (0, _NUM_REQS - 1):
                self._assert_matches_legacy(online=True, dtype=dtype, slot=slot)

    def test_offline_layout_matches_legacy_bytes(self):
        for dtype in _DTYPES:
            for slot in (0, _NUM_REQS - 1):
                self._assert_matches_legacy(online=False, dtype=dtype, slot=slot)

    def test_other_slots_untouched(self):
        for online in (True, False):
            with mock.patch.object(pool_mod, "ONLINE_C128", online):
                state = _make_state(online=online, dtype=torch.float32)
                before = state.clone()
                span = 1 if online else _RING
                touched = slice(1 * span, 2 * span)

                _make_kv_pool([state]).clear_c128_req_state(1)

                untouched = torch.ones(state.shape[0], dtype=torch.bool)
                untouched[touched] = False
                self.assertTrue(torch.equal(state[untouched], before[untouched]))
                self.assertFalse(torch.equal(state[touched], before[touched]))

    def test_plan_built_once_and_reused(self):
        with mock.patch.object(pool_mod, "ONLINE_C128", True):
            kv_pool = _make_kv_pool([_make_state(online=True, dtype=torch.float32)])
            self.assertFalse(kv_pool._c128_clear_plan_built)

            kv_pool.clear_c128_req_state(0)
            self.assertTrue(kv_pool._c128_clear_plan_built)
            plan = kv_pool._c128_clear_plan
            self.assertIsNotNone(plan)

            kv_pool.clear_c128_req_state(1)
            self.assertIs(kv_pool._c128_clear_plan, plan)

    def test_heterogeneous_pools_fall_back_to_loop(self):
        for online in (True, False):
            with mock.patch.object(pool_mod, "ONLINE_C128", online):
                wide = _make_state(online=online, dtype=torch.float32, width_mult=2)
                narrow = _make_state(online=online, dtype=torch.float32)
                fast_states = [narrow, wide]
                ref_states = [s.clone() for s in fast_states]

                kv_pool = _make_kv_pool(fast_states)
                kv_pool.clear_c128_req_state(0)
                self.assertIsNone(kv_pool._c128_clear_plan, "mismatched widths")

                _legacy_clear(_make_kv_pool(ref_states), 0)
                for fast, ref in zip(fast_states, ref_states):
                    self.assertTrue(
                        torch.equal(fast.view(torch.uint8), ref.view(torch.uint8))
                    )

    def test_dtype_mismatch_falls_back_to_loop(self):
        with mock.patch.object(pool_mod, "ONLINE_C128", True):
            kv_pool = _make_kv_pool(
                [
                    _make_state(online=True, dtype=torch.float32),
                    _make_state(online=True, dtype=torch.bfloat16),
                ]
            )
            kv_pool.clear_c128_req_state(0)
            self.assertIsNone(kv_pool._c128_clear_plan)

    def test_non_c128_and_missing_pools_are_skipped(self):
        with mock.patch.object(pool_mod, "ONLINE_C128", True):
            c128 = _make_state(online=True, dtype=torch.float32)
            other = _make_state(online=True, dtype=torch.float32)
            other_before = other.clone()
            kv_pool = _make_kv_pool([c128], extra=(None, _make_pool(other, ratio=4)))

            kv_pool.clear_c128_req_state(0)

            self.assertIsNotNone(kv_pool._c128_clear_plan)
            self.assertEqual(len(kv_pool._c128_clear_plan[0]), 1)
            self.assertTrue(torch.equal(other, other_before))

    def test_no_c128_pool_yields_no_plan(self):
        with mock.patch.object(pool_mod, "ONLINE_C128", True):
            kv_pool = _make_kv_pool([], extra=(None,))
            kv_pool.clear_c128_req_state(0)
            self.assertIsNone(kv_pool._c128_clear_plan)


if __name__ == "__main__":
    unittest.main()
