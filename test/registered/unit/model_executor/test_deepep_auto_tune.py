"""Unit tests for DeepEP low_latency dispatch-cap auto-tuning.

All CPU-only and fully mocked:
  - The dispatcher resolves SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK
    lazily (honoring a value auto-tuned after construction) and caches it.
  - ModelRunner._maybe_auto_tune_deepep_num_max_dispatch_tokens sizes the cap to
    the scheduler's decode concurrency (request-pool size, capped at 1024), only
    for the "deepep" backend low_latency path, never overrides a user env, and only
    raises num_max when the auto mem_fraction reserved a ceiling (without one it
    stays at the static default, since the larger buffer was never reserved).
    The spec multiplier uses max_speculative_num_draft_tokens (the adaptive-spec
    upper bound), not just the startup value.
  - _clamp_deepep_low_latency_concurrency caps per-rank decode concurrency to the
    buffer's num_max (reserved ceiling, or the env/default when no reservation ran)
    and takes the EP-group minimum so the collective dispatch buffer is uniform and
    no rank overruns it.
"""

from __future__ import annotations

import contextlib
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ENV = envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK

_MIXIN_MODULE = "sglang.srt.model_executor.model_runner_kv_cache_mixin"

try:
    from sglang.srt.layers.moe.token_dispatcher.deepep import _DeepEPDispatcherImplBase

    _HAS_DEEPEP_MODULE = True
except Exception:
    _HAS_DEEPEP_MODULE = False

try:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
        ModelRunnerKVCacheMixin,
    )

    _HAS_MODEL_RUNNER = True
except Exception:
    _HAS_MODEL_RUNNER = False


@contextlib.contextmanager
def _env_unset():
    """Run with the env cleared, restoring whatever was there afterward."""
    was_set = _ENV.is_set()
    old = _ENV.get() if was_set else None
    _ENV.clear()
    try:
        yield
    finally:
        if was_set:
            _ENV.set(old)
        else:
            _ENV.clear()


def _deepep_runner(
    backend="deepep",
    mode="auto",
    num_draft_tokens=None,
    max_draft_tokens=None,
    reserved_num_max=None,
    **extra,
):
    # max_draft_tokens models the adaptive-spec ceiling; reserved_num_max models the
    # ceiling published by ServerArgs._adjust_mem_fraction_for_deepep_capture.
    runner = SimpleNamespace(
        server_args=SimpleNamespace(
            moe_a2a_backend=backend,
            deepep_mode=mode,
            speculative_num_draft_tokens=num_draft_tokens,
            max_speculative_num_draft_tokens=(
                max_draft_tokens if max_draft_tokens is not None else num_draft_tokens
            ),
            _deepep_reserved_num_max=reserved_num_max,
        ),
        device="cuda",
        gpu_id=0,
        **extra,
    )
    if _HAS_MODEL_RUNNER:
        runner._is_deepep_low_latency = types.MethodType(
            ModelRunnerKVCacheMixin._is_deepep_low_latency, runner
        )
        # The buffer-size drift-guard is covered by test_deepep_buffer_size; stub it
        # so these num_max tests need no full model_config / native deep_ep.
        runner._warn_on_deepep_buffer_size_drift = lambda num_max: None
    return runner


@unittest.skipUnless(_HAS_DEEPEP_MODULE, "deepep token dispatcher not importable")
class TestDeepEPNumMaxDispatchTokensProperty(unittest.TestCase):
    """The dispatcher field resolves from the env lazily, then caches."""

    def _stub(self):
        # Skip __init__ (needs DeepEP + a process group); only the lazy field matters.
        stub = _DeepEPDispatcherImplBase.__new__(_DeepEPDispatcherImplBase)
        stub._num_max_dispatch_tokens_per_rank = None
        return stub

    def test_resolves_default_when_unset(self):
        with _env_unset():
            self.assertEqual(self._stub().num_max_dispatch_tokens_per_rank, 128)

    def test_resolves_value_set_after_construction(self):
        stub = self._stub()
        with _ENV.override(512):
            self.assertEqual(stub.num_max_dispatch_tokens_per_rank, 512)

    def test_caches_first_read(self):
        stub = self._stub()
        with _ENV.override(256):
            self.assertEqual(stub.num_max_dispatch_tokens_per_rank, 256)
        # Env moved after the first read; the cached value must not follow it.
        with _ENV.override(512):
            self.assertEqual(stub.num_max_dispatch_tokens_per_rank, 256)

    def test_rejects_above_hard_cap(self):
        stub = self._stub()
        with _ENV.override(2048):
            with self.assertRaises(AssertionError):
                _ = stub.num_max_dispatch_tokens_per_rank


@unittest.skipUnless(_HAS_MODEL_RUNNER, "model_runner not importable")
class TestDeepEPAutoTune(unittest.TestCase):
    """ModelRunner._maybe_auto_tune_deepep_num_max_dispatch_tokens behavior."""

    def _runner(
        self,
        backend="deepep",
        mode="auto",
        pool_size=4096,
        num_draft_tokens=None,
        max_draft_tokens=None,
        reserved_num_max=None,
    ):
        return _deepep_runner(
            backend=backend,
            mode=mode,
            num_draft_tokens=num_draft_tokens,
            max_draft_tokens=max_draft_tokens,
            reserved_num_max=reserved_num_max,
            req_to_token_pool=SimpleNamespace(size=pool_size),
        )

    def _run(self, runner):
        ModelRunner._maybe_auto_tune_deepep_num_max_dispatch_tokens(runner)

    def test_user_env_always_wins(self):
        with _ENV.override(700):
            self._run(self._runner(pool_size=4096))
            self.assertEqual(_ENV.get(), 700)

    def test_no_op_for_non_deepep_backend(self):
        with _env_unset():
            self._run(self._runner(backend="none", pool_size=4096))
            self.assertFalse(_ENV.is_set())

    def test_no_op_for_normal_mode(self):
        with _env_unset():
            self._run(self._runner(mode="normal", pool_size=4096))
            self.assertFalse(_ENV.is_set())

    def test_raises_to_decode_concurrency(self):
        # With a non-binding reservation, num_max tracks decode concurrency.
        with _env_unset():
            self._run(self._runner(pool_size=512, reserved_num_max=1024))
            self.assertEqual(_ENV.get(), 512)

    def test_caps_at_finished_sum_tag(self):
        # req pool above the 1024 FINISHED_SUM_TAG ceiling clamps to 1024.
        with _env_unset():
            self._run(self._runner(pool_size=4096, reserved_num_max=1024))
            self.assertEqual(_ENV.get(), 1024)

    def test_no_reservation_keeps_default(self):
        # No reserved ceiling (kill-switch off / user-set mem_fraction / unreadable
        # config): the buffer for a larger num_max was never reserved, so auto-tune
        # must NOT raise it — stay at the static default instead of OOMing at capture.
        with _env_unset():
            self._run(self._runner(pool_size=4096, reserved_num_max=None))
            self.assertFalse(_ENV.is_set())
            self.assertEqual(_ENV.get(), 128)

    def test_mooncake_backend_not_tuned(self):
        # B2: only the "deepep" backend allocates the nvshmem low_latency buffer and
        # reads SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK; mooncake sizes its
        # own buffer from its own env, so the auto-tune leaves the env unset.
        with _env_unset():
            self._run(self._runner(backend="mooncake", pool_size=512))
            self.assertFalse(_ENV.is_set())

    def test_low_concurrency_keeps_default(self):
        # No reservation: env stays at the static default (never written).
        with _env_unset():
            self._run(self._runner(pool_size=64))
            self.assertFalse(_ENV.is_set())
            self.assertEqual(_ENV.get(), 128)

    def test_reserved_ceiling_caps_num_max(self):
        # The auto mem_fraction sized the buffer for reserved_num_max; the runtime
        # must not auto-tune above it even when decode concurrency is far higher,
        # or the larger buffer would OOM at capture.
        with _env_unset():
            self._run(self._runner(pool_size=4096, reserved_num_max=128))
            self.assertEqual(_ENV.get(), 128)

    def test_reserved_ceiling_below_default_is_written(self):
        # A reserved ceiling below the env default must still be written verbatim
        # (the buffer was reserved for exactly that num_max), unlike the no-
        # reservation low-concurrency case which keeps the default.
        with _env_unset():
            self._run(self._runner(pool_size=4096, reserved_num_max=64))
            self.assertTrue(_ENV.is_set())
            self.assertEqual(_ENV.get(), 64)

    def test_spec_scales_num_max_by_draft_tokens(self):
        # Spec verify dispatches num_draft_tokens per request, so num_max tracks
        # concurrency * num_draft_tokens, still clamped to the 1024 ceiling.
        with _env_unset():
            self._run(
                self._runner(pool_size=100, num_draft_tokens=4, reserved_num_max=1024)
            )
            self.assertEqual(_ENV.get(), 400)
        with _env_unset():
            self._run(
                self._runner(pool_size=512, num_draft_tokens=4, reserved_num_max=1024)
            )
            self.assertEqual(_ENV.get(), 1024)

    def test_adaptive_spec_uses_max_draft_tokens(self):
        # B1: adaptive spec can grow draft tokens to max_speculative_num_draft_tokens
        # at runtime, so num_max must track the max (8), not the startup value (2):
        # 100 * 8 = 800.
        with _env_unset():
            self._run(
                self._runner(
                    pool_size=100,
                    num_draft_tokens=2,
                    max_draft_tokens=8,
                    reserved_num_max=1024,
                )
            )
            self.assertEqual(_ENV.get(), 800)


@unittest.skipUnless(_HAS_MODEL_RUNNER, "model_runner not importable")
class TestDeepEPConcurrencyClamp(unittest.TestCase):
    """_clamp_deepep_low_latency_concurrency caps + EP-group-syncs concurrency."""

    def _clamp(self, runner, max_num_reqs):
        return ModelRunnerKVCacheMixin._clamp_deepep_low_latency_concurrency(
            runner, max_num_reqs
        )

    def test_passthrough_for_non_deepep(self):
        self.assertEqual(self._clamp(_deepep_runner(backend="none"), 2048), 2048)

    def test_passthrough_for_normal_mode(self):
        self.assertEqual(self._clamp(_deepep_runner(mode="normal"), 2048), 2048)

    def test_caps_to_reserved_ceiling_single_rank(self):
        # With a reservation at the 1024 ceiling, concurrency caps to it.
        with _env_unset(), patch(
            f"{_MIXIN_MODULE}.get_moe_ep_group",
            return_value=SimpleNamespace(world_size=1, cpu_group=None),
        ):
            self.assertEqual(
                self._clamp(_deepep_runner(reserved_num_max=1024), 2048), 1024
            )

    def test_no_reservation_caps_to_default_buffer(self):
        # No reservation (kill-switch off / user mem_fraction): the buffer is the
        # static default, so concurrency caps to it — NOT the loose FINISHED_SUM_TAG
        # bound, which would let the decode batch overrun the small default buffer.
        with _env_unset(), patch(
            f"{_MIXIN_MODULE}.get_moe_ep_group",
            return_value=SimpleNamespace(world_size=1, cpu_group=None),
        ):
            self.assertEqual(
                self._clamp(_deepep_runner(reserved_num_max=None), 2048), 128
            )

    def test_below_cap_unchanged_single_rank(self):
        with _env_unset(), patch(
            f"{_MIXIN_MODULE}.get_moe_ep_group",
            return_value=SimpleNamespace(world_size=1, cpu_group=None),
        ):
            self.assertEqual(
                self._clamp(_deepep_runner(reserved_num_max=1024), 512), 512
            )

    def test_mooncake_backend_not_capped(self):
        # B2: the concurrency clamp is deepep-only; mooncake passes through.
        with patch(
            f"{_MIXIN_MODULE}.get_moe_ep_group",
            return_value=SimpleNamespace(world_size=1, cpu_group=None),
        ):
            self.assertEqual(
                self._clamp(_deepep_runner(backend="mooncake"), 2048), 2048
            )

    def test_spec_divides_cap_by_draft_tokens(self):
        # num_draft_tokens=4 means batch * 4 tokens dispatched, so the ceiling
        # (reserved 1024) is divided: 1024 // 4 = 256.
        with _env_unset(), patch(
            f"{_MIXIN_MODULE}.get_moe_ep_group",
            return_value=SimpleNamespace(world_size=1, cpu_group=None),
        ):
            self.assertEqual(
                self._clamp(
                    _deepep_runner(num_draft_tokens=4, reserved_num_max=1024), 2048
                ),
                256,
            )

    def test_adaptive_spec_divides_cap_by_max_draft_tokens(self):
        # B1: the clamp divides by the adaptive max (8), not the startup value (2):
        # 1024 // 8 = 128.
        with _env_unset(), patch(
            f"{_MIXIN_MODULE}.get_moe_ep_group",
            return_value=SimpleNamespace(world_size=1, cpu_group=None),
        ):
            self.assertEqual(
                self._clamp(
                    _deepep_runner(
                        num_draft_tokens=2, max_draft_tokens=8, reserved_num_max=1024
                    ),
                    2048,
                ),
                128,
            )

    def test_takes_ep_group_minimum(self):
        # Simulate a peer rank contributing a smaller concurrency: cap is 800,
        # the group MIN drives it to 600.
        def fake_all_reduce(tensor, op=None, group=None):
            tensor.fill_(600)

        with _env_unset(), patch(
            f"{_MIXIN_MODULE}.get_moe_ep_group",
            return_value=SimpleNamespace(world_size=2, cpu_group=object()),
        ), patch("torch.distributed.all_reduce", side_effect=fake_all_reduce):
            self.assertEqual(
                self._clamp(_deepep_runner(reserved_num_max=1024), 800), 600
            )


if __name__ == "__main__":
    unittest.main()
