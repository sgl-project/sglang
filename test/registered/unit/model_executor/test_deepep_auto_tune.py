"""Unit tests for DeepEP num_max_dispatch_tokens_per_rank auto-tuning.

Two pieces, both CPU-only and fully mocked:
  - The dispatcher resolves SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK
    lazily (honoring a value auto-tuned after construction) and caches it.
  - ModelRunner._maybe_auto_tune_deepep_num_max_dispatch_tokens picks the cap
    from post-KV free memory, only for the DeepEP low_latency path, and never
    overrides an explicit user env.
"""

from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ENV = envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK

try:
    from sglang.srt.layers.moe.token_dispatcher.deepep import _DeepEPDispatcherImplBase

    _HAS_DEEPEP_MODULE = True
except Exception:
    _HAS_DEEPEP_MODULE = False

try:
    from sglang.srt.model_executor.model_runner import ModelRunner

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

    def _runner(self, backend="deepep", mode="auto"):
        return SimpleNamespace(
            server_args=SimpleNamespace(moe_a2a_backend=backend, deepep_mode=mode),
            device="cuda",
            gpu_id=0,
        )

    def _run(self, runner, free_gb):
        with patch(
            "sglang.srt.model_executor.model_runner.get_available_gpu_memory",
            return_value=free_gb,
        ), patch(
            "sglang.srt.model_executor.model_runner.get_world_group",
            return_value=SimpleNamespace(world_size=1, cpu_group=None),
        ):
            ModelRunner._maybe_auto_tune_deepep_num_max_dispatch_tokens(runner)

    def test_user_env_always_wins(self):
        with _ENV.override(700):
            self._run(self._runner(), free_gb=200)
            self.assertEqual(_ENV.get(), 700)

    def test_no_op_for_non_deepep_backend(self):
        with _env_unset():
            self._run(self._runner(backend="none"), free_gb=200)
            self.assertFalse(_ENV.is_set())

    def test_no_op_for_normal_mode(self):
        with _env_unset():
            self._run(self._runner(mode="normal"), free_gb=200)
            self.assertFalse(_ENV.is_set())

    def test_picks_512_with_ample_free_mem(self):
        with _env_unset():
            self._run(self._runner(), free_gb=40)
            self.assertEqual(_ENV.get(), 512)

    def test_picks_512_at_threshold(self):
        with _env_unset():
            self._run(self._runner(), free_gb=32)
            self.assertEqual(_ENV.get(), 512)

    def test_picks_256_in_mid_band(self):
        with _env_unset():
            self._run(self._runner(mode="low_latency"), free_gb=25)
            self.assertEqual(_ENV.get(), 256)

    def test_picks_256_at_threshold(self):
        with _env_unset():
            self._run(self._runner(), free_gb=20)
            self.assertEqual(_ENV.get(), 256)

    def test_keeps_default_128_when_low(self):
        with _env_unset():
            self._run(self._runner(), free_gb=15)
            # 128 == default: never written, so the env stays unset.
            self.assertFalse(_ENV.is_set())
            self.assertEqual(_ENV.get(), 128)


if __name__ == "__main__":
    unittest.main()
