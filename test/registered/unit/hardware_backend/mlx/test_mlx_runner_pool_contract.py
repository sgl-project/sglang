"""Guard the MLX stub's ``alloc_memory_pool`` override against drift.

The base ``ModelRunner.alloc_memory_pool`` runs ``_init_pools`` which
asserts ``is_draft_worker`` (model_runner_kv_cache_mixin.py:409); the
MLX stub manages its own KV cache via ``MlxAttentionKVPool`` and must
short-circuit that GPU-allocation path.  If the override is lost, every
MLX startup crashes inside ``Scheduler.init_target_memory_pool``.

The checks are signature/identity-only and MLX-gated because importing
the stub pulls in ``mlx.core``.
"""

from __future__ import annotations

import importlib.util
import inspect
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"

if _HAS_MLX:
    from sglang.srt.hardware_backend.mlx.model_runner_stub import MlxModelRunnerStub
    from sglang.srt.model_executor.model_runner import ModelRunner


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxRunnerPoolContract(unittest.TestCase):
    """``MlxModelRunnerStub.alloc_memory_pool`` must override the base."""

    def test_stub_overrides_base_alloc_memory_pool(self):
        self.assertIn(
            "alloc_memory_pool",
            vars(MlxModelRunnerStub),
            msg=(
                "MlxModelRunnerStub lost its alloc_memory_pool override. "
                "Without it the base ModelRunner.alloc_memory_pool runs "
                "_init_pools, which asserts is_draft_worker "
                "(model_runner_kv_cache_mixin.py:409) and crashes every "
                "MLX startup. Re-add the no-op override."
            ),
        )
        self.assertIsNot(
            MlxModelRunnerStub.alloc_memory_pool,
            ModelRunner.alloc_memory_pool,
            msg="alloc_memory_pool must be overridden on the MLX stub, "
            "not inherited from ModelRunner.",
        )

    def test_stub_alloc_memory_pool_binds_with_no_args(self):
        sig = inspect.signature(MlxModelRunnerStub.alloc_memory_pool)
        try:
            sig.bind(object())
        except TypeError as exc:
            self.fail(
                "MlxModelRunnerStub.alloc_memory_pool must accept a no-arg "
                f"call (scheduler default): {exc}"
            )

    def test_stub_alloc_memory_pool_binds_with_optional_config(self):
        class _FakeConfig:
            pass

        sig = inspect.signature(MlxModelRunnerStub.alloc_memory_pool)
        try:
            sig.bind(object(), _FakeConfig())
        except TypeError as exc:
            self.fail(
                "MlxModelRunnerStub.alloc_memory_pool must accept an "
                f"optional MemoryPoolConfig argument: {exc}"
            )


if __name__ == "__main__":
    unittest.main()