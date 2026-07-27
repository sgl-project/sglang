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

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

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

    def test_initialize_sets_dp_size_for_max_running_requests(self):
        """Regression: serving with ``--max-running-requests`` crashed with
        ``AttributeError: 'MlxModelRunnerStub' object has no attribute
        'dp_size'``.

        ``_resolve_max_running_requests`` splits the requested concurrency
        across dp workers via ``self.dp_size``, but the base ``ModelRunner``
        only ever sets ``dcp_size`` (never ``dp_size``) and the stub overrides
        ``initialize()`` without calling ``super().initialize()``, so
        ``initialize`` must derive ``dp_size`` from ``server_args`` itself.
        Every MLX e2e correctness test passes ``--max-running-requests 1``,
        so without this the whole MLX serving lane is broken.
        """
        import inspect
        from unittest.mock import MagicMock, patch

        import sglang.srt.configs.hybrid_arch as hybrid_arch

        init_src = inspect.getsource(MlxModelRunnerStub.initialize)
        self.assertIn(
            "self.dp_size = self.server_args.dp_size",
            init_src,
            msg=(
                "MlxModelRunnerStub.initialize must set self.dp_size from "
                "server_args before _resolve_max_running_requests runs. The "
                "base ModelRunner never sets dp_size (only dcp_size) and "
                "this override skips super().initialize()."
            ),
        )

        # The split must divide the requested cap across dp workers without
        # touching the missing attribute.
        stub = MlxModelRunnerStub.__new__(MlxModelRunnerStub)
        stub.max_total_num_tokens = 10000
        stub.dp_size = 2
        stub.server_args = MagicMock(
            max_running_requests=4, dp_size=2, max_mamba_cache_size=None
        )
        stub.model_config = MagicMock()
        with patch.object(hybrid_arch, "mambaish_config", return_value=None):
            # 4 // dp_size(2) = 2; capacity_cap = 10000 // 2 = 5000; -> 2
            self.assertEqual(stub._resolve_max_running_requests(), 2)


if __name__ == "__main__":
    unittest.main()
