"""Regression: MlxModelRunnerStub must honor --max-running-requests instead of
hardcoding min(pool // 2, 4096). The stub previously ignored the flag, so MLX
concurrency silently diverged from what the user requested (and from CUDA).

Guards `_resolve_max_running_requests`, which mirrors the base runner's clamp
(model_runner_kv_cache_mixin._resolve_max_num_reqs). MLX-gated because importing
the stub pulls in mlx.core; the logic itself is pure integer arithmetic.
"""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"

if _HAS_MLX:
    from sglang.srt.hardware_backend.mlx.model_runner_stub import MlxModelRunnerStub


def _stub(max_running_requests, max_total_num_tokens, dp_size=1):
    """A stub carrying only what _resolve_max_running_requests reads."""
    stub = MlxModelRunnerStub.__new__(MlxModelRunnerStub)
    stub.server_args = SimpleNamespace(max_running_requests=max_running_requests)
    stub.max_total_num_tokens = max_total_num_tokens
    stub.dp_size = dp_size
    return stub


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxMaxRunningRequests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
