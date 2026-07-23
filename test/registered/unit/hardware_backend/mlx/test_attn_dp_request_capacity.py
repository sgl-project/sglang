"""Request-capacity accounting for pure DP and DP attention on MLX.

``max_running_requests`` is partitioned only when attention DP partitions a
logical batch across multiple KV-cache owners. Pure data-parallel replicas own
independent schedulers and caches, so each replica retains the full configured
limit even when the system DP size is greater than one.
"""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"

if _HAS_MLX:
    from sglang.srt.hardware_backend.mlx.model_runner_stub import (
        MlxModelRunnerStub,
    )


def _resolve_request_capacity(*, dp_size: int, attn_dp_size: int) -> int:
    stub = MlxModelRunnerStub.__new__(MlxModelRunnerStub)
    stub.dp_size = dp_size
    stub.ps = SimpleNamespace(attn_dp_size=attn_dp_size)
    stub.max_total_num_tokens = 64
    stub.model_config = SimpleNamespace()
    stub.server_args = SimpleNamespace(
        max_running_requests=8,
        max_mamba_cache_size=None,
        disable_radix_cache=False,
    )

    with mock.patch(
        "sglang.srt.hardware_backend.mlx.model_runner_stub.mambaish_config",
        return_value=None,
    ):
        return stub._resolve_max_running_requests()


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestAttentionDpRequestCapacity(unittest.TestCase):
    def test_pure_dp_replica_retains_full_request_limit(self):
        self.assertEqual(
            _resolve_request_capacity(dp_size=4, attn_dp_size=1),
            8,
        )

    def test_attention_dp_partitions_request_limit(self):
        self.assertEqual(
            _resolve_request_capacity(dp_size=4, attn_dp_size=4),
            2,
        )


if __name__ == "__main__":
    unittest.main()
