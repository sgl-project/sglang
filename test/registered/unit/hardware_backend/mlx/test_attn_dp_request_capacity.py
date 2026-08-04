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

from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"

if _HAS_MLX:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.hardware_backend.mlx.model_runner_stub import (
        MLX_AUX_STATE_SIZE_MAX_RUNNING_REQUESTS_RATIO as RATIO,
    )
    from sglang.srt.hardware_backend.mlx.model_runner_stub import (
        MlxModelRunnerStub,
    )


def _arch(*, hybrid: bool):
    return mock.patch(
        "sglang.srt.hardware_backend.mlx.model_runner_stub.mambaish_config",
        return_value=object() if hybrid else None,
    )


def _stub_for_initialize(
    test,
    *,
    dp_size: int,
    attn_dp_size: int,
    max_running_requests: int = 8,
    max_mamba_cache_size: int | None = None,
    pool_size: int = 64,
):
    # ``initialize`` reads the config namespaces, so the config has to be
    # published rather than stubbed onto the runner.
    override = get_context().override_server_args(
        enable_memory_saver=False,
        max_running_requests=max_running_requests,
        max_mamba_cache_size=max_mamba_cache_size,
        disable_radix_cache=False,
    )
    server_args = override.install()
    test.addCleanup(override.restore)

    stub = MlxModelRunnerStub.__new__(MlxModelRunnerStub)
    stub._mlx_pool_size = pool_size
    stub.device = "cpu"
    stub.ps = ParallelState.trivial(dp_size=dp_size, attn_dp_size=attn_dp_size)
    stub.server_args = server_args
    stub.model_config = SimpleNamespace(
        is_hybrid_swa=False,
        sliding_window_size=None,
        attention_chunk_size=None,
        dtype="float16",
        num_hidden_layers=1,
        num_attention_layers=1,
        context_len=64,
        use_ngram_embedding=False,
    )
    return stub


def _initialize_stub(stub, *, hybrid: bool = False):
    with _arch(hybrid=hybrid):
        stub.initialize()
    return stub


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestAttentionDpRequestCapacity(CustomTestCase):
    def test_pure_dp_replica_retains_full_request_limit(self):
        stub = _initialize_stub(
            _stub_for_initialize(self, dp_size=4, attn_dp_size=1),
        )
        self.assertEqual(stub.max_running_requests, 8)
        self.assertEqual(stub.req_to_token_pool.size, 8)

    def test_attention_dp_partitions_request_limit(self):
        stub = _initialize_stub(
            _stub_for_initialize(self, dp_size=4, attn_dp_size=4),
        )
        self.assertEqual(stub.max_running_requests, 2)
        self.assertEqual(stub.req_to_token_pool.size, 2)

    def test_attention_dp_partitions_explicit_auxiliary_state_limit(self):
        stub = _initialize_stub(
            _stub_for_initialize(
                self,
                dp_size=4,
                attn_dp_size=4,
                max_running_requests=8,
                max_mamba_cache_size=4 * RATIO,
            ),
            hybrid=True,
        )

        self.assertEqual(stub.max_running_requests, 1)
        self.assertEqual(stub.req_to_token_pool.size, 1)
        self.assertEqual(stub.req_to_token_pool.auxiliary_state_pool.size, RATIO)

    def test_attention_dp_auxiliary_error_reports_global_cli_units(self):
        stub = _stub_for_initialize(
            self,
            dp_size=4,
            attn_dp_size=4,
            max_running_requests=8,
            max_mamba_cache_size=4 * RATIO - 1,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "max_mamba_cache_size=15.*per-worker auxiliary-state cap=3.*" "at least 16",
        ):
            _initialize_stub(stub, hybrid=True)


if __name__ == "__main__":
    unittest.main()
