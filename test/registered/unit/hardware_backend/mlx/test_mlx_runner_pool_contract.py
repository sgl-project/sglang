"""Guard the MLX stub's ``ModelRunner`` overrides against drift.

The base ``ModelRunner.alloc_memory_pool`` runs ``_init_pools`` which
asserts ``is_draft_worker`` (model_runner_kv_cache_mixin.py:409); the
MLX stub manages its own KV cache via ``MlxAttentionKVPool`` and must
short-circuit that GPU-allocation path.  If the override is lost, every
MLX startup crashes inside ``Scheduler.init_target_memory_pool``.

Similarly, the base ``init_attention_backends`` constructs the torch
attention backend named by ``server_args.attention_backend``; MLX never
uses one, and some backends read real KV buffers in ``__init__``, which
crashes on ``_DummyKVCache``.

The base ``preloaded_weights_bytes`` property reads the Torch model loader.
The MLX stub never creates one because its native runner owns model loading
and KV sizing, so it must report zero for the Torch accounting hook.

The checks are MLX-gated because importing the stub pulls in ``mlx.core``.
"""

from __future__ import annotations

import importlib.util
import inspect
import unittest
from types import SimpleNamespace

from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"

if _HAS_MLX:
    from sglang.srt.hardware_backend.mlx.model_runner_stub import MlxModelRunnerStub
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxRunnerPoolContract(CustomTestCase):
    """Guard the stub's scheduler-facing ``ModelRunner`` contracts."""

    def test_worker_startup_uses_mlx_token_capacity(self):
        """MLX startup must report its own pool capacity without a Torch configurator."""
        runner = object.__new__(MlxModelRunnerStub)
        runner.max_total_num_tokens = 64
        runner.max_running_requests = 2
        runner.is_hybrid_swa = False
        runner.forward_stream = None
        runner.req_to_token_pool = ReqToTokenPool(
            size=2,
            max_context_len=128,
            device="cpu",
            enable_memory_saver=False,
        )
        runner.token_to_kv_pool = SimpleNamespace(size=64)
        worker = object.__new__(TpModelWorker)
        worker._model_runner = runner
        worker.model_runner_list = [runner]
        worker.model_config = SimpleNamespace(context_len=128)
        worker.dllm_algorithm = None
        worker.random_seed = 0
        worker.device = "cpu"

        with get_context().override_server_args(
            max_prefill_tokens=128, max_queued_requests=None
        ):
            worker.alloc_memory_pool()
            info = worker.get_worker_info()

        self.assertEqual(info[0], 64)
        self.assertEqual(info[4], 63)
        self.assertEqual(info[5], 58)

    def test_stub_reports_no_preloaded_torch_weights(self):
        runner = object.__new__(MlxModelRunnerStub)
        runner.pre_model_load_memory = 1.0
        self.assertEqual(runner.preloaded_weights_bytes, 0)
        runner.account_preloaded_weights(runner.preloaded_weights_bytes)
        self.assertEqual(runner.pre_model_load_memory, 1.0)

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

    def test_stub_overrides_base_init_attention_backends(self):
        self.assertIn(
            "init_attention_backends",
            vars(MlxModelRunnerStub),
            msg=(
                "MlxModelRunnerStub lost its init_attention_backends "
                "override. The base implementation constructs the backend "
                "named by server_args.attention_backend; some backends "
                "read real KV buffers in __init__, which crashes on "
                "_DummyKVCache. MLX never uses a torch attention backend "
                "— re-add the override that keeps attn_backend = None."
            ),
        )
        self.assertIsNot(
            MlxModelRunnerStub.init_attention_backends,
            ModelRunner.init_attention_backends,
            msg="init_attention_backends must be overridden on the MLX "
            "stub, not inherited from ModelRunner.",
        )

    def test_stub_init_attention_backends_keeps_attn_backend_none(self):
        runner = object.__new__(MlxModelRunnerStub)
        MlxModelRunnerStub.init_attention_backends(runner)
        self.assertIsNone(runner.attn_backend)


if __name__ == "__main__":
    unittest.main()
