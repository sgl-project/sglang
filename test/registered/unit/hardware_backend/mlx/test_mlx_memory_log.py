"""SGLANG_MLX_MEMORY_LOG reports MLX allocator state at request lifecycle
points and costs nothing when unset; SGLANG_MLX_CACHE_LIMIT_GB caps the
buffer cache before the model loads."""

from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")

_HAS_MLX = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm") is not None
)

if _HAS_MLX:
    from mlx_lm.models import llama

    from sglang.srt.environ import envs
    from sglang.srt.hardware_backend.mlx import model_runner as runner_module
    from sglang.srt.hardware_backend.mlx.aot import MlxAOTKernelSet
    from sglang.srt.hardware_backend.mlx.kv_cache import (
        find_attention_layers,
        get_layer_window_sizes,
        patch_model_attention,
    )
    from sglang.srt.hardware_backend.mlx.kv_cache.layout import MlxModelCacheLayout
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

LOGGER = "sglang.srt.hardware_backend.mlx.model_runner"
PROMPT = [3, 14, 15, 9, 2]


def _runner(memory_log: bool, clear_steps: int = 0):
    model = llama.Model(
        llama.ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=64,
            head_dim=16,
            num_key_value_heads=1,
        )
    )
    patch_model_attention(model)
    layers, attrs = find_attention_layers(model)
    runner = MlxModelRunner.__new__(MlxModelRunner)
    runner.model = model
    runner.disable_radix_cache = False
    runner._cache_layout = MlxModelCacheLayout.from_attention_discovery(
        layers, attrs, layer_window_sizes=get_layer_window_sizes(model)
    )
    runner._max_seq_len = 64
    runner._cache_pool = []
    runner._req_caches = {}
    runner._req_token_ids = {}
    runner._req_sampling = {}
    runner._req_pool_idx = {}
    runner._req_synced_offset = {}
    runner._attention_kv_pool = None
    runner._decode_step_ct = 0
    runner._clear_steps = clear_steps
    runner._memory_log = memory_log
    runner._aot_kernels = MlxAOTKernelSet()
    runner._pool_size = 32
    runner.init_cache_pools(
        ReqToTokenPool(
            size=2, max_context_len=64, device="cpu", enable_memory_saver=False
        )
    )
    return runner


def _serve_one(runner, decode_steps: int):
    runner.prefill(
        req_id="r",
        new_token_ids=PROMPT,
        full_token_ids=PROMPT,
        prefix_slot_ids=[],
        new_slot_ids=list(range(1, len(PROMPT) + 1)),
        req_pool_idx=0,
    )
    for _ in range(decode_steps):
        runner.decode_batch(["r"])
    runner.remove_request("r")


@unittest.skipUnless(_HAS_MLX, "requires mlx + mlx_lm")
class TestMlxMemoryLog(CustomTestCase):
    def test_disabled_reads_no_allocator_state(self):
        runner = _runner(memory_log=False, clear_steps=2)
        with (
            mock.patch.object(runner_module.mx, "get_active_memory") as active,
            mock.patch.object(runner_module.mx, "get_cache_memory") as cache,
            mock.patch.object(runner_module.mx, "get_peak_memory") as peak,
            self.assertNoLogs(LOGGER, level="INFO"),
        ):
            _serve_one(runner, decode_steps=4)
        active.assert_not_called()
        cache.assert_not_called()
        peak.assert_not_called()

    def test_enabled_logs_each_lifecycle_point(self):
        runner = _runner(memory_log=True, clear_steps=2)
        with self.assertLogs(LOGGER, level="INFO") as logs:
            _serve_one(runner, decode_steps=2)
        events = [line.split("[", 1)[1].split("]", 1)[0] for line in logs.output]
        self.assertEqual(
            events,
            ["prefill", "before_clear_cache", "after_clear_cache", "release"],
        )
        for field in ("active=", "cache=", "peak=", "requests=", "idle_kv="):
            self.assertIn(field, logs.output[0])

    def test_release_reports_the_parked_request_kv(self):
        runner = _runner(memory_log=True)
        with self.assertLogs(LOGGER, level="INFO") as logs:
            _serve_one(runner, decode_steps=1)
        release = logs.output[-1]
        self.assertIn("requests=0", release)
        self.assertIn("idle_kv_lists=1", release)
        parked = sum(
            a.nbytes for a in MlxModelRunner.cache_state_arrays(runner._cache_pool)
        )
        self.assertGreater(parked, 0)

    def test_startup_reports_the_effective_cache_limit(self):
        runner = _runner(memory_log=True)
        limit = 2 * 1024**3
        previous = runner_module.mx.set_cache_limit(limit)
        try:
            with self.assertLogs(LOGGER, level="INFO") as logs:
                runner.log_memory("startup")
        finally:
            restored = runner_module.mx.set_cache_limit(previous)
        self.assertIn("(limit=2.00 GiB)", logs.output[0])
        self.assertEqual(restored, limit)


class _StopAfterCacheLimit(Exception):
    pass


@unittest.skipUnless(_HAS_MLX, "requires mlx + mlx_lm")
class TestMlxCacheLimit(CustomTestCase):
    def _construct(self):
        with mock.patch.object(
            MlxModelRunner, "_load_model", side_effect=_StopAfterCacheLimit
        ):
            with self.assertRaises(_StopAfterCacheLimit):
                MlxModelRunner(model_path="unused")

    def test_cap_is_set_before_model_load(self):
        with (
            envs.SGLANG_MLX_CACHE_LIMIT_GB.override(2.0),
            mock.patch.object(runner_module.mx, "set_cache_limit") as set_limit,
        ):
            self._construct()
        set_limit.assert_called_once_with(2 * 1024**3)

    def test_negative_cap_is_rejected(self):
        with (
            envs.SGLANG_MLX_CACHE_LIMIT_GB.override(-1.0),
            self.assertRaises(ValueError),
        ):
            MlxModelRunner(model_path="unused")


if __name__ == "__main__":
    unittest.main()
