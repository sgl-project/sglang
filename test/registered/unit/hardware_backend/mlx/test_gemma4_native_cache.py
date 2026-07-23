"""Correctness-first Gemma 4 coverage for the MLX native-cache fallback.

The initial Apple Silicon path intentionally delegates cache semantics to
``mlx-lm``.  Gemma 4 combines YOCO K/V sharing, sliding attention, K=V full
attention, and heterogeneous head dimensions; the uniform SGLang MLX radix
pool cannot represent that layout yet.  These tests use a tiny randomly
initialised Gemma 4 model, so they exercise the real architecture without a
checkpoint download.
"""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

_HAS_GEMMA4_MLX = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm.models.gemma4_text") is not None
)
_SKIP_REASON = "requires mlx-lm with Gemma 4 support"

if _HAS_GEMMA4_MLX:
    from registered.unit.hardware_backend.mlx.gemma4_test_utils import (
        build_runner as _build_runner,
    )
    from registered.unit.hardware_backend.mlx.gemma4_test_utils import (
        reference_tokens as _reference_tokens,
    )
    from registered.unit.hardware_backend.mlx.gemma4_test_utils import (
        tiny_gemma4 as _tiny_gemma4,
    )

    from sglang.srt.hardware_backend.mlx.model_runner_stub import MlxModelRunnerStub
    from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker


@unittest.skipUnless(_HAS_GEMMA4_MLX, _SKIP_REASON)
class TestGemma4NativeCacheFallback(unittest.TestCase):
    def test_wrapped_model_cross_window_matches_raw_mlx_lm(self):
        # Exercise mlx-lm's conditional-generation shell as loaded from the
        # public checkpoints, and cross the synthetic sliding window boundary.
        model = _tiny_gemma4(wrapped=True)
        prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expected = _reference_tokens(model, prompt, steps=3)
        runner = _build_runner(model)

        first = runner.prefill(
            req_id="request",
            new_token_ids=prompt,
            full_token_ids=prompt,
            prefix_slot_ids=[],
            new_slot_ids=[],
            req_pool_idx=0,
        )
        actual = [first]
        for _ in range(2):
            actual.append(runner.decode_batch(["request"])[0])

        self.assertTrue(runner.native_cache_fallback)
        self.assertEqual(actual, expected)
        # Four transformer layers own only two caches; the remaining YOCO
        # layers reuse the most recent sliding/full cache respectively.
        self.assertEqual(len(runner._req_caches["request"]), 2)

    def test_multi_request_decode_preserves_cache_isolation(self):
        model = _tiny_gemma4()
        expected_a = _reference_tokens(model, [1, 2], steps=2)
        expected_b = _reference_tokens(model, [3, 4, 5], steps=2)
        runner = _build_runner(model)

        first_a = runner.prefill("a", [1, 2], [1, 2], [], [], 0)
        first_b = runner.prefill("b", [3, 4, 5], [3, 4, 5], [], [], 1)
        second_a, second_b = runner.decode_batch(["a", "b"])

        self.assertEqual([first_a, second_a], expected_a)
        self.assertEqual([first_b, second_b], expected_b)

    def test_radix_cache_fails_with_actionable_message(self):
        model = _tiny_gemma4()
        with self.assertRaisesRegex(NotImplementedError, "disable-radix-cache"):
            _build_runner(
                model,
                disable_radix_cache=False,
            )

    def test_default_scheduler_capacity_is_conservative(self):
        model = _tiny_gemma4()
        runner = _build_runner(model, pool_size=None)

        self.assertEqual(runner.pool_size, 2048)

    def test_native_cache_defaults_to_one_live_request(self):
        stub = MlxModelRunnerStub.__new__(MlxModelRunnerStub)
        stub._mlx_native_cache_fallback = True
        stub.max_total_num_tokens = 2048
        stub.model_config = SimpleNamespace()
        schedule = SimpleNamespace(
            max_running_requests=None,
            max_mamba_cache_size=None,
        )

        with (
            mock.patch(
                "sglang.srt.hardware_backend.mlx.model_runner_stub.get_schedule",
                return_value=schedule,
            ),
            mock.patch(
                "sglang.srt.hardware_backend.mlx.model_runner_stub.mambaish_config",
                return_value=None,
            ),
        ):
            self.assertEqual(stub._resolve_max_running_requests(), 1)

    def test_explicit_concurrency_uses_attention_dp_width(self):
        stub = MlxModelRunnerStub.__new__(MlxModelRunnerStub)
        stub._mlx_native_cache_fallback = True
        stub.max_total_num_tokens = 2048
        stub.model_config = SimpleNamespace()
        stub.ps = SimpleNamespace(attn_dp_size=2)
        schedule = SimpleNamespace(
            max_running_requests=8,
            max_mamba_cache_size=None,
        )

        with (
            mock.patch(
                "sglang.srt.hardware_backend.mlx.model_runner_stub.get_schedule",
                return_value=schedule,
            ),
            mock.patch(
                "sglang.srt.hardware_backend.mlx.model_runner_stub.mambaish_config",
                return_value=None,
            ),
        ):
            self.assertEqual(stub._resolve_max_running_requests(), 4)

    def test_finished_native_request_is_released_immediately(self):
        worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
        worker._mlx_runner = mock.Mock(native_cache_fallback=True)
        worker._mlx_runner.has_request.return_value = True
        worker._mlx_active_rids = {"done"}
        req = SimpleNamespace(rid="done", mamba_last_track_seqlen=128)

        worker.prepare_for_kv_cache_release(req)

        worker._mlx_runner.store_auxiliary_state_for_request.assert_called_once_with(
            "done"
        )
        worker._mlx_runner.remove_request.assert_called_once_with("done")
        self.assertNotIn("done", worker._mlx_active_rids)
        self.assertIsNone(req.mamba_last_track_seqlen)


if __name__ == "__main__":
    unittest.main()
