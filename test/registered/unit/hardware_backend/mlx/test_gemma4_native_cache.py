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
from dataclasses import asdict
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
    import mlx.core as mx
    from mlx_lm.models import gemma4, gemma4_text

    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
    from sglang.srt.hardware_backend.mlx.model_runner_stub import MlxModelRunnerStub
    from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker


def _tiny_gemma4(*, wrapped: bool = False):
    """Build a four-layer model containing every target cache quirk."""
    args = gemma4_text.ModelArgs(
        hidden_size=16,
        num_hidden_layers=4,
        intermediate_size=32,
        num_attention_heads=2,
        head_dim=4,
        global_head_dim=8,
        vocab_size=32,
        vocab_size_per_layer_input=32,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        num_kv_shared_layers=2,
        hidden_size_per_layer_input=0,
        sliding_window=8,
        sliding_window_pattern=2,
        max_position_embeddings=4096,
        attention_k_eq_v=True,
        use_double_wide_mlp=False,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
    )
    model = (
        gemma4.Model(
            gemma4.ModelArgs(
                text_config=asdict(args),
                vocab_size=args.vocab_size,
            )
        )
        if wrapped
        else gemma4_text.Model(args)
    )
    mx.eval(model.parameters())
    return model


def _build_runner(model, **kwargs):
    options = {
        "model_path": "tiny-gemma4",
        "disable_radix_cache": True,
        "pool_size": 64,
    }
    options.update(kwargs)
    with mock.patch.object(
        MlxModelRunner,
        "_load_model",
        new=lambda runner: setattr(runner, "model", model),
    ):
        return MlxModelRunner(**options)


def _reference_tokens(model, prompt: list[int], steps: int) -> list[int]:
    cache = model.make_cache()
    input_ids = mx.array([prompt], dtype=mx.int32)
    output = []
    for _ in range(steps):
        logits = model(input_ids, cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token, *[value for entry in cache for value in entry.state])
        output.append(int(token.item()))
        input_ids = token[:, None]
    return output


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
