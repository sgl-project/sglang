from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_mlx_ci

register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

_HAS_MLX = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm.models.gemma4_text") is not None
)

if _HAS_MLX:
    import mlx.core as mx
    from registered.unit.hardware_backend.mlx.gemma4_test_utils import (
        assert_native_cache_equal,
        tiny_gemma4,
    )

from sglang.srt.hardware_backend.mlx.model_adapter import (
    Gemma4TargetAdapter,
    MlxTargetForwardOutput,
)
from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner


@unittest.skipUnless(_HAS_MLX, "requires MLX and mlx-lm Gemma 4")
class TestGemma4TargetAdapter(unittest.TestCase):
    def test_verify_queries_preserve_one_token_decode_shape(self):
        calls = []

        class RecordingAdapter:
            def forward(self, input_ids, *, cache, collect_hidden_states):
                calls.append((tuple(input_ids.shape), collect_hidden_states))
                width = input_ids.shape[1]
                return MlxTargetForwardOutput(
                    logits=mx.zeros((1, width, 7)),
                    hidden_states=(
                        mx.zeros((1, width, 5)) if collect_hidden_states else None
                    ),
                )

        runner = object.__new__(MlxModelRunner)
        runner._target_adapter = RecordingAdapter()
        output = runner._forward_native_queries_sequential(
            [], (11, 12), collect_hidden_states=True
        )
        self.assertEqual(calls, [((1, 1), True), ((1, 1), True)])
        self.assertEqual(output.logits.shape, (1, 2, 7))
        self.assertEqual(output.hidden_states.shape, (1, 2, 5))

        calls.clear()
        output = runner._forward_native_queries_sequential(
            [], (13, 14), collect_hidden_states=False
        )
        self.assertEqual(calls, [((1, 1), False), ((1, 1), False)])
        self.assertEqual(output.logits.shape, (1, 2, 7))
        self.assertIsNone(output.hidden_states)

    def test_wrapped_and_unwrapped_capture_preserve_exact_logits(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                model = tiny_gemma4(wrapped=wrapped)
                adapter = Gemma4TargetAdapter(model)
                ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

                normal_cache = model.make_cache()
                normal = model(ids, cache=normal_cache)
                uncaptured_cache = model.make_cache()
                uncaptured = adapter.forward(ids, cache=uncaptured_cache)
                captured_cache = model.make_cache()
                captured = adapter.forward(
                    ids, cache=captured_cache, collect_hidden_states=True
                )
                mx.eval(
                    normal, uncaptured.logits, captured.logits, captured.hidden_states
                )

                np.testing.assert_array_equal(
                    np.array(normal), np.array(uncaptured.logits)
                )
                np.testing.assert_array_equal(
                    np.array(normal), np.array(captured.logits)
                )
                np.testing.assert_array_equal(
                    np.array(mx.argmax(normal, axis=-1)),
                    np.array(mx.argmax(captured.logits, axis=-1)),
                )
                self.assertIsNone(uncaptured.hidden_states)
                self.assertEqual(captured.hidden_states.shape, (1, 4, 16))
                assert_native_cache_equal(self, uncaptured_cache, normal_cache)
                assert_native_cache_equal(self, captured_cache, normal_cache)

    def test_scaled_embeddings_and_seed_row_selection(self):
        model = tiny_gemma4()
        adapter = Gemma4TargetAdapter(model)
        ids = mx.array([[1, 2, 3]], dtype=mx.int32)
        output = adapter.forward(
            ids, cache=model.make_cache(), collect_hidden_states=True
        )

        scaled = adapter.input_embeddings(ids)
        expected = model.model.embed_tokens(ids) * model.model.embed_scale
        seed = adapter.make_seed(output, hidden_row_index=1, emitted_token_id=7)
        mx.eval(scaled, expected, seed.hidden_state, seed.token_embedding)

        np.testing.assert_array_equal(np.array(scaled), np.array(expected))
        np.testing.assert_array_equal(
            np.array(seed.hidden_state), np.array(output.hidden_states[:, 1:2, :])
        )
        self.assertEqual(seed.hidden_state.shape, (1, 1, 16))
        self.assertEqual(seed.token_embedding.shape, (1, 1, 16))

    def test_final_prefill_and_verify_seed_semantics(self):
        model = tiny_gemma4()
        adapter = Gemma4TargetAdapter(model)
        prefill = adapter.forward(
            mx.array([[1, 2, 3]], dtype=mx.int32),
            cache=model.make_cache(),
            collect_hidden_states=True,
        )
        prefill_seed = adapter.make_seed(
            prefill,
            hidden_row_index=prefill.hidden_states.shape[1] - 1,
            emitted_token_id=4,
        )
        self.assertEqual(prefill_seed.hidden_state.shape, (1, 1, 16))

        verify = adapter.forward(
            mx.array([[4, 5]], dtype=mx.int32),
            cache=model.make_cache(),
            collect_hidden_states=True,
        )
        # Rejection commits row 0 and pairs it with the sampled mismatch;
        # acceptance commits row 1 and pairs it with the sampled bonus.
        reject_seed = adapter.make_seed(verify, hidden_row_index=0, emitted_token_id=6)
        accept_seed = adapter.make_seed(verify, hidden_row_index=1, emitted_token_id=7)
        np.testing.assert_array_equal(
            np.array(reject_seed.hidden_state),
            np.array(verify.hidden_states[:, 0:1, :]),
        )
        np.testing.assert_array_equal(
            np.array(accept_seed.hidden_state),
            np.array(verify.hidden_states[:, 1:2, :]),
        )


if __name__ == "__main__":
    unittest.main()
