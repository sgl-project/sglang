from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sglang.test.ci.ci_register import register_mlx_ci

register_mlx_ci(est_time=2, suite="stage-a-unit-test-mlx")

_HAS_PROVIDER = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_vlm.speculative.drafters.gemma4_assistant")
    is not None
)

if _HAS_PROVIDER:
    import mlx.core as mx
    from mlx_vlm.speculative.drafters.gemma4_assistant.masks import (
        make_drafter_masks,
    )
    from registered.unit.hardware_backend.mlx.gemma4_test_utils import (
        native_cache_snapshot,
        tiny_gemma4,
        write_tiny_assistant_checkpoint,
    )

    from sglang.srt.hardware_backend.mlx.gemma4_mtp import (
        Gemma4MTPAssistantLoader,
    )
    from sglang.srt.hardware_backend.mlx.model_adapter import (
        Gemma4TargetAdapter,
        MlxTargetSeed,
    )


def _eager_reference(model, target_seed, shared, position, softcap):
    inputs = mx.concatenate(
        (target_seed.token_embedding, target_seed.hidden_state), axis=-1
    )
    h = model.pre_projection(inputs)
    masks = make_drafter_masks(
        shared,
        query_len=1,
        query_offset=position,
        sliding_window=model.config.text_config.sliding_window,
        dtype=h.dtype,
    )
    offset = mx.array(position)
    for layer in model.model.layers:
        h, _, _ = layer(
            h,
            mask=masks[layer.layer_type],
            cache=None,
            per_layer_input=None,
            shared_kv=shared[layer.layer_type],
            offset=offset,
        )
    h = model.model.norm(h)
    _projected = model.post_projection(h)
    logits = model._lm_head_fn(h)
    if softcap is not None:
        logits = mx.tanh(logits / softcap) * softcap
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)
    return int(token.item())


@unittest.skipUnless(_HAS_PROVIDER, "requires mlx-vlm Gemma 4 assistant provider")
class TestGemma4MTPModel(unittest.TestCase):
    def _loaded_case(self, root: Path, *, ordered: bool, prompt_len: int = 11):
        target = tiny_gemma4()
        write_tiny_assistant_checkpoint(root, ordered=ordered)
        runtime = Gemma4MTPAssistantLoader(target).load(str(root))
        adapter = Gemma4TargetAdapter(target)
        prompt = [1 + (index % 29) for index in range(prompt_len)]
        cache = target.make_cache()
        output = adapter.forward(
            mx.array([prompt], dtype=mx.int32),
            cache=cache,
            collect_hidden_states=True,
        )
        token = int(mx.argmax(output.logits[:, -1, :], axis=-1).item())
        seed = runtime.bind_seed(
            "request",
            adapter.make_seed(
                output,
                hidden_row_index=len(prompt) - 1,
                emitted_token_id=token,
            ),
        )
        view = runtime.bind_request("request", cache)
        return runtime, cache, seed, view

    def test_dense_and_ordered_tokens_match_eager_q_only_reference(self):
        for ordered in (False, True):
            with self.subTest(ordered=ordered), tempfile.TemporaryDirectory() as temp:
                runtime, cache, seed, view = self._loaded_case(
                    Path(temp), ordered=ordered
                )
                before = native_cache_snapshot(cache)
                shared = view.shared_kv_states()
                expected = _eager_reference(
                    runtime._model,
                    seed.target_seed,
                    shared,
                    view.position,
                    runtime.metadata.final_logit_softcapping,
                )
                actual = runtime.propose_one(seed, view)
                self.assertEqual(actual, expected)

                after = native_cache_snapshot(cache)
                for left, right in zip(before, after):
                    self.assertEqual(
                        (left["offset"], left["idx"]),
                        (right["offset"], right["idx"]),
                    )
                    np.testing.assert_array_equal(left["keys"], right["keys"])
                    np.testing.assert_array_equal(left["values"], right["values"])

    def test_seed_shape_and_request_identity_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime, cache, seed, view = self._loaded_case(Path(temp), ordered=False)
            bad_target_seed = MlxTargetSeed(
                token_id=seed.target_seed.token_id,
                hidden_state=mx.zeros((1, 2, 16)),
                token_embedding=seed.target_seed.token_embedding,
            )
            bad_seed = runtime.bind_seed("request", bad_target_seed)
            with self.assertRaisesRegex(ValueError, "shapes"):
                runtime.propose_one(bad_seed, view)

            other_view = runtime.bind_request("other", cache)
            with self.assertRaisesRegex(ValueError, "different requests"):
                runtime.propose_one(seed, other_view)


if __name__ == "__main__":
    unittest.main()
