from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_mlx_ci

register_mlx_ci(est_time=2, suite="stage-a-unit-test-mlx")

_HAS_PROVIDER = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_vlm.speculative.drafters.gemma4_assistant")
    is not None
)

if _HAS_PROVIDER:
    import mlx.core as mx
    from registered.unit.hardware_backend.mlx.gemma4_test_utils import (
        tiny_assistant_config,
        tiny_gemma4,
        write_tiny_assistant_checkpoint,
    )

    from sglang.srt.hardware_backend.mlx.gemma4_mtp import (
        Gemma4MTPAssistantLoader,
        validate_gemma4_assistant_config,
    )
    from sglang.srt.hardware_backend.mlx.model_adapter import Gemma4TargetAdapter


@unittest.skipUnless(_HAS_PROVIDER, "requires mlx-vlm Gemma 4 assistant provider")
class TestGemma4MTPConfigAndLoader(unittest.TestCase):
    def setUp(self):
        self.target = tiny_gemma4()

    def test_canonical_and_compatibility_architectures(self):
        config = tiny_assistant_config()
        metadata = validate_gemma4_assistant_config(config, self.target)
        self.assertEqual(metadata.backbone_hidden_size, 16)
        self.assertEqual(metadata.assistant_hidden_size, 8)
        self.assertEqual(
            metadata.layer_types, tuple(config["text_config"]["layer_types"])
        )

        alias = copy.deepcopy(config)
        alias["architectures"] = ["Gemma4UnifiedAssistantForCausalLM"]
        self.assertEqual(
            validate_gemma4_assistant_config(alias, self.target).architecture,
            "Gemma4UnifiedAssistantForCausalLM",
        )

    def test_incompatible_metadata_fails_before_loading(self):
        base = tiny_assistant_config(ordered=True)
        mutations = {
            "target_config": lambda value: value.update(model_type="gemma4_text"),
            "remote_hook": lambda value: value.update(auto_map={"AutoModel": "x.py"}),
            "vocab": lambda value: value["text_config"].update(vocab_size=31),
            "backbone": lambda value: value.update(backbone_hidden_size=15),
            "assistant_hidden": lambda value: value["text_config"].update(
                hidden_size=0
            ),
            "layer_count": lambda value: value["text_config"].update(
                num_hidden_layers=3
            ),
            "tail": lambda value: value["text_config"]["layer_types"].reverse(),
            "tied": lambda value: value["text_config"].update(
                tie_word_embeddings=False
            ),
            "centroids": lambda value: value.update(num_centroids=3),
            "centroid_topk": lambda value: value.update(centroid_intermediate_top_k=5),
            "window": lambda value: value["text_config"].update(sliding_window=9),
            "head_dim": lambda value: value["text_config"].update(head_dim=5),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                config = copy.deepcopy(base)
                mutate(config)
                with self.assertRaises(ValueError):
                    validate_gemma4_assistant_config(config, self.target)

    def test_strict_weight_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_tiny_assistant_checkpoint(root)
            loader = Gemma4MTPAssistantLoader(self.target)
            runtime = loader.load(str(root))
            self.assertEqual(loader.load_count, 1)
            self.assertEqual(runtime.metadata.vocab_size, 32)

            weight_file = root / "model.safetensors"
            original = dict(mx.load(str(weight_file)))
            cases = {}
            missing = dict(original)
            missing.pop(sorted(missing)[0])
            cases["missing"] = missing
            unexpected = dict(original)
            unexpected["unexpected.weight"] = mx.zeros((1,))
            cases["unexpected"] = unexpected
            wrong = dict(original)
            first = sorted(wrong)[0]
            wrong[first] = wrong[first].reshape(-1)[:1]
            cases["wrong_shape"] = wrong

            for expected_message, weights in cases.items():
                with self.subTest(expected_message=expected_message):
                    mx.save_safetensors(str(weight_file), weights)
                    with self.assertRaisesRegex(ValueError, expected_message):
                        loader.load(str(root))

            # Duplicate names in separate shards are rejected explicitly.
            mx.save_safetensors(str(weight_file), original)
            duplicate_name = sorted(original)[0]
            mx.save_safetensors(
                str(root / "model-00002.safetensors"),
                {duplicate_name: original[duplicate_name]},
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                loader.load(str(root))

    def test_reload_replace_unload_invalidate_every_old_handle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint_a = root / "a"
            checkpoint_b = root / "b"
            write_tiny_assistant_checkpoint(checkpoint_a)
            write_tiny_assistant_checkpoint(checkpoint_b, weight_delta=2.0)
            loader = Gemma4MTPAssistantLoader(self.target)
            runtime_a = loader.load(str(checkpoint_a), revision="revision-a")
            handle_a = runtime_a.model_handle

            cache = self.target.make_cache()
            adapter = Gemma4TargetAdapter(self.target)
            output = adapter.forward(
                mx.array([[1, 2, 3]], dtype=mx.int32),
                cache=cache,
                collect_hidden_states=True,
            )
            token = int(mx.argmax(output.logits[:, -1, :], axis=-1).item())
            target_seed = adapter.make_seed(
                output, hidden_row_index=2, emitted_token_id=token
            )
            view_a = runtime_a.bind_request("request", cache)
            seed_a = runtime_a.bind_seed("request", target_seed)
            fingerprint_a = runtime_a.fingerprint

            # Same path/revision reload after changing the provider payload must
            # materialize fresh tensors and invalidate all generation-A objects.
            write_tiny_assistant_checkpoint(checkpoint_a, weight_delta=1.0)
            runtime_a2 = loader.load(str(checkpoint_a), revision="revision-a")
            self.assertNotEqual(runtime_a2.fingerprint, fingerprint_a)
            self.assertGreater(runtime_a2.generation, runtime_a.generation)
            for operation in (
                lambda: runtime_a.bind_request("late", cache),
                lambda: view_a.position,
                seed_a.validate,
                lambda: handle_a.fingerprint,
            ):
                with self.assertRaisesRegex(RuntimeError, "stale"):
                    operation()

            # Replacing A with B invalidates A2 even without an explicit unload.
            stale_a2 = runtime_a2.model_handle
            runtime_b = loader.replace_assistant(
                str(checkpoint_b), revision="revision-b"
            )
            self.assertNotEqual(runtime_b.fingerprint, runtime_a2.fingerprint)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                _ = stale_a2.fingerprint

            generation_b = runtime_b.generation
            loader.unload_assistant()
            self.assertIsNone(loader.runtime)
            self.assertGreater(loader.generation, generation_b)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                runtime_b.bind_request("late", cache)

    def test_request_flush_retains_loaded_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp)
            write_tiny_assistant_checkpoint(checkpoint)
            loader = Gemma4MTPAssistantLoader(self.target)
            runtime = loader.load(str(checkpoint))
            cache = self.target.make_cache()
            self.target(mx.array([[1]], dtype=mx.int32), cache=cache)
            view = runtime.bind_request("r", cache)
            fingerprint = runtime.fingerprint
            load_count = loader.load_count
            loader.clear_request_bindings()

            self.assertEqual(loader.runtime.fingerprint, fingerprint)
            self.assertEqual(loader.load_count, load_count)
            self.assertEqual(runtime.request_binding_count, 0)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                _ = view.position


if __name__ == "__main__":
    unittest.main()
