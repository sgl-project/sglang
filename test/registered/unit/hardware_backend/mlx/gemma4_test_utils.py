"""Shared tiny Gemma 4 fixtures for offline MLX architecture tests."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import mlx.core as mx
import numpy as np
from mlx_lm.models import gemma4, gemma4_text

from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner


def tiny_gemma4(*, wrapped: bool = False):
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


def build_runner(model, **kwargs):
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


def reference_tokens(model, prompt: list[int], steps: int) -> list[int]:
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


def cache_logical_length(cache) -> int:
    offsets = {int(entry.offset) for entry in cache}
    if len(offsets) != 1:
        raise AssertionError(f"native cache offsets disagree: {sorted(offsets)}")
    return offsets.pop()


def native_cache_snapshot(cache):
    """Capture logical K/V in temporal order plus all visible ring metadata."""

    snapshot = []
    for entry in cache:
        keys = getattr(entry, "keys", None)
        values = getattr(entry, "values", None)
        if keys is None:
            logical_keys = logical_values = None
        elif hasattr(entry, "_temporal_order"):
            logical_keys = entry._temporal_order(keys)
            logical_values = entry._temporal_order(values)
            valid = min(int(entry.offset), int(entry.max_size))
            logical_keys = (
                logical_keys[..., -valid:, :] if valid else logical_keys[..., :0, :]
            )
            logical_values = (
                logical_values[..., -valid:, :] if valid else logical_values[..., :0, :]
            )
        else:
            valid = int(entry.offset)
            logical_keys = keys[..., :valid, :]
            logical_values = values[..., :valid, :]

        arrays = [item for item in (logical_keys, logical_values) if item is not None]
        if arrays:
            mx.eval(*arrays)
        snapshot.append(
            {
                "type": type(entry).__name__,
                "offset": int(entry.offset),
                "idx": getattr(entry, "_idx", None),
                "keep": getattr(entry, "keep", None),
                "max_size": getattr(entry, "max_size", None),
                "keys": None if logical_keys is None else np.array(logical_keys),
                "values": None if logical_values is None else np.array(logical_values),
            }
        )
    return snapshot


def assert_native_cache_equal(testcase, actual, expected) -> None:
    actual_snapshot = native_cache_snapshot(actual)
    expected_snapshot = native_cache_snapshot(expected)
    testcase.assertEqual(len(actual_snapshot), len(expected_snapshot))
    for index, (left, right) in enumerate(zip(actual_snapshot, expected_snapshot)):
        with testcase.subTest(cache_index=index):
            for key in ("type", "offset", "idx", "keep", "max_size"):
                testcase.assertEqual(left[key], right[key], key)
            if left["keys"] is None or right["keys"] is None:
                testcase.assertIs(left["keys"], right["keys"])
                testcase.assertIs(left["values"], right["values"])
            else:
                np.testing.assert_array_equal(left["keys"], right["keys"])
                np.testing.assert_array_equal(left["values"], right["values"])


def tiny_assistant_config(*, ordered: bool = False) -> dict:
    return {
        "architectures": ["Gemma4AssistantForCausalLM"],
        "model_type": "gemma4_assistant",
        "backbone_hidden_size": 16,
        "use_ordered_embeddings": ordered,
        "num_centroids": 4 if ordered else 0,
        "centroid_intermediate_top_k": 2 if ordered else 0,
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "head_dim": 4,
            "global_head_dim": 8,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 0,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "num_kv_shared_layers": 4,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 8,
            "sliding_window_pattern": 2,
            "max_position_embeddings": 4096,
            "attention_k_eq_v": False,
            "use_double_wide_mlp": False,
            "final_logit_softcapping": None,
            "tie_word_embeddings": True,
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        },
    }


def write_tiny_assistant_checkpoint(
    directory: Path,
    *,
    ordered: bool = False,
    weight_delta: float = 0.0,
) -> dict:
    """Materialize deterministic-shaped local provider weights for Stage A."""

    from mlx.utils import tree_flatten
    from mlx_vlm.speculative.drafters.gemma4_assistant.config import (
        Gemma4AssistantConfig,
    )
    from mlx_vlm.speculative.drafters.gemma4_assistant.gemma4_assistant import (
        Gemma4AssistantDraftModel,
    )

    directory.mkdir(parents=True, exist_ok=True)
    config = tiny_assistant_config(ordered=ordered)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    provider_config = Gemma4AssistantConfig.from_dict(config)
    model = Gemma4AssistantDraftModel(provider_config)
    weights = dict(tree_flatten(model.parameters()))
    if weight_delta:
        first = sorted(weights)[0]
        weights[first] = weights[first] + weight_delta
    mx.save_safetensors(str(directory / "model.safetensors"), weights)
    return config
