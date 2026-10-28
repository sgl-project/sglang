"""A draft must not share (and mutate) the target's cached RoPE module, and its
private copy must follow the lifecycle of get_rope()'s own cache."""

import gc
import weakref
from unittest.mock import patch

import pytest
import torch
from transformers import PretrainedConfig

from sglang.srt.layers.rotary_embedding.factory import _ROPE_DICT
from sglang.srt.models import dflash
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def single_rank_native():
    # A fresh shared module is a fresh key for the draft-copy cache too.
    _ROPE_DICT.clear()
    try:
        with (
            get_context().override_server_args(),
            get_parallel().override(tp_size=1, tp_rank=0, tp_group=None),
            patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True),
        ):
            yield
    finally:
        _ROPE_DICT.clear()


def _config():
    # Partial RoPE as in a draft config copied from a Qwen3.5-style target.
    return PretrainedConfig(
        hidden_size=8,
        intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=16,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        max_position_embeddings=32,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        dflash_config=dict(
            block_size=4,
            selector_rank=2,
            selector_top_k=2,
            target_layer_ids=[0, 2],
            conv_kernel_size=2,
            conv_group_size=2,
        ),
    )


def _build_draft():
    """Build a draft, also returning the module get_rope() shares process-wide
    (and so with a target using the same RoPE config)."""
    shared = []
    real_get_rope = dflash.get_rope

    def recording_get_rope(*args, **kwargs):
        rope = real_get_rope(*args, **kwargs)
        shared.append(rope)
        return rope

    with patch.object(dflash, "get_rope", recording_get_rope):
        model = dflash.DFlash2DraftModel(_config())
    assert len({id(rope) for rope in shared}) == 1
    return model, shared[0]


def _draft_rope(model):
    ropes = [layer.self_attn.rotary_emb for layer in model.layers]
    assert all(rope is ropes[0] for rope in ropes)
    return ropes[0]


def test_draft_rope_is_private_and_casts_without_touching_target():
    model, target_rope = _build_draft()
    rope = _draft_rope(model)
    assert rope is not target_rope

    # The draft's dtype match must leave the target's buffer (captured by its
    # CUDA graphs) in place.
    target_cache = target_rope.cos_sin_cache
    target_ptr = target_cache.data_ptr()
    rope._match_cos_sin_cache_dtype(torch.zeros(1, 8, dtype=torch.bfloat16))
    assert rope.cos_sin_cache.dtype == torch.bfloat16
    assert target_rope.cos_sin_cache is target_cache
    assert target_rope.cos_sin_cache.data_ptr() == target_ptr


def test_private_copy_is_released_with_the_draft():
    model, target_rope = _build_draft()
    released = weakref.ref(_draft_rope(model))
    del model
    gc.collect()
    assert released() is None

    reloaded, reloaded_target = _build_draft()
    assert reloaded_target is target_rope
    assert _draft_rope(reloaded) is not target_rope


@pytest.mark.parametrize("teardown", ["meta", "storage"])
def test_dead_private_copy_is_rebuilt(teardown):
    model, target_rope = _build_draft()
    rope = _draft_rope(model)
    expected = target_rope.cos_sin_cache.clone()
    # A teardown kills the old draft's copy while the module is still alive.
    if teardown == "meta":
        rope.to(device="meta")
    else:
        rope.cos_sin_cache.untyped_storage().resize_(0)

    reloaded, _ = _build_draft()
    rebuilt = _draft_rope(reloaded)
    assert rebuilt is not rope and rebuilt is not target_rope
    assert rebuilt.cos_sin_cache.device.type != "meta"
    assert torch.equal(rebuilt.cos_sin_cache, expected)


def test_meta_build_shares_one_copy_and_real_build_rebuilds():
    with torch.device("meta"):
        meta_model, _ = _build_draft()
    meta_rope = _draft_rope(meta_model)
    assert meta_rope.cos_sin_cache.device.type == "meta"

    model, target_rope = _build_draft()
    rope = _draft_rope(model)
    assert rope is not meta_rope and rope is not target_rope
    assert rope.cos_sin_cache.device.type != "meta"
