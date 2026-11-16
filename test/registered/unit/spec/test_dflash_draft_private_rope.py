"""A DFlash/DSpark draft must not share (and mutate) the target's cached RoPE
module, and its private copy must follow the lifecycle of get_rope()'s cache."""

import copy
import gc
import sys
import weakref
from unittest.mock import patch

import pytest
import torch
from transformers import PretrainedConfig

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.layers.rotary_embedding.factory import _ROPE_DICT
from sglang.srt.models import dflash
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Partial RoPE as in a draft config copied from a Qwen3.5-style target, with
# and without its M-RoPE fields.
_ROPE_CONFIGS = {
    "rope": dict(head_dim=4, partial_rotary_factor=0.5),
    "mrope": dict(
        head_dim=16,
        partial_rotary_factor=0.5,
        rope_scaling=dict(
            rope_type="default", mrope_section=[2, 1, 1], mrope_interleaved=True
        ),
    ),
}


@pytest.fixture(autouse=True)
def single_rank():
    # A fresh shared module is a fresh key for the draft-copy cache too.
    _ROPE_DICT.clear()
    try:
        with (
            get_context().override_server_args(),
            get_parallel().override(tp_size=1, tp_rank=0, tp_group=None),
        ):
            yield
    finally:
        _ROPE_DICT.clear()


@pytest.fixture(autouse=True)
def native_rope():
    # CPU runners have no CUDA rope kernel; keep RoPE on the native path. The
    # CUDA tests below opt back into the real dispatch.
    with patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True):
        yield


def _config(rope="rope"):
    return PretrainedConfig(
        hidden_size=32,
        intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=16,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        max_position_embeddings=32,
        rope_theta=10000.0,
        dflash_config=dict(
            block_size=4,
            selector_rank=2,
            selector_top_k=2,
            target_layer_ids=[0, 2],
            conv_kernel_size=2,
            conv_group_size=2,
        ),
        **_ROPE_CONFIGS[rope],
    )


def _build_draft(rope="rope"):
    """Build a draft, also returning the module get_rope() shares process-wide
    (and so with a target using the same RoPE config)."""
    shared = []
    real_get_rope = dflash.get_rope

    def recording_get_rope(*args, **kwargs):
        module = real_get_rope(*args, **kwargs)
        shared.append(module)
        return module

    with patch.object(dflash, "get_rope", recording_get_rope):
        model = dflash.DFlash2DraftModel(_config(rope))
    assert len({id(module) for module in shared}) == 1
    return model, shared[0]


def _draft_rope(model):
    ropes = [layer.self_attn.rotary_emb for layer in model.layers]
    assert all(rope is ropes[0] for rope in ropes)
    return ropes[0]


def _qk(rope, num_tokens, dtype, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    head_size = rope.head_size
    q = torch.randn(num_tokens, 2 * head_size, generator=gen, device=device)
    k = torch.randn(num_tokens, head_size, generator=gen, device=device)
    return q.to(dtype), k.to(dtype)


def _snapshot(rope):
    cache = rope.cos_sin_cache
    return cache, cache.data_ptr(), cache.dtype, cache.clone()


def _assert_unchanged(rope, snapshot):
    cache, ptr, dtype, values = snapshot
    assert rope.cos_sin_cache is cache
    assert rope.cos_sin_cache.data_ptr() == ptr
    assert rope.cos_sin_cache.dtype == dtype
    assert torch.equal(rope.cos_sin_cache, values)


@pytest.mark.parametrize("rope_kind", ["rope", "mrope"])
def test_draft_rope_is_private_with_isolated_storage(rope_kind):
    model, target_rope = _build_draft(rope_kind)
    rope = _draft_rope(model)
    assert type(rope) is type(target_rope)
    if rope_kind == "mrope":
        assert isinstance(rope, MRotaryEmbedding)
    assert rope is not target_rope
    assert (
        rope.cos_sin_cache.untyped_storage().data_ptr()
        != target_rope.cos_sin_cache.untyped_storage().data_ptr()
    )


def test_mrope_bf16_forward_leaves_target_cache_intact():
    # On CUDA, MRotaryEmbedding dispatches to forward_cuda, which casts
    # cos_sin_cache to the activation dtype before the native math; on a shared
    # module that frees the fp32 buffer a target CUDA graph reads. It runs on
    # CPU tensors too (the real CUDA dispatch is covered below).
    model, target_rope = _build_draft("mrope")
    rope = _draft_rope(model)
    target = _snapshot(target_rope)
    q, k = _qk(rope, 8, torch.bfloat16, "cpu")
    rope.forward_cuda(torch.arange(8), q, k)
    assert rope.cos_sin_cache.dtype == torch.bfloat16
    _assert_unchanged(target_rope, target)


@pytest.mark.parametrize("rope_kind", ["rope", "mrope"])
def test_draft_rope_matches_an_independent_reference(rope_kind):
    model, target_rope = _build_draft(rope_kind)
    rope = _draft_rope(model)
    # Same initial state, separate storage; inputs are cloned per call since
    # RoPE rotates in place.
    reference = copy.deepcopy(target_rope)
    positions = torch.tensor([0, 1, 5, 17, 31])
    q, k = _qk(rope, positions.numel(), torch.float32, "cpu")
    actual = rope.forward_native(positions, q.clone(), k.clone())
    expected = reference.forward_native(positions, q.clone(), k.clone())
    for a, e in zip(actual, expected):
        assert torch.equal(a, e)


def test_live_drafts_with_the_same_rope_share_one_copy():
    # The cache is process-wide and keyed by the shared module: drafts alive at
    # the same time share one copy (isolated from the target, not from each
    # other), and the copy is collectable once its last holder is gone.
    first, target_rope = _build_draft()
    second, second_target = _build_draft()
    assert second_target is target_rope
    copy_ref = weakref.ref(_draft_rope(first))
    assert _draft_rope(second) is copy_ref()
    del first
    gc.collect()
    assert copy_ref() is not None
    del second
    gc.collect()
    assert copy_ref() is None


def test_private_copy_is_released_with_its_last_holder():
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


@requires_cuda
@pytest.mark.parametrize(
    "rope_kind,positions_ndim", [("rope", 1), ("mrope", 1), ("mrope", 2)]
)
def test_cuda_bf16_forward_isolates_target_and_matches_reference(
    rope_kind, positions_ndim
):
    with (
        patch("sglang.srt.layers.rotary_embedding.base._is_cpu", False),
        torch.device("cuda"),
    ):
        model, target_rope = _build_draft(rope_kind)
    rope = _draft_rope(model)
    target = _snapshot(target_rope)
    reference = copy.deepcopy(target_rope)
    num_tokens = 6
    positions = torch.tensor([0, 1, 3, 7, 15, 31], device="cuda")
    if positions_ndim == 2:
        positions = torch.stack([positions, positions.flip(0), positions // 2])
    q, k = _qk(rope, num_tokens, torch.bfloat16, "cuda", seed=1)

    with patch("sglang.srt.layers.rotary_embedding.base._is_cpu", False):
        actual = rope(positions, q.clone(), k.clone())
        expected = reference(positions, q.clone(), k.clone())
    torch.cuda.synchronize()

    _assert_unchanged(target_rope, target)
    for a, e in zip(actual, expected):
        assert torch.equal(a, e)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
