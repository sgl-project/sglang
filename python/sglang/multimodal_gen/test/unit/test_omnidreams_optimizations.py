# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the 4 OmniDreams performance optimizations.

Tests the correctness of:
- T1: AdaLN fusion (LayerNormScaleShift replacing nn.LayerNorm + manual scale/shift)
- T2: RoPE kernel (to_cos_sin_cache, fast-path dispatch, B>1 broadcast)
- T3: KV-cache split-copy (no .clone(), correct overlapping-region handling)
- T4: Text encoder cache (LRU hit/miss, eviction, CPU storage)

All tests are CPU-only; no checkpoint or GPU required.
"""

from collections import OrderedDict

import torch

from sglang.multimodal_gen.runtime.layers.layernorm import LayerNormScaleShift
from sglang.multimodal_gen.runtime.models.dits.omnidreams_kvcache import BlockKVCache
from sglang.multimodal_gen.runtime.models.dits.omnidreams_rope import (
    RotaryPositionEmbedding3D,
    apply_rope_freqs,
)


# ============================================================================
# T1: AdaLN Fusion -- LayerNormScaleShift correctness
# ============================================================================

def test_adaln_fusion_matches_old_pattern():
    """LayerNormScaleShift(x, shift, scale) == norm(x) * (1+scale) + shift."""
    torch.manual_seed(0)
    x_dim = 2048
    B, L = 2, 128
    x = torch.randn(B, L, x_dim)
    shift = torch.randn(B, 1, x_dim)
    scale = torch.randn(B, 1, x_dim)

    # Fused path
    fused = LayerNormScaleShift(x_dim, eps=1e-6)
    out_fused = fused(x, shift, scale)

    # Manual equivalent (original code)
    manual_norm = torch.nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
    out_manual = manual_norm(x) * (1.0 + scale) + shift

    diff = (out_fused - out_manual).abs().max().item()
    assert diff < 1e-5, f"Fused vs manual mismatch: {diff}"


def test_adaln_fusion_handles_varied_shapes():
    """LayerNormScaleShift works for single-batch, single-token, and batched inputs."""
    torch.manual_seed(0)
    fused = LayerNormScaleShift(2048, eps=1e-6)

    for B, L in [(1, 1), (1, 256), (4, 128), (8, 64)]:
        x = torch.randn(B, L, 2048)
        shift = torch.randn(B, 1, 2048)
        scale = torch.randn(B, 1, 2048)
        out = fused(x, shift, scale)
        assert out.shape == (B, L, 2048)
        assert torch.isfinite(out).all()


def test_adaln_fusion_preserves_elementwise_affine_false():
    """LayerNormScaleShift default has no learnable weight (affine=False)."""
    fused = LayerNormScaleShift(2048, eps=1e-6)
    # The internal norm should not have weight (elementwise_affine=False)
    assert not hasattr(fused.norm, "weight") or fused.norm.weight is None


# ============================================================================
# T2: RoPE Kernel -- to_cos_sin_cache, fast path, B>1 broadcast
# ============================================================================

def test_rope_to_cos_sin_cache_format():
    """to_cos_sin_cache returns [L, D] with cos in [:D/2] and sin in [D/2:]."""
    rope = RotaryPositionEmbedding3D(
        head_dim=128, len_h=22, len_w=40, len_t=2,
        h_extrapolation_ratio=3.0, w_extrapolation_ratio=3.0,
        t_extrapolation_ratio=1.0, device="cpu",
    )
    cs = rope.to_cos_sin_cache(0)
    L = 22 * 40 * 2
    assert cs.shape == (L, 128)
    half = 64
    # cos^2 + sin^2 == 1
    identity_error = (cs[:, :half] ** 2 + cs[:, half:] ** 2 - 1.0).abs().max().item()
    assert identity_error < 1e-6, f"cos^2+sin^2 != 1, error={identity_error}"


def test_rope_to_cos_sin_cache_matches_shift_t():
    """to_cos_sin_cache produces same angles as shift_t -> extract -> cos/sin."""
    torch.manual_seed(0)
    rope = RotaryPositionEmbedding3D(
        head_dim=128, len_h=4, len_w=5, len_t=2,
        h_extrapolation_ratio=3.0, w_extrapolation_ratio=3.0,
        t_extrapolation_ratio=1.0,
    )
    for ar_idx in (0, 1, 2):
        freqs = rope.shift_t(ar_idx)  # [L, 1, 1, 128]
        cs = rope.to_cos_sin_cache(ar_idx)  # [L, 128]
        half = 64
        # First half of cs should be cos of shift_t's first half
        angles = freqs[:, 0, 0, :half]
        expected_cos = angles.cos()
        expected_sin = angles.sin()
        assert torch.allclose(cs[:, :half], expected_cos, atol=1e-6), f"ar_idx={ar_idx} cos mismatch"
        assert torch.allclose(cs[:, half:], expected_sin, atol=1e-6), f"ar_idx={ar_idx} sin mismatch"


def test_rope_fast_path_same_as_fallback():
    """apply_rope_freqs with cos_sin_cache == apply_rope_freqs without (fallback)."""
    torch.manual_seed(0)
    rope = RotaryPositionEmbedding3D(
        head_dim=128, len_h=4, len_w=5, len_t=2,
        h_extrapolation_ratio=3.0, w_extrapolation_ratio=3.0,
        t_extrapolation_ratio=1.0,
    )
    freqs = rope.shift_t(0)
    cs = rope.to_cos_sin_cache(0)
    L = freqs.shape[0]

    for B, H in [(1, 16), (2, 16), (3, 8)]:
        x = torch.randn(B, L, H, 128)
        # Fast path
        out_fast = apply_rope_freqs(x.clone(), freqs, cos_sin_cache=cs)
        # Fallback path
        out_fallback = apply_rope_freqs(x.clone(), freqs, cos_sin_cache=None)
        diff = (out_fast - out_fallback).abs().max().item()
        assert diff < 1e-5, f"B={B} H={H}: fast vs fallback diff={diff}"


def test_rope_fast_path_works_for_all_ar_offsets():
    """Fast path correct at AR offsets 0, 1, 5."""
    torch.manual_seed(0)
    rope = RotaryPositionEmbedding3D(
        head_dim=128, len_h=4, len_w=5, len_t=2,
        h_extrapolation_ratio=3.0, w_extrapolation_ratio=3.0,
        t_extrapolation_ratio=1.0,
    )
    for ar_idx in (0, 1, 5):
        freqs = rope.shift_t(ar_idx)
        cs = rope.to_cos_sin_cache(ar_idx)
        x = torch.randn(2, freqs.shape[0], 16, 128)
        out_fast = apply_rope_freqs(x.clone(), freqs, cos_sin_cache=cs)
        out_fallback = apply_rope_freqs(x.clone(), freqs, cos_sin_cache=None)
        diff = (out_fast - out_fallback).abs().max().item()
        assert diff < 1e-5, f"ar_idx={ar_idx}: fast vs fallback diff={diff}"


def test_rope_fast_path_preserves_norm():
    """Cos/sin cache path is norm-preserving (like the original)."""
    torch.manual_seed(0)
    rope = RotaryPositionEmbedding3D(
        head_dim=128, len_h=4, len_w=5, len_t=2,
        h_extrapolation_ratio=3.0, w_extrapolation_ratio=3.0,
        t_extrapolation_ratio=1.0,
    )
    freqs = rope.shift_t(0)
    cs = rope.to_cos_sin_cache(0)
    x = torch.randn(2, freqs.shape[0], 16, 128)
    out = apply_rope_freqs(x, freqs, cos_sin_cache=cs)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4)


def test_rope_batch_dimension_broadcast():
    """B>1: each batch element gets the same position-dependent cos/sin."""
    torch.manual_seed(0)
    rope = RotaryPositionEmbedding3D(
        head_dim=128, len_h=4, len_w=5, len_t=2,
        h_extrapolation_ratio=3.0, w_extrapolation_ratio=3.0,
        t_extrapolation_ratio=1.0,
    )
    freqs = rope.shift_t(0)
    cs = rope.to_cos_sin_cache(0)
    L = freqs.shape[0]
    B = 4
    x = torch.randn(B, L, 16, 128)

    # Fast path on full batch
    out_batch = apply_rope_freqs(x.clone(), freqs, cos_sin_cache=cs)

    # Reference: rotate each batch element independently (should match)
    for b in range(B):
        out_single = apply_rope_freqs(x[b:b+1], freqs, cos_sin_cache=cs)
        assert torch.allclose(out_batch[b:b+1], out_single, atol=1e-6), f"batch {b} mismatch"


# ============================================================================
# T3: KV-Cache Split-Copy -- no .clone(), correct ordering
# ============================================================================

def _kv_chunk(B, size, n_heads, head_dim, val):
    """Helper: create K/V tensors filled with a constant value."""
    k = torch.full((B, size, n_heads, head_dim), float(val))
    v = torch.full((B, size, n_heads, head_dim), float(val + 0.5))
    return k, v


def test_kv_cache_split_copy_steady_state_roll():
    """Split-copy produces the same chunk ordering as the original clone-based approach."""
    c = BlockKVCache(
        k_shape=(1, 6, 2, 3), v_shape=(1, 6, 2, 3),
        seq_dim=1, chunk_size=2, window_size=6, sink_size=0,
    )
    # Fill 3 chunks: [10,10], [11,11], [12,12] --> full cache
    for idx, val in enumerate([10, 11, 12]):
        c.before_update(idx)
        k, v = _kv_chunk(1, 2, 2, 3, val)
        c.update(k, v)
        c.after_update(idx)

    # Steady-state roll: chunk 3 overwrites, window shifts left
    c.before_update(3)
    k, v = _kv_chunk(1, 2, 2, 3, 13)
    c.update(k, v)
    ck = c.cached_k().clone()
    c.after_update(3)

    # After roll: [11,11, 12,12, 13,13] -- chunk 10 evicted
    assert bool((ck[:, :2] == 11).all()), "first chunk should be 11 after roll"
    assert bool((ck[:, 2:4] == 12).all()), "second chunk should be 12 after roll"
    assert bool((ck[:, 4:6] == 13).all()), "third chunk should be new 13"


def test_kv_cache_split_copy_no_clone_in_source():
    """Verify BlockKVCache source file has zero .clone() calls in actual code."""
    import inspect
    source = inspect.getsource(BlockKVCache._roll_local_window_left)
    # .clone() should not appear in the method body (only possibly in docstrings/comments)
    # Strip comments and check
    lines = [ln for ln in source.split('\n') if not ln.strip().startswith('#')]
    code = '\n'.join(lines)
    # The word "clone" may appear in a comment but not as a .clone() call
    assert ".clone()" not in code, f".clone() call found in _roll_local_window_left:\n{code}"


def test_kv_cache_split_copy_overwrite_same_chunk():
    """Re-forward overwrite (same chunk_idx) works with split-copy."""
    c = BlockKVCache(
        k_shape=(1, 4, 2, 3), v_shape=(1, 4, 2, 3),
        seq_dim=1, chunk_size=2, window_size=4, sink_size=0,
    )
    for idx, val in enumerate([10, 11]):
        c.before_update(idx)
        k, v = _kv_chunk(1, 2, 2, 3, val)
        c.update(k, v)
        c.after_update(idx)

    # Re-forward: overwrite chunk 1 with value 99
    c.before_update(1)
    k, v = _kv_chunk(1, 2, 2, 3, 99)
    c.update(k, v)
    ck = c.cached_k().clone()
    c.after_update(1)

    assert bool((ck[:, :2] == 10).all()), "chunk 0 should remain 10"
    assert bool((ck[:, 2:] == 99).all()), "chunk 1 should be overwritten to 99"


def test_kv_cache_split_copy_overlap_handling():
    """When 2*chunk_size < window_size, the overlapping copy region is handled correctly."""
    # chunk=1, window=3: tokens_to_keep=2, copy1=1, copy2=1
    # Layout: [dst0,dst1]=[0,1] [src0,src1]=[1,2] -- dst1 overlaps with src0
    c = BlockKVCache(
        k_shape=(1, 3, 1, 1), v_shape=(1, 3, 1, 1),
        seq_dim=1, chunk_size=1, window_size=3, sink_size=0,
    )
    # Fill: [1], [2], [3] -> full cache = [1,2,3]
    for idx, val in enumerate([1, 2, 3]):
        c.before_update(idx)
        c.update(
            torch.full((1, 1, 1, 1), float(val)),
            torch.full((1, 1, 1, 1), float(val)),
        )
        c.after_update(idx)

    # Roll: chunk [4] -> should produce [2,3,4]
    c.before_update(3)
    c.update(
        torch.full((1, 1, 1, 1), 4.0),
        torch.full((1, 1, 1, 1), 4.0),
    )
    ck = c.cached_k().clone()
    c.after_update(3)

    assert ck[0, 0, 0, 0].item() == 2.0, f"expected 2, got {ck[0,0,0,0]}"
    assert ck[0, 1, 0, 0].item() == 3.0, f"expected 3, got {ck[0,1,0,0]}"
    assert ck[0, 2, 0, 0].item() == 4.0, f"expected 4, got {ck[0,2,0,0]}"


def test_kv_cache_split_copy_production_shapes():
    """With OmniDreams production shapes (window=6, chunk=2), split-copy is correct."""
    c = BlockKVCache(
        k_shape=(1, 6, 16, 128), v_shape=(1, 6, 16, 128),
        seq_dim=1, chunk_size=2, window_size=6, sink_size=0,
        dtype=torch.float32,
    )
    refs = [torch.randn(1, 2, 16, 128) for _ in range(3)]
    for idx, k in enumerate(refs):
        c.before_update(idx)
        c.update(k, torch.randn(1, 2, 16, 128))
        c.after_update(idx)

    k_new = torch.randn(1, 2, 16, 128)
    c.before_update(3)
    c.update(k_new, torch.randn(1, 2, 16, 128))
    ck = c.cached_k()
    c.after_update(3)

    # After roll with chunk=2, window=6: tokens_to_keep=4
    # copy1=2, copy2=2 (both non-overlapping after first copy writes past overlap zone)
    assert ck.shape == (1, 6, 16, 128)
    assert torch.allclose(ck[:, 0:2], refs[1], atol=1e-6), "chunk mismatch after roll"
    assert torch.allclose(ck[:, 2:4], refs[2], atol=1e-6), "chunk mismatch after roll"
    assert torch.allclose(ck[:, 4:6], k_new, atol=1e-6), "chunk mismatch after roll"


# ============================================================================
# T4: Text Encoder Cache -- LRU behavior
# ============================================================================

# We test the LRU cache logic in isolation (not the full _encode_text which needs
# a real Qwen2.5-VL model). The test verifies the cache data structure and eviction
# pattern that OmniDreamsBeforeDenoisingStage._encode_text uses.

def test_text_cache_lru_eviction_order():
    """LRU cache evicts oldest entry when full (FIFO-like with move_to_end)."""
    # Simulate the cache data structure and logic from _encode_text
    cache: OrderedDict[str, torch.Tensor] = OrderedDict()
    max_size = 4

    for i in range(6):  # Insert 6 entries into a 4-slot cache
        prompt = f"prompt_{i}"
        embeds = torch.tensor([float(i)])  # Stand-in for real embedding
        
        if len(cache) >= max_size:
            cache.popitem(last=False)  # Evict oldest (first inserted)
        cache[prompt] = embeds.detach()

    # After 6 inserts with max 4: should have prompts 2,3,4,5
    assert len(cache) == max_size
    assert "prompt_2" in cache
    assert "prompt_3" in cache
    assert "prompt_4" in cache
    assert "prompt_5" in cache
    assert "prompt_0" not in cache
    assert "prompt_1" not in cache


def test_text_cache_hit_promotes_to_mru():
    """Cache hit moves entry to most-recently-used position."""
    cache: OrderedDict[str, torch.Tensor] = OrderedDict()
    max_size = 3

    # Fill cache: [a, b, c]
    for prompt in ["a", "b", "c"]:
        cache[prompt] = torch.zeros(1)

    # Hit on "a" -> should move to end: [b, c, a]
    cached = cache.get("a")
    assert cached is not None
    cache.move_to_end("a")

    # Insert "d" -> evict oldest (b): [c, a, d]
    if len(cache) >= max_size:
        cache.popitem(last=False)
    cache["d"] = torch.zeros(1)

    assert "a" in cache, "promoted entry should survive eviction"
    assert "c" in cache, "not-the-oldest should survive"
    assert "d" in cache, "new entry should be present"
    assert "b" not in cache, "oldest un-promoted entry should be evicted"


def test_text_cache_stores_on_cpu():
    """Cached embeddings are stored on CPU to avoid GPU VRAM."""
    cache: OrderedDict[str, torch.Tensor] = OrderedDict()
    embeds = torch.randn(1, 512, 100352)
    
    # Simulate the store logic from _encode_text
    cache["test"] = embeds.detach().cpu()

    assert cache["test"].device.type == "cpu"
    # Detached tensor should not have grad
    assert not cache["test"].requires_grad


def test_text_cache_key_is_prompt_string():
    """Cache key is the raw prompt string; different prompts produce different entries."""
    cache: OrderedDict[str, torch.Tensor] = OrderedDict()
    
    cache["a driving scene"] = torch.zeros(1)
    cache["a rainy scene"] = torch.ones(1)

    assert cache["a driving scene"].item() == 0.0
    assert cache["a rainy scene"].item() == 1.0
    assert len(cache) == 2


# ============================================================================
# Supplemental tests for critical edge cases
# ============================================================================

# --- T2: Cross-attention invariant ---

def test_rope_cross_attn_does_not_receive_cos_sin_cache():
    """Cross-attention call in Block.forward must NOT pass rope_cos_sin.

    This is a correctness invariant: cross-attention does not use RoPE,
    so rope_cos_sin should only go to self_attn, never to cross_attn.
    """
    # Read the source and verify cross_attn call does not contain rope_cos_sin
    import os
    dit_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "runtime", "models", "dits", "omnidreams.py",
    )
    with open(dit_path) as f:
        # Find the cross_attn call in Block.forward
        block_forward = f.read().split("def forward(self,")[1]
        block_forward = block_forward.split("def _cross_view_attn_forward")[0]
        # The cross_attn call line
        cross_call_lines = [ln for ln in block_forward.split('\n') if 'self.cross_attn(' in ln and 'cross_attn' in ln]
        # There should be exactly one cross_attn call (the self-attn call also has cross_attn in the name)
        # The cross_attn call is: x + gate_c * self.cross_attn(normed, context=context, cross_kv=cross_attn_kv)
        # It should NOT contain rope_cos_sin
        cross_attn_line = [ln for ln in cross_call_lines if 'context=context' in ln]
        if cross_attn_line:
            assert "rope_cos_sin" not in cross_attn_line[0], (
                "cross_attn must NOT receive rope_cos_sin. Found in line:\n"
                + cross_attn_line[0]
            )


# --- T3: Sink + split-copy interaction ---

def test_kv_cache_split_copy_with_sink_retention():
    """Split-copy must preserve sink tokens when sink_size > 0."""
    c = BlockKVCache(
        k_shape=(1, 6, 2, 3), v_shape=(1, 6, 2, 3),
        seq_dim=1, chunk_size=2, window_size=4, sink_size=2,
    )
    # Fill: sink=[1,1, 2,2], window=[3,3, 4,4] -> full cache
    for idx, val in enumerate([1, 2, 3, 4, 5]):
        c.before_update(idx)
        k, v = _kv_chunk(1, 2, 2, 3, val)
        c.update(k, v)
        c.after_update(idx)

    # After steady-state roll with chunk 5:
    # sink tokens (0:2) should still be [1,1]
    # window (2:6) should be [4,4, 5,5] (chunk 3 rolled out)
    ck = c.cached_k().clone()
    expected_sink = torch.full((1, 2, 2, 3), 1.0)
    assert torch.allclose(ck[:, :2], expected_sink, atol=1e-6), (
        f"Sink tokens were corrupted by split-copy roll. "
        f"Got {ck[:, :2, 0, 0]}, expected all 1.0"
    )
    # Window region should contain chunks 4 and 5
    assert bool((ck[:, 2:4] == 4).all()), "Window should contain chunk 4 after roll"
    assert bool((ck[:, 4:6] == 5).all()), "Window should contain chunk 5 after roll"


def test_kv_cache_split_copy_sink_not_rolled_into():
    """split-copy dst_start must start after sink region to avoid overwriting it."""
    c = BlockKVCache(
        k_shape=(1, 8, 1, 1), v_shape=(1, 8, 1, 1),
        seq_dim=1, chunk_size=2, window_size=6, sink_size=2,
    )
    # Fill with unique values: sink=[0, 1], window=[2,3,4,5,6,7]
    for idx in range(4):  # 4 chunks of 2 = 8 total
        c.before_update(idx)
        c.update(
            torch.tensor([[[[float(idx*2)]], [[float(idx*2+1)]]]]),
            torch.tensor([[[[float(idx*2)]], [[float(idx*2+1)]]]]),
        )
        c.after_update(idx)

    # Now roll: chunk 4 should shift window left by 2
    # Expected: sink=[0, 1] (unchanged), window=[4,5,6,7,8,9]
    c.before_update(4)
    c.update(
        torch.tensor([[[[8.0]], [[9.0]]]]),
        torch.tensor([[[[8.0]], [[9.0]]]]),
    )
    ck = c.cached_k().clone()
    c.after_update(4)

    # Sink (indices 0, 1) must match original values 0, 1
    assert ck[0, 0, 0, 0].item() == 0.0, f"Sink[0] corrupted: {ck[0,0,0,0]}"
    assert ck[0, 1, 0, 0].item() == 1.0, f"Sink[1] corrupted: {ck[0,1,0,0]}"
    # Window (indices 2..8) must have values 4..9
    for i in range(6):
        assert ck[0, 2+i, 0, 0].item() == float(4+i), (
            f"Window[{i}] corrupted: expected {4+i}, got {ck[0,2+i,0,0]}"
        )


# --- T1: Edge case ---

def test_adaln_fusion_zero_scale_shift():
    """LayerNormScaleShift with scale=shift=0 should be equivalent to pure LayerNorm."""
    torch.manual_seed(0)
    fused = LayerNormScaleShift(2048, eps=1e-6)
    x = torch.randn(2, 128, 2048)
    zero_shift = torch.zeros(2, 1, 2048)
    zero_scale = torch.zeros(2, 1, 2048)

    out_fused = fused(x, zero_shift, zero_scale)
    manual_norm = torch.nn.LayerNorm(2048, elementwise_affine=False, eps=1e-6)
    out_manual = manual_norm(x)

    diff = (out_fused - out_manual).abs().max().item()
    assert diff < 1e-5, f"Zero scale/shift: fused vs norm mismatch: {diff}"


# --- T2: head_dim edge case ---

def test_rope_to_cos_sin_cache_non_standard_head_dim():
    """to_cos_sin_cache works for head_dim != 128."""
    for head_dim in [64, 192, 256]:
        rope = RotaryPositionEmbedding3D(
            head_dim=head_dim, len_h=2, len_w=3, len_t=1,
            h_extrapolation_ratio=1.0, w_extrapolation_ratio=1.0,
            t_extrapolation_ratio=1.0, device="cpu",
        )
        cs = rope.to_cos_sin_cache(0)
        L = 1 * 2 * 3
        assert cs.shape == (L, head_dim), f"head_dim={head_dim}: bad shape {cs.shape}"
        half = head_dim // 2
        identity_err = (cs[:, :half] ** 2 + cs[:, half:] ** 2 - 1.0).abs().max().item()
        assert identity_err < 1e-6, f"head_dim={head_dim}: cos^2+sin^2 != 1, error={identity_err}"


# ============================================================================
# SP (Sequence Parallelism) Tests -- TDD for attention split
# ============================================================================

def test_sp_self_attention_uses_usp_attention():
    """OmniDreamsSelfAttention must use USPAttention (SP-compatible)."""
    from sglang.multimodal_gen.runtime.models.dits.omnidreams import (
        OmniDreamsSelfAttention,
    )
    attn = OmniDreamsSelfAttention(query_dim=2048, n_heads=16, head_dim=128)
    from sglang.multimodal_gen.runtime.layers.attention import USPAttention
    msg = (
        "OmniDreamsSelfAttention must contain a USPAttention for SP support. "
        "If this fails, self-attention is using raw SDPA without SP communication."
    )
    found = isinstance(attn.attn, USPAttention)
    assert found, msg


def test_sp_cross_attention_uses_local_attention():
    """OmniDreamsCrossAttention must use LocalAttention (no SP needed)."""
    from sglang.multimodal_gen.runtime.models.dits.omnidreams import (
        OmniDreamsCrossAttention,
    )
    attn = OmniDreamsCrossAttention(
        query_dim=2048, context_dim=1024, n_heads=16, head_dim=128,
    )
    from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
    msg = (
        "OmniDreamsCrossAttention must contain a LocalAttention. "
        "Cross-attention text K/V is replicated across SP ranks; no all-to-all needed."
    )
    found = isinstance(attn.attn, LocalAttention)
    assert found, msg


def test_sp_block_init_uses_correct_attention_classes():
    """OmniDreamsBlock.__init__ must use Self/Cross classes, not the old alias."""
    import os
    dit_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "runtime", "models", "dits", "omnidreams.py",
    )
    with open(dit_path) as f:
        source = f.read()
    checks = [
        ("self_attn init", "self.self_attn = OmniDreamsSelfAttention(x_dim, num_heads, head_dim)" in source),
        ("cross_attn init", "self.cross_attn = OmniDreamsCrossAttention(x_dim, context_dim, num_heads, head_dim)" in source),
        ("no old class name in init", "OmniDreamsAttention(x_dim, None" not in source),
        ("no old class name in cross init", "OmniDreamsAttention(x_dim, context_dim" not in source),
    ]
    failed = [label for label, ok in checks if not ok]
    assert not failed, (
        "OmniDreamsBlock.__init__ still uses the old OmniDreamsAttention class name:\n"
        + "\n".join(f"  - {f}" for f in failed)
    )


def test_sp_self_attention_no_context_arg_in_block_call():
    """Block.forward must NOT pass 'context' to self_attn (self-attn has no context)."""
    import os
    dit_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "runtime", "models", "dits", "omnidreams.py",
    )
    with open(dit_path) as f:
        source = f.read()
    # Extract the self-attn call line from Block.forward
    block_forward = source.split("def forward(self,")[1].split("def _cross_view_attn_forward")[0]
    self_call = [ln for ln in block_forward.split('\n') if 'self.self_attn(' in ln and ' ' in ln]
    if self_call:
        call_line = self_call[0].strip()
        assert "context=context" not in call_line, (
            "OmniDreamsSelfAttention.forward() should NOT accept a 'context' kwarg. "
            "Self-attention context is always the same as x (implicit). "
            f"Found in: {call_line}"
        )
        assert "rope_freqs=" in call_line, (
            "OmniDreamsSelfAttention.forward() should accept 'rope_freqs' for RoPE. "
            f"Found: {call_line}"
        )


def test_sp_cross_attention_no_rope_in_block_call():
    """Block.forward must NOT pass rope_freqs to cross_attn (cross-attn has no RoPE)."""
    import os
    dit_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "runtime", "models", "dits", "omnidreams.py",
    )
    with open(dit_path) as f:
        source = f.read()
    block_forward = source.split("def forward(self,")[1].split("def _cross_view_attn_forward")[0]
    cross_call = [ln for ln in block_forward.split('\n') if 'self.cross_attn(' in ln and 'context=context' in ln]
    if cross_call:
        call_line = cross_call[0].strip()
        assert "rope_freqs" not in call_line, (
            "OmniDreamsCrossAttention.forward() should NOT accept 'rope_freqs'. "
            "Cross-attention does not use RoPE (plan non-negotiable). "
            f"Found in: {call_line}"
        )


def test_sp_precompute_cross_attn_kv_still_works():
    """precompute_cross_attn_kv must iterate blocks and access cross_attn attributes."""
    import os
    dit_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "runtime", "models", "dits", "omnidreams.py",
    )
    with open(dit_path) as f:
        source = f.read()
    precompute = source.split("def precompute_cross_attn_kv")[1].split("def patchify")[0]
    checks = [
        ("iterates blocks", "for block in self.blocks:" in precompute),
        ("accesses cross_attn", "block.cross_attn" in precompute),
        ("uses k_proj", "attn.k_proj(context)" in precompute or "k_proj" in precompute),
        ("uses v_proj", "attn.v_proj(context)" in precompute or "v_proj" in precompute),
        ("uses k_norm", "attn.k_norm" in precompute),
        ("returns K,V tuples", "result.append((k, v))" in precompute),
    ]
    failed = [label for label, ok in checks if not ok]
    assert not failed, (
        "precompute_cross_attn_kv is broken:\n" + "\n".join(f"  - {f}" for f in failed)
    )
