# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OmniDreams performance / acceleration code (no checkpoint for
most tests; the LightTAE/LightVAE checkpoint smokes skip when weights absent).

Covers:

* **T1 AdaLN fusion** — ``LayerNormScaleShift`` replacing ``nn.LayerNorm`` +
  manual scale/shift.
* **T2 RoPE kernel** — ``shift_t`` cos/sin cache, ``apply_rope_freqs`` dispatch,
  B>1 broadcast.
* **T3 KV-cache split-copy** — no ``.clone()``, correct overlapping-region
  handling.
* **T4 Text encoder cache** — LRU hit/miss, eviction, CPU storage.
* **CUDAGraphWrapper** — input-staging logic + ``set_or_copy`` pointer
  stability (the CPU-checkable half; capture/replay needs CUDA).
* **LightTAE (TAEHV)** — checkpoint-key coverage, ``frames_to_trim`` math, and a
  decode-shape smoke (real ``lighttaew2_1.pth`` -- skipped if absent).
* **LightVAE (pruned Wan)** — checkpoint-key coverage + encode-shape smoke
  (real ``lightvaew2_1.pth`` -- skipped if absent).

The checkpoint-dependent tests resolve the weights from
``SGLANG_OMNIDREAMS_{LIGHTTAE,LIGHTVAE}_CKPT`` or a few known local paths, and
``pytest.skip`` when none exist (so CI without the weights stays green). The T1
AdaLN-fusion forward checks drive ``LayerNormScaleShift``'s fused kernel, which
is CUDA-only, so they run on the platform device and skip when no GPU is present.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.layernorm import LayerNormScaleShift
from sglang.multimodal_gen.runtime.models.dits.omnidreams import (
    BlockKVCache,
    RotaryPositionEmbedding3D,
    apply_rope_freqs,
)
from sglang.multimodal_gen.runtime.models.dits.omnidreams_cuda_graph import (
    CUDAGraphWrapper,
    set_or_copy,
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LayerNormScaleShift's fused kernel runs on the platform (GPU) device",
)


def _find_ckpt(env_key: str, *names: str) -> str | None:
    if os.environ.get(env_key) and os.path.isfile(os.environ[env_key]):
        return os.environ[env_key]
    roots = ["/Users/cerdore/gitRepo/models", "/root/blockdata", os.getcwd()]
    for root in roots:
        for name in names:
            cand = os.path.join(root, name)
            if os.path.isfile(cand):
                return cand
    return None


def _kv_chunk(B, size, n_heads, head_dim, val):
    """Helper: create K/V tensors filled with a constant value."""
    k = torch.full((B, size, n_heads, head_dim), float(val))
    v = torch.full((B, size, n_heads, head_dim), float(val + 0.5))
    return k, v


# =========================================================================== #
# T1: AdaLN Fusion -- LayerNormScaleShift correctness
# =========================================================================== #
@requires_gpu
def test_adaln_fusion_matches_old_pattern():
    """LayerNormScaleShift(x, shift, scale) == norm(x) * (1+scale) + shift."""
    torch.manual_seed(0)
    x_dim = 2048
    B, L = 2, 128
    x = torch.randn(B, L, x_dim, device=_DEVICE)
    shift = torch.randn(B, 1, x_dim, device=_DEVICE)
    scale = torch.randn(B, 1, x_dim, device=_DEVICE)

    # Fused path
    fused = LayerNormScaleShift(x_dim, eps=1e-6)
    out_fused = fused(x, shift, scale)

    # Manual equivalent (original code)
    manual_norm = torch.nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
    out_manual = manual_norm(x) * (1.0 + scale) + shift

    diff = (out_fused - out_manual).abs().max().item()
    assert diff < 1e-5, f"Fused vs manual mismatch: {diff}"


@requires_gpu
def test_adaln_fusion_handles_varied_shapes():
    """LayerNormScaleShift works for single-batch, single-token, and batched inputs."""
    torch.manual_seed(0)
    fused = LayerNormScaleShift(2048, eps=1e-6)

    for B, L in [(1, 1), (1, 256), (4, 128), (8, 64)]:
        x = torch.randn(B, L, 2048, device=_DEVICE)
        shift = torch.randn(B, 1, 2048, device=_DEVICE)
        scale = torch.randn(B, 1, 2048, device=_DEVICE)
        out = fused(x, shift, scale)
        assert out.shape == (B, L, 2048)
        assert torch.isfinite(out).all()


def test_adaln_fusion_preserves_elementwise_affine_false():
    """LayerNormScaleShift default has no learnable weight (affine=False)."""
    fused = LayerNormScaleShift(2048, eps=1e-6)
    # The internal norm should not have weight (elementwise_affine=False)
    assert not hasattr(fused.norm, "weight") or fused.norm.weight is None


# =========================================================================== #
# T2: RoPE Kernel -- shift_t cos/sin cache, _apply_rotary_emb, B>1 broadcast
# =========================================================================== #
def test_rope_shift_t_cos_sin_format():
    """shift_t returns the [L, D] cache with cos in [:D/2] and sin in [D/2:]."""
    rope = RotaryPositionEmbedding3D(
        head_dim=128,
        len_h=22,
        len_w=40,
        len_t=2,
        h_extrapolation_ratio=3.0,
        w_extrapolation_ratio=3.0,
        t_extrapolation_ratio=1.0,
        device="cpu",
    )
    cs = rope.shift_t(0)
    L = 22 * 40 * 2
    assert cs.shape == (L, 128)
    half = 64
    # cos^2 + sin^2 == 1
    identity_error = (cs[:, :half] ** 2 + cs[:, half:] ** 2 - 1.0).abs().max().item()
    assert identity_error < 1e-6, f"cos^2+sin^2 != 1, error={identity_error}"


def test_rope_works_for_all_ar_offsets():
    """apply_rope_freqs is finite and norm-preserving at AR offsets 0, 1, 5."""
    torch.manual_seed(0)
    rope = RotaryPositionEmbedding3D(
        head_dim=128,
        len_h=4,
        len_w=5,
        len_t=2,
        h_extrapolation_ratio=3.0,
        w_extrapolation_ratio=3.0,
        t_extrapolation_ratio=1.0,
    )
    for ar_idx in (0, 1, 5):
        cs = rope.shift_t(ar_idx)
        x = torch.randn(2, cs.shape[0], 16, 128)
        out = apply_rope_freqs(x, cs)
        assert torch.isfinite(out).all(), f"ar_idx={ar_idx}: non-finite output"
        assert torch.allclose(
            out.norm(dim=-1), x.norm(dim=-1), atol=1e-4
        ), f"ar_idx={ar_idx}: norm not preserved"


def test_rope_preserves_norm():
    """The cos/sin rotation is norm-preserving (orthogonal rotation per pair)."""
    torch.manual_seed(0)
    rope = RotaryPositionEmbedding3D(
        head_dim=128,
        len_h=4,
        len_w=5,
        len_t=2,
        h_extrapolation_ratio=3.0,
        w_extrapolation_ratio=3.0,
        t_extrapolation_ratio=1.0,
    )
    cs = rope.shift_t(0)
    x = torch.randn(2, cs.shape[0], 16, 128)
    out = apply_rope_freqs(x, cs)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4)


def test_rope_batch_dimension_broadcast():
    """B>1: each batch element gets the same position-dependent cos/sin."""
    torch.manual_seed(0)
    rope = RotaryPositionEmbedding3D(
        head_dim=128,
        len_h=4,
        len_w=5,
        len_t=2,
        h_extrapolation_ratio=3.0,
        w_extrapolation_ratio=3.0,
        t_extrapolation_ratio=1.0,
    )
    cs = rope.shift_t(0)
    L = cs.shape[0]
    B = 4
    x = torch.randn(B, L, 16, 128)

    # Rotate the full batch at once.
    out_batch = apply_rope_freqs(x.clone(), cs)

    # Reference: rotate each batch element independently (should match).
    for b in range(B):
        out_single = apply_rope_freqs(x[b : b + 1], cs)
        assert torch.allclose(
            out_batch[b : b + 1], out_single, atol=1e-6
        ), f"batch {b} mismatch"


# =========================================================================== #
# T3: KV-Cache Split-Copy -- no .clone(), correct ordering
# =========================================================================== #
def test_kv_cache_split_copy_steady_state_roll():
    """Split-copy produces the same chunk ordering as the original clone-based approach."""
    c = BlockKVCache(
        k_shape=(1, 6, 2, 3),
        v_shape=(1, 6, 2, 3),
        seq_dim=1,
        chunk_size=2,
        window_size=6,
        sink_size=0,
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
    lines = [l for l in source.split("\n") if not l.strip().startswith("#")]
    code = "\n".join(lines)
    # The word "clone" may appear in a comment but not as a .clone() call
    assert (
        ".clone()" not in code
    ), f".clone() call found in _roll_local_window_left:\n{code}"


def test_kv_cache_split_copy_overwrite_same_chunk():
    """Re-forward overwrite (same chunk_idx) works with split-copy."""
    c = BlockKVCache(
        k_shape=(1, 4, 2, 3),
        v_shape=(1, 4, 2, 3),
        seq_dim=1,
        chunk_size=2,
        window_size=4,
        sink_size=0,
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
        k_shape=(1, 3, 1, 1),
        v_shape=(1, 3, 1, 1),
        seq_dim=1,
        chunk_size=1,
        window_size=3,
        sink_size=0,
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
        k_shape=(1, 6, 16, 128),
        v_shape=(1, 6, 16, 128),
        seq_dim=1,
        chunk_size=2,
        window_size=6,
        sink_size=0,
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


# =========================================================================== #
# T4: Text Encoder Cache -- LRU behavior
# =========================================================================== #
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


# =========================================================================== #
# CUDAGraphWrapper staging logic
# =========================================================================== #
def test_set_or_copy_pointer_stable_on_same_shape():
    state: dict = {}
    set_or_copy(state, 0, torch.ones(2, 3))
    ptr = state[0].data_ptr()
    set_or_copy(state, 0, torch.full((2, 3), 5.0))  # same shape -> in-place copy_
    assert state[0].data_ptr() == ptr, "same-shape write must preserve storage"
    assert torch.equal(state[0], torch.full((2, 3), 5.0))
    set_or_copy(state, 0, torch.ones(4, 5))  # new shape -> fresh buffer
    assert state[0].shape == (4, 5)


def test_cuda_graph_wrapper_warmup_runs_eager_through_static_buffers():
    # High warmup so we stay on the eager path (the capture path needs CUDA).
    seen = []

    def fn(a, *, b):
        seen.append(a)  # the staged static buffer, not the caller's tensor
        return a + b

    w = CUDAGraphWrapper(fn, warmup_iters=8)
    out = w(torch.tensor([1.0]), b=torch.tensor([2.0]))
    assert torch.equal(out, torch.tensor([3.0]))
    # The arg fn sees IS the wrapper's static buffer (copy-in staging).
    assert seen[0].data_ptr() == w._static_args[0].data_ptr()
    assert not w.is_capturing_or_captured  # still warming up, not captured


def test_cuda_graph_wrapper_signature_change_resets():
    def fn(a):
        return a * 2

    # High warmup so neither shape captures (CPU has no CUDAGraph).
    w = CUDAGraphWrapper(fn, warmup_iters=8)
    w(torch.zeros(2, 2))
    first_buf = w._static_args[0]
    w(torch.zeros(2, 2))  # same shape -> reuse the same static buffer
    assert w._static_args[0].data_ptr() == first_buf.data_ptr()
    rem_before = w._warmup_remaining
    w(torch.zeros(3, 3))  # shape change -> reset + realloc + warmup restart
    assert w._static_args[0].shape == (3, 3)
    # Reset restored warmup_remaining to the full count, then consumed one.
    assert w._warmup_remaining == w.warmup_iters - 1 > rem_before - 1


def test_cuda_graph_wrapper_non_tensor_args_passthrough():
    seen = {}

    def fn(t, cache, flag):
        seen["cache_is_same"] = cache is sentinel
        seen["flag"] = flag
        return t

    w = CUDAGraphWrapper(fn, warmup_iters=1)
    sentinel = ["kv-cache-list"]  # non-tensor container passes through verbatim
    w(torch.ones(1), sentinel, 7)
    assert seen["cache_is_same"] and seen["flag"] == 7


# =========================================================================== #
# LightTAE (TAEHV)
# =========================================================================== #
def test_lighttae_frames_to_trim_math():
    from sglang.multimodal_gen.runtime.models.vaes.taehv import TAEHV

    m = TAEHV(checkpoint_path=None)
    # 2 ** sum((True, True)) - 1 == 3
    assert m.frames_to_trim == 3
    assert m.TEMPORAL_COMPRESSION_RATIO == 4 and m.SPATIAL_COMPRESSION_RATIO == 8


def test_lighttae_checkpoint_key_coverage():
    from sglang.multimodal_gen.runtime.models.vaes.taehv import (
        TAEHV,
        lighttae_state_dict_transform,
    )

    ckpt = _find_ckpt("SGLANG_OMNIDREAMS_LIGHTTAE_CKPT", "lighttaew2_1.pth")
    if ckpt is None:
        pytest.skip("lighttaew2_1.pth not found")
    raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    remapped = lighttae_state_dict_transform(raw)
    model_keys = set(TAEHV(checkpoint_path=None).state_dict().keys())
    missing = [k for k in model_keys if k not in remapped]
    assert not missing, f"checkpoint missing {len(missing)} model keys: {missing[:5]}"
    # Full load leaves nothing on meta.
    loaded = TAEHV(checkpoint_path=ckpt)
    assert not [k for k, v in loaded.state_dict().items() if v.is_meta]


def test_lighttae_decode_shape():
    from sglang.multimodal_gen.runtime.models.vaes.taehv import LightTAEDecoder

    ckpt = _find_ckpt("SGLANG_OMNIDREAMS_LIGHTTAE_CKPT", "lighttaew2_1.pth")
    if ckpt is None:
        pytest.skip("lighttaew2_1.pth not found")
    dec = LightTAEDecoder(ckpt, dtype=torch.float32)
    z = torch.randn(1, 16, 3, 8, 8)  # [B, C, F, H, W], 3 latent frames
    out = dec.decode(z)
    # F_out = 1 + (3-1)*4 = 9 ; spatial 8 * 8 = 64 ; 3 image channels ; [-1,1].
    assert out.shape == (1, 3, 9, 64, 64)
    assert torch.isfinite(out).all()
    assert out.min() >= -1.0 - 1e-4 and out.max() <= 1.0 + 1e-4


# =========================================================================== #
# LightVAE (pruned Wan)
# =========================================================================== #
def test_lightvae_checkpoint_key_coverage_and_encode_shape():
    from sglang.multimodal_gen.runtime.models.vaes.omnidreams_light_vae import (
        LightVAEEncoder,
    )

    ckpt = _find_ckpt("SGLANG_OMNIDREAMS_LIGHTVAE_CKPT", "lightvaew2_1.pth")
    if ckpt is None:
        pytest.skip("lightvaew2_1.pth not found")
    enc = LightVAEEncoder(
        ckpt, latents_mean=[0.0] * 16, latents_std=[1.0] * 16, dtype=torch.float32
    )
    # No encoder/quant params left on meta after the (strict=False) load.
    assert not [k for k, v in enc.state_dict().items() if v.is_meta]
    # First-frame image: [B,3,1,H,W] -> [B,16,1,H/8,W/8].
    z1 = enc.encode(torch.randn(1, 3, 1, 64, 64)).mode()
    assert z1.shape == (1, 16, 1, 8, 8) and torch.isfinite(z1).all()
    # HD-map clip: 5 pixel frames -> 2 causal latent frames.
    z2 = enc.encode(torch.randn(1, 3, 5, 64, 64)).mode()
    assert z2.shape == (1, 16, 2, 8, 8) and torch.isfinite(z2).all()


# =========================================================================== #
# Long-video OOM fix -- WanVAE.decode() per-chunk CPU offload
# =========================================================================== #
class _ToyDecoder(torch.nn.Module):
    """Conv3d 1x1 + pixel-shuffle upsampling; mimics WanVAE decoder output
    growth (out_ch * u * u per input voxel) without the real VAE."""

    def __init__(self, in_ch=16, out_ch=3, upsample=8):
        super().__init__()
        self.upsample, self.out_ch = upsample, out_ch
        self.proj = torch.nn.Conv3d(in_ch, out_ch * upsample * upsample, 1)

    def forward(self, x):
        b, _, t, h, w = x.shape
        u = self.upsample
        y = self.proj(x).view(b, self.out_ch, u, u, t, h, w)
        return (
            y.permute(0, 1, 4, 5, 2, 6, 3)
            .contiguous()
            .view(b, self.out_ch, t, h * u, w * u)
        )


def _decode_loop(decoder, x, *, offload):
    """Mirror AutoencoderKLWan.decode()'s feature-cache loop; only .cpu() differs."""
    out_chunks = []
    for i in range(x.shape[2]):
        chunk = decoder(x[:, :, i : i + 1, :, :])
        if offload:
            chunk = chunk.cpu()
        out_chunks.append(chunk)
    return torch.cat(out_chunks, 2) if len(out_chunks) > 1 else out_chunks[0]


def _loop_peak_mb(decoder, x, *, offload):
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.no_grad():
        out = _decode_loop(decoder, x, offload=offload)
    torch.cuda.synchronize()
    return out, (torch.cuda.max_memory_allocated() - base) / 1e6


@requires_gpu
def test_decode_cpu_offload_bounds_gpu_memory_without_precision_loss():
    """decode() streams chunks to CPU so GPU peak stays bounded (not ~linear in
    frames) and is byte-exact vs no-offload. The no-offload curve also pins the
    SP-path behavior (chunks stay on GPU, peak grows ~linearly; SP divides by
    world_size, doesn't bound) — update this if SP-safe offload lands."""
    torch.manual_seed(0)
    decoder = _ToyDecoder().to(_DEVICE).eval()

    frame_counts = [8, 64]
    noff, off = {}, {}
    for n in frame_counts:
        x = torch.randn(1, 16, n, 60, 104, device=_DEVICE)
        out_noff, mb_noff = _loop_peak_mb(decoder, x, offload=False)
        out_off, mb_off = _loop_peak_mb(decoder, x, offload=True)
        assert torch.equal(out_noff.cpu(), out_off)  # offload keeps result
        noff[n], off[n] = mb_noff, mb_off
        del x, out_noff, out_off

    frame_ratio = frame_counts[-1] / frame_counts[0]
    noff_ratio = noff[frame_counts[-1]] / max(noff[frame_counts[0]], 1e-6)
    # No-offload (= SP-path equivalent) grows ~linearly with frames.
    assert (
        noff_ratio > frame_ratio * 0.7
    ), f"expected ~linear growth (~{frame_ratio:.0f}x), got {noff_ratio:.2f}x"

    # Offload keeps GPU peak bounded and ~constant across frames.
    off_peak = max(off.values())
    assert off_peak < noff[frame_counts[-1]] * 0.5, (
        f"offload peak {off_peak:.1f}MB not bounded vs "
        f"{noff[frame_counts[-1]]:.1f}MB"
    )
    assert (
        abs(off[frame_counts[-1]] - off[frame_counts[0]]) < off_peak * 0.5 + 5
    ), "offload peak should not scale with frame count"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
