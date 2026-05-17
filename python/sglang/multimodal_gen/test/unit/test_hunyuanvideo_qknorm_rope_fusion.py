"""CPU parity tests for HunyuanVideo's QK-Norm + RoPE fusion.

HunyuanVideo's ``MMDoubleStreamBlock`` and ``MMSingleStreamBlock`` have been
refactored to call :func:`apply_qk_norm_with_optional_rope` (the same helper
that Flux / Qwen-Image use) instead of a hand-rolled sequence of two
``RMSNorm`` passes followed by two ``_apply_rotary_emb`` calls. These tests
pin the math: each test reconstructs the *pre-refactor* recipe and asserts the
fused-helper output matches it element-wise on float32 CPU.

Why CPU + float32: the fused CUDA kernel kicks in only when
``current_platform.is_cuda() and q.dtype in (fp16, bf16)``. On CPU we
exercise the helper's eager fallback (RMSNorm forward_native + flashinfer
rope's triton fallback, which itself swaps to ``apply_rotary_embedding_native``
on CPU). That fallback shares the same arithmetic kernels the pre-refactor
``_apply_rotary_emb`` resolved to, so float32 parity is exact.
"""

from __future__ import annotations

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.layernorm import (
    RMSNorm,
    apply_qk_norm_with_optional_rope,
)


# -- Fixtures / helpers ----------------------------------------------------


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """Inlined copy of the eager GPT-J style rotary kernel.

    Importing the real ``_apply_rotary_emb`` from
    ``sglang.multimodal_gen.runtime.layers.rotary_embedding.utils`` triggers
    that module's top-level ``register_custom_op_from_extern(...)`` against
    a stubbed ``flashinfer`` symbol, which fails at collection time on the
    CPU-only test harness (torch's ``infer_schema`` reads
    ``__globals__`` on the stub). The math below mirrors
    ``apply_rotary_embedding_native`` from
    ``sglang/jit_kernel/diffusion/triton/torch_fallback.py`` for
    ``is_neox_style=False`` with ``cos.shape[-1] == head_dim // 2`` — which
    is exactly what HunyuanVideo passes through and what the fused-helper
    fallback resolves to on CPU.
    """
    assert not is_neox_style, "test only exercises GPT-J interleaved rotary"
    cos_b = cos.unsqueeze(-2).to(x.dtype)
    sin_b = sin.unsqueeze(-2).to(x.dtype)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos_b - x2 * sin_b
    o2 = x2 * cos_b + x1 * sin_b
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def _make_qk_norms(head_dim: int, dtype: torch.dtype) -> tuple[RMSNorm, RMSNorm]:
    """Build a (q_norm, k_norm) pair with deliberately non-trivial weights.

    All-ones weights would make the test pass even if the implementation
    silently dropped the multiplication; uniform jitter forces the kernel to
    actually apply the gain.
    """
    torch.manual_seed(0)
    q_norm = RMSNorm(head_dim, eps=1e-6, dtype=dtype)
    k_norm = RMSNorm(head_dim, eps=1e-6, dtype=dtype)
    with torch.no_grad():
        q_norm.weight.copy_(torch.empty(head_dim).uniform_(0.9, 1.1))
        k_norm.weight.copy_(torch.empty(head_dim).uniform_(0.85, 1.15))
    return q_norm, k_norm


def _make_cos_sin(
    num_tokens: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct cos / sin in the layout HunyuanVideo passes around.

    Returns (cos, sin, cos_sin_cache) where:
      - ``cos``, ``sin`` are ``[num_tokens, head_dim // 2]`` (the layout
        ``_apply_rotary_emb`` expects)
      - ``cos_sin_cache`` is ``[num_tokens, head_dim]`` packed for the
        flashinfer-style helper (matches how Flux / Wan / hunyuanvideo build
        it just before calling the fused helper).
    """
    torch.manual_seed(1)
    half = head_dim // 2
    cos = torch.randn(num_tokens, half) * 0.1 + 0.9
    sin = torch.randn(num_tokens, half) * 0.1
    cos_sin_cache = torch.cat(
        [cos.to(torch.float32).contiguous(), sin.to(torch.float32).contiguous()],
        dim=-1,
    )
    return cos, sin, cos_sin_cache


# -- Double-stream image path: fused QK-Norm + RoPE -----------------------


@pytest.mark.parametrize(
    "batch,img_len,num_heads,head_dim",
    [
        (1, 256, 4, 64),  # narrow shape, single batch
        (2, 64, 2, 32),  # multi-batch, smaller
    ],
)
def test_double_stream_img_norm_rope_parity(batch, img_len, num_heads, head_dim):
    """img-stream: 2x RMSNorm + 2x _apply_rotary_emb == fused helper."""
    dtype = torch.float32
    q_norm, k_norm = _make_qk_norms(head_dim, dtype)
    cos, sin, cos_sin_cache = _make_cos_sin(img_len, head_dim)

    torch.manual_seed(2)
    img_q = torch.randn(batch, img_len, num_heads, head_dim, dtype=dtype)
    img_k = torch.randn(batch, img_len, num_heads, head_dim, dtype=dtype)
    img_v = torch.randn(batch, img_len, num_heads, head_dim, dtype=dtype)

    # Pre-refactor recipe: 2x RMSNorm + 2x rope.
    ref_q = q_norm(img_q.contiguous()).to(img_v)
    ref_k = k_norm(img_k.contiguous()).to(img_v)
    ref_q = _apply_rotary_emb(ref_q, cos, sin, is_neox_style=False)
    ref_k = _apply_rotary_emb(ref_k, cos, sin, is_neox_style=False)

    # New: fused helper.
    out_q, out_k = apply_qk_norm_with_optional_rope(
        q=img_q.clone().contiguous(),
        k=img_k.clone().contiguous(),
        q_norm=q_norm,
        k_norm=k_norm,
        head_dim=head_dim,
        cos_sin_cache=cos_sin_cache,
        is_neox=False,
        allow_inplace=True,
    )

    assert out_q.shape == ref_q.shape
    assert out_q.dtype == ref_q.dtype
    torch.testing.assert_close(out_q, ref_q, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_k, ref_k, atol=1e-5, rtol=1e-5)


# -- Double-stream text path: fused QK-Norm only (no RoPE) -----------------


@pytest.mark.parametrize(
    "batch,txt_len,num_heads,head_dim",
    [
        (1, 64, 4, 64),
        (2, 256, 8, 32),
    ],
)
def test_double_stream_txt_norm_only_parity(batch, txt_len, num_heads, head_dim):
    """txt-stream: 2x RMSNorm == fused helper with cos_sin_cache=None."""
    dtype = torch.float32
    q_norm, k_norm = _make_qk_norms(head_dim, dtype)

    torch.manual_seed(3)
    txt_q = torch.randn(batch, txt_len, num_heads, head_dim, dtype=dtype)
    txt_k = torch.randn(batch, txt_len, num_heads, head_dim, dtype=dtype)

    ref_q = q_norm(txt_q.contiguous())
    ref_k = k_norm(txt_k.contiguous())

    out_q, out_k = apply_qk_norm_with_optional_rope(
        q=txt_q.clone().contiguous(),
        k=txt_k.clone().contiguous(),
        q_norm=q_norm,
        k_norm=k_norm,
        head_dim=head_dim,
        cos_sin_cache=None,
        allow_inplace=True,
    )

    torch.testing.assert_close(out_q, ref_q, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_k, ref_k, atol=1e-5, rtol=1e-5)


# -- Single-stream split-first parity -------------------------------------


@pytest.mark.parametrize(
    "batch,img_len,txt_len,num_heads,head_dim",
    [
        (1, 256, 64, 4, 64),
        (2, 128, 32, 2, 32),
    ],
)
def test_single_stream_split_then_fuse_parity(
    batch, img_len, txt_len, num_heads, head_dim
):
    """Split-then-fuse must match (full-RMSNorm -> split -> rope) numerically.

    The refactor moves the image/text split before QK-Norm so the image
    portion can route through fused QK-Norm + RoPE while the text portion
    routes through fused QK-Norm only. RMSNorm is a per-token operation,
    so reordering split vs. norm is mathematically a no-op; this test
    pins that property.
    """
    dtype = torch.float32
    seq_len = img_len + txt_len
    q_norm, k_norm = _make_qk_norms(head_dim, dtype)
    cos, sin, cos_sin_cache = _make_cos_sin(img_len, head_dim)

    torch.manual_seed(4)
    qkv = torch.randn(batch, seq_len, 3, num_heads, head_dim, dtype=dtype)

    # Reference (pre-refactor): RMSNorm on full q/k, then split, then rope.
    q_full = qkv[:, :, 0]
    k_full = qkv[:, :, 1]
    v_full = qkv[:, :, 2]
    q_ref = q_norm(q_full.contiguous()).to(v_full.dtype)
    k_ref = k_norm(k_full.contiguous()).to(v_full.dtype)
    img_q_ref, txt_q_ref = q_ref[:, :img_len], q_ref[:, img_len:]
    img_k_ref, txt_k_ref = k_ref[:, :img_len], k_ref[:, img_len:]
    img_q_ref = _apply_rotary_emb(img_q_ref, cos, sin, is_neox_style=False)
    img_k_ref = _apply_rotary_emb(img_k_ref, cos, sin, is_neox_style=False)

    # New: split first, then fused per-stream.
    img_q = qkv[:, :img_len, 0].contiguous()
    img_k = qkv[:, :img_len, 1].contiguous()
    txt_q = qkv[:, img_len:, 0].contiguous()
    txt_k = qkv[:, img_len:, 1].contiguous()

    img_q, img_k = apply_qk_norm_with_optional_rope(
        q=img_q,
        k=img_k,
        q_norm=q_norm,
        k_norm=k_norm,
        head_dim=head_dim,
        cos_sin_cache=cos_sin_cache,
        is_neox=False,
        allow_inplace=True,
    )
    txt_q, txt_k = apply_qk_norm_with_optional_rope(
        q=txt_q,
        k=txt_k,
        q_norm=q_norm,
        k_norm=k_norm,
        head_dim=head_dim,
        cos_sin_cache=None,
        allow_inplace=True,
    )

    torch.testing.assert_close(img_q, img_q_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(img_k, img_k_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(txt_q, txt_q_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(txt_k, txt_k_ref, atol=1e-5, rtol=1e-5)


# -- Helper-level sanity checks -------------------------------------------


def test_helper_no_rope_path_matches_apply_qk_norm():
    """Passing ``cos_sin_cache=None`` routes to plain QK-Norm (no rope)."""
    dtype = torch.float32
    head_dim = 64
    q_norm, k_norm = _make_qk_norms(head_dim, dtype)
    torch.manual_seed(5)
    q = torch.randn(1, 32, 4, head_dim, dtype=dtype)
    k = torch.randn(1, 32, 4, head_dim, dtype=dtype)

    ref_q = q_norm(q.contiguous())
    ref_k = k_norm(k.contiguous())

    out_q, out_k = apply_qk_norm_with_optional_rope(
        q=q.clone().contiguous(),
        k=k.clone().contiguous(),
        q_norm=q_norm,
        k_norm=k_norm,
        head_dim=head_dim,
        cos_sin_cache=None,
        allow_inplace=True,
    )
    torch.testing.assert_close(out_q, ref_q, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_k, ref_k, atol=1e-5, rtol=1e-5)


def test_helper_preserves_input_dtype_fp32():
    """The helper must not silently upcast to float32 (the model relies on
    bf16/fp16 outputs flowing into attention).

    Restricted to fp32 in the CPU-only suite because sglang's ``RMSNorm``
    dispatches the low-precision path through a triton CUDA-only custom op
    (``sglang::triton_one_pass_rms_norm_cuda``) when
    ``current_platform.is_cuda()`` is True — which is the case on a GPU
    host even when our test tensors live on CPU. The bf16 variant lives
    in ``test_helper_preserves_input_dtype_bf16_cuda``, gated on an
    actually-reachable CUDA device.
    """
    head_dim = 64
    dtype = torch.float32
    q_norm, k_norm = _make_qk_norms(head_dim, dtype)
    torch.manual_seed(6)
    q = torch.randn(1, 16, 4, head_dim, dtype=dtype)
    k = torch.randn(1, 16, 4, head_dim, dtype=dtype)
    _, _, cos_sin_cache = _make_cos_sin(16, head_dim)
    out_q, out_k = apply_qk_norm_with_optional_rope(
        q=q.contiguous(),
        k=k.contiguous(),
        q_norm=q_norm,
        k_norm=k_norm,
        head_dim=head_dim,
        cos_sin_cache=cos_sin_cache,
        is_neox=False,
        allow_inplace=True,
    )
    assert out_q.dtype == dtype
    assert out_k.dtype == dtype


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="bf16 RMSNorm path runs through a CUDA-only triton kernel",
)
def test_helper_preserves_input_dtype_bf16_cuda():
    """bf16 dtype-preservation check on an actual CUDA device.

    Skipped on CPU-only hosts: sglang's ``RMSNorm.forward_cuda`` routes
    bf16/fp16 inputs through ``triton_one_pass_rms_norm`` for
    ``hidden_size <= 128``, which is registered only for the CUDA backend.
    """
    device = torch.device("cuda")
    head_dim = 64
    dtype = torch.bfloat16
    q_norm, k_norm = _make_qk_norms(head_dim, dtype)
    q_norm = q_norm.to(device)
    k_norm = k_norm.to(device)
    torch.manual_seed(6)
    q = torch.randn(1, 16, 4, head_dim, dtype=dtype, device=device)
    k = torch.randn(1, 16, 4, head_dim, dtype=dtype, device=device)
    _, _, cos_sin_cache = _make_cos_sin(16, head_dim)
    cos_sin_cache = cos_sin_cache.to(device)
    out_q, out_k = apply_qk_norm_with_optional_rope(
        q=q.contiguous(),
        k=k.contiguous(),
        q_norm=q_norm,
        k_norm=k_norm,
        head_dim=head_dim,
        cos_sin_cache=cos_sin_cache,
        is_neox=False,
        allow_inplace=True,
    )
    assert out_q.dtype == dtype
    assert out_k.dtype == dtype


# -- Routing test ----------------------------------------------------------


def test_hunyuanvideo_module_wires_fused_helper():
    """Smoke test: after the refactor, hunyuanvideo.py imports the fused
    helper and has stopped importing the per-pass rope helper."""
    import sglang.multimodal_gen.runtime.models.dits.hunyuanvideo as hv

    # The fused helper is bound at module scope (via layernorm imports).
    assert hasattr(hv, "apply_qk_norm_with_optional_rope")

    # The pre-refactor per-pass rope helper should no longer be a top-level
    # symbol — the refactor's intent is a single fused call site.
    assert not hasattr(hv, "_apply_rotary_emb")
