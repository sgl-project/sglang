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

    NOTE: sglang's ``RMSNorm.__init__`` accepts a ``dtype`` kwarg but ignores
    it for weight construction -- the parameter is hardcoded as
    ``nn.Parameter(torch.ones(hidden_size))``, which defaults to fp32. We
    explicitly cast the modules at the end so the weight dtype matches the
    requested ``dtype``. Without this, the fused CUDA path's
    ``q_norm.weight.dtype == q.dtype`` gate silently fails for bf16/fp16
    inputs and ``apply_qk_norm_rope`` falls back to the eager path -- i.e.,
    the test would *not* exercise the fused kernel it claims to test.
    """
    torch.manual_seed(0)
    q_norm = RMSNorm(head_dim, eps=1e-6, dtype=dtype)
    k_norm = RMSNorm(head_dim, eps=1e-6, dtype=dtype)
    with torch.no_grad():
        q_norm.weight.copy_(torch.empty(head_dim).uniform_(0.9, 1.1))
        k_norm.weight.copy_(torch.empty(head_dim).uniform_(0.85, 1.15))
    q_norm = q_norm.to(dtype)
    k_norm = k_norm.to(dtype)
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
    """bf16 dtype-preservation check on an actual CUDA device, *and* proof
    that the fused JIT kernel actually fires.

    Skipped on CPU-only hosts: sglang's ``RMSNorm.forward_cuda`` routes
    bf16/fp16 inputs through ``triton_one_pass_rms_norm`` for
    ``hidden_size <= 128``, which is registered only for the CUDA backend.

    A spy on ``fused_inplace_qknorm_rope`` asserts the fused-CUDA path was
    actually taken -- if the helper falls back to the eager + flashinfer
    rope branch (e.g. because the norm weights were left as fp32 by
    `_make_qk_norms`), the spy's call_count stays 0 and the test fails
    instead of silently passing through the wrong code path.
    """
    from unittest.mock import patch

    import sglang.multimodal_gen.runtime.layers.layernorm as ln_mod

    device = torch.device("cuda")
    head_dim = 64
    dtype = torch.bfloat16
    q_norm, k_norm = _make_qk_norms(head_dim, dtype)
    # NB: `.to(device=..., dtype=...)` -- moving dtype too is what makes the
    # weight match q.dtype downstream. Plain `.to(device)` would keep the
    # weight fp32 (see `_make_qk_norms` note) and bypass the fused gate.
    q_norm = q_norm.to(device=device, dtype=dtype)
    k_norm = k_norm.to(device=device, dtype=dtype)
    assert q_norm.weight.dtype == dtype
    assert k_norm.weight.dtype == dtype

    torch.manual_seed(6)
    q = torch.randn(1, 16, 4, head_dim, dtype=dtype, device=device)
    k = torch.randn(1, 16, 4, head_dim, dtype=dtype, device=device)
    _, _, cos_sin_cache = _make_cos_sin(16, head_dim)
    cos_sin_cache = cos_sin_cache.to(device)

    real = ln_mod.fused_inplace_qknorm_rope
    with patch.object(
        ln_mod, "fused_inplace_qknorm_rope", wraps=real
    ) as mock_fused:
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
    assert mock_fused.call_count >= 1, (
        "Expected fused_inplace_qknorm_rope to fire at least once on CUDA "
        f"bf16; got call_count={mock_fused.call_count} (helper silently "
        "fell back to the eager + flashinfer rope branch)"
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


# -- Block-level tests with mocked attention -------------------------------
#
# The helper-level tests above pin the *math* of the fusion. The tests
# below pin the *wiring* -- that ``MMDoubleStreamBlock.forward`` and
# ``MMSingleStreamBlock.forward`` actually invoke
# ``apply_qk_norm_with_optional_rope`` at the intended sites, with the
# intended arguments. They construct the production block modules on CPU,
# patch the attention call with a shape-preserving mock, spy on the helper,
# and assert the expected call count + arg shapes.
#
# Construction requires a 1-rank tensor-parallel group because the blocks
# instantiate ``ReplicatedLinear`` / ``ColumnParallelLinear`` which call
# ``get_tp_group()``. A module-scoped fixture initializes a minimal
# gloo-backed single-rank group once and tears it down at the end.


@pytest.fixture(scope="module", autouse=True)
def _init_minimal_distributed():
    """Initialize a 1-rank gloo distributed + TP group so the block
    constructors can resolve ``get_tp_group()`` on CPU. Single-process,
    single-rank -- no real collective work happens."""
    import os
    import socket

    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    was_initialized = dist.is_initialized()
    if not was_initialized:
        # Pick a free port so concurrent test runs / leftover sockets don't clash.
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = str(free_port)
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{free_port}",
            local_rank=0,
            backend="gloo",
        )

    initialize_model_parallel(
        data_parallel_size=1,
        classifier_free_guidance_degree=1,
        sequence_parallel_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
    )

    yield

    # Only tear down what this fixture created -- leave any pre-existing
    # distributed state untouched.
    if not was_initialized and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
#
# In particular, ``test_single_stream_block_*_splits_before_norm`` is the
# only test that actually proves the single-stream split-then-fuse refactor
# went through ``MMSingleStreamBlock.forward`` (not just through a
# hand-rolled math reference): the first spy call must see a Q tensor of
# shape ``[B, img_len, H, head_dim]``, not the full ``[B, seq_len, ...]``
# that the pre-refactor full-norm-then-split path would have produced.


class _StubAttention(torch.nn.Module):
    """Stand-in for ``UlyssesAttention`` during block construction. Takes any
    constructor args (so the block doesn't need ServerArgs / attn-backend
    setup), and on forward returns zero tensors matching the V shapes --
    enough for ``MMDoubleStreamBlock`` / ``MMSingleStreamBlock`` to flow
    through their post-attention residual / MLP layers without changing
    what we're testing (which is the QK-Norm + RoPE call sites)."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, img_q, img_k, img_v, txt_q, txt_k, txt_v):
        return torch.zeros_like(img_v), torch.zeros_like(txt_v)


def _block_test_device() -> torch.device:
    """Pick the device for the block-level tests.

    On a CUDA-capable host sglang's ``CustomOp`` dispatch (e.g.
    ``LayerNormScaleShift.forward_cuda`` -> ``fuse_scale_shift_kernel``,
    and ``RMSNorm`` -> ``triton_one_pass_rms_norm``) is selected at module
    init time from ``current_platform.is_cuda()``, NOT from the input
    tensor's device. Running these dispatch paths against CPU tensors
    blows up with ``assert (x.is_cuda and scale.is_cuda)``-style errors
    even though the test isn't trying to exercise CUDA kernels.

    So: run the block forward on CUDA when CUDA is available; fall back to
    CPU (which has its own ``forward_cpu`` / ``forward_native`` paths) on
    CPU-only hosts. The assertions we care about (helper call count,
    helper kwarg shapes, V contiguity) hold on either device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_block_with_stub_attention(block_cls, **kwargs):
    """Construct ``block_cls`` with the production ``UlyssesAttention``
    constructor temporarily swapped for ``_StubAttention``, so the block
    can be instantiated without requiring a global ServerArgs."""
    from unittest.mock import patch

    import sglang.multimodal_gen.runtime.models.dits.hunyuanvideo as hv_mod

    with patch.object(hv_mod, "UlyssesAttention", _StubAttention):
        return block_cls(**kwargs).eval()


def test_double_stream_block_fuses_img_and_txt_sites():
    """MMDoubleStreamBlock.forward calls apply_qk_norm_with_optional_rope
    exactly twice: img stream with cos_sin_cache (norm + rope), txt stream
    with cos_sin_cache=None (norm only). Output shapes match input shapes."""
    from unittest.mock import patch

    import sglang.multimodal_gen.runtime.models.dits.hunyuanvideo as hv_mod
    from sglang.multimodal_gen.runtime.models.dits.hunyuanvideo import (
        MMDoubleStreamBlock,
    )

    hidden_size = 128
    num_attention_heads = 4
    head_dim = hidden_size // num_attention_heads
    batch_size = 1
    img_seq_len = 16
    txt_seq_len = 8

    device = _block_test_device()
    block = _build_block_with_stub_attention(
        MMDoubleStreamBlock,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        mlp_ratio=2.0,
    ).to(device)

    torch.manual_seed(0)
    img = torch.randn(batch_size, img_seq_len, hidden_size, device=device)
    txt = torch.randn(batch_size, txt_seq_len, hidden_size, device=device)
    vec = torch.randn(batch_size, hidden_size, device=device)
    cos = torch.randn(img_seq_len, head_dim // 2, device=device)
    sin = torch.randn(img_seq_len, head_dim // 2, device=device)
    freqs_cis = (cos, sin)

    real = hv_mod.apply_qk_norm_with_optional_rope
    with patch.object(
        hv_mod, "apply_qk_norm_with_optional_rope", wraps=real
    ) as spy:
        with torch.no_grad():
            out_img, out_txt = block(img, txt, vec, freqs_cis)

    assert spy.call_count == 2, (
        f"MMDoubleStreamBlock should fuse exactly 2 sites; got {spy.call_count}"
    )
    img_call = spy.call_args_list[0].kwargs
    txt_call = spy.call_args_list[1].kwargs
    assert img_call["cos_sin_cache"] is not None, (
        "img stream call must pass cos_sin_cache for rope"
    )
    assert txt_call["cos_sin_cache"] is None, (
        "txt stream call must pass cos_sin_cache=None (no rope on text)"
    )
    # Q tensors at the helper boundary are [B, seq_len, H, head_dim].
    assert img_call["q"].shape == (
        batch_size, img_seq_len, num_attention_heads, head_dim,
    )
    assert txt_call["q"].shape == (
        batch_size, txt_seq_len, num_attention_heads, head_dim,
    )
    # Output residual connections preserve input shape.
    assert out_img.shape == (batch_size, img_seq_len, hidden_size)
    assert out_txt.shape == (batch_size, txt_seq_len, hidden_size)


def test_single_stream_block_fuses_img_and_txt_and_splits_before_norm():
    """MMSingleStreamBlock.forward calls apply_qk_norm_with_optional_rope
    twice, and *the split happens before the norm*: the first (img) call
    receives Q with shape [B, img_len, H, head_dim], not the full
    [B, seq_len, H, head_dim].

    This is the property that pins the single-stream split-then-fuse
    refactor in place. Pre-refactor (full-RMSNorm-then-split-then-rope)
    would have seen full ``seq_len`` here -- so this test would fail if
    the wiring ever reverts to the old shape."""
    from unittest.mock import patch

    import sglang.multimodal_gen.runtime.models.dits.hunyuanvideo as hv_mod
    from sglang.multimodal_gen.runtime.models.dits.hunyuanvideo import (
        MMSingleStreamBlock,
    )

    hidden_size = 128
    num_attention_heads = 4
    head_dim = hidden_size // num_attention_heads
    batch_size = 1
    img_len = 16
    txt_len = 8
    seq_len = img_len + txt_len

    device = _block_test_device()
    block = _build_block_with_stub_attention(
        MMSingleStreamBlock,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        mlp_ratio=2.0,
    ).to(device)

    torch.manual_seed(0)
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    vec = torch.randn(batch_size, hidden_size, device=device)
    cos = torch.randn(img_len, head_dim // 2, device=device)
    sin = torch.randn(img_len, head_dim // 2, device=device)
    freqs_cis = (cos, sin)

    real = hv_mod.apply_qk_norm_with_optional_rope
    with patch.object(
        hv_mod, "apply_qk_norm_with_optional_rope", wraps=real
    ) as spy:
        with torch.no_grad():
            out = block(x, vec, txt_len, freqs_cis)

    assert spy.call_count == 2, (
        f"MMSingleStreamBlock should fuse exactly 2 sites; got {spy.call_count}"
    )
    img_call = spy.call_args_list[0].kwargs
    txt_call = spy.call_args_list[1].kwargs
    # Split-before-norm pin: img Q has img_len tokens, NOT the full seq_len.
    assert img_call["q"].shape == (
        batch_size, img_len, num_attention_heads, head_dim,
    ), (
        f"img call q shape {tuple(img_call['q'].shape)} does not match the "
        f"split-before-norm contract [B, img_len, H, head_dim]"
    )
    assert txt_call["q"].shape == (
        batch_size, txt_len, num_attention_heads, head_dim,
    )
    # img call applies rope; txt call does not.
    assert img_call["cos_sin_cache"] is not None
    assert txt_call["cos_sin_cache"] is None
    # Block output preserves [B, seq_len, hidden] shape.
    assert out.shape == (batch_size, seq_len, hidden_size)


def test_single_stream_block_does_not_redundantly_contiguify_v():
    """MMSingleStreamBlock.forward must not allocate fresh contiguous V
    tensors. The fused QK-Norm+RoPE kernel only mutates Q and K (per
    ``mutates_args=["q", "k"]`` on ``fused_inplace_qknorm_rope``);
    UlyssesAttention.forward then does ``torch.cat([q, k, v], dim=0)``
    downstream, which handles non-contiguous V fine. Adding ``.contiguous()``
    on img_v / txt_v would silently double the V-sized HBM traffic per
    single-stream block and erode the fusion's measured speedup at
    HunyuanVideo / LTX-2 token counts.

    This test pins the property by capturing V via the mocked attention
    call and asserting it shares storage with the linear1 output (i.e., it
    is a strided view, not a fresh allocation)."""
    import sglang.multimodal_gen.runtime.models.dits.hunyuanvideo as hv_mod  # noqa: F401  (kept for symmetry)
    from sglang.multimodal_gen.runtime.models.dits.hunyuanvideo import (
        MMSingleStreamBlock,
    )

    hidden_size = 128
    num_attention_heads = 4
    batch_size = 1
    img_len = 16
    txt_len = 8
    seq_len = img_len + txt_len

    captured: dict[str, torch.Tensor] = {}

    class _CapturingAttn(_StubAttention):
        def forward(self, img_q, img_k, img_v, txt_q, txt_k, txt_v):
            captured["img_v"] = img_v
            captured["txt_v"] = txt_v
            return torch.zeros_like(img_v), torch.zeros_like(txt_v)

    from unittest.mock import patch as _patch

    import sglang.multimodal_gen.runtime.models.dits.hunyuanvideo as _hv_mod

    device = _block_test_device()
    with _patch.object(_hv_mod, "UlyssesAttention", _CapturingAttn):
        block = (
            MMSingleStreamBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                mlp_ratio=2.0,
            )
            .to(device)
            .eval()
        )

    torch.manual_seed(0)
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    vec = torch.randn(batch_size, hidden_size, device=device)
    head_dim = hidden_size // num_attention_heads
    cos = torch.randn(img_len, head_dim // 2, device=device)
    sin = torch.randn(img_len, head_dim // 2, device=device)
    with torch.no_grad():
        _ = block(x, vec, txt_len, (cos, sin))

    img_v = captured["img_v"]
    txt_v = captured["txt_v"]
    # Non-contiguous strided views (split out of the linear1 qkv tensor).
    # If someone re-adds .contiguous() to the img_v / txt_v lines in
    # MMSingleStreamBlock.forward, both of these tensors flip to
    # is_contiguous() == True and the assertion fires.
    assert not img_v.is_contiguous(), (
        "img_v should be a strided view of qkv (not a fresh contiguous "
        "copy); .contiguous() on img_v doubles V-sized HBM traffic per block"
    )
    assert not txt_v.is_contiguous(), (
        "txt_v should be a strided view of qkv (not a fresh contiguous "
        "copy); .contiguous() on txt_v doubles V-sized HBM traffic per block"
    )
