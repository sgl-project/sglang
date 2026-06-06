from __future__ import annotations

import os
import socket
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.layernorm import (
    RMSNorm,
    apply_qk_norm_with_optional_rope,
)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    # Inlined eager GPT-J interleaved rotary; importing the real
    # ``_apply_rotary_emb`` triggers a ``register_custom_op_from_extern``
    # against a stubbed ``flashinfer`` symbol that crashes pytest collection
    # on the CPU-only harness.
    assert not is_neox_style, "test only exercises GPT-J interleaved rotary"
    cos_b = cos.unsqueeze(-2).to(x.dtype)
    sin_b = sin.unsqueeze(-2).to(x.dtype)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos_b - x2 * sin_b
    o2 = x2 * cos_b + x1 * sin_b
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def _make_qk_norms(head_dim: int, dtype: torch.dtype) -> tuple[RMSNorm, RMSNorm]:
    # Non-trivial gains so a silently-dropped weight multiply would fail.
    # NB: ``RMSNorm.__init__`` ignores its ``dtype`` kwarg for weights (always
    # fp32); the explicit ``.to(dtype)`` is required for the fused CUDA path's
    # ``q_norm.weight.dtype == q.dtype`` gate to pass.
    torch.manual_seed(0)
    q_norm = RMSNorm(head_dim, eps=1e-6, dtype=dtype)
    k_norm = RMSNorm(head_dim, eps=1e-6, dtype=dtype)
    with torch.no_grad():
        q_norm.weight.copy_(torch.empty(head_dim).uniform_(0.9, 1.1))
        k_norm.weight.copy_(torch.empty(head_dim).uniform_(0.85, 1.15))
    return q_norm.to(dtype), k_norm.to(dtype)


def _make_cos_sin(
    num_tokens: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    half = head_dim // 2
    cos = torch.randn(num_tokens, half) * 0.1 + 0.9
    sin = torch.randn(num_tokens, half) * 0.1
    cos_sin_cache = torch.cat(
        [cos.to(torch.float32).contiguous(), sin.to(torch.float32).contiguous()],
        dim=-1,
    )
    return cos, sin, cos_sin_cache


@pytest.mark.parametrize(
    "batch,img_len,num_heads,head_dim",
    [
        (1, 256, 4, 64),
        (2, 64, 2, 32),
    ],
)
def test_double_stream_img_norm_rope_parity(batch, img_len, num_heads, head_dim):
    dtype = torch.float32
    q_norm, k_norm = _make_qk_norms(head_dim, dtype)
    cos, sin, cos_sin_cache = _make_cos_sin(img_len, head_dim)

    torch.manual_seed(2)
    img_q = torch.randn(batch, img_len, num_heads, head_dim, dtype=dtype)
    img_k = torch.randn(batch, img_len, num_heads, head_dim, dtype=dtype)
    img_v = torch.randn(batch, img_len, num_heads, head_dim, dtype=dtype)

    ref_q = q_norm(img_q.contiguous()).to(img_v)
    ref_k = k_norm(img_k.contiguous()).to(img_v)
    ref_q = _apply_rotary_emb(ref_q, cos, sin, is_neox_style=False)
    ref_k = _apply_rotary_emb(ref_k, cos, sin, is_neox_style=False)

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

    torch.testing.assert_close(out_q, ref_q, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_k, ref_k, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "batch,txt_len,num_heads,head_dim",
    [
        (1, 64, 4, 64),
        (2, 256, 8, 32),
    ],
)
def test_double_stream_txt_norm_only_parity(batch, txt_len, num_heads, head_dim):
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
    # Pin: pre-refactor (RMSNorm-then-split) == post-refactor (split-then-RMSNorm).
    # RMSNorm is per-token so this reordering is mathematically a no-op.
    dtype = torch.float32
    seq_len = img_len + txt_len
    q_norm, k_norm = _make_qk_norms(head_dim, dtype)
    cos, sin, cos_sin_cache = _make_cos_sin(img_len, head_dim)

    torch.manual_seed(4)
    qkv = torch.randn(batch, seq_len, 3, num_heads, head_dim, dtype=dtype)

    q_full = qkv[:, :, 0]
    k_full = qkv[:, :, 1]
    v_full = qkv[:, :, 2]
    q_ref = q_norm(q_full.contiguous()).to(v_full.dtype)
    k_ref = k_norm(k_full.contiguous()).to(v_full.dtype)
    img_q_ref, txt_q_ref = q_ref[:, :img_len], q_ref[:, img_len:]
    img_k_ref, txt_k_ref = k_ref[:, :img_len], k_ref[:, img_len:]
    img_q_ref = _apply_rotary_emb(img_q_ref, cos, sin, is_neox_style=False)
    img_k_ref = _apply_rotary_emb(img_k_ref, cos, sin, is_neox_style=False)

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


def test_helper_preserves_input_dtype_fp32():
    # CPU path restricted to fp32: ``RMSNorm.forward_cuda`` routes bf16/fp16
    # through a CUDA-only triton custom op when ``current_platform.is_cuda()``,
    # which fires on any GPU host even with CPU tensors. The bf16 sibling
    # below exercises that path on an actual CUDA device.
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
    # Spy on ``fused_inplace_qknorm_rope`` to confirm the fused CUDA kernel
    # actually fires; otherwise the helper would silently fall back to the
    # eager + flashinfer-rope branch and the test would pass for the wrong
    # reason.
    import sglang.multimodal_gen.runtime.layers.layernorm as ln_mod

    device = torch.device("cuda")
    head_dim = 64
    dtype = torch.bfloat16
    q_norm, k_norm = _make_qk_norms(head_dim, dtype)
    q_norm = q_norm.to(device=device, dtype=dtype)
    k_norm = k_norm.to(device=device, dtype=dtype)

    torch.manual_seed(6)
    q = torch.randn(1, 16, 4, head_dim, dtype=dtype, device=device)
    k = torch.randn(1, 16, 4, head_dim, dtype=dtype, device=device)
    _, _, cos_sin_cache = _make_cos_sin(16, head_dim)
    cos_sin_cache = cos_sin_cache.to(device)

    with patch.object(
        ln_mod,
        "fused_inplace_qknorm_rope",
        wraps=ln_mod.fused_inplace_qknorm_rope,
    ) as spy:
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
    assert spy.call_count >= 1
    assert out_q.dtype == dtype
    assert out_k.dtype == dtype


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused QK-Norm+RoPE kernel requires CUDA",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_kernel_matches_eager_reference_cuda(dtype):
    # Numerical parity on the actual fused fast path. The fp32 parity tests
    # above only exercise the eager fallback because the fused kernel gates
    # on ``q.dtype in (fp16, bf16)``; this test runs at the production dtype
    # and uses a spy to confirm the fused branch fires (not silently falling
    # back to eager+flashinfer-rope and passing for the wrong reason).
    import sglang.multimodal_gen.runtime.layers.layernorm as ln_mod

    device = torch.device("cuda")
    head_dim = 64
    num_tokens = 256
    num_heads = 4
    batch = 1
    q_norm, k_norm = _make_qk_norms(head_dim, dtype)
    q_norm = q_norm.to(device=device, dtype=dtype)
    k_norm = k_norm.to(device=device, dtype=dtype)

    torch.manual_seed(7)
    q = torch.randn(batch, num_tokens, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, num_tokens, num_heads, head_dim, dtype=dtype, device=device)
    cos, sin, cos_sin_cache = _make_cos_sin(num_tokens, head_dim)
    cos = cos.to(device)
    sin = sin.to(device)
    cos_sin_cache = cos_sin_cache.to(device)

    # Eager reference: 2x RMSNorm + 2x rotary on the SAME inputs, in dtype.
    ref_q = q_norm(q.contiguous()).to(dtype)
    ref_k = k_norm(k.contiguous()).to(dtype)
    ref_q = _apply_rotary_emb(ref_q, cos, sin, is_neox_style=False)
    ref_k = _apply_rotary_emb(ref_k, cos, sin, is_neox_style=False)

    # Fused path. allow_inplace=True so we clone inputs to avoid mutating
    # the (already-consumed) eager-reference tensors.
    with patch.object(
        ln_mod,
        "fused_inplace_qknorm_rope",
        wraps=ln_mod.fused_inplace_qknorm_rope,
    ) as spy:
        out_q, out_k = apply_qk_norm_with_optional_rope(
            q=q.clone().contiguous(),
            k=k.clone().contiguous(),
            q_norm=q_norm,
            k_norm=k_norm,
            head_dim=head_dim,
            cos_sin_cache=cos_sin_cache,
            is_neox=False,
            allow_inplace=True,
        )
    assert spy.call_count >= 1, (
        "expected fused_inplace_qknorm_rope to fire on bf16/fp16 CUDA inputs; "
        "helper silently fell back to eager (likely a dtype-gate regression)"
    )

    # Tolerances chosen for fused vs eager at the head_dim reductions used in
    # HunyuanVideo. bf16 has ~7 mantissa bits so ULP near 1.0 is ~7e-3; the
    # fused kernel does its reduction in fp32 internally then casts back, so
    # 2e-2 atol/rtol is the loosest credible parity bar. fp16 keeps ~5e-3.
    tol = {torch.bfloat16: 2e-2, torch.float16: 5e-3}[dtype]
    torch.testing.assert_close(out_q, ref_q, atol=tol, rtol=tol)
    torch.testing.assert_close(out_k, ref_k, atol=tol, rtol=tol)


# Block-level tests --------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _init_minimal_distributed():
    # The blocks construct ``ReplicatedLinear`` / ``ColumnParallelLinear``
    # which call ``get_tp_group()``. Initialise a 1-rank gloo group so
    # construction works on CPU; tear down only if we created it.
    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    was_initialized = dist.is_initialized()
    if not was_initialized:
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

    if not was_initialized and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


class _StubAttention(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, img_q, img_k, img_v, txt_q, txt_k, txt_v):
        return torch.zeros_like(img_v), torch.zeros_like(txt_v)


def _block_test_device() -> torch.device:
    # ``CustomOp.forward_cuda`` (``fuse_scale_shift_kernel``,
    # ``triton_one_pass_rms_norm``) is selected at module init from
    # ``current_platform.is_cuda()``, not from tensor device. On CUDA hosts
    # those dispatches assert ``x.is_cuda``, so run blocks on the device
    # that matches the platform.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_block_with_stub_attention(block_cls, **kwargs):
    import sglang.multimodal_gen.runtime.models.dits.hunyuanvideo as hv_mod

    with patch.object(hv_mod, "UlyssesAttention", _StubAttention):
        return block_cls(**kwargs).eval()


def test_double_stream_block_fuses_img_and_txt_sites():
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

    with patch.object(
        hv_mod,
        "apply_qk_norm_with_optional_rope",
        wraps=hv_mod.apply_qk_norm_with_optional_rope,
    ) as spy:
        with torch.no_grad():
            out_img, out_txt = block(img, txt, vec, freqs_cis)

    assert spy.call_count == 2
    img_call = spy.call_args_list[0].kwargs
    txt_call = spy.call_args_list[1].kwargs
    assert img_call["cos_sin_cache"] is not None
    assert txt_call["cos_sin_cache"] is None
    assert img_call["q"].shape == (
        batch_size,
        img_seq_len,
        num_attention_heads,
        head_dim,
    )
    assert txt_call["q"].shape == (
        batch_size,
        txt_seq_len,
        num_attention_heads,
        head_dim,
    )
    assert out_img.shape == (batch_size, img_seq_len, hidden_size)
    assert out_txt.shape == (batch_size, txt_seq_len, hidden_size)


def test_single_stream_block_fuses_img_and_txt_and_splits_before_norm():
    # Pin: the refactor must split *before* QK-Norm. The first spy call
    # must see Q with shape [B, img_len, H, head_dim], not the full
    # [B, seq_len, H, head_dim] that a pre-refactor full-norm-then-split
    # would produce.
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

    with patch.object(
        hv_mod,
        "apply_qk_norm_with_optional_rope",
        wraps=hv_mod.apply_qk_norm_with_optional_rope,
    ) as spy:
        with torch.no_grad():
            out = block(x, vec, txt_len, freqs_cis)

    assert spy.call_count == 2
    img_call = spy.call_args_list[0].kwargs
    txt_call = spy.call_args_list[1].kwargs
    assert img_call["q"].shape == (
        batch_size,
        img_len,
        num_attention_heads,
        head_dim,
    )
    assert txt_call["q"].shape == (
        batch_size,
        txt_len,
        num_attention_heads,
        head_dim,
    )
    assert img_call["cos_sin_cache"] is not None
    assert txt_call["cos_sin_cache"] is None
    assert out.shape == (batch_size, seq_len, hidden_size)


def test_single_stream_block_does_not_redundantly_contiguify_v():
    # Pin: the fused QK-Norm+RoPE kernel only mutates Q/K
    # (``mutates_args=["q", "k"]``); ``UlyssesAttention`` does
    # ``torch.cat([q, k, v])`` which accepts non-contiguous V. Adding
    # ``.contiguous()`` on img_v / txt_v would silently double V-sized HBM
    # traffic per single-stream block.
    import sglang.multimodal_gen.runtime.models.dits.hunyuanvideo as hv_mod
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

    device = _block_test_device()
    with patch.object(hv_mod, "UlyssesAttention", _CapturingAttn):
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
        block(x, vec, txt_len, (cos, sin))

    assert not captured["img_v"].is_contiguous()
    assert not captured["txt_v"].is_contiguous()
