import inspect

import pytest
import torch


def _has_nvfp4_fa4_support() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() not in [(10, 0), (10, 3)]:
        return False
    try:
        from flash_attn.cute import flash_attn_func
        from flashinfer.quantization import SfLayout, nvfp4_quantize  # noqa: F401

        return "mSFQ" in inspect.signature(flash_attn_func).parameters
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_nvfp4_fa4_support(),
    reason="requires Blackwell and flash_attn.cute with mSFQ/mSFK support",
)


def test_nvfp4_fa4_attention_matches_bf16_shape_and_dtype(monkeypatch):
    monkeypatch.setenv("CUTE_DSL_ENABLE_TVM_FFI", "1")
    monkeypatch.setenv("SGLANG_DIFFUSION_NVFP4_FA4", "1")

    from flash_attn.cute import flash_attn_func

    from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
        FlashAttentionImpl,
        FlashAttentionMetadata,
        set_fa_ver,
    )
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )

    set_fa_ver(4)
    batch, seqlen, nheads, headdim = 1, 256, 12, 128
    torch.manual_seed(42)
    q = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)

    impl = FlashAttentionImpl(
        num_heads=nheads,
        head_size=headdim,
        causal=False,
        softmax_scale=headdim**-0.5,
        nvfp4_fa4=True,
    )
    metadata = FlashAttentionMetadata(max_seqlen_q=None, max_seqlen_k=None)
    with set_forward_context(0, metadata):
        out_fp4 = impl.forward(q, k, v)

    out_bf16 = flash_attn_func(q, k, v, softmax_scale=headdim**-0.5, causal=False)
    if isinstance(out_bf16, tuple):
        out_bf16 = out_bf16[0]

    assert out_fp4.shape == (batch, seqlen, nheads, headdim)
    assert out_fp4.dtype == torch.bfloat16
    assert not torch.isnan(out_fp4).any()

    cos = torch.nn.functional.cosine_similarity(
        out_fp4.flatten().float().unsqueeze(0),
        out_bf16.flatten().float().unsqueeze(0),
    ).item()
    assert cos > 0.95
