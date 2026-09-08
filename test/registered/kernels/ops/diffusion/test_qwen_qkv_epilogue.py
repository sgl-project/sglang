import sys
from unittest.mock import patch

import pytest
import torch

import sglang.multimodal_gen.runtime.models.dits.qwen_image as qwen_image
from sglang.kernels.ops.diffusion import try_fused_qwen_qkv_epilogue
from sglang.multimodal_gen.runtime.layers.layernorm import (
    RMSNorm,
    apply_qk_norm_with_optional_rope,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="Qwen-Image QKV epilogue requires SM90+",
)


@pytest.fixture(autouse=True)
def _seed_cuda():
    torch.cuda.manual_seed(0)


@pytest.mark.parametrize("img_tokens,txt_tokens,heads", [(17, 7, 4), (8152, 1365, 24)])
def test_qwen_qkv_epilogue_is_bit_exact(img_tokens, txt_tokens, heads):
    head_dim = 128
    img_qkv = [
        torch.randn(
            1,
            img_tokens,
            heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(3)
    ]
    txt_qkv = [
        torch.randn(
            1,
            txt_tokens,
            heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(3)
    ]
    norms = [
        RMSNorm(head_dim, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
        for _ in range(4)
    ]
    with torch.no_grad():
        for norm in norms:
            norm.weight.copy_(torch.randn_like(norm.weight))

    def cache(tokens):
        angles = torch.randn(tokens, head_dim // 2, device="cuda")
        return torch.cat([angles.cos(), angles.sin()], dim=-1).contiguous()

    img_cache = cache(img_tokens)
    txt_cache = cache(txt_tokens)

    img_reference = [tensor.clone() for tensor in img_qkv]
    txt_reference = [tensor.clone() for tensor in txt_qkv]
    img_reference[0], img_reference[1] = apply_qk_norm_with_optional_rope(
        img_reference[0],
        img_reference[1],
        norms[0],
        norms[1],
        head_dim,
        img_cache,
        is_neox=False,
    )
    txt_reference[0], txt_reference[1] = apply_qk_norm_with_optional_rope(
        txt_reference[0],
        txt_reference[1],
        norms[2],
        norms[3],
        head_dim,
        txt_cache,
        is_neox=False,
    )
    expected = tuple(
        torch.cat([txt_reference[index], img_reference[index]], dim=1)
        for index in range(3)
    )

    actual = try_fused_qwen_qkv_epilogue(
        *img_qkv,
        *txt_qkv,
        norms[0].weight,
        norms[1].weight,
        norms[2].weight,
        norms[3].weight,
        img_cache,
        txt_cache,
        1e-6,
        1e-6,
    )
    assert actual is not None
    assert all(
        torch.equal(result, reference) for result, reference in zip(actual, expected)
    )

    # ModelOpt FP8 produces one packed QKV GEMM output. Its chunked Q/K/V
    # tensors are zero-copy views with a 3x token stride; the epilogue must
    # consume those views directly rather than launching six contiguous copies.
    img_packed = torch.cat([tensor.flatten(2) for tensor in img_qkv], dim=-1)
    txt_packed = torch.cat([tensor.flatten(2) for tensor in txt_qkv], dim=-1)
    img_views = [
        tensor.unflatten(-1, (heads, head_dim))
        for tensor in img_packed.chunk(3, dim=-1)
    ]
    txt_views = [
        tensor.unflatten(-1, (heads, head_dim))
        for tensor in txt_packed.chunk(3, dim=-1)
    ]
    assert all(not tensor.is_contiguous() for tensor in (*img_views, *txt_views))

    # Unquantized image projections and packed text projections can use
    # different token strides; each family also works when both are packed.
    for img_inputs, txt_inputs in (
        (img_views, txt_views),
        (img_qkv, txt_views),
        (img_views, txt_qkv),
    ):
        packed_actual = try_fused_qwen_qkv_epilogue(
            *img_inputs,
            *txt_inputs,
            norms[0].weight,
            norms[1].weight,
            norms[2].weight,
            norms[3].weight,
            img_cache,
            txt_cache,
            1e-6,
            1e-6,
        )
        assert packed_actual is not None
        assert all(
            torch.equal(result, reference)
            for result, reference in zip(packed_actual, expected)
        )


def test_qwen_qkv_epilogue_rejects_compile():
    tensor = torch.empty(1, 1, 1, 128, device="cuda", dtype=torch.bfloat16)
    row = torch.empty(128, device="cuda", dtype=torch.bfloat16)
    cache = torch.empty(1, 128, device="cuda", dtype=torch.float32)
    with patch("torch.compiler.is_compiling", return_value=True):
        assert (
            try_fused_qwen_qkv_epilogue(
                tensor,
                tensor,
                tensor,
                tensor,
                tensor,
                tensor,
                row,
                row,
                row,
                row,
                cache,
                cache,
                1e-6,
                1e-6,
            )
            is None
        )


def test_qwen_qkv_epilogue_rejects_unsupported_head_dim():
    tensor = torch.empty(1, 1, 1, 64, device="cuda", dtype=torch.bfloat16)
    row = torch.empty(64, device="cuda", dtype=torch.bfloat16)
    cache = torch.empty(1, 64, device="cuda", dtype=torch.float32)
    assert (
        try_fused_qwen_qkv_epilogue(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            row,
            row,
            row,
            row,
            cache,
            cache,
            1e-6,
            1e-6,
        )
        is None
    )


@pytest.mark.parametrize("mode", ["dense", "attention_mask", "text_mask", "sharded"])
def test_qwen_attention_preserves_normalized_segments(mode):
    heads, head_dim, img_tokens, txt_tokens = 4, 128, 17, 7
    dim = heads * head_dim
    img = torch.randn(1, img_tokens, dim, device="cuda", dtype=torch.bfloat16)
    txt = torch.randn(1, txt_tokens, dim, device="cuda", dtype=torch.bfloat16)
    projections = [torch.randn_like(img) for _ in range(3)] + [
        torch.randn_like(txt) for _ in range(3)
    ]

    class IdentityProjection(torch.nn.Module):
        def forward(self, x):
            return x, None

    class CaptureAttention(torch.nn.Module):
        sp_attention_mode = "kv_gather"

        def forward(self, q, k, v, **kwargs):
            tensors = [q, k, v]
            if kwargs["q_prefix"] is not None:
                tensors = [
                    torch.cat([kwargs[name], tensor], dim=1)
                    for name, tensor in zip(
                        ("q_prefix", "k_prefix", "v_prefix"), tensors
                    )
                ]
            self.inputs = tuple(tensor.clone() for tensor in tensors)
            return tensors[0]

    module = object.__new__(qwen_image.QwenImageCrossAttention)
    torch.nn.Module.__init__(module)
    module._unquantized_added_qkv_is_packed = False
    module.local_num_heads = heads
    module.head_dim = head_dim
    module.qk_norm = True
    for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
        setattr(module, name, RMSNorm(head_dim).to(device="cuda", dtype=img.dtype))
    module.attn = CaptureAttention()
    module.to_out = torch.nn.ModuleList([IdentityProjection()])
    module.to_add_out = IdentityProjection()

    def cache(tokens):
        angle = torch.randn(tokens, head_dim // 2, device="cuda")
        return torch.cat([angle.cos(), angle.sin()], dim=-1)

    caches = (cache(img_tokens), cache(txt_tokens))
    kwargs = {}
    if mode == "attention_mask":
        kwargs["attn_mask"] = torch.ones(1, img_tokens + txt_tokens, device="cuda")
    elif mode == "text_mask":
        kwargs["encoder_hidden_states_mask"] = torch.ones(1, txt_tokens, device="cuda")
    elif mode == "sharded":
        kwargs["sp_text_sharded"] = True

    with patch.object(
        qwen_image,
        "_get_qkv_projections",
        side_effect=lambda *a, **kw: tuple(t.clone() for t in projections),
    ):
        module.use_fused_qkv_epilogue = False
        expected = module(img, txt, image_rotary_emb=caches, **kwargs)
        expected_inputs = module.attn.inputs
        module.use_fused_qkv_epilogue = True
        with patch.object(
            qwen_image,
            "try_fused_qwen_qkv_epilogue",
            wraps=try_fused_qwen_qkv_epilogue,
        ) as fused:
            actual = module(img, txt, image_rotary_emb=caches, **kwargs)
        assert fused.call_count == (1 if mode == "dense" else 0)
    assert all(torch.equal(a, b) for a, b in zip(actual[:2], expected[:2]))
    assert all(torch.equal(a, b) for a, b in zip(module.attn.inputs, expected_inputs))


@pytest.mark.parametrize("misaligned_image", [True, False])
def test_qwen_qkv_epilogue_rejects_misaligned_cache(misaligned_image):
    tensor = torch.empty(1, 1, 1, 128, device="cuda", dtype=torch.bfloat16)
    row = torch.empty(128, device="cuda", dtype=torch.bfloat16)
    cache = torch.empty(1, 128, device="cuda", dtype=torch.float32)
    misaligned = torch.empty(129, device="cuda", dtype=torch.float32)[1:].view(1, 128)
    assert misaligned.is_contiguous() and misaligned.data_ptr() % 8 != 0
    assert (
        try_fused_qwen_qkv_epilogue(
            *([tensor] * 6),
            *([row] * 4),
            misaligned if misaligned_image else cache,
            cache if misaligned_image else misaligned,
            1e-6,
            1e-6,
        )
        is None
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
