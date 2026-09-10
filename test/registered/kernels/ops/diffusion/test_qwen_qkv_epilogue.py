import sys
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.diffusion import try_fused_qwen_qkv_epilogue
from sglang.multimodal_gen.runtime.layers.layernorm import (
    RMSNorm,
    apply_qk_norm_with_optional_rope,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="Qwen-Image QKV epilogue requires SM100+",
)


@pytest.fixture(autouse=True)
def _seed_cuda():
    torch.cuda.manual_seed(0)


def test_qwen_qkv_epilogue_is_bit_exact():
    heads = 4
    head_dim = 128
    img_tokens = 17
    txt_tokens = 7
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

    packed_actual = try_fused_qwen_qkv_epilogue(
        *img_views,
        *txt_views,
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
