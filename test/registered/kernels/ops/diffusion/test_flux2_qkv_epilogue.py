import sys
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    try_fused_flux2_qkv_epilogue,
)
from sglang.multimodal_gen.runtime.layers.layernorm import (
    RMSNorm,
    apply_qk_norm_with_optional_rope,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

DEVICE = "cuda"
DTYPE = torch.bfloat16
HEAD_DIM = 128


def _packed_qkv(tokens: int, heads: int, generator: torch.Generator):
    source = [
        torch.randn(
            (1, tokens, heads, HEAD_DIM),
            dtype=DTYPE,
            device=DEVICE,
            generator=generator,
        )
        for _ in range(3)
    ]
    packed = torch.cat([tensor.flatten(2) for tensor in source], dim=-1)
    views = [
        tensor.unflatten(-1, (heads, HEAD_DIM)) for tensor in packed.chunk(3, dim=-1)
    ]
    assert all(not tensor.is_contiguous() for tensor in views)
    return views


@pytest.mark.parametrize("img_tokens,txt_tokens,heads", [(17, 7, 4), (256, 64, 8)])
def test_flux2_qkv_epilogue_is_bit_exact(
    img_tokens: int, txt_tokens: int, heads: int
) -> None:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(20260831 + img_tokens)
    img_qkv = _packed_qkv(img_tokens, heads, generator)
    txt_qkv = _packed_qkv(txt_tokens, heads, generator)
    norms = [
        RMSNorm(HEAD_DIM, eps=1e-6).to(device=DEVICE, dtype=DTYPE) for _ in range(4)
    ]
    for norm in norms:
        norm.weight.data.normal_(generator=generator)

    angles = torch.randn(
        (img_tokens + txt_tokens, HEAD_DIM // 2),
        device=DEVICE,
        generator=generator,
    )
    cache = torch.cat([angles.cos(), angles.sin()], dim=-1).contiguous()

    img_reference = [tensor.contiguous() for tensor in img_qkv]
    txt_reference = [tensor.contiguous() for tensor in txt_qkv]
    txt_reference[0], txt_reference[1] = apply_qk_norm_with_optional_rope(
        txt_reference[0],
        txt_reference[1],
        norms[2],
        norms[3],
        HEAD_DIM,
        cache,
        is_neox=False,
    )
    img_reference[0], img_reference[1] = apply_qk_norm_with_optional_rope(
        img_reference[0],
        img_reference[1],
        norms[0],
        norms[1],
        HEAD_DIM,
        cache,
        is_neox=False,
        position_offset=txt_tokens,
    )
    expected = tuple(
        torch.cat([txt_reference[index], img_reference[index]], dim=1)
        for index in range(3)
    )

    actual = try_fused_flux2_qkv_epilogue(
        *img_qkv,
        *txt_qkv,
        norms[0].weight,
        norms[1].weight,
        norms[2].weight,
        norms[3].weight,
        cache,
        1e-6,
        1e-6,
    )

    assert actual is not None
    assert all(
        torch.equal(result, reference)
        for result, reference in zip(actual, expected, strict=True)
    )


def test_flux2_qkv_epilogue_rejects_compile() -> None:
    tensor = torch.empty((1, 1, 1, HEAD_DIM), device=DEVICE, dtype=DTYPE)
    weight = torch.empty((HEAD_DIM,), device=DEVICE, dtype=DTYPE)
    cache = torch.empty((2, HEAD_DIM), device=DEVICE, dtype=torch.float32)
    with patch("torch.compiler.is_compiling", return_value=True):
        assert (
            try_fused_flux2_qkv_epilogue(
                tensor,
                tensor,
                tensor,
                tensor,
                tensor,
                tensor,
                weight,
                weight,
                weight,
                weight,
                cache,
                1e-6,
                1e-6,
            )
            is None
        )


def test_flux2_qkv_epilogue_rejects_cuda_graph_capture() -> None:
    tensor = torch.empty((1, 1, 1, HEAD_DIM), device=DEVICE, dtype=DTYPE)
    weight = torch.empty((HEAD_DIM,), device=DEVICE, dtype=DTYPE)
    cache = torch.empty((2, HEAD_DIM), device=DEVICE, dtype=torch.float32)
    with patch("torch.cuda.is_current_stream_capturing", return_value=True):
        assert (
            try_fused_flux2_qkv_epilogue(
                tensor,
                tensor,
                tensor,
                tensor,
                tensor,
                tensor,
                weight,
                weight,
                weight,
                weight,
                cache,
                1e-6,
                1e-6,
            )
            is None
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
