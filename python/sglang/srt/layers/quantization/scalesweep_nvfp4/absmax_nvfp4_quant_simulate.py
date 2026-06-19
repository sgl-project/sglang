# SPDX-License-Identifier: MIT
# Ported from https://github.com/efsotr/nvfp4quant_test (MIT, Copyright (c) 2026
# Li Lin) for sgl-project/sglang#27246. ScaleSweep MSE NVFP4 quantization.

import torch

try:
    from .scalesweep_mse_nvfp4_quant_simulate import (
        BLOCK_SIZE,
        create_fp4_output_tensors,
        round_up,
    )
except ImportError:
    from scalesweep_mse_nvfp4_quant_simulate import (
        BLOCK_SIZE,
        create_fp4_output_tensors,
        round_up,
    )


def _torch_round_to_fp4_code(x: torch.Tensor) -> torch.Tensor:
    ax = torch.abs(x)
    le2 = ax <= 2.0
    le4 = ax <= 4.0
    exp = torch.where(le2, 0.5, torch.where(le4, 1.0, 2.0))
    r = torch.floor(ax / exp + 0.5)
    mag = torch.where(le2, r, torch.where(le4, r + 2.0, torch.clamp(r + 4.0, max=7.0))).to(torch.uint8)
    sign = (x < 0.0).to(torch.uint8) << 3
    return mag | sign


def _swizzled_scale_indices(
    num_blocks: int,
    blocks_per_out: int,
    blocks_per_out_pad: int,
    device: torch.device,
) -> torch.Tensor:
    block_offsets = torch.arange(num_blocks, device=device, dtype=torch.int64)
    row = block_offsets // blocks_per_out
    col = block_offsets % blocks_per_out

    major_m = row >> 7
    row_in_tile = row & 127
    tile_m = row_in_tile >> 5
    inner_m = row_in_tile & 31

    major_k = col >> 2
    inner_k = col & 3

    return (
        major_m * (blocks_per_out_pad * 128)
        + major_k * 512
        + inner_m * 16
        + tile_m * 4
        + inner_k
    )


@torch.inference_mode()
def absmax_nvfp4_quant_simulate(
    input: torch.Tensor,
    global_scale_inv: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    backend: str = "absmax_simulate",
) -> tuple[torch.Tensor, torch.Tensor]:
    del backend

    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape

    assert n % BLOCK_SIZE == 0, (
        f"last dim has to be multiple of {BLOCK_SIZE}, but got {n}."
    )
    assert input.dtype in (torch.float16, torch.bfloat16), (
        f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."
    )
    assert global_scale_inv.dtype == torch.float32, (
        f"global_scale_inv.dtype needs to be fp32 but got {global_scale_inv.dtype}."
    )

    output, output_scale = create_fp4_output_tensors(
        m,
        n,
        input.device,
        is_sf_swizzled_layout,
    )

    blocks_per_out = n // BLOCK_SIZE
    blocks = input.contiguous().view(m, blocks_per_out, BLOCK_SIZE).to(torch.float32)
    blocks = blocks * global_scale_inv

    scale = (torch.amax(torch.abs(blocks), dim=-1) * (1.0 / 6.0)).to(torch.float8_e4m3fn)
    scale_f32 = scale.to(torch.float32)
    inv_scale = torch.where(scale_f32 == 0.0, 0.0, 1.0 / scale_f32)

    code = _torch_round_to_fp4_code(blocks * inv_scale.unsqueeze(-1))
    packed = code[:, :, 0::2] | (code[:, :, 1::2] << 4)
    output.copy_(packed.reshape(m, n // 2))

    if is_sf_swizzled_layout:
        indices = _swizzled_scale_indices(
            m * blocks_per_out,
            blocks_per_out,
            round_up(blocks_per_out, 4),
            input.device,
        )
        output_scale.view(-1)[indices] = scale.reshape(-1)
    else:
        output_scale.copy_(scale)

    return output, output_scale


scaled_fp4_quant_simulate = absmax_nvfp4_quant_simulate
