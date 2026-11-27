# type: ignore

"""Load-time expansion of Inkling-2's block-scaled expert weights.

Pre-Hopper GPUs (A100/SM80) have neither block-scaled MMA (SM90) nor tcgen05
(SM100), so the NVFP4 / MXFP8 routed expert kernels cannot run and the generic
BF16 grouped GEMM is the only option. These helpers decode the checkpoint
formats exactly once, while the weights are loaded.
"""

import torch

NVFP4_E2M1_TO_FLOAT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Element budget per slice, bounding the transient fp32 copies of a big expert.
DEQUANT_CHUNK_ELEMS = 1 << 22


def _rows(rows: int, cols: int):
    step = max(1, DEQUANT_CHUNK_ELEMS // max(cols, 1))
    for start in range(0, rows, step):
        yield slice(start, min(start + step, rows))


def dequantize_nvfp4_experts(weight, scale, amax, denominator, dtype):
    """Expand NVFP4 experts (E, N, K/2) uint8 to `dtype` (E, N, K).

    Every 16-element block shares one fp8_e4m3 `scale`; the per-expert outer
    scale is ``amax / denominator`` (the checkpoint's denominator is already
    ``448 * 6``, so it must not be rescaled again). Low nibbles are the even
    elements (see ``break_fp4_bytes``).
    """
    table = torch.tensor(NVFP4_E2M1_TO_FLOAT, device=weight.device)
    out = torch.empty(
        (weight.shape[0], weight.shape[1], weight.shape[2] * 2),
        dtype=dtype,
        device=weight.device,
    )
    global_scale = (amax.float() / denominator.float()).tolist()
    for expert in range(weight.shape[0]):
        for window in _rows(weight.shape[1], weight.shape[2]):
            packed = weight[expert, window]
            nibbles = torch.empty(
                (packed.shape[0], packed.shape[1] * 2),
                dtype=torch.int64,
                device=weight.device,
            )
            nibbles[:, 0::2] = packed & 0x0F
            nibbles[:, 1::2] = packed >> 4
            values = table[nibbles & 0x7]
            values = torch.where(nibbles & 0x8 != 0, -values, values)
            block_scale = scale[expert, window].float().repeat_interleave(16, dim=-1)
            out[expert, window] = (values * block_scale * global_scale[expert]).to(
                dtype
            )
    return out


def dequantize_mxfp8_experts(weight, scale, dtype):
    """Expand MXFP8 experts (E, N, K) fp8 + ue8m0 scales (E, N, K/32) to `dtype`."""
    out = torch.empty(weight.shape, dtype=dtype, device=weight.device)
    for expert in range(weight.shape[0]):
        for window in _rows(weight.shape[1], weight.shape[2]):
            factors = torch.exp2(scale[expert, window].float() - 127.0)
            out[expert, window] = (
                weight[expert, window].float() * factors.repeat_interleave(32, dim=-1)
            ).to(dtype)
    return out
