import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernel_api_logging import debug_kernel_api
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.srt.utils.custom_op import register_custom_op


# Adapted from https://github.com/ModelTC/LightX2V/blob/main/lightx2v/common/ops/norm/triton_ops.py#L905-L956
@triton.jit
def _rms_norm_tiled_onepass(
    y_ptr,
    x_ptr,
    w_ptr,
    SEQ: tl.constexpr,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    seq_blk_id = tl.program_id(0)
    seq_id = seq_blk_id * BLOCK_SIZE_SEQ

    seq_offset = seq_id + tl.arange(0, BLOCK_SIZE_SEQ)[:, None]
    s_mask = seq_offset < SEQ
    d_offset = tl.arange(0, BLOCK_SIZE_DIM)[None, :]
    d_mask = d_offset < DIM
    y_blk = y_ptr + seq_offset * DIM + d_offset
    x_blk = x_ptr + seq_offset * DIM + d_offset
    mask = s_mask & d_mask

    x = tl.load(x_blk, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=1, keep_dims=True) / DIM
    rstd = tl.math.rsqrt(mean_square + EPS)
    w = tl.load(w_ptr + d_offset, mask=d_mask)
    tl.store(y_blk, x * rstd * w, mask=mask)


@register_custom_op(op_name="triton_one_pass_rms_norm_cuda", out_shape="x")
def _triton_one_pass_rms_norm_cuda(
    x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    shape = x.shape
    x = x.contiguous()
    y = torch.empty_like(x)
    x_view = x.reshape(-1, shape[-1])
    y_view = y.reshape(-1, shape[-1])
    S, D = x_view.shape

    block_size_seq = min(16, triton.next_power_of_2(max(1, S // 512)))
    grid = (triton.cdiv(S, block_size_seq),)

    with torch.get_device_module().device(x.device):
        _rms_norm_tiled_onepass[grid](
            y_view,
            x_view,
            w,
            S,
            D,
            eps,
            BLOCK_SIZE_DIM=triton.next_power_of_2(D),
            BLOCK_SIZE_SEQ=block_size_seq,
        )
    return y


def triton_one_pass_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    return _triton_one_pass_rms_norm_cuda(x, w, eps)


if current_platform.is_mps():
    from .mps_fallback import triton_one_pass_rms_norm_native

    @debug_kernel_api
    def triton_one_pass_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
        return triton_one_pass_rms_norm_native(x, w, eps)
