# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    marlin_make_workspace,
    marlin_permute_scales,
    should_use_atomic_add_reduce,
)
from sglang.srt.layers.quantization.utils import get_scalar_types
from sglang.srt.utils import is_cuda
from sglang.srt.utils.custom_op import register_custom_op

_is_cuda = is_cuda()
if _is_cuda:
    from sglang.kernels.ops.quantization.gptq_marlin import gptq_marlin_gemm
    from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack

ScalarType, scalar_types = get_scalar_types()

logger = logging.getLogger(__name__)


def fp8_fused_exponent_bias_into_scales(scales):
    fp8_exponent = 4
    if scales.dtype == torch.half:
        target_exponent = 5
    elif scales.dtype == torch.bfloat16:
        target_exponent = 8
    # exponent_bias_fp16 = 2 ** 4 - 2 ** 3 = 8
    # exponent_bias_bf16 = 2 ** 7 - 2 ** 3 = 120
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp8_exponent - 1)
    s = torch.ones_like(scales) * 2
    s = s**exponent_bias
    return scales * s


def fake_apply_fp8_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor],
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    out_shape = input.shape[:-1] + (size_n,)
    fake_output = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    return fake_output


@register_custom_op(fake_impl=fake_apply_fp8_marlin_linear)
def apply_fp8_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor],
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    # For GPUs that lack FP8 hardware support, we can leverage the
    # Marlin kernel for fast weight-only FP8 quantization

    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0), n=size_n, k=size_k, device=input.device, dtype=input.dtype
    )

    output = gptq_marlin_gemm(
        a=reshaped_x,
        c=None,
        b_q_weight=weight,
        b_scales=weight_scale,
        global_scale=None,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=workspace,
        b_q_type=scalar_types.float8_e4m3fn,
        size_m=reshaped_x.size(0),
        size_n=size_n,
        size_k=size_k,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
    )

    if bias is not None:
        output.add_(bias)

    return output.reshape(out_shape)


def prepare_fp8_layer_for_marlin(
    layer: torch.nn.Module, size_k_first: bool = True
) -> None:
    logger.warning_once(
        "Your GPU does not have native support for FP8 computation but "
        "FP8 quantization is being used. Weight-only FP8 compression will "
        "be used leveraging the Marlin kernel. This may degrade "
        "performance for compute-heavy workloads."
    )

    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    weight_block_size = getattr(layer, "weight_block_size", None)

    if size_k_first:
        assert layer.weight.shape == (part_size_k, part_size_n)
    else:
        assert layer.weight.shape == (part_size_n, part_size_k)

    device = layer.weight.device

    # WORKSPACE
    layer.workspace = marlin_make_workspace(device)

    if "weight_scale" in dir(layer):
        scales = layer.weight_scale.to(layer.orig_dtype)
    elif "weight_scale_inv" in dir(layer):
        scales = layer.weight_scale_inv.to(layer.orig_dtype)
        del layer.weight_scale_inv

    # CP decode attention TP swaps in a rank-local shard of this linear during
    # decode. A Marlin-packed tensor cannot be sliced along its logical axes, so
    # the shard has to be packed here, while the logical weight is still around.
    decode_shard = getattr(layer, "cp_decode_attn_tp_shard", None)
    if decode_shard is not None:
        layer.cp_decode_attn_tp_packed_views = _pack_fp8_marlin_decode_shard(
            layer, scales, decode_shard, size_k_first
        )

    # WEIGHT
    # Repack weights to marlin format
    marlin_qweight = _repack_fp8_weight_for_marlin(
        layer.weight, part_size_n, part_size_k, size_k_first
    )
    layer.weight = torch.nn.Parameter(marlin_qweight, requires_grad=False)

    # WEIGHT SCALES
    marlin_scales = _permute_fp8_scales_for_marlin(
        scales, part_size_n, part_size_k, weight_block_size, size_k_first
    )
    layer.weight_scale = torch.nn.Parameter(marlin_scales, requires_grad=False)

    # The dense FP8 Marlin wrapper adds bias after the kernel returns, so the
    # bias must remain in logical output-channel order. Only scales need the
    # Marlin tile permutation.
    if hasattr(layer, "bias") and layer.bias is not None:
        assert layer.bias.shape == (part_size_n,)
        layer.bias = torch.nn.Parameter(layer.bias.detach(), requires_grad=False)


def _repack_fp8_weight_for_marlin(
    weight: torch.Tensor, size_n: int, size_k: int, size_k_first: bool
) -> torch.Tensor:
    perm = torch.empty(0, dtype=torch.int, device=weight.device)
    qweight = pack_fp8_to_int32(weight, size_k_first)
    if not size_k_first:
        qweight = qweight.T.contiguous()

    return gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=size_k,
        size_n=size_n,
        num_bits=8,
    )


def _permute_fp8_scales_for_marlin(
    scales: torch.Tensor,
    size_n: int,
    size_k: int,
    weight_block_size: Optional[list[int]],
    size_k_first: bool,
) -> torch.Tensor:
    group_size = -1 if weight_block_size is None else weight_block_size[1]

    # marlin kernel only support channel-wise and group-wise quantization
    # we need to convert the scales
    if weight_block_size is None:
        if scales.nelement() == 1:
            # tensor-wise quantization -> channel-wise quantization
            # (1, 1) =>(repeat)=> (1, size_n)
            scales = scales.view(1, 1).repeat_interleave(size_n, 1)
        elif scales.nelement() > 1 and scales.nelement() != size_n:
            assert size_n % scales.nelement() == 0
            s_size = scales.nelement()
            # tensor-wise quantization (for gate-up proj)
            #     -> channel-wise quantization
            # (1, s_size) =>(repeat)=> (1, size_n)
            scales = scales.view(1, s_size)
            scales = scales.repeat_interleave(size_n // s_size, 1)
        else:
            # channel-wise quantization
            # (1, size_n)
            scales = scales.view(1, size_n)
    else:
        # block-wise quantization -> group-wise quantization
        # (size_k // block_size[1], ceil(size_n / block_size[0]))
        #  =>(repeat)=> (size_k // block_size[1], size_n)
        if not size_k_first:
            scales = scales.T.contiguous()
        block_n = weight_block_size[0]
        scales = scales.repeat_interleave(block_n, 1)
        # size_n may not divisible by block_size[0]
        scales = scales[:, :size_n]

    marlin_scales = marlin_permute_scales(
        s=scales, size_k=size_k, size_n=size_n, group_size=group_size
    )
    return fp8_fused_exponent_bias_into_scales(marlin_scales)


def _pack_fp8_marlin_decode_shard(
    layer: torch.nn.Module,
    scales: torch.Tensor,
    decode_shard,
    size_k_first: bool,
) -> dict[str, torch.Tensor]:
    """Marlin-pack the rank-local CP decode attention TP shard of ``layer``.

    ``decode_shard`` carries ``split`` ("output" for column-parallel, "input"
    for row-parallel), ``rank`` and ``size``. ``scales`` are the logical
    (unpermuted) scales. Returns the packed ``weight`` / ``weight_scale`` to
    swap in while decode attention TP is active.
    """
    split, rank, size = decode_shard.split, decode_shard.rank, decode_shard.size
    assert split in ("output", "input"), split
    assert size > 1 and 0 <= rank < size, (rank, size)
    assert getattr(layer, "bias", None) is None, (
        "CP decode attention TP does not shard linear biases"
    )

    size_n = layer.output_size_per_partition
    size_k = layer.input_size_per_partition
    n_dim, k_dim = (1, 0) if size_k_first else (0, 1)
    if split == "output":
        dim = n_dim
        assert size_n % size == 0, (size_n, size)
        size_n //= size
        chunk = size_n
    else:
        dim = k_dim
        assert size_k % size == 0, (size_k, size)
        size_k //= size
        chunk = size_k
    weight = layer.weight.narrow(dim, rank * chunk, chunk)

    weight_block_size = getattr(layer, "weight_block_size", None)
    if weight_block_size is not None:
        # Block scales share the weight's axis order, one entry per block.
        block = weight_block_size[0 if split == "output" else 1]
        assert chunk % block == 0, (
            "CP decode attention TP shard would split an FP8 quantization "
            f"block: split={split}, chunk={chunk}, block={block}"
        )
        assert scales.shape[dim] % size == 0, (tuple(scales.shape), dim, size)
        scale_chunk = scales.shape[dim] // size
        assert scale_chunk * block == chunk, (scale_chunk, block, chunk)
        scales = scales.narrow(dim, rank * scale_chunk, scale_chunk)
    elif split == "output" and scales.nelement() > 1:
        # Channel-wise (or per-logical-shard) scales follow output channels.
        # Per-tensor scales and input-dim sharding keep the scales unchanged.
        flat_scales = scales.reshape(-1)
        assert flat_scales.nelement() % size == 0, (flat_scales.nelement(), size)
        scale_chunk = flat_scales.nelement() // size
        scales = flat_scales.narrow(0, rank * scale_chunk, scale_chunk)

    return {
        "weight": _repack_fp8_weight_for_marlin(weight, size_n, size_k, size_k_first),
        "weight_scale": _permute_fp8_scales_for_marlin(
            scales.contiguous(), size_n, size_k, weight_block_size, size_k_first
        ),
    }


def pack_fp8_to_int32(
    fp8_tensor: torch.Tensor, size_k_first: bool = True
) -> torch.Tensor:
    """
    Repack FP8 weights to gptq format (packed int32 elements)
    """
    assert fp8_tensor.dtype == torch.float8_e4m3fn
    assert fp8_tensor.ndim == 2

    fp8_tensor = fp8_tensor.T if size_k_first else fp8_tensor
    fp8_tensor = fp8_tensor.contiguous()
    # fp8_tensor is contiguous and have shape (N, K) now
    # with `.view(torch.int32)`, it become (N, K // 4)
    int32_tensor = fp8_tensor.view(torch.int32)
    return int32_tensor.T.contiguous() if size_k_first else int32_tensor
