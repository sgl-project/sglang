# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py

"""NVFP4 Marlin fallback: run FP4-quantized models on non-Blackwell GPUs via Marlin kernel."""

import logging
from typing import Optional

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    marlin_make_workspace,
    marlin_permute_bias,
    marlin_permute_scales,
    should_use_atomic_add_reduce,
)
from sglang.srt.layers.quantization.utils import get_scalar_types
from sglang.srt.utils import direct_register_custom_op, get_device_capability, is_cuda

_is_cuda = is_cuda()
if _is_cuda:
    from sglang.jit_kernel.gptq_marlin import gptq_marlin_gemm
    from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack

ScalarType, scalar_types = get_scalar_types()

logger = logging.getLogger(__name__)

# NVFP4 always uses group_size=16
FP4_MARLIN_GROUP_SIZE = 16


def is_fp4_marlin_supported() -> bool:
    """Check if the current GPU supports FP4 Marlin fallback (CUDA SM >= 75)."""
    if not _is_cuda:
        return False
    if torch.version.hip is not None:
        return False
    major, minor = get_device_capability()
    if major is None or minor is None:
        return False
    return (major * 10 + minor) >= 75


def should_use_fp4_marlin_fallback() -> bool:
    """True if non-Blackwell (or forced) AND Marlin kernel available (SM >= 75)."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.quantization.fp8_utils import is_blackwell_supported

    force = envs.SGLANG_FORCE_NVFP4_MARLIN.get()
    return (force or not is_blackwell_supported()) and is_fp4_marlin_supported()


def nvfp4_marlin_process_scales(marlin_scales: torch.Tensor) -> torch.Tensor:
    """Convert NVFP4 scales from FP8-S1E4M3 to FP8-S0E5M3 format for Marlin.

    The int16 <<1 may wrap for large scales (e.g. 448*128=57344), but the BIT
    PATTERN is preserved correctly — the kernel reads raw bytes, not int16 values.
    """
    marlin_scales = marlin_scales.to(torch.half)

    if not (marlin_scales >= 0).all():
        logger.warning_once(
            "NVFP4 Marlin assumes scales >= 0, but encountered negative scales. "
            "Accuracy may be degraded. The scales are converted from FP8-S1E4M3 "
            "to a special FP8-S0E5M3 format to speed up dequantization."
        )

    marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
        marlin_scales.size(0), -1
    )

    marlin_scales = (marlin_scales * (2**7)).view(torch.int16) << 1
    marlin_scales = marlin_scales.view(torch.float8_e4m3fn)
    marlin_scales = marlin_scales[:, 1::2].contiguous()

    return marlin_scales


def nvfp4_marlin_process_global_scale(global_scale: torch.Tensor) -> torch.Tensor:
    """Pre-adjust global scale with FP4/FP16/BF16 exponent bias for Marlin kernel."""
    assert global_scale.dtype in [
        torch.half,
        torch.bfloat16,
    ], f"global_scale dtype must be half or bfloat16, got {global_scale.dtype}"
    fp4_exponent = 2
    if global_scale.dtype == torch.half:
        target_exponent = 5
    elif global_scale.dtype == torch.bfloat16:
        target_exponent = 8
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    return global_scale * (2.0 ** (exponent_bias - 7))


def apply_fp4_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: Optional[torch.Tensor],
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor] = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    """Apply FP4-quantized linear via Marlin kernel (non-Blackwell fallback)."""
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=size_n,
        k=size_k,
        device=input.device,
        dtype=input.dtype,
    )

    output = gptq_marlin_gemm(
        a=reshaped_x,
        c=None,
        b_q_weight=weight,
        b_scales=weight_scale,
        global_scale=weight_global_scale.reshape(-1),
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=workspace,
        b_q_type=scalar_types.float4_e2m1f,
        size_m=reshaped_x.size(0),
        size_n=size_n,
        size_k=size_k,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
    )

    if bias is not None:
        output.add_(bias)

    return output.reshape(out_shape)


def fake_apply_fp4_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: Optional[torch.Tensor],
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor] = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    out_shape = input.shape[:-1] + (size_n,)
    return torch.empty(out_shape, dtype=input.dtype, device=input.device)


direct_register_custom_op(
    op_name="apply_fp4_marlin_linear",
    op_func=apply_fp4_marlin_linear,
    mutates_args=[],
    fake_impl=fake_apply_fp4_marlin_linear,
)


def prepare_fp4_layer_for_marlin(
    layer: torch.nn.Module,
    weight_attr: str = "weight",
    weight_scale_attr: str = "weight_scale",
    weight_global_scale_attr: str = "weight_global_scale",
) -> None:
    """Repack NVFP4 linear layer weights into Marlin format in-place."""
    logger.warning_once(
        "Your GPU does not have native support for FP4 computation but "
        "FP4 quantization is being used. Weight-only FP4 compression will "
        "be used leveraging the Marlin kernel. This may degrade "
        "performance for compute-heavy workloads."
    )

    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    param_dtype = layer.params_dtype

    weight = getattr(layer, weight_attr)
    assert weight.shape == (part_size_n, part_size_k // 2), (
        f"Expected {weight_attr} shape ({part_size_n}, {part_size_k // 2}), "
        f"got {weight.shape}"
    )

    device = weight.device

    # WORKSPACE
    layer.marlin_workspace = marlin_make_workspace(device)

    # WEIGHT: repack from NVFP4 native layout to Marlin tile layout
    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = weight.data.view(torch.int32).T.contiguous()
    del weight
    marlin_qweight = gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=part_size_k,
        size_n=part_size_n,
        num_bits=4,
    )
    del qweight
    setattr(layer, weight_attr, torch.nn.Parameter(marlin_qweight, requires_grad=False))

    # WEIGHT SCALES: transpose, permute, convert to FP8-S0E5M3
    weight_scale = getattr(layer, weight_scale_attr)
    weight_scale = weight_scale.data.T.contiguous().to(param_dtype)
    weight_scale = marlin_permute_scales(
        s=weight_scale,
        size_k=part_size_k,
        size_n=part_size_n,
        group_size=FP4_MARLIN_GROUP_SIZE,
    )
    weight_scale = nvfp4_marlin_process_scales(weight_scale)
    setattr(
        layer, weight_scale_attr, torch.nn.Parameter(weight_scale, requires_grad=False)
    )

    # GLOBAL SCALE: Pre-adjust exponent bias for Marlin kernel.
    weight_global_scale = getattr(layer, weight_global_scale_attr)
    weight_global_scale = weight_global_scale.to(param_dtype)
    weight_global_scale = nvfp4_marlin_process_global_scale(weight_global_scale)
    setattr(
        layer,
        weight_global_scale_attr,
        torch.nn.Parameter(weight_global_scale, requires_grad=False),
    )

    # BIAS (if present): Permute for Marlin's fast access pattern
    if hasattr(layer, "bias") and layer.bias is not None:
        assert layer.bias.shape == (part_size_n,)
        bias = marlin_permute_bias(layer.bias)
        layer.bias = torch.nn.Parameter(bias, requires_grad=False)


def prepare_moe_fp4_layer_for_marlin(layer: torch.nn.Module) -> None:
    """Repack NVFP4 MoE weights into Marlin format in-place (per-expert)."""
    logger.warning_once(
        "Your GPU does not have native support for FP4 computation but "
        "FP4 quantization is being used. Weight-only FP4 compression will "
        "be used leveraging the Marlin kernel for MoE layers. This may "
        "degrade performance for compute-heavy workloads."
    )

    e = layer.num_local_experts
    k = layer.w13_weight.shape[2] * 2  # hidden_size (packed: K//2 per uint8)
    n = layer.intermediate_size_per_partition
    param_dtype = layer.params_dtype
    num_shards = 2 if layer.moe_runner_config.is_gated else 1

    device = layer.w13_weight.device
    perm = torch.empty(0, dtype=torch.int, device=device)

    # (size_n, size_k) for each projection
    sizes = {"w13": (n * num_shards, k), "w2": (k, n)}

    # --- WEIGHT REPACKING ---
    for name in ["w13_weight", "w2_weight"]:
        prefix = name.split("_")[0]  # "w13" or "w2"
        size_n, size_k = sizes[prefix]
        weight = getattr(layer, name)

        assert weight.shape == (e, size_n, size_k // 2), (
            f"Expected {name} shape ({e}, {size_n}, {size_k // 2}), "
            f"got {weight.shape}"
        )

        repacked = []
        for i in range(e):
            qweight = weight.data[i].view(torch.int32).T.contiguous()
            repacked.append(
                gptq_marlin_repack(
                    b_q_weight=qweight,
                    perm=perm,
                    size_k=size_k,
                    size_n=size_n,
                    num_bits=4,
                )
            )

        del weight
        setattr(
            layer, name, torch.nn.Parameter(torch.stack(repacked), requires_grad=False)
        )

    # --- WEIGHT SCALE PROCESSING ---
    for prefix in ["w13", "w2"]:
        size_n, size_k = sizes[prefix]
        scales = getattr(layer, prefix + "_weight_scale").to(param_dtype)
        global_scale = getattr(layer, prefix + "_weight_scale_2").to(param_dtype)

        processed = []
        for i in range(e):
            s = marlin_permute_scales(
                s=scales.data[i].T,
                size_k=size_k,
                size_n=size_n,
                group_size=FP4_MARLIN_GROUP_SIZE,
            )
            processed.append(nvfp4_marlin_process_scales(s))

        del scales
        setattr(
            layer,
            prefix + "_weight_scale",
            torch.nn.Parameter(torch.stack(processed), requires_grad=False),
        )

        if global_scale.dim() > 1:
            global_scale = global_scale.max(dim=-1).values
        global_scale = nvfp4_marlin_process_global_scale(global_scale)
        setattr(
            layer,
            prefix + "_weight_scale_2",
            torch.nn.Parameter(global_scale, requires_grad=False),
        )
