from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple, Optional

import torch

from sglang.kernels.ops.gemm.rdna4_nvfp4 import (
    is_rdna4_nvfp4_device,
    rdna4_nvfp4_linear,
    try_warmup_rdna4_nvfp4,
)

PETIT_NVFP4_BACKEND = "petit"
RDNA4_NVFP4_BACKEND = "rdna4_jit"


class _PetitOps(NamedTuple):
    mul_nvfp4_a16: object
    process_nvfp4_scales: object
    repack_nvfp4: object


@lru_cache(maxsize=1)
def _load_petit_ops() -> _PetitOps:
    try:
        from petit_kernel import (
            mul_nvfp4_a16,
            process_nvfp4_scales,
            repack_nvfp4,
        )
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "The Petit NVFP4 backend is unavailable or ABI-incompatible. "
            "Install a petit-kernel build matching the active ROCm runtime. "
            f"Original import error: {error}"
        ) from error
    return _PetitOps(mul_nvfp4_a16, process_nvfp4_scales, repack_nvfp4)


def _check_petit_nvfp4_supported(
    quant_method: str, group_size: Optional[int]
) -> tuple[bool, Optional[str]]:
    if quant_method != "NVFP4":
        return (
            False,
            "The dense ROCm NVFP4 path only supports quant_algo=NVFP4. "
            "Check the model hf_quant_config.json file.",
        )
    if group_size is not None and group_size != 16:
        return (
            False,
            "The dense ROCm NVFP4 path only supports group_size=16.",
        )
    return (True, None)


def verify_petit_nvfp4_supported(quant_method: str, group_size: Optional[int]) -> None:
    supported, error_msg = _check_petit_nvfp4_supported(quant_method, group_size)
    if not supported:
        raise ValueError(error_msg)


def select_nvfp4_backend(device: torch.device | int | None = None) -> str:
    if not is_rdna4_nvfp4_device(device):
        return PETIT_NVFP4_BACKEND
    return RDNA4_NVFP4_BACKEND


def prepare_nvfp4_layer_for_petit(layer: torch.nn.Module) -> None:
    backend = select_nvfp4_backend(layer.weight.device)
    layer.nvfp4_backend = backend

    if backend == PETIT_NVFP4_BACKEND:
        petit_ops = _load_petit_ops()
        part_size_n = layer.output_size_per_partition
        part_size_k = layer.input_size_per_partition
        qweight = layer.weight.view(torch.int32).contiguous()
        petit_qweight = petit_ops.repack_nvfp4(
            qweight, size_n=part_size_n, size_k=part_size_k
        )
        layer.weight = torch.nn.Parameter(petit_qweight, requires_grad=False)

        weight_scale = petit_ops.process_nvfp4_scales(
            scales=layer.weight_scale, size_k=part_size_k, size_n=part_size_n
        )
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        return

    if backend != RDNA4_NVFP4_BACKEND:
        raise ValueError(f"Unknown NVFP4 backend: {backend}")

    # The RDNA4 JIT consumes the canonical checkpoint layout directly.
    # Keep this decision on the layer so apply does not query the device.
    layer.weight = torch.nn.Parameter(
        layer.weight.data.contiguous(), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        layer.weight_scale.data.contiguous(), requires_grad=False
    )
    # Force the HIP and Triton compiles now: loading a layer is the last point
    # before CUDA graph capture where a plain launch is still safe.
    try_warmup_rdna4_nvfp4(
        getattr(layer, "params_dtype", None) or torch.get_default_dtype(),
        layer.weight.device,
        layer.output_size_per_partition,
        layer.input_size_per_partition,
    )


def apply_petit_nvfp4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor] = None,
    backend: str = PETIT_NVFP4_BACKEND,
) -> torch.Tensor:
    if backend == PETIT_NVFP4_BACKEND:
        reshaped_x = input.reshape(-1, input.shape[-1])
        out_shape = input.shape[:-1] + (size_n,)
        petit_ops = _load_petit_ops()
        output = petit_ops.mul_nvfp4_a16(
            a=reshaped_x,
            b=weight,
            s=weight_scale,
            global_scale=weight_scale_2,
            size_m=reshaped_x.size(0),
            size_n=size_n,
            size_k=size_k,
            solution_id=-1,
        )
        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)

    if backend != RDNA4_NVFP4_BACKEND:
        raise ValueError(f"Unknown NVFP4 backend: {backend}")

    output = rdna4_nvfp4_linear(
        input=input,
        weight=weight,
        weight_scale=weight_scale,
        weight_global_scale=weight_scale_2,
    )
    if bias is not None:
        output.add_(bias)
    return output
