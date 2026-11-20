"""MXFP4 knowledge that only SAIL needs.

``layers/quantization/mxfp4.py`` describes the checkpoint format and is shared
with the NVIDIA backends. The pieces that depend on SAIL's DeepGEMM build live
here instead: the dense FP4xFF4 linear method, the W4A16 mode guards and the
E8M0 scale preprocessing. The shared module imports them on the SAIL path only,
so neither the vendor entrypoint nor the platform guards have to appear in a
file every platform loads.

The W4A16 guards are shared with ``arg_groups/platform_hook``: the hook checks
the argument combination before the runtime flags exist, while
``Mxfp4MoEMethod`` reads both opt-ins and applies that same check plus the
hardware requirements, because the quant method can also be built on its own.
"""

from typing import List, Optional, Tuple

import torch
from torch.nn.parameter import Parameter

from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend
from sglang.srt.layers.parameter import GroupQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.base_config import LinearMethodBase
from sglang.srt.utils import get_device_sm, is_sail

_is_sail = is_sail()


def preprocess_sail_mxfp4_scales(scales: torch.Tensor) -> torch.Tensor:
    from deep_gemm import preprocess_mxfp4_scales

    return preprocess_mxfp4_scales(scales)


def _uses_deepgemm(backend) -> bool:
    """Accept either the resolved enum or the raw server argument."""
    if isinstance(backend, str):
        return backend == "deep_gemm"
    return backend is not None and backend.is_deep_gemm()


def _is_none_or_deepep(backend) -> bool:
    if isinstance(backend, str):
        return backend in ("none", "deepep")
    return backend is None or backend.is_none() or backend.is_deepep()


def validate_w4a16_modes(
    valu_enabled: bool,
    mma_enabled: bool,
    *,
    runner_backend=None,
    a2a_backend=None,
    check_hardware: bool = True,
) -> None:
    """Reject the W4A16 mode combinations SAIL cannot serve.

    ``runner_backend`` and ``a2a_backend`` default to the resolved flags. The
    argument hook passes the values it is still resolving, which the flags do
    not carry yet, together with ``check_hardware=False``: probing the device and
    the DeepGEMM build while the arguments are being parsed would fail before
    the hook could report the combination, so that half of the check stays with
    ``Mxfp4MoEMethod``.
    """
    if valu_enabled and mma_enabled:
        raise ValueError("SAIL MXFP4 W4A16 VALU and MMA modes are mutually exclusive.")
    if not (valu_enabled or mma_enabled):
        return
    if check_hardware and mma_enabled and get_device_sm() < 89:
        raise ValueError("SAIL MXFP4 W4A16 MMA requires SM89+.")
    if not _uses_deepgemm(
        get_moe_runner_backend() if runner_backend is None else runner_backend
    ):
        raise ValueError("SAIL MXFP4 W4A16 requires --moe-runner-backend deep_gemm.")
    if not _is_none_or_deepep(
        get_moe_a2a_backend() if a2a_backend is None else a2a_backend
    ):
        raise ValueError(
            "SAIL MXFP4 W4A16 only supports --moe-a2a-backend none or deepep."
        )
    if check_hardware and not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        raise RuntimeError(
            "SAIL MXFP4 W4A16 requires an available DeepGEMM; check the "
            "dependencies and SGLANG_ENABLE_JIT_DEEPGEMM."
        )


def resolved_w4a16_modes() -> Tuple[bool, bool]:
    """Read both W4A16 opt-ins and reject what SAIL cannot serve."""
    valu_enabled = envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16.get()
    mma_enabled = envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA.get()
    validate_w4a16_modes(valu_enabled, mma_enabled)
    return valu_enabled, mma_enabled


def deepgemm_moe_runner_backend_enabled() -> bool:
    """Only select SAIL FP4xFP4; the NVIDIA W4A8 backend is left unchanged."""
    moe_runner_backend = get_moe_runner_backend()
    sm_supported = get_device_sm() >= 89
    if moe_runner_backend.is_deep_gemm():
        if not (sm_supported and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM):
            raise RuntimeError(
                "SAIL MXFP4 MoE requires SM89+ and an available DeepGEMM."
            )
        return True
    if not (moe_runner_backend.is_auto() and sm_supported):
        return False

    # Which layers the DeepGEMM engines take -- the DeepEP-family rule and the
    # opt-in one -- is their own knowledge. Imported here rather than at module
    # scope so the argument hook, which imports this module to validate the
    # W4A16 flags, stays off the engine path.
    from sglang.srt.hardware_backend.sail.moe.deepgemm_moe import (
        deepgemm_moe_selected_for_a2a,
        deepgemm_moe_selected_for_auto,
    )

    return deepgemm_moe_selected_for_a2a() or deepgemm_moe_selected_for_auto()


class Mxfp4LinearMethod(LinearMethodBase):
    def __init__(
        self,
        prefix: str,
    ):
        if not (
            _is_sail and get_device_sm() >= 89 and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
        ):
            raise RuntimeError(
                "SAIL MXFP4 dense requires SM89+ and an available DeepGEMM."
            )
        self.prefix = prefix

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_loader = extra_weight_attrs.get("weight_loader")
        self.logical_widths = output_partition_sizes
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8
        mxfp4_block = 32
        extra_weight_attrs.update({"quant_method": "group"})

        if input_size_per_partition % mxfp4_block != 0:
            raise ValueError(
                f"Weight input_size_per_partition = {input_size_per_partition} is not divisible by {mxfp4_block}."
            )

        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // mxfp4_block,
                dtype=scale_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer):
        layer.weight = Parameter(layer.weight.t(), requires_grad=False)

        weight_scale = layer.weight_scale
        processed_scale = preprocess_sail_mxfp4_scales(weight_scale)
        layer.weight_scale = Parameter(processed_scale, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        from sglang.kernels.ops.quantization.sail_mxfp4 import downcast_to_mxfp4
        from sglang.srt.hardware_backend.sail.deep_gemm.sail_fused_deepgemm_ops import (
            gemm_nt_f4f4bf16,
        )

        output_shape = (*x.shape[:-1], layer.weight.shape[1])
        x_flat = x.reshape(-1, x.shape[-1])
        if x_flat.shape[0] == 0:
            return torch.empty(output_shape, dtype=x.dtype, device=x.device)
        if bias is not None:
            bias = bias.to(torch.float32)
        x_q, x_scale = downcast_to_mxfp4(x_flat, axis=1)
        out = torch.empty(
            (x_q.shape[0], layer.weight.shape[1]),
            dtype=torch.bfloat16,
            device=x_q.device,
        )
        gemm_nt_f4f4bf16(x_q, x_scale, layer.weight.t(), layer.weight_scale, bias, out)
        return out.to(x.dtype).reshape(output_shape)
