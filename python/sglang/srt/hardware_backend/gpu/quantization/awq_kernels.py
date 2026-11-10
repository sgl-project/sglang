from __future__ import annotations

import logging
import os
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe import MoeRunner
from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.layers.quantization.marlin_utils import (
    apply_awq_marlin_linear,
    awq_to_marlin_zero_points,
    marlin_make_empty_g_idx,
    marlin_make_workspace,
    marlin_moe_permute_scales,
    marlin_permute_scales,
    moe_awq_to_marlin_zero_points,
    moe_awq_to_sm70_marlin_zero_points_float,
    sm70_marlin_moe_logical_scales,
)
from sglang.srt.layers.quantization.utils import get_scalar_types, replace_parameter
from sglang.srt.utils import is_hip, is_xpu
from sglang.kernels.sm70_paths import sm70_prebuilt

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

awq_marlin_moe_repack = None
awq_marlin_repack = None


def _unsupported_awq_dequantize(*args, **kwargs):
    raise RuntimeError("AWQ GPU kernels are unavailable on the current platform.")


awq_dequantize = _unsupported_awq_dequantize

logger = logging.getLogger(__name__)

_SM70_AWQ_PREFILL_DENSE_M = 4096
_SM70_AWQ_PREFILL_DENSE_SHAPES = {
    "gate_up_proj": (5120, 8704),
    "down_proj": (4352, 5120),
    "in_proj_qkvz": (5120, 4096),
    "out_proj": (1536, 5120),
    "o_proj": (1536, 5120),
}
_SM70_AWQ_PREFILL_DENSE_WORKSPACE_ELEMENTS = max(
    k * n for k, n in _SM70_AWQ_PREFILL_DENSE_SHAPES.values()
)
_SM70_AWQ_PREFILL_DENSE_WORKSPACES: weakref.WeakValueDictionary[
    tuple[int, torch.dtype], torch.Tensor
] = weakref.WeakValueDictionary()
_SM70_TURBOMIND_OPS_LOAD_ATTEMPTED = False
_SM70_TURBOMIND_OPS_AVAILABLE = False
_SM70_AWQ_PREFILL_DENSE_OOM_WARNED = False
_SM70_AWQ_PREFILL_DENSE_LOGGED = False
_SM70_AWQ_TURBOMIND_LOGGED = False
_SM70_TURBOMIND_OPS_PATH = sm70_prebuilt("_sm70_turbomind_v100.so")


def _env_flag(name: str, default: str) -> bool:
    value = os.environ.get(name, default).strip().lower()
    if value not in (
        "0",
        "false",
        "off",
        "no",
        "1",
        "true",
        "on",
        "yes",
    ):
        raise ValueError(f"{name} must be a boolean value, got {value!r}.")
    return value in ("1", "true", "on", "yes")


def _sm70_awq_prefill_exact_dense_enabled() -> bool:
    return _env_flag("SGLANG_SM70_AWQ_PREFILL_EXACT_DENSE", "1")


def _sm70_turbomind_awq_enabled() -> bool:
    return _env_flag("SGLANG_SM70_AWQ_TURBOMIND", "1")


def _load_sm70_turbomind_awq_ops() -> bool:
    global _SM70_TURBOMIND_OPS_AVAILABLE, _SM70_TURBOMIND_OPS_LOAD_ATTEMPTED
    if _SM70_TURBOMIND_OPS_LOAD_ATTEMPTED:
        return _SM70_TURBOMIND_OPS_AVAILABLE
    _SM70_TURBOMIND_OPS_LOAD_ATTEMPTED = True
    path = Path(
        os.environ.get("SGLANG_SM70_TURBOMIND_OPS_PATH", _SM70_TURBOMIND_OPS_PATH)
    )
    try:
        torch.ops.load_library(str(path))
        ops = torch.ops.sglang_sm70_turbomind
        _SM70_TURBOMIND_OPS_AVAILABLE = all(
            hasattr(ops, name)
            for name in ("awq_prepare", "awq_gemm_out", "awq_dequantize_out")
        )
    except Exception as exc:
        logger.warning("SM70 TurboMind AWQ operators unavailable: %s", exc)
        _SM70_TURBOMIND_OPS_AVAILABLE = False
    return _SM70_TURBOMIND_OPS_AVAILABLE


def can_use_sm70_turbomind_awq() -> bool:
    return (
        _sm70_turbomind_awq_enabled()
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (7, 0)
        and _load_sm70_turbomind_awq_ops()
    )


def _is_sm70_awq_prefill_exact_dense_layer(layer: torch.nn.Module) -> bool:
    if getattr(layer, "tp_size", 1) != 4:
        return False
    suffix = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
    expected = _SM70_AWQ_PREFILL_DENSE_SHAPES.get(suffix)
    if expected is None:
        return False
    return (layer.qweight.shape[0], layer.qweight.shape[1] * 8) == expected


def _get_sm70_awq_prefill_dense_workspace(weight: torch.Tensor):
    global _SM70_AWQ_PREFILL_DENSE_OOM_WARNED
    device_index = weight.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (device_index, torch.float16)
    workspace = _SM70_AWQ_PREFILL_DENSE_WORKSPACES.get(key)
    if workspace is None:
        try:
            workspace = torch.empty(
                _SM70_AWQ_PREFILL_DENSE_WORKSPACE_ELEMENTS,
                dtype=torch.float16,
                device=weight.device,
            )
        except torch.OutOfMemoryError:
            if not _SM70_AWQ_PREFILL_DENSE_OOM_WARNED:
                logger.warning(
                    "Insufficient memory for the bounded SM70 AWQ prefill "
                    "workspace; retaining compact TurboMind AWQ."
                )
                _SM70_AWQ_PREFILL_DENSE_OOM_WARNED = True
            return None
        _SM70_AWQ_PREFILL_DENSE_WORKSPACES[key] = workspace
    return workspace

if is_xpu():
    try:
        from sgl_kernel import awq_dequantize
    except ImportError:
        pass
elif is_hip():
    try:
        from sglang.kernels.ops.quantization.awq_triton import (
            awq_dequantize_triton as awq_dequantize,
        )
    except ImportError:
        pass
else:
    try:
        from sglang.kernels.ops.quantization.awq_dequantize import awq_dequantize
        from sglang.kernels.ops.quantization.awq_marlin_repack import (
            awq_marlin_moe_repack,
            awq_marlin_repack,
        )
        from sglang.srt.utils.custom_op import register_custom_op_from_extern

        # On SM70 (V100) the stock JIT awq_marlin_repack is an __CUDA_ARCH__<800
        # stub that writes nothing -> expert weights would be repacked to zeros.
        # Prefer marlin_v100's real SM70 repack when available.
        try:
            from sglang.srt.layers.quantization.marlin_utils import (
                _sm70_marlin_v100_repack_ops,
            )

            _, _sm70_awq_repack = _sm70_marlin_v100_repack_ops()
            if _sm70_awq_repack is not None:
                awq_marlin_repack = _sm70_awq_repack

                def _sm70_awq_moe_repack(
                    b_q_weight, _perm, size_k, size_n, num_bits
                ):
                    # marlin_v100 exposes the real SM70 dense repack op.  Its
                    # MoE helper is a host-side loop over that same op; mirror
                    # it here because SGLang's JIT MoE helper closes over the
                    # stock repack function, whose CUDA kernel is a no-op on
                    # SM70.
                    output = torch.empty(
                        (
                            b_q_weight.shape[0],
                            size_k // 16,
                            size_n * (num_bits // 2),
                        ),
                        dtype=b_q_weight.dtype,
                        device=b_q_weight.device,
                    )
                    for expert_id in range(b_q_weight.shape[0]):
                        output[expert_id] = _sm70_awq_repack(
                            b_q_weight[expert_id],
                            size_k,
                            size_n,
                            num_bits,
                        )
                    return output

                awq_marlin_moe_repack = _sm70_awq_moe_repack
        except Exception:
            pass

        awq_dequantize = register_custom_op_from_extern(
            awq_dequantize,
            fake_impl=lambda qweight, scales, qzeros: qweight.new_empty(
                qweight.shape[:-1] + (qweight.shape[-1] * 8,), dtype=scales.dtype
            ),
        )
    except ImportError:
        try:
            from sglang.kernels.ops.quantization.awq_triton import (
                awq_dequantize_triton as awq_dequantize,
            )
        except ImportError:
            pass

_, scalar_types = get_scalar_types()


class AWQLinearKernel:
    def __init__(self, quant_config: Optional[QuantizationConfig] = None):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_awq_sm70_prepared", False):
            return

        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

        if not _sm70_turbomind_awq_enabled() or not layer.qweight.is_cuda:
            return
        if torch.cuda.get_device_capability(layer.qweight.device) != (7, 0):
            return

        group_size = self.quant_config.group_size
        if group_size == -1:
            group_size = layer.qweight.shape[0]
        if group_size not in (32, 64, 128):
            raise RuntimeError(
                "SM70 TurboMind AWQ supports group_size 32/64/128, "
                f"but got {group_size}."
            )
        if not _load_sm70_turbomind_awq_ops():
            raise RuntimeError(
                "SGLANG_SM70_AWQ_TURBOMIND=1 requires the v1.3.0 "
                "SM70 TurboMind extension."
            )

        use_exact_dense = (
            _sm70_awq_prefill_exact_dense_enabled()
            and group_size == 128
            and _is_sm70_awq_prefill_exact_dense_layer(layer)
        )
        workspace = (
            _get_sm70_awq_prefill_dense_workspace(layer.qweight)
            if use_exact_dense
            else None
        )
        output_size = layer.qweight.shape[1] * self.quant_config.pack_factor
        tm_weight, tm_scales, meta = (
            torch.ops.sglang_sm70_turbomind.awq_prepare(
                layer.qweight,
                layer.scales,
                layer.qzeros,
                group_size,
                False,
            )
        )
        layer._awq_sm70_weight = tm_weight
        layer._awq_sm70_scales = tm_scales
        layer._awq_sm70_k_ld = int(meta[0].item())
        layer._awq_sm70_q_ld = int(meta[1].item())
        layer._awq_sm70_group_size = group_size
        layer._awq_sm70_input_size = layer.qweight.shape[0]
        layer._awq_sm70_output_size = output_size
        if workspace is not None:
            layer._awq_sm70_prefill_dense_workspace = workspace
        layer._awq_sm70_prepared = True

        # The packed TurboMind copy services decode and partial chunks too, so
        # the original AWQ tensors can be released without resident duplication.
        layer.qweight = torch.nn.Parameter(
            torch.empty(0, dtype=torch.int32, device=tm_weight.device),
            requires_grad=False,
        )
        layer.qzeros = torch.nn.Parameter(
            torch.empty(0, dtype=torch.int32, device=tm_weight.device),
            requires_grad=False,
        )
        layer.scales = torch.nn.Parameter(
            torch.empty(0, dtype=tm_scales.dtype, device=tm_weight.device),
            requires_grad=False,
        )
        global _SM70_AWQ_PREFILL_DENSE_LOGGED, _SM70_AWQ_TURBOMIND_LOGGED
        if workspace is not None and not _SM70_AWQ_PREFILL_DENSE_LOGGED:
            logger.info(
                "SM70 AWQ: enabled bounded 85 MiB exact-dense M=4096 path."
            )
            _SM70_AWQ_PREFILL_DENSE_LOGGED = True
        if not _SM70_AWQ_TURBOMIND_LOGGED:
            logger.info("SM70 AWQ: enabled compact TurboMind dense path.")
            _SM70_AWQ_TURBOMIND_LOGGED = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if getattr(layer, "_awq_sm70_prepared", False):
            reshaped_x = x.reshape(-1, x.shape[-1])
            out_shape = x.shape[:-1] + (layer._awq_sm70_output_size,)
            if (
                reshaped_x.shape[0] == _SM70_AWQ_PREFILL_DENSE_M
                and reshaped_x.dtype == torch.float16
                and hasattr(layer, "_awq_sm70_prefill_dense_workspace")
            ):
                k = layer._awq_sm70_input_size
                n = layer._awq_sm70_output_size
                dense_weight = layer._awq_sm70_prefill_dense_workspace[: k * n].view(
                    k, n
                )
                torch.ops.sglang_sm70_turbomind.awq_dequantize_out(
                    dense_weight,
                    layer._awq_sm70_weight,
                    layer._awq_sm70_scales,
                    layer._awq_sm70_group_size,
                )
                out = torch.mm(reshaped_x, dense_weight)
            else:
                out = torch.empty(
                    (reshaped_x.shape[0], layer._awq_sm70_output_size),
                    dtype=x.dtype,
                    device=x.device,
                )
                torch.ops.sglang_sm70_turbomind.awq_gemm_out(
                    out,
                    reshaped_x,
                    layer._awq_sm70_weight,
                    layer._awq_sm70_scales,
                    layer._awq_sm70_group_size,
                    layer._awq_sm70_k_ld,
                    layer._awq_sm70_q_ld,
                    False,
                )
            if bias is not None:
                out.add_(bias)
            return out.reshape(out_shape)

        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = awq_dequantize(qweight, scales, qzeros)
        out = torch.matmul(reshaped_x, out)

        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)


class AWQMarlinLinearKernel:
    def __init__(self, quant_config: Optional[QuantizationConfig] = None):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.qweight.device
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

        layer.workspace = marlin_make_workspace(device)

        marlin_qweight = awq_marlin_repack(
            layer.qweight,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            num_bits=self.quant_config.quant_type.size_bits,
        )
        replace_parameter(layer, "qweight", marlin_qweight)

        marlin_scales = marlin_permute_scales(
            layer.scales,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            group_size=self.quant_config.group_size,
        )
        replace_parameter(layer, "scales", marlin_scales)

        marlin_zp = awq_to_marlin_zero_points(
            layer.qzeros,
            size_k=layer.num_groups,
            size_n=layer.output_size_per_partition,
            num_bits=self.quant_config.quant_type.size_bits,
        )
        replace_parameter(layer, "qzeros", marlin_zp)

        layer.g_idx = marlin_make_empty_g_idx(device)
        layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return apply_awq_marlin_linear(
            input=x,
            weight=layer.qweight,
            weight_scale=layer.scales,
            weight_zp=layer.qzeros,
            g_idx=layer.g_idx,
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=layer.workspace,
            quant_type=self.quant_config.quant_type,
            output_size_per_partition=layer.output_size_per_partition,
            input_size_per_partition=layer.input_size_per_partition,
            bias=bias,
        )


class AWQMoEKernel:
    def __init__(self, quant_config: Optional[QuantizationConfig] = None):
        self.quant_config = quant_config
        self.runner: Optional[MoeRunner] = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        num_experts = layer.w13_qweight.shape[0]
        device = layer.w13_qweight.device
        is_sm70 = torch.cuda.get_device_capability(device)[0] == 7

        layer.w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        layer.w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )

        marlin_w13_qweight = awq_marlin_moe_repack(
            layer.w13_qweight,
            layer.w13_g_idx_sort_indices,
            size_k=layer.w13_qweight.shape[1],
            size_n=layer.w13_qweight.shape[2] * self.quant_config.pack_factor,
            num_bits=self.quant_config.weight_bits,
        )
        replace_parameter(layer, "w13_qweight", marlin_w13_qweight)

        marlin_w2_qweight = awq_marlin_moe_repack(
            layer.w2_qweight,
            layer.w2_g_idx_sort_indices,
            size_k=layer.w2_qweight.shape[1],
            size_n=layer.w2_qweight.shape[2] * self.quant_config.pack_factor,
            num_bits=self.quant_config.weight_bits,
        )
        replace_parameter(layer, "w2_qweight", marlin_w2_qweight)

        w13_scales = layer.w13_scales.data.contiguous()
        w2_scales = layer.w2_scales.data.contiguous()
        scale_transform = (
            sm70_marlin_moe_logical_scales if is_sm70 else marlin_moe_permute_scales
        )
        marlin_w13_scales = scale_transform(
            s=w13_scales,
            size_k=layer.intermediate_size_per_partition,
            size_n=w13_scales.shape[2],
            group_size=self.quant_config.group_size,
        )
        replace_parameter(layer, "w13_scales", marlin_w13_scales)

        marlin_w2_scales = scale_transform(
            s=w2_scales,
            size_k=layer.intermediate_size_per_partition,
            size_n=w2_scales.shape[2],
            group_size=self.quant_config.group_size,
        )
        replace_parameter(layer, "w2_scales", marlin_w2_scales)

        if is_sm70:
            marlin_w13_zp = moe_awq_to_sm70_marlin_zero_points_float(
                layer.w13_qzeros,
                w13_scales,
                size_k=layer.w13_qzeros.shape[1],
                size_n=layer.w13_qzeros.shape[2] * self.quant_config.pack_factor,
                num_bits=self.quant_config.weight_bits,
            )
        else:
            marlin_w13_zp = moe_awq_to_marlin_zero_points(
                layer.w13_qzeros,
                size_k=layer.w13_qzeros.shape[1],
                size_n=layer.w13_qzeros.shape[2] * self.quant_config.pack_factor,
                num_bits=self.quant_config.weight_bits,
            )
        replace_parameter(layer, "w13_qzeros", marlin_w13_zp)

        if is_sm70:
            marlin_w2_zp = moe_awq_to_sm70_marlin_zero_points_float(
                layer.w2_qzeros,
                w2_scales,
                size_k=layer.w2_qzeros.shape[1],
                size_n=layer.w2_qzeros.shape[2] * self.quant_config.pack_factor,
                num_bits=self.quant_config.weight_bits,
            )
        else:
            marlin_w2_zp = moe_awq_to_marlin_zero_points(
                layer.w2_qzeros,
                size_k=layer.w2_qzeros.shape[1],
                size_n=layer.w2_qzeros.shape[2] * self.quant_config.pack_factor,
                num_bits=self.quant_config.weight_bits,
            )
        replace_parameter(layer, "w2_qzeros", marlin_w2_zp)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        if self.runner is None:
            raise RuntimeError("moe runner is not initialized")

        quant_info = MarlinMoeQuantInfo(
            w13_qweight=layer.w13_qweight,
            w2_qweight=layer.w2_qweight,
            w13_scales=layer.w13_scales,
            w2_scales=layer.w2_scales,
            w13_g_idx_sort_indices=layer.w13_g_idx_sort_indices,
            w2_g_idx_sort_indices=layer.w2_g_idx_sort_indices,
            w13_qzeros=layer.w13_qzeros,
            w2_qzeros=layer.w2_qzeros,
            weight_bits=self.quant_config.weight_bits,
        )
        return self.runner.run(dispatch_output, quant_info)
