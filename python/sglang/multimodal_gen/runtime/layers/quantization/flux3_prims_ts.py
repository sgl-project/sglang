# SPDX-License-Identifier: Apache-2.0
"""FLUX.3 PrimsTS specializations of rowwise-FP8 and ModelOpt NVFP4.

The FLUX.3 loader replaces the generic checkpoint configs with these
subclasses. Each linear method plans, at construction, whether its TP-local
projection can run a FlashInfer PrimsTS epilogue, prepares that layout after
loading, and exposes the format-specific kernel calls that
:mod:`sglang.multimodal_gen.runtime.layers.flux3_fused_ops` dispatches to.
Projections without a plan keep the generic method's layout and ``apply``.
"""

from __future__ import annotations

import functools
import inspect
import logging
from enum import Enum
from typing import Any, ClassVar

import torch
from torch import nn

from sglang.kernels.fused_op import get_fused_op_backend
from sglang.kernels.spec import (
    CapabilityRequirement,
    KernelBackend,
    PlatformInfo,
    capabilities_satisfied,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8LinearMethod
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    _linear_nvfp4_scales_to_tma,
    _prepare_nvfp4_weight_bytes,
    _rebase_partition_weight_scales,
)
from sglang.multimodal_gen.runtime.layers.quantization.rowwise_fp8 import (
    RowwiseFp8Config,
)
from sglang.srt.environ import envs
from sglang.srt.layers.utils.common import copy_or_rebind_param

logger = logging.getLogger(__name__)

# PrimsTS emits TCGEN05 instructions: every SM10x part (SM100/103/107) has
# them, SM120 does not.
FLUX3_PRIMS_TS_CAPABILITIES = frozenset(
    {CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 9))}
)


class Flux3PrimsTsRole(str, Enum):
    """PrimsTS epilogue planned for one FLUX.3 projection."""

    LINEAR = "linear"
    QKV_QKNORM_ROPE = "qkv_qknorm_rope"
    SWIGLU = "swiglu"


def flux3_prims_ts_available() -> bool:
    """Whether this process may prepare FLUX.3 projections for PrimsTS."""
    forced = get_fused_op_backend()
    if forced is not None and forced is not KernelBackend.FLASHINFER:
        return False
    return capabilities_satisfied(FLUX3_PRIMS_TS_CAPABILITIES, PlatformInfo.detect())


@functools.cache
def flux3_prims_ts_supports_lora() -> bool:
    """Whether FlashInfer's PrimsTS GEMMs take the ``delta`` epilogue input."""
    try:
        from flashinfer.prims_ts import gemm as prims_ts_gemm
    except ImportError:
        return False
    entry_points = (
        prims_ts_gemm.fp8_linear,
        prims_ts_gemm.fp8_linear_swiglu,
        prims_ts_gemm.fp8_qkv_qknorm_rope,
        prims_ts_gemm.fp4_linear,
        prims_ts_gemm.fp4_linear_swiglu,
        prims_ts_gemm.fp4_qkv_qknorm_rope,
        prims_ts_gemm.PreparedFp4Linear.__call__,
    )
    return all("delta" in inspect.signature(fn).parameters for fn in entry_points)


def _use_separate_qkv_output() -> bool:
    """Whether TRT-LLM Gen Sage should receive directly separated Q/K/V."""
    return envs.SGLANG_USE_SAGE_ATTENTION_IMPL.get().lower() in {
        "trtllm_gen",
        "trtllm-gen",
    }


def _pack_qkv_layer_name(name: str) -> str:
    for leaf in (".q_proj", ".k_proj", ".v_proj"):
        if name.endswith(leaf):
            return name[: -len(leaf)] + ".to_qkv"
    return name


class Flux3PrimsTsConfigMixin:
    """FLUX.3 deployment targets shared by both quantized formats."""

    # Layer-prefix roots whose projections are PrimsTS deployment targets.
    prims_ts_targets: ClassVar[tuple[str, ...]]
    # Additional roots where only packed QKV uses PrimsTS.
    prims_ts_qkv_targets: ClassVar[tuple[str, ...]] = ()
    head_dim: int | None = None
    enable_prims_ts: bool = True

    @classmethod
    def from_base_config(
        cls,
        base: Any,
        *,
        head_dim: int | None = None,
        enable_prims_ts: bool = True,
    ) -> Any:
        """Specialize a generic checkpoint config for the native FLUX.3 DiT."""
        config = cls.__new__(cls)
        # Copy instance state rather than re-running ``__init__`` so values
        # adjusted after construction (e.g. ``swap_weight_nibbles``) survive.
        config.__dict__.update(vars(base))
        config.head_dim = head_dim
        config.enable_prims_ts = enable_prims_ts
        return config

    def verify_prims_ts_coverage(self, model: nn.Module) -> None:
        """Hook for formats that must not silently fall back."""
        return None


class Flux3PrimsTsMethodMixin:
    """Plan, prepare, and run one FLUX.3 projection on PrimsTS.

    The planned role is the only switch the fused operators consult. Subclasses
    provide the format-specific eligibility checks and kernel calls.
    """

    prims_ts_role: Flux3PrimsTsRole | None = None
    # Operand format of a quantized activation shared between projections.
    activation_format: ClassVar[str]
    # Format of packed MLP-up output that MLP-down can consume directly.
    packed_activation_format: ClassVar[str | None] = None

    def _prims_ts_layer_supported(self, layer: nn.Module) -> bool:
        raise NotImplementedError

    def _prims_ts_shape_supported(self, k: int, n: int) -> bool:
        raise NotImplementedError

    def _prims_ts_fused_qkv_supported(self) -> bool:
        return True

    def _plan_prims_ts_role(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
    ) -> Flux3PrimsTsRole | None:
        prefix = getattr(layer, "prefix", "")
        leaf = prefix.rsplit(".", 1)[-1]
        is_target = prefix.startswith(self.quant_config.prims_ts_targets) or (
            leaf == "to_qkv"
            and prefix.startswith(self.quant_config.prims_ts_qkv_targets)
        )
        if not (
            self.quant_config.enable_prims_ts
            and is_target
            and flux3_prims_ts_available()
            and self._prims_ts_layer_supported(layer)
            # Shapes are TP-local: each rank's shard must fit the GEMM tiles.
            and self._prims_ts_shape_supported(
                input_size_per_partition, sum(output_partition_sizes)
            )
        ):
            return None
        if leaf == "to_qkv":
            if self._prims_ts_fused_qkv_supported():
                return Flux3PrimsTsRole.QKV_QKNORM_ROPE
            return Flux3PrimsTsRole.LINEAR
        if leaf == "mlp_in":
            # Only the merged [gate | value] projection of a gated MLP.
            gated = (
                len(output_partition_sizes) == 2
                and output_partition_sizes[0] == output_partition_sizes[1]
            )
            return Flux3PrimsTsRole.SWIGLU if gated else None
        if leaf in ("attn_out", "mlp_out"):
            return Flux3PrimsTsRole.LINEAR
        return None

    def _finalize_prims_ts_role(self, layer: nn.Module) -> None:
        # The bias parameter is created after ``create_weights``.
        if self.prims_ts_role is not None and layer.bias is not None:
            self.prims_ts_role = None

    def quantize_prims_ts_activation(self, layer: nn.Module, x: torch.Tensor):
        raise NotImplementedError

    def apply_prims_ts_linear(
        self, layer: nn.Module, x: torch.Tensor, quantized=None, *, delta=None
    ) -> torch.Tensor:
        raise NotImplementedError

    def apply_prims_ts_linear_from_packed(self, layer: nn.Module, activation):
        raise NotImplementedError(
            f"{layer.prefix}: {type(self).__name__} has no packed activation input"
        )

    def apply_prims_ts_swiglu(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        quantized=None,
        *,
        packed_output: bool = False,
        delta=None,
    ):
        raise NotImplementedError

    def apply_prims_ts_qkv_qknorm_rope(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        quantized=None,
        *,
        delta=None,
    ) -> torch.Tensor:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Rowwise FP8
# --------------------------------------------------------------------------- #


class Flux3RowwiseFp8Config(Flux3PrimsTsConfigMixin, RowwiseFp8Config):
    """Rowwise FP8 for the native FLUX.3 DiT and its packed ``to_qkv``."""

    prims_ts_targets = ("content_mode_blocks.image.", "single_blocks.")
    prims_ts_qkv_targets = ("txt_mode_blocks.",)

    @classmethod
    def from_base_config(
        cls,
        base: RowwiseFp8Config,
        *,
        head_dim: int | None = None,
        enable_prims_ts: bool = True,
    ) -> Flux3RowwiseFp8Config:
        config = super().from_base_config(
            base,
            head_dim=head_dim,
            enable_prims_ts=enable_prims_ts,
        )
        # The checkpoint serializes split q/k/v projections; the native tree
        # loads them into one packed ``to_qkv`` linear.
        config.fp8_layers = frozenset(map(_pack_qkv_layer_name, base.fp8_layers))
        config.input_scale_layers = frozenset(
            map(_pack_qkv_layer_name, base.input_scale_layers)
        )
        return config

    def get_quant_method(self, layer: nn.Module, prefix: str):
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase

        if isinstance(layer, LinearBase) and prefix in self.fp8_layers:
            return Flux3Fp8LinearMethod(self)
        return super().get_quant_method(layer, prefix)


class Flux3Fp8LinearMethod(Flux3PrimsTsMethodMixin, Fp8LinearMethod):
    """Serialized rowwise FP8 with optional PrimsTS FP8 epilogues.

    The generic method leaves ``layer.weight`` as a ``[K, N]`` view of the
    checkpoint's contiguous ``[N, K]`` storage, so ``layer.weight.t()`` is
    already the PrimsTS operand. Only SwiGLU rewrites that storage, once.
    """

    activation_format = "fp8"

    def _prims_ts_layer_supported(self, layer: nn.Module) -> bool:
        # PrimsTS quantizes activations per token; static scales stay generic.
        return self.per_channel_serialized and not self.quant_config.has_input_scale(
            layer.prefix
        )

    def _prims_ts_shape_supported(self, k: int, n: int) -> bool:
        return k % 128 == 0 and n % 512 == 0

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        self.prims_ts_role = self._plan_prims_ts_role(
            layer, input_size_per_partition, output_partition_sizes
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        super().process_weights_after_loading(layer)
        self._finalize_prims_ts_role(layer)
        if self.prims_ts_role is Flux3PrimsTsRole.SWIGLU:
            from sglang.kernels.ops.diffusion import pack_flux3_mlp_weight

            # Interleave in place so the epilogue's [value_i, gate_i] layout
            # replaces, rather than duplicates, the checkpoint's [gate | value].
            weight, weight_scale = pack_flux3_mlp_weight(
                layer.weight, layer.weight_scale
            )
            copy_or_rebind_param(layer, "weight", weight.t())
            copy_or_rebind_param(layer, "weight_scale", weight_scale)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.prims_ts_role is Flux3PrimsTsRole.SWIGLU:
            raise RuntimeError(
                f"{layer.prefix}: weights are interleaved for the PrimsTS SwiGLU "
                "epilogue; call the FLUX.3 fused operator"
            )
        return super().apply(layer, x, bias)

    def quantize_prims_ts_activation(self, layer: nn.Module, x: torch.Tensor):
        from sglang.kernels.ops.diffusion import quantize_flux3_activation

        return quantize_flux3_activation(x)

    def apply_prims_ts_linear(
        self, layer: nn.Module, x: torch.Tensor, quantized=None, *, delta=None
    ) -> torch.Tensor:
        from sglang.kernels.ops.diffusion import flux3_clustered_fp8_linear

        return flux3_clustered_fp8_linear(
            x, layer.weight.t(), layer.weight_scale, quantized, delta=delta
        )

    def apply_prims_ts_swiglu(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        quantized=None,
        *,
        packed_output: bool = False,
        delta=None,
    ) -> torch.Tensor:
        from sglang.kernels.ops.diffusion import flux3_fused_mlp_up

        if packed_output:
            raise NotImplementedError(f"{layer.prefix}: FP8 SwiGLU emits BF16 only")
        return flux3_fused_mlp_up(
            x, layer.weight.t(), layer.weight_scale, quantized, delta=delta
        )

    def apply_prims_ts_qkv_qknorm_rope(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        quantized=None,
        *,
        delta=None,
    ) -> torch.Tensor:
        from sglang.kernels.ops.diffusion import (
            flux3_clustered_fp8_qkv_qknorm_rope,
        )

        return flux3_clustered_fp8_qkv_qknorm_rope(
            x,
            layer.weight.t(),
            layer.weight_scale,
            q_norm_weight,
            k_norm_weight,
            cos_sin_cache,
            positions,
            quantized,
            separate_qkv_output=_use_separate_qkv_output(),
            delta=delta,
        )


# --------------------------------------------------------------------------- #
# ModelOpt NVFP4
# --------------------------------------------------------------------------- #

_NVFP4_COVERAGE = (
    ("to_qkv", Flux3PrimsTsRole.QKV_QKNORM_ROPE),
    ("attn_out", Flux3PrimsTsRole.LINEAR),
    ("mlp_in", Flux3PrimsTsRole.SWIGLU),
    ("mlp_out", Flux3PrimsTsRole.LINEAR),
)


class Flux3ModelOptFp4Config(Flux3PrimsTsConfigMixin, ModelOptFp4Config):
    """ModelOpt NVFP4 for the native FLUX.3 DiT.

    Static-activation checkpoints use PrimsTS on every image/text/joint block
    projection; dynamic-activation checkpoints keep the generic backend.
    """

    prims_ts_targets = ("content_mode_blocks.", "txt_mode_blocks.", "single_blocks.")

    def get_quant_method(self, layer: nn.Module, prefix: str):
        return self._get_quant_method(layer, prefix, Linear=Flux3Nvfp4LinearMethod)

    def verify_prims_ts_coverage(self, model: nn.Module) -> None:
        """Fail closed unless every static-NVFP4 block GEMM uses PrimsTS.

        The FLUX.3 NVFP4 checkpoint is a fixed-shape deployment target.
        Silently falling back for one projection would make a benchmark appear
        to use the new path while retaining a conventional FP4 GEMM in the
        denoiser. Unsupported hardware never plans PrimsTS, so it is exempt.
        """
        if (
            not self.enable_prims_ts
            or self.dynamic_input_scale
            or not flux3_prims_ts_available()
        ):
            return
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase

        linears = {
            module.prefix: module
            for module in model.modules()
            if isinstance(module, LinearBase)
        }
        blocks = sorted(
            prefix[: -len(".to_qkv")]
            for prefix, module in linears.items()
            if prefix.endswith(".to_qkv")
            and prefix.startswith(self.prims_ts_targets)
            and isinstance(module.quant_method, Flux3Nvfp4LinearMethod)
        )
        if not blocks:
            return
        missing: list[str] = []
        for block in blocks:
            for leaf, role in _NVFP4_COVERAGE:
                prefix = f"{block}.{leaf}"
                layer = linears.get(prefix)
                planned = getattr(
                    getattr(layer, "quant_method", None), "prims_ts_role", None
                )
                if planned is role:
                    continue
                if role is not Flux3PrimsTsRole.QKV_QKNORM_ROPE or planned is None:
                    missing.append(prefix)
                if role is Flux3PrimsTsRole.QKV_QKNORM_ROPE:
                    missing.append(f"{prefix} (QKV/QKNorm/RoPE epilogue)")
        if missing:
            raise RuntimeError(
                "FLUX.3 NVFP4 requires PrimsTS FP4 GEMM for every image/joint "
                f"projection; unsupported projections: {', '.join(missing)}"
            )
        logger.info(
            "FLUX.3 NVFP4 PrimsTS coverage verified: %d image/joint blocks, "
            "%d QKV+QKNorm+RoPE, %d fused MLP-up, %d attention-out, and %d MLP-down GEMMs",
            *([len(blocks)] * 5),
        )


def _interleave_swiglu_rows(
    weight: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder ``[gate_0..H | value_0..H]`` rows as ``[value_i, gate_i]`` pairs.

    PrimsTS' SwiGLU epilogue reads adjacent ``[value_i, gate_i]`` rows and
    computes ``value_i * silu(gate_i)``. Packed rows and their per-row block
    scales move together, before the scales are converted to 128x4.
    """
    n = weight.shape[0]
    half = n // 2
    weight = (
        torch.stack((weight[half:], weight[:half]), dim=1)
        .reshape(n, weight.shape[1])
        .contiguous()
    )
    scales = (
        torch.stack((scales[half:], scales[:half]), dim=1)
        .reshape(n, scales.shape[1])
        .contiguous()
    )
    return weight, scales


class Flux3Nvfp4LinearMethod(Flux3PrimsTsMethodMixin, ModelOptFp4LinearMethod):
    """ModelOpt NVFP4 with optional static-activation PrimsTS epilogues."""

    activation_format = "nvfp4"
    packed_activation_format = "nvfp4"

    def _prims_ts_layer_supported(self, layer: nn.Module) -> bool:
        config = self.quant_config
        return config.is_checkpoint_nvfp4_serialized and not config.dynamic_input_scale

    def _prims_ts_shape_supported(self, k: int, n: int) -> bool:
        return k % 256 == 0 and n % 512 == 0

    def _prims_ts_fused_qkv_supported(self) -> bool:
        return self.quant_config.head_dim == 128

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        self.prims_ts_role = self._plan_prims_ts_role(
            layer, input_size_per_partition, output_partition_sizes
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        self._finalize_prims_ts_role(layer)
        if self.prims_ts_role is None:
            super().process_weights_after_loading(layer)
            return

        input_scale_2, weight_scale_2 = self._process_global_scales(layer)
        # PrimsTS accepts capture-safe host scalars; read them once here
        # instead of synchronizing on every GEMM.
        input_global_scale = float(input_scale_2.item())
        scales = self._logical_weight_scales(layer)
        qkv_scale = None
        if layer.weight_scale_2.numel() > 1:
            if self.prims_ts_role is Flux3PrimsTsRole.QKV_QKNORM_ROPE:
                # The fused epilogue applies exact per-partition Q/K/V scales.
                qkv_scale = (
                    layer.weight_scale_2.float().contiguous() * input_global_scale
                )
            else:
                scales = _rebase_partition_weight_scales(
                    scales, layer.logical_widths, layer.weight_scale_2, weight_scale_2
                )

        n, k = layer.weight.shape[0], layer.weight.shape[1] * 2
        if k % 256 or n % 512:
            raise ValueError(
                f"{layer.prefix}: FLUX.3 PrimsTS NVFP4 requires K % 256 == 0 "
                f"and N % 512 == 0, got N={n}, K={k}"
            )
        weight = _prepare_nvfp4_weight_bytes(
            layer.weight.data,
            swap_weight_nibbles=getattr(
                self.quant_config, "swap_weight_nibbles", False
            ),
        )
        if self.prims_ts_role is Flux3PrimsTsRole.SWIGLU:
            weight, scales = _interleave_swiglu_rows(weight, scales)
        # Dense PrimsTS consumes an unshuffled ``[N, K/2]`` weight and a flat
        # canonical 128x4 scale buffer, unlike the TRT-LLM/CUTLASS layouts.
        copy_or_rebind_param(layer, "weight", weight)
        copy_or_rebind_param(
            layer,
            "weight_scale_prims_ts_128x4",
            _linear_nvfp4_scales_to_tma(scales)
            .view(torch.uint8)
            .reshape(-1)
            .contiguous(),
        )
        if qkv_scale is not None:
            copy_or_rebind_param(layer, "prims_ts_qkv_scale", qkv_scale)
        layer.prims_ts_input_global_scale = input_global_scale
        layer.prims_ts_weight_global_scale = float(weight_scale_2.item())
        layer.weights_padding_cols = 0

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.prims_ts_role is not None:
            raise RuntimeError(
                f"{layer.prefix}: weights are prepared for PrimsTS "
                f"({self.prims_ts_role.value}); call the FLUX.3 fused operator"
            )
        return super().apply(layer, x, bias)

    def quantize_prims_ts_activation(self, layer: nn.Module, x: torch.Tensor):
        from sglang.kernels.ops.diffusion import (
            quantize_flux3_nvfp4_activation,
        )

        return quantize_flux3_nvfp4_activation(x, layer.prims_ts_input_global_scale)

    def apply_prims_ts_linear(
        self, layer: nn.Module, x: torch.Tensor, quantized=None, *, delta=None
    ) -> torch.Tensor:
        from sglang.kernels.ops.diffusion import flux3_nvfp4_linear

        return flux3_nvfp4_linear(
            x,
            layer.weight,
            layer.weight_scale_prims_ts_128x4,
            layer.prims_ts_input_global_scale,
            layer.prims_ts_weight_global_scale,
            quantized,
            delta=delta,
        )

    def apply_prims_ts_linear_from_packed(
        self, layer: nn.Module, activation
    ) -> torch.Tensor:
        from sglang.kernels.ops.diffusion import (
            flux3_nvfp4_linear_from_packed_activation,
        )

        return flux3_nvfp4_linear_from_packed_activation(
            activation,
            layer.weight,
            layer.weight_scale_prims_ts_128x4,
            layer.prims_ts_weight_global_scale,
        )

    def apply_prims_ts_swiglu(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        quantized=None,
        *,
        packed_output: bool = False,
        delta=None,
    ):
        from sglang.kernels.ops.diffusion import (
            flux3_nvfp4_linear_swiglu,
            flux3_nvfp4_linear_swiglu_fp4,
        )

        if packed_output:
            if delta is not None:
                raise NotImplementedError(
                    f"{layer.prefix}: packed NVFP4 SwiGLU output has no delta input"
                )
            return flux3_nvfp4_linear_swiglu_fp4(
                x,
                layer.weight,
                layer.weight_scale_prims_ts_128x4,
                layer.prims_ts_input_global_scale,
                layer.prims_ts_weight_global_scale,
                quantized,
            )
        return flux3_nvfp4_linear_swiglu(
            x,
            layer.weight,
            layer.weight_scale_prims_ts_128x4,
            layer.prims_ts_input_global_scale,
            layer.prims_ts_weight_global_scale,
            quantized,
            delta=delta,
        )

    def apply_prims_ts_qkv_qknorm_rope(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        quantized=None,
        *,
        delta=None,
    ) -> torch.Tensor:
        from sglang.kernels.ops.diffusion import (
            flux3_nvfp4_qkv_qknorm_rope,
        )

        return flux3_nvfp4_qkv_qknorm_rope(
            x,
            layer.weight,
            layer.weight_scale_prims_ts_128x4,
            layer.prims_ts_input_global_scale,
            layer.prims_ts_weight_global_scale,
            q_norm_weight,
            k_norm_weight,
            cos_sin_cache,
            positions,
            getattr(layer, "prims_ts_qkv_scale", None),
            quantized,
            separate_qkv_output=_use_separate_qkv_output(),
            delta=delta,
        )
