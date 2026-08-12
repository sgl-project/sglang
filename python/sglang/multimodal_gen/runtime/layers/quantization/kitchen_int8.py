"""INT8 weight-only-storage linear backed by comfy_kitchen's fused ConvRot kernel.

On Ada (RTX 4090) INT8 is only worth doing with the right kernel: on MiniMax H3
shapes `torch._int_mm` measures 0.46-0.90x of BF16 (i.e. slower) and a Triton
INT8 GEMM roughly ties BF16, while `comfy_kitchen.int8_linear` reaches 2.49x.
The difference is that it is a single fused op -- it takes a BF16 activation and
does the Hadamard rotation, dynamic per-row activation quantization, IMMA GEMM,
dequantization and bias add without ever materializing the intermediates.

Quantization is data-free (group-wise Hadamard rotation + per-output-channel
absmax), so weights are quantized here after loading rather than read from a
pre-quantized checkpoint. That keeps this usable with the stock BF16 checkpoint
and avoids depending on any external file layout.

Registered CLI name: ``kitchen_int8``.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch.nn.parameter import Parameter

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs
from sglang.srt.layers.quantization.utils import is_layer_skipped

logger = init_logger(__name__)

# comfy_kitchen's dtype codes for the fused op's output.
_OUT_DTYPE_CODE = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}

_SUPPORTED_GROUP_SIZES = (16, 64, 256)

# comfy_kitchen picks its CUTLASS tile configuration from a threshold tree
# (select_fused_int8_config in cutlass_gemm_int8.cu). Shapes whose N falls under
# its 24832 cutoff but whose M is large get a Stream-K schedule, which exists to
# balance load when there are too few tiles to fill the GPU. At H3's 32700 tokens
# qkv_proj already launches ~21k CTAs over the 4090's 128 SMs, so Stream-K's
# workspace and fixup reduction are pure overhead: 26.5 ms against 17.9 ms for
# the identical tile without it. Capping rows per call keeps the plain
# data-parallel config, and is bit-exact because splitting rows does not change
# any single row's arithmetic.
_MAX_ROWS_PER_CALL = int(os.environ.get("SGLANG_KITCHEN_INT8_MAX_ROWS", "8192"))
# Narrow outputs do not recover the cost of writing results back through a
# preallocated buffer; H3's out_proj and fc2 (N=5376) both measure slower split.
_MIN_SPLIT_OUTPUT = int(os.environ.get("SGLANG_KITCHEN_INT8_MIN_SPLIT_N", "8192"))


def _row_split(rows: int, out_features: int) -> int | None:
    """Rows per `int8_linear` call, or None to issue one call for everything."""
    if _MAX_ROWS_PER_CALL <= 0 or rows <= _MAX_ROWS_PER_CALL:
        return None
    if out_features < _MIN_SPLIT_OUTPUT:
        return None
    return _MAX_ROWS_PER_CALL


def _load_comfy_kitchen():
    try:
        import comfy_kitchen  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "kitchen_int8 quantization requires the `comfy-kitchen` package "
            "(pip install comfy-kitchen). It is a self-contained abi3 extension "
            "and does not link against libtorch, so any torch version works."
        ) from exc
    if not hasattr(torch.ops.comfy_kitchen, "int8_linear"):
        raise RuntimeError(
            "comfy_kitchen is installed but did not register "
            "torch.ops.comfy_kitchen.int8_linear"
        )


class KitchenInt8Config(QuantizationConfig):
    """Config for online INT8 ConvRot quantization via comfy_kitchen.

    A no-arg ``KitchenInt8Config()`` is the only supported form: weights load in
    their source dtype and are quantized in ``process_weights_after_loading``.
    """

    def __init__(
        self,
        group_size: int = 256,
        ignored_layers: list[str] | None = None,
        packed_modules_mapping: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        if group_size not in _SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"kitchen_int8 group_size must be one of {_SUPPORTED_GROUP_SIZES}, "
                f"got {group_size}"
            )
        self.group_size = group_size
        self.ignored_layers = ignored_layers or []
        self.packed_modules_mapping = packed_modules_mapping or {}
        # Which layers actually got quantized is worth stating plainly in the
        # log: a silent fallback to BF16 looks exactly like a slow kernel.
        self.selected: list[str] = []
        self.skipped: list[str] = []
        self._processed = 0
        self._quantized_bytes = 0

    @classmethod
    def get_name(cls) -> str:
        return "kitchen_int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # INT8 tensor cores land on Turing.
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> KitchenInt8Config:
        return cls(
            group_size=cls.get_from_keys_or(config, ["group_size"], 256),
            ignored_layers=cls.get_from_keys_or(config, ["ignored_layers"], None),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase

        if not isinstance(layer, LinearBase):
            return None
        if is_layer_skipped(
            prefix, self.ignored_layers, fused_mapping=self.packed_modules_mapping
        ):
            self.skipped.append(prefix)
            return UnquantizedLinearMethod()
        # The rotation partitions the input dim into fixed-size groups, so a
        # layer whose input does not divide evenly simply stays in BF16 rather
        # than failing the whole model. H3's adaln projections (in=2688) are
        # the case this exists for, and they cost 0.2% of a step anyway.
        if layer.input_size % self.group_size:
            self.skipped.append(f"{prefix}(in={layer.input_size})")
            return UnquantizedLinearMethod()
        self.selected.append(prefix)
        return KitchenInt8LinearMethod(self)

    def note_quantized(self, saved_bytes: int) -> None:
        self._processed += 1
        self._quantized_bytes += saved_bytes
        if self._processed == len(self.selected):
            logger.info(
                "kitchen_int8: quantized %d linear layers (%.2f GiB of BF16 weights "
                "-> %.2f GiB INT8), left %d in BF16",
                self._processed,
                self._quantized_bytes / 1024**3,
                self._quantized_bytes / 2 / 1024**3,
                len(self.skipped),
            )
            logger.debug("kitchen_int8: layers left in BF16: %s", self.skipped)

    def get_scaled_act_names(self) -> list[str]:
        return []


class KitchenInt8LinearMethod(LinearMethodBase):
    """Quantizes BF16 weights to INT8 after load and runs the fused kernel."""

    def __init__(self, quant_config: KitchenInt8Config) -> None:
        self.quant_config = quant_config
        _load_comfy_kitchen()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        # get_quant_method already screened the unsharded input size, so this
        # only fires under TP > 1, where a row-parallel layer splits the very
        # dimension the rotation groups over.
        if input_size_per_partition % self.quant_config.group_size:
            raise ValueError(
                f"kitchen_int8 needs input_size_per_partition "
                f"({input_size_per_partition}) divisible by group_size "
                f"{self.quant_config.group_size}"
            )

        # Deliberately identical to UnquantizedLinearMethod: weights load as
        # BF16 through the model's existing loaders (H3 for instance installs a
        # custom qkv loader that reorders the grouped checkpoint layout), and
        # only then get replaced by their quantized form.
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

        weight = layer.weight.data
        if weight.dtype == torch.int8:  # already processed
            return

        # Quantization runs on CUDA, but the model may still be staged on CPU
        # for offload. Round-trip one layer at a time rather than relying on
        # the loader's whole-model device move, which would not fit in VRAM.
        home = weight.device
        qdata, params = TensorWiseINT8Layout.quantize(
            weight.to("cuda", non_blocking=True),
            is_weight=True,
            per_channel=True,
            convrot=True,
            convrot_groupsize=self.quant_config.group_size,
            stochastic_rounding=0,
        )
        layer.weight = Parameter(qdata.to(home), requires_grad=False)
        layer.register_parameter(
            "weight_scale",
            Parameter(
                params.scale.to(device=home, dtype=torch.float32), requires_grad=False
            ),
        )
        self.quant_config.note_quantized(weight.numel() * weight.element_size())
        del qdata, params
        torch.cuda.empty_cache()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_code = _OUT_DTYPE_CODE.get(x.dtype)
        if out_code is None:
            raise ValueError(
                f"kitchen_int8 does not support activation dtype {x.dtype}"
            )

        # The kernel takes 2D activations; callers may pass [..., K].
        orig_shape = x.shape
        if x.dim() != 2:
            x = x.reshape(-1, orig_shape[-1])
        x = x.contiguous()

        def run(rows: torch.Tensor) -> torch.Tensor:
            return torch.ops.comfy_kitchen.int8_linear(
                rows,
                layer.weight,
                layer.weight_scale,
                bias,
                out_code,
                True,  # convrot
                self.quant_config.group_size,
            )

        n_rows, n_out = x.shape[0], layer.weight.shape[0]
        split = _row_split(n_rows, n_out)
        if split is None:
            out = run(x)
        else:
            # Row slices of a contiguous 2D tensor are themselves contiguous, so
            # this splits without copying the activation.
            out = torch.empty(n_rows, n_out, dtype=x.dtype, device=x.device)
            for start in range(0, n_rows, split):
                out[start : start + split] = run(x[start : start + split])

        if len(orig_shape) != 2:
            out = out.reshape(*orig_shape[:-1], out.shape[-1])
        return out
