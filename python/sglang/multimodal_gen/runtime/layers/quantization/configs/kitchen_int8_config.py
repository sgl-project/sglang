# SPDX-License-Identifier: Apache-2.0
"""Config for online or serialized INT8 ConvRot, run by sgl-kernel or comfy_kitchen.

Both backends implement the same quantization (arXiv:2512.03673): group-wise
regular Hadamard rotation, per-row dynamic INT8 activations and per-output-
channel INT8 weights, with interchangeable weight and scale tensors. ``auto``
picks sgl-kernel's fused ``convrot_int8_*`` ops where they run (CC 9.0, 10.0,
12.0 and 12.1; group size 64, 128, 256 or 512; output width a multiple of 8)
and ``comfy_kitchen.int8_linear`` where that package is installed (Turing and
later; group size 16, 64 or 256); a layer neither backend serves stays BF16.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.layers.quantization.utils import is_layer_skipped

logger = init_logger(__name__)

COMFY_KITCHEN = "comfy_kitchen"
SGL_KERNEL = "sgl_kernel"
BACKENDS = ("auto", COMFY_KITCHEN, SGL_KERNEL)
# Group widths each backend's kernels are instantiated for.
BACKEND_GROUP_SIZES = {COMFY_KITCHEN: (16, 64, 256), SGL_KERNEL: (64, 128, 256, 512)}
_SUPPORTED_GROUP_SIZES = (16, 64, 128, 256, 512)


def _sgl_kernel_available() -> bool:
    from sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_sgl_kernel import (
        sgl_kernel_convrot_unavailable_reason,
    )

    reason = sgl_kernel_convrot_unavailable_reason()
    if reason is not None:
        _log_sgl_kernel_fallback(reason)
    return reason is None


@functools.cache
def _log_sgl_kernel_fallback(reason: str) -> None:
    # Once per process: the refused-GPU reason (for example CC 10.3) would
    # otherwise only surface with an explicit sgl_kernel backend.
    logger.info(
        "kitchen_int8: sgl-kernel backend unavailable (%s); auto uses comfy_kitchen",
        reason,
    )


def _comfy_kitchen_available() -> bool:
    from sglang.multimodal_gen.runtime.layers.quantization.kitchen_int8 import (
        comfy_kitchen_available,
    )

    return comfy_kitchen_available()


class KitchenInt8Config(QuantizationConfig):
    """Dispatch online or serialized Comfy ConvRot layers to a kernel backend."""

    def __init__(
        self,
        group_size: int = 256,
        ignored_layers: list[str] | None = None,
        packed_modules_mapping: dict[str, list[str]] | None = None,
        layer_markers: dict[str, dict[str, Any]] | None = None,
        backend: str | None = None,
    ) -> None:
        super().__init__()
        if group_size not in _SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"kitchen_int8 group_size must be one of {_SUPPORTED_GROUP_SIZES}, "
                f"got {group_size}"
            )
        if backend is None:
            backend = envs.SGLANG_DIFFUSION_KITCHEN_INT8_BACKEND
        if backend not in BACKENDS:
            raise ValueError(
                f"kitchen_int8 backend must be one of {BACKENDS}, got {backend!r}"
            )
        if backend != "auto" and group_size not in BACKEND_GROUP_SIZES[backend]:
            raise ValueError(
                f"kitchen_int8 backend {backend} has kernels for group sizes "
                f"{BACKEND_GROUP_SIZES[backend]}, got {group_size}"
            )
        self.group_size = group_size
        self.backend = backend
        self.ignored_layers = ignored_layers or []
        self.packed_modules_mapping = packed_modules_mapping or {}
        self.layer_markers = layer_markers
        self.is_checkpoint_int8_serialized = layer_markers is not None
        self.checkpoint_uses_native_qkv_layout = self.is_checkpoint_int8_serialized
        self._serialized_group_sizes: dict[str, int] = {}
        if layer_markers is not None:
            for prefix, marker in layer_markers.items():
                if marker.get("format") != "int8_tensorwise":
                    raise ValueError(
                        f"Unsupported Comfy INT8 format for {prefix!r}: "
                        f"{marker.get('format')!r}"
                    )
                if marker.get("convrot") is not True:
                    raise ValueError(
                        f"Serialized kitchen_int8 layer {prefix!r} must set "
                        "convrot=true"
                    )
                marker_group_size = marker.get("convrot_groupsize")
                if marker_group_size not in BACKEND_GROUP_SIZES[COMFY_KITCHEN]:
                    raise ValueError(
                        f"Serialized kitchen_int8 layer {prefix!r} must declare "
                        f"convrot_groupsize in {BACKEND_GROUP_SIZES[COMFY_KITCHEN]}, "
                        f"got {marker_group_size!r}"
                    )
                self._serialized_group_sizes[prefix] = marker_group_size
        elif backend == "auto" and self.resolve_backend() is None:
            raise ValueError(
                f"kitchen_int8: no backend on this machine serves group size "
                f"{group_size} (sgl-kernel needs CC 9.0/10.0/12.0/12.1 and one of "
                f"{BACKEND_GROUP_SIZES[SGL_KERNEL]}; comfy_kitchen serves "
                f"{BACKEND_GROUP_SIZES[COMFY_KITCHEN]})"
            )
        # Logged per backend at the end of loading: a silent fallback to BF16
        # or to the slower backend looks exactly like a slow kernel.
        self.selected: list[str] = []
        self.selected_by_backend: dict[str, list[str]] = {
            COMFY_KITCHEN: [],
            SGL_KERNEL: [],
        }
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
        # INT8 tensor cores land on Turing (comfy_kitchen); the sgl-kernel
        # backend checks its own table when it is selected.
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> KitchenInt8Config:
        return cls(
            group_size=cls.get_from_keys_or(config, ["group_size"], 256),
            ignored_layers=cls.get_from_keys_or(config, ["ignored_layers"], None),
            backend=cls.get_from_keys_or(config, ["backend"], None),
        )

    def resolve_backend(self, group_size: int | None = None) -> str | None:
        """Backend serving ``group_size`` (default: the online group size), or
        None when the requested backend has no kernel for it. ``auto`` prefers
        sgl-kernel where its ops run on this GPU; an explicit backend is never
        substituted."""
        group_size = self.group_size if group_size is None else group_size
        if self.backend == "auto":
            if (
                group_size in BACKEND_GROUP_SIZES[SGL_KERNEL]
                and _sgl_kernel_available()
            ):
                return SGL_KERNEL
            if group_size in BACKEND_GROUP_SIZES[COMFY_KITCHEN]:
                return COMFY_KITCHEN
            return None
        return self.backend if group_size in BACKEND_GROUP_SIZES[self.backend] else None

    def require_comfy_kitchen(self, reason: str) -> None:
        """Serve every layer with comfy_kitchen; an explicit sgl_kernel request is refused."""
        if self.backend == SGL_KERNEL:
            raise ValueError(
                f"kitchen_int8 backend sgl_kernel {reason}; use backend "
                "comfy_kitchen or TP and/or sequence parallelism instead"
            )
        if self.group_size not in BACKEND_GROUP_SIZES[COMFY_KITCHEN]:
            raise ValueError(
                f"kitchen_int8 backend comfy_kitchen has no kernel for group size "
                f"{self.group_size}, and the sgl_kernel backend {reason}"
            )
        if self.backend == "auto":
            logger.info(
                "kitchen_int8: using the comfy_kitchen backend because %s", reason
            )
        self.backend = COMFY_KITCHEN

    def _backend_for(self, *, group_size: int, out_size: int) -> str | None:
        backend = self.resolve_backend(group_size)
        if backend == SGL_KERNEL and out_size % 8:
            # The sgl-kernel GEMM epilogue stores 8 BF16 outputs at a time.
            if (
                self.backend == "auto"
                and group_size in BACKEND_GROUP_SIZES[COMFY_KITCHEN]
                and _comfy_kitchen_available()
            ):
                return COMFY_KITCHEN
            return None
        return backend

    def _no_kernel_reason(self, *, group_size: int, out_size: int) -> str:
        comfy = (
            "installed"
            if _comfy_kitchen_available()
            else "not installed (pip install comfy-kitchen)"
        )
        return (
            f"ConvRot group {group_size}, {out_size} outputs; backend {self.backend!r}; "
            f"sgl_kernel serves group sizes {BACKEND_GROUP_SIZES[SGL_KERNEL]} with an "
            f"output width that is a multiple of 8, comfy_kitchen serves group sizes "
            f"{BACKEND_GROUP_SIZES[COMFY_KITCHEN]} and is {comfy}"
        )

    def _select(
        self, *, prefix: str, backend: str, group_size: int, serialized: bool
    ) -> QuantizeMethodBase:
        self.selected.append(prefix)
        self.selected_by_backend[backend].append(prefix)
        if backend == SGL_KERNEL:
            from sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_sgl_kernel import (
                ConvRotInt8SglKernelLinearMethod,
            )

            return ConvRotInt8SglKernelLinearMethod(
                self, group_size=group_size, is_checkpoint_serialized=serialized
            )
        from sglang.multimodal_gen.runtime.layers.quantization.kitchen_int8 import (
            KitchenInt8LinearMethod,
        )

        return KitchenInt8LinearMethod(
            self, group_size=group_size, is_checkpoint_serialized=serialized
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        # A column-parallel layer shards its output; every rank sees the same
        # per-shard width, so the decision below is rank-consistent.
        out_size = (
            layer.output_size_per_partition
            if isinstance(layer, ColumnParallelLinear)
            else layer.output_size
        )
        if self.layer_markers is not None:
            marker_group_size = self._serialized_group_sizes.get(prefix)
            if marker_group_size is None:
                return UnquantizedLinearMethod()
            if layer.input_size % marker_group_size:
                raise ValueError(
                    f"Serialized kitchen_int8 layer {prefix!r} has input size "
                    f"{layer.input_size}, which is not divisible by its "
                    f"ConvRot group size {marker_group_size}"
                )
            backend = self._backend_for(group_size=marker_group_size, out_size=out_size)
            if backend is None:
                raise ValueError(
                    f"Serialized kitchen_int8 layer {prefix!r} has no kernel: "
                    + self._no_kernel_reason(
                        group_size=marker_group_size, out_size=out_size
                    )
                )
            return self._select(
                prefix=prefix,
                backend=backend,
                group_size=marker_group_size,
                serialized=True,
            )
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
        backend = self._backend_for(group_size=self.group_size, out_size=out_size)
        if backend is None:
            self.skipped.append(f"{prefix}(out={out_size})")
            return UnquantizedLinearMethod()
        return self._select(
            prefix=prefix, backend=backend, group_size=self.group_size, serialized=False
        )

    def note_quantized(self, saved_bytes: int) -> None:
        self._processed += 1
        self._quantized_bytes += saved_bytes
        if self._processed == len(self.selected):
            logger.info(
                "kitchen_int8: quantized %d linear layers (%.2f GiB of BF16 weights "
                "-> %.2f GiB INT8; sgl_kernel %d, comfy_kitchen %d), left %d in BF16",
                self._processed,
                self._quantized_bytes / 1024**3,
                self._quantized_bytes / 2 / 1024**3,
                len(self.selected_by_backend[SGL_KERNEL]),
                len(self.selected_by_backend[COMFY_KITCHEN]),
                len(self.skipped),
            )
            logger.debug("kitchen_int8: layers left in BF16: %s", self.skipped)

    def note_loaded(self) -> None:
        """A serialized layer's INT8 weights are in place; logs the backend split once."""
        self._processed += 1
        if self._processed == len(self.selected):
            logger.info(
                "kitchen_int8: loaded %d serialized INT8 linear layers "
                "(sgl_kernel %d, comfy_kitchen %d)",
                self._processed,
                len(self.selected_by_backend[SGL_KERNEL]),
                len(self.selected_by_backend[COMFY_KITCHEN]),
            )

    def get_scaled_act_names(self) -> list[str]:
        return []

    def supports_input_partition(
        self, prefix: str, input_size_per_partition: int
    ) -> bool:
        group_size = self.group_size
        if self.layer_markers is not None:
            marker_group_size = self._serialized_group_sizes.get(prefix)
            if marker_group_size is None:
                return True
            group_size = marker_group_size
        return input_size_per_partition % group_size == 0
