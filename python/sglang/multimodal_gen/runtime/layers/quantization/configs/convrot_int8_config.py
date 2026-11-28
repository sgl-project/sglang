# SPDX-License-Identifier: Apache-2.0
"""Config for online or serialized ConvRot INT8 on native XPU, JIT or comfy_kitchen.

The backends implement the same quantization scheme (arXiv:2512.03673): group-wise
regular Hadamard rotation, per-row dynamic INT8 activations and per-output-
channel INT8 weights, with interchangeable weight and scale tensors. ``auto``
picks SGLang's JIT-compiled fused ``convrot_int8`` ops where they build and run
(CC 9.0, 10.0, 12.0 and 12.1 with a CUDA toolchain; group size 64, 128, 256 or
512; output width a multiple of 8) and ``comfy_kitchen.int8_linear`` otherwise
(Turing and later; group size 16, 64 or 256; the load fails with an install hint
when the package is missing). Only a layer whose output width the JIT epilogue
cannot store falls to comfy_kitchen or, without it, stays BF16.

On XPU, ``auto`` selects sgl-kernel-xpu's two native kernels (BMG, BF16,
group 256, per-shard K in [256, 8192] and per-shard N divisible by 32). As on
CUDA, an online layer outside that range stays BF16; a serialized one fails,
since there is no comfy_kitchen or eager fallback. The native butterfly
rotation is not bit-exact with dense-matmul rotation.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

from sglang.kernels.ops.gemm.convrot_int8 import GROUP_SIZES as JIT_GROUP_SIZES
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.comfy_int8 import (
    ComfyInt8EmbeddingMethod,
    is_comfy_int8_embedding,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.layers.quantization.utils import is_layer_skipped

logger = init_logger(__name__)

COMFY_KITCHEN = "comfy_kitchen"
JIT = "jit"
XPU = "xpu"
BACKENDS = ("auto", COMFY_KITCHEN, JIT, XPU)
# Per-shard GEMM shapes the sgl-kernel-xpu ConvRot ops accept.
XPU_GROUP_SIZE = 256
XPU_MAX_INPUT_SIZE = 8192
XPU_OUTPUT_ALIGNMENT = 32
# Group widths each backend's kernels are instantiated for.
BACKEND_GROUP_SIZES = {
    COMFY_KITCHEN: (16, 64, 256),
    JIT: JIT_GROUP_SIZES,
    XPU: (XPU_GROUP_SIZE,),
}
_SUPPORTED_GROUP_SIZES = tuple(
    sorted({*BACKEND_GROUP_SIZES[COMFY_KITCHEN], *JIT_GROUP_SIZES})
)


def xpu_supports_input_size(input_size: int) -> bool:
    """Whether the XPU ops accept a per-shard input (K) width."""
    return (
        XPU_GROUP_SIZE <= input_size <= XPU_MAX_INPUT_SIZE
        and input_size % XPU_GROUP_SIZE == 0
    )


def xpu_supports_shape(*, input_size: int, output_size: int) -> bool:
    """Whether the XPU ops accept a per-shard ``[output_size, input_size]`` weight."""
    return (
        xpu_supports_input_size(input_size)
        and output_size > 0
        and output_size % XPU_OUTPUT_ALIGNMENT == 0
    )


def _jit_available() -> bool:
    from sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_jit import (
        jit_convrot_unavailable_reason,
    )

    reason = jit_convrot_unavailable_reason()
    if reason is not None:
        _log_jit_fallback(reason)
    return reason is None


@functools.cache
def _log_jit_fallback(reason: str) -> None:
    # Once per process: the refused-GPU reason (for example CC 10.3) or a failed
    # build would otherwise only surface with an explicit jit backend.
    logger.info(
        "convrot_int8: JIT backend unavailable (%s); auto uses comfy_kitchen",
        reason,
    )


def _comfy_kitchen_available() -> bool:
    from sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_comfy_kitchen import (
        comfy_kitchen_available,
    )

    return comfy_kitchen_available()


class ConvRotInt8Config(QuantizationConfig):
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
                f"convrot_int8 group_size must be one of {_SUPPORTED_GROUP_SIZES}, "
                f"got {group_size}"
            )
        if backend is None:
            backend = envs.SGLANG_DIFFUSION_CONVROT_INT8_BACKEND
        if backend not in BACKENDS:
            raise ValueError(
                f"convrot_int8 backend must be one of {BACKENDS}, got {backend!r}"
            )
        on_xpu = current_platform.is_xpu()
        if on_xpu and backend not in ("auto", XPU):
            raise ValueError(
                "convrot_int8 on XPU requires backend auto or xpu; "
                "CUDA JIT and comfy_kitchen fallback are not supported"
            )
        if backend != "auto" and group_size not in BACKEND_GROUP_SIZES[backend]:
            raise ValueError(
                f"convrot_int8 backend {backend} has kernels for group sizes "
                f"{BACKEND_GROUP_SIZES[backend]}, got {group_size}"
            )
        self.group_size = group_size
        self.backend = backend
        # Every layer runs on the native XPU ops; decided once, so the
        # per-layer dispatch below never re-probes the platform.
        self.uses_xpu = backend == XPU or (backend == "auto" and on_xpu)
        self.ignored_layers = ignored_layers or []
        self.packed_modules_mapping = packed_modules_mapping or {}
        self.layer_markers = layer_markers
        self.is_checkpoint_int8_serialized = layer_markers is not None
        self.checkpoint_uses_native_qkv_layout = self.is_checkpoint_int8_serialized
        self._serialized_group_sizes: dict[str, int] = {}
        if layer_markers is not None:
            for prefix, marker in layer_markers.items():
                if is_comfy_int8_embedding(marker):
                    continue
                if marker.get("format") != "int8_tensorwise":
                    raise ValueError(
                        f"Unsupported Comfy INT8 format for {prefix!r}: "
                        f"{marker.get('format')!r}"
                    )
                if marker.get("convrot") is not True:
                    raise ValueError(
                        f"Serialized convrot_int8 layer {prefix!r} must set "
                        "convrot=true"
                    )
                marker_group_size = marker.get("convrot_groupsize")
                if marker_group_size not in BACKEND_GROUP_SIZES[COMFY_KITCHEN]:
                    raise ValueError(
                        f"Serialized convrot_int8 layer {prefix!r} must declare "
                        f"convrot_groupsize in {BACKEND_GROUP_SIZES[COMFY_KITCHEN]}, "
                        f"got {marker_group_size!r}"
                    )
                self._serialized_group_sizes[prefix] = marker_group_size
        elif backend == "auto" and self.resolve_backend() is None:
            if self.uses_xpu:
                raise ValueError(
                    f"convrot_int8 backend xpu supports only group size "
                    f"{XPU_GROUP_SIZE}; got {group_size}. No comfy_kitchen/eager "
                    "fallback is used"
                )
            raise ValueError(
                f"convrot_int8: no backend on this machine serves group size "
                f"{group_size} (the JIT backend needs CC 9.0/10.0/12.0/12.1, a CUDA "
                f"toolchain and one of {BACKEND_GROUP_SIZES[JIT]}; comfy_kitchen serves "
                f"{BACKEND_GROUP_SIZES[COMFY_KITCHEN]})"
            )
        # Logged per backend at the end of loading: a silent fallback to BF16
        # or to the slower backend looks exactly like a slow kernel.
        self.selected: list[str] = []
        # Embeddings count as selected for the loader's marker check but never
        # report back through note_quantized / note_loaded.
        self.selected_embeddings: list[str] = []
        self.selected_by_backend: dict[str, list[str]] = {
            COMFY_KITCHEN: [],
            JIT: [],
            XPU: [],
        }
        self.skipped: list[str] = []
        self._processed = 0
        self._quantized_bytes = 0

    @classmethod
    def get_name(cls) -> str:
        return "convrot_int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        if current_platform.is_xpu():
            return [torch.bfloat16]
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # INT8 tensor cores land on Turing (comfy_kitchen); the JIT backend
        # checks its own table when it is selected.
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ConvRotInt8Config:
        return cls(
            group_size=cls.get_from_keys_or(config, ["group_size"], 256),
            ignored_layers=cls.get_from_keys_or(config, ["ignored_layers"], None),
            backend=cls.get_from_keys_or(config, ["backend"], None),
        )

    def resolve_backend(self, group_size: int | None = None) -> str | None:
        """Backend serving ``group_size`` (default: the online group size), or
        None when the requested backend has no kernel for it. ``auto`` selects
        native XPU ops on XPU, or prefers JIT on supported CUDA GPUs.
        An explicit backend is never substituted."""
        group_size = self.group_size if group_size is None else group_size
        if self.backend == "auto":
            if self.uses_xpu:
                return XPU if group_size in BACKEND_GROUP_SIZES[XPU] else None
            if group_size in BACKEND_GROUP_SIZES[JIT] and _jit_available():
                return JIT
            if group_size in BACKEND_GROUP_SIZES[COMFY_KITCHEN]:
                return COMFY_KITCHEN
            return None
        return self.backend if group_size in BACKEND_GROUP_SIZES[self.backend] else None

    def require_comfy_kitchen(self, reason: str) -> None:
        """Pin comfy_kitchen on CUDA; refuse native XPU and explicit JIT."""
        if self.uses_xpu:
            raise ValueError(
                f"convrot_int8 backend xpu {reason}; use TP and/or sequence "
                "parallelism instead. No comfy_kitchen fallback is used"
            )
        if self.backend == JIT:
            raise ValueError(
                f"convrot_int8 backend jit {reason}; use backend "
                "comfy_kitchen or TP and/or sequence parallelism instead"
            )
        if self.group_size not in BACKEND_GROUP_SIZES[COMFY_KITCHEN]:
            raise ValueError(
                f"convrot_int8 backend comfy_kitchen has no kernel for group size "
                f"{self.group_size}, and the jit backend {reason}"
            )
        if self.backend == "auto":
            logger.info(
                "convrot_int8: using the comfy_kitchen backend because %s", reason
            )
        self.backend = COMFY_KITCHEN

    def _backend_for(
        self, *, group_size: int, in_size: int, out_size: int
    ) -> str | None:
        backend = self.resolve_backend(group_size)
        if backend == XPU and not xpu_supports_shape(
            input_size=in_size, output_size=out_size
        ):
            return None
        if backend == JIT and out_size % 8:
            # The JIT GEMM epilogue stores 8 BF16 outputs at a time.
            if (
                self.backend == "auto"
                and group_size in BACKEND_GROUP_SIZES[COMFY_KITCHEN]
                and _comfy_kitchen_available()
            ):
                return COMFY_KITCHEN
            return None
        return backend

    def _no_kernel_reason(self, *, group_size: int, in_size: int, out_size: int) -> str:
        if self.uses_xpu:
            return (
                f"XPU requires ConvRot group {XPU_GROUP_SIZE}, a per-shard input "
                f"width in [{XPU_GROUP_SIZE}, {XPU_MAX_INPUT_SIZE}] divisible by "
                f"{XPU_GROUP_SIZE} and a positive per-shard output width "
                f"divisible by {XPU_OUTPUT_ALIGNMENT}; got group {group_size}, "
                f"{in_size} inputs, {out_size} outputs. No comfy_kitchen/eager "
                "fallback is used"
            )
        comfy = (
            "installed"
            if _comfy_kitchen_available()
            else "not installed (pip install comfy-kitchen)"
        )
        return (
            f"ConvRot group {group_size}, {out_size} outputs; backend {self.backend!r}; "
            f"jit serves group sizes {BACKEND_GROUP_SIZES[JIT]} with an "
            f"output width that is a multiple of 8, comfy_kitchen serves group sizes "
            f"{BACKEND_GROUP_SIZES[COMFY_KITCHEN]} and is {comfy}"
        )

    def _select(
        self, *, prefix: str, backend: str, group_size: int, serialized: bool
    ) -> QuantizeMethodBase:
        self.selected.append(prefix)
        self.selected_by_backend[backend].append(prefix)
        if backend == XPU:
            from sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_xpu import (
                ConvRotInt8XpuLinearMethod,
            )

            return ConvRotInt8XpuLinearMethod(
                self, group_size=group_size, is_checkpoint_serialized=serialized
            )
        if backend == JIT:
            from sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_jit import (
                ConvRotInt8JitLinearMethod,
            )

            return ConvRotInt8JitLinearMethod(
                self, group_size=group_size, is_checkpoint_serialized=serialized
            )
        from sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_comfy_kitchen import (
            ConvRotInt8ComfyKitchenLinearMethod,
        )

        return ConvRotInt8ComfyKitchenLinearMethod(
            self, group_size=group_size, is_checkpoint_serialized=serialized
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, VocabParallelEmbedding) and self.quantizes_embedding(
            prefix
        ):
            self.selected.append(prefix)
            self.selected_embeddings.append(prefix)
            return ComfyInt8EmbeddingMethod(
                tensorwise=bool(self.layer_markers[prefix].get("_is_tensorwise_scalar"))
            )
        if not isinstance(layer, LinearBase):
            return None
        # A column-parallel layer shards its output and a row-parallel one its
        # input; every rank sees the same per-shard widths, so the decisions
        # below are rank-consistent.
        out_size = (
            layer.output_size_per_partition
            if isinstance(layer, ColumnParallelLinear)
            else layer.output_size
        )
        in_size = getattr(layer, "input_size_per_partition", layer.input_size)
        if self.layer_markers is not None:
            marker_group_size = self._serialized_group_sizes.get(prefix)
            if marker_group_size is None:
                return UnquantizedLinearMethod()
            if layer.input_size % marker_group_size:
                raise ValueError(
                    f"Serialized convrot_int8 layer {prefix!r} has input size "
                    f"{layer.input_size}, which is not divisible by its "
                    f"ConvRot group size {marker_group_size}"
                )
            backend = self._backend_for(
                group_size=marker_group_size, in_size=in_size, out_size=out_size
            )
            if backend is None:
                raise ValueError(
                    f"Serialized convrot_int8 layer {prefix!r} has no kernel: "
                    + self._no_kernel_reason(
                        group_size=marker_group_size,
                        in_size=in_size,
                        out_size=out_size,
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
        backend = self._backend_for(
            group_size=self.group_size, in_size=in_size, out_size=out_size
        )
        if backend is None:
            if self.uses_xpu:
                logger.warning(
                    "convrot_int8: the XPU ops do not serve %s (per-shard in=%d, "
                    "out=%d); keeping it in BF16",
                    prefix,
                    in_size,
                    out_size,
                )
            self.skipped.append(
                f"{prefix}(in={in_size}, out={out_size})"
                if self.uses_xpu
                else f"{prefix}(out={out_size})"
            )
            return UnquantizedLinearMethod()
        return self._select(
            prefix=prefix, backend=backend, group_size=self.group_size, serialized=False
        )

    def note_quantized(self, saved_bytes: int) -> None:
        self._processed += 1
        self._quantized_bytes += saved_bytes
        if self._processed == len(self.selected) - len(self.selected_embeddings):
            logger.info(
                "convrot_int8: quantized %d linear layers (%.2f GiB of BF16 weights "
                "-> %.2f GiB INT8; jit %d, comfy_kitchen %d, xpu %d), "
                "left %d in BF16",
                self._processed,
                self._quantized_bytes / 1024**3,
                self._quantized_bytes / 2 / 1024**3,
                len(self.selected_by_backend[JIT]),
                len(self.selected_by_backend[COMFY_KITCHEN]),
                len(self.selected_by_backend[XPU]),
                len(self.skipped),
            )
            logger.debug("convrot_int8: layers left in BF16: %s", self.skipped)

    def note_loaded(self) -> None:
        """A serialized layer's INT8 weights are in place; logs the backend split once."""
        self._processed += 1
        if self._processed == len(self.selected) - len(self.selected_embeddings):
            logger.info(
                "convrot_int8: loaded %d serialized INT8 linear layers "
                "(jit %d, comfy_kitchen %d, xpu %d)",
                self._processed,
                len(self.selected_by_backend[JIT]),
                len(self.selected_by_backend[COMFY_KITCHEN]),
                len(self.selected_by_backend[XPU]),
            )

    def quantizes_embedding(self, prefix: str) -> bool:
        return self.layer_markers is not None and is_comfy_int8_embedding(
            self.layer_markers.get(prefix)
        )

    def supports_input_partition(
        self, prefix: str, input_size_per_partition: int
    ) -> bool:
        group_size = self.group_size
        if self.layer_markers is not None:
            marker_group_size = self._serialized_group_sizes.get(prefix)
            if marker_group_size is None:
                return True
            group_size = marker_group_size
        if self.uses_xpu:
            return group_size == XPU_GROUP_SIZE and xpu_supports_input_size(
                input_size_per_partition
            )
        return input_size_per_partition % group_size == 0
