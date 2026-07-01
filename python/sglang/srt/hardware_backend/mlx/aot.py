"""AOT kernel selection and decode-context helpers for the MLX backend."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

import mlx.core as mx

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_cache import (
        ContiguousAttentionKVCache,
    )


def _load_metal_rope_pool_fused():
    try:
        from sgl_kernel import metal  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "sgl_kernel.metal is not importable. Install sgl-kernel in the "
            "active environment before enabling SGLANG_MLX_USE_CUSTOM_ROPE."
        ) from exc

    import_error = getattr(metal, "_IMPORT_ERROR", None)
    if getattr(metal, "_metal", None) is None or import_error is not None:
        reason = f" Reason: {import_error}." if import_error is not None else ""
        raise ImportError(
            "sgl_kernel.metal is importable, but the native Metal extension "
            f"or metallib is not available.{reason} Install the Metal kernels "
            "with `uv run sgl-kernel/setup_metal.py install` from the SGLang "
            "repo root in the active environment."
        ) from import_error
    return metal.rope_pool_fused


def _load_metal_paged_attention_decode():
    try:
        from sgl_kernel import metal  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "sgl_kernel.metal is not importable. Install sgl-kernel in the "
            "active environment before enabling SGLANG_MLX_USE_PAGED_ATTENTION."
        ) from exc

    import_error = getattr(metal, "_IMPORT_ERROR", None)
    if getattr(metal, "_metal", None) is None or import_error is not None:
        reason = f" Reason: {import_error}." if import_error is not None else ""
        raise ImportError(
            "sgl_kernel.metal is importable, but the native Metal extension "
            f"or metallib is not available.{reason} Install the Metal kernels "
            "with `uv run sgl-kernel/setup_metal.py install` from the SGLang "
            "repo root in the active environment."
        ) from import_error
    return metal.paged_attention_decode


@dataclass
class MlxAOTRoPEKernel:
    base: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)
    rope_pool_fused: Optional[Any] = None

    @property
    def enabled(self) -> bool:
        return (
            self.base > 0.0 and bool(self.config) and self.rope_pool_fused is not None
        )


@dataclass
class MlxAOTPagedAttentionKernel:
    paged_attention_decode: Optional[Any] = None

    @property
    def enabled(self) -> bool:
        return self.paged_attention_decode is not None


@dataclass
class MlxAOTKernelBuildInputs:
    sample_attn: Any
    n_kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class MlxAOTKernelSpec:
    name: str
    kernel_attr: str
    is_enabled: Callable[[], bool]
    build: Callable[[MlxAOTKernelBuildInputs], Any]


@dataclass
class MlxAOTKernelSet:
    rope: MlxAOTRoPEKernel = field(default_factory=MlxAOTRoPEKernel)
    paged_attention: MlxAOTPagedAttentionKernel = field(
        default_factory=MlxAOTPagedAttentionKernel
    )
    selected_kernel_names: tuple[str, ...] = ()


class MlxAOTKernelRegistry:
    """Registry for optional MLX AOT kernels.

    Each spec owns one kernel field on ``MlxAOTKernelSet``. The registry is the
    only place that checks kernel enablement policy and model support.
    """

    def __init__(self, specs: tuple[MlxAOTKernelSpec, ...]):
        self._specs = specs

    @property
    def registered_kernel_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._specs)

    def build_kernel_set(
        self,
        *,
        sample_attn: Any,
        n_kv_heads: int,
        head_dim: int,
    ) -> MlxAOTKernelSet:
        inputs = MlxAOTKernelBuildInputs(
            sample_attn=sample_attn,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )
        kernel_set = MlxAOTKernelSet()
        selected_kernel_names = []
        for spec in self._specs:
            if not spec.is_enabled():
                continue
            kernel = spec.build(inputs)
            if getattr(kernel, "enabled", False):
                if not hasattr(kernel_set, spec.kernel_attr):
                    raise ValueError(
                        f"AOT kernel {spec.name} targets unknown kernel-set "
                        f"attribute {spec.kernel_attr}"
                    )
                setattr(kernel_set, spec.kernel_attr, kernel)
                selected_kernel_names.append(spec.name)
        kernel_set.selected_kernel_names = tuple(selected_kernel_names)
        if kernel_set.selected_kernel_names:
            logger.info(
                "MLX AOT kernels selected: %s",
                ", ".join(kernel_set.selected_kernel_names),
            )
        return kernel_set


def _build_rope_kernel(inputs: MlxAOTKernelBuildInputs) -> MlxAOTRoPEKernel:
    from sglang.srt.hardware_backend.mlx.kv_cache.attention_contract import (
        get_num_heads,
    )

    sample_attn = getattr(inputs.sample_attn, "_inner", inputs.sample_attn)
    rope = getattr(sample_attn, "rope", None)
    if rope is None or getattr(rope, "traditional", False):
        return MlxAOTRoPEKernel()

    rope_dim = int(getattr(rope, "dims", 0))
    if rope_dim == 0:
        return MlxAOTRoPEKernel()
    if rope_dim != inputs.head_dim:
        # AOT kernel currently requires rope_dim == head_dim.
        return MlxAOTRoPEKernel()

    base = float(getattr(rope, "base", 10000.0))
    num_qo_heads = get_num_heads(sample_attn)
    if num_qo_heads is None:
        return MlxAOTRoPEKernel()
    config = {
        "head_dim": int(inputs.head_dim),
        "rope_dim": rope_dim,
        "num_qo_heads": int(num_qo_heads),
        "num_kv_heads": int(inputs.n_kv_heads),
    }
    try:
        rope_pool_fused = _load_metal_rope_pool_fused()
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "AOT Metal RoPE kernel not available (%s) - falling back to "
            "mx.fast.rope.",
            exc,
        )
        return MlxAOTRoPEKernel()

    logger.info(
        f"AOT Metal RoPE kernel ENABLED: head_dim={inputs.head_dim}, "
        f"n_heads={config['num_qo_heads']}, n_kv={config['num_kv_heads']}, "
        f"base={base}"
    )
    return MlxAOTRoPEKernel(
        base=base,
        config=config,
        rope_pool_fused=rope_pool_fused,
    )


def _build_paged_attention_kernel(
    inputs: MlxAOTKernelBuildInputs,
) -> MlxAOTPagedAttentionKernel:
    if inputs.head_dim > 128:
        return MlxAOTPagedAttentionKernel()
    try:
        paged_attention_decode = _load_metal_paged_attention_decode()
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "AOT Metal paged attention kernel not available (%s) - falling "
            "back to padded SDPA.",
            exc,
        )
        return MlxAOTPagedAttentionKernel()

    logger.info(
        f"AOT Metal paged attention decode ENABLED: head_dim={inputs.head_dim}, "
        f"n_kv={inputs.n_kv_heads}"
    )
    return MlxAOTPagedAttentionKernel(
        paged_attention_decode=paged_attention_decode,
    )


MLX_AOT_KERNEL_REGISTRY = MlxAOTKernelRegistry(
    specs=(
        MlxAOTKernelSpec(
            name="metal_rope_pool_fused",
            kernel_attr="rope",
            is_enabled=lambda: envs.SGLANG_MLX_USE_CUSTOM_ROPE.get()
            or envs.SGLANG_MLX_USE_PAGED_ATTENTION.get(),
            build=_build_rope_kernel,
        ),
        MlxAOTKernelSpec(
            name="metal_paged_attention_decode",
            kernel_attr="paged_attention",
            is_enabled=lambda: envs.SGLANG_MLX_USE_PAGED_ATTENTION.get(),
            build=_build_paged_attention_kernel,
        ),
    )
)


@dataclass
class MlxAOTRoPEContext:
    kernel: MlxAOTRoPEKernel
    kv_pool: Any
    new_token_slots: Optional[mx.array] = None


@dataclass
class MlxAOTPagedAttentionContext:
    kernel: MlxAOTPagedAttentionKernel
    kv_pool: Any


@dataclass
class MlxAOTKernelContext:
    rope: Optional[MlxAOTRoPEContext] = None
    paged_attention: Optional[MlxAOTPagedAttentionContext] = None

    @classmethod
    def from_decode(
        cls,
        *,
        aot_kernels: MlxAOTKernelSet,
        kv_pool: Any | None,
        req_ids: list[str],
        req_pool_idx: dict[str, int],
        req_to_token_pool: Any | None,
        layer_caches: list[list[ContiguousAttentionKVCache]],
    ) -> MlxAOTKernelContext:
        """Build optional AOT context for one batched decode step."""
        if kv_pool is None:
            return cls()

        new_token_slots = None
        if aot_kernels.rope.enabled and req_to_token_pool is not None:
            try:
                slot_ids = []
                for req_idx, req_id in enumerate(req_ids):
                    pool_idx = req_pool_idx.get(req_id)
                    if pool_idx is None:
                        raise KeyError(req_id)
                    slot = int(
                        req_to_token_pool.req_to_token[
                            pool_idx, layer_caches[0][req_idx].offset
                        ].item()
                    )
                    slot_ids.append(slot)
                new_token_slots = mx.array(slot_ids, dtype=mx.int32)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "AOT RoPE: failed to resolve new-token slots (%s); "
                    "falling back to RoPE-only for this decode step",
                    exc,
                )

        rope_ctx = None
        if aot_kernels.rope.enabled:
            rope_ctx = MlxAOTRoPEContext(
                kernel=aot_kernels.rope,
                kv_pool=kv_pool,
                new_token_slots=new_token_slots,
            )

        paged_ctx = None
        if aot_kernels.paged_attention.enabled:
            paged_ctx = MlxAOTPagedAttentionContext(
                kernel=aot_kernels.paged_attention,
                kv_pool=kv_pool,
            )

        return cls(rope=rope_ctx, paged_attention=paged_ctx)
