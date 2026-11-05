"""KV cache storage capability of the DSA KV path on Ascend NPUs.

The configurator, the pool sizer, the device pool and the attention backend
read the layout decisions from here instead of re-deriving them from the dtype
and the SoC.
"""

import functools
import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_NPU_NON_ARCH35_KV_CACHE_DTYPES = (
    "bf16, bfloat16, int8, or auto with a non-FP8 checkpoint"
)
_NPU_ARCH35_KV_CACHE_DTYPES = "bf16, bfloat16, fp8_e4m3, or auto"

# An unquantized indexer K cache keeps the dtype the sparse attention ops read
# queries in, which is BF16 on every packed path. It must not follow a packed
# one-byte main KV dtype, or bf16 index_k would be cast to it on write.
_UNQUANTIZED_INDEXER_KV_DTYPE = torch.bfloat16


@dataclass(frozen=True)
class PackedQuantKvCapability:
    kv_cache_dtype: torch.dtype
    # Main MLA KV is stored as one packed record per token:
    # quantized latent | bf16 rope | per-128-tile scales.
    main_kv_packed: bool
    main_kv_scale_dtype: Optional[torch.dtype]
    # The DSA indexer K cache is quantized with a per-token scale buffer.
    indexer_quant: bool
    indexer_kv_dtype: torch.dtype
    indexer_scale_dtype: Optional[torch.dtype]
    unsupported_reason: Optional[str] = None

    @property
    def supported(self) -> bool:
        return self.unsupported_reason is None


def _rejected(*, kv_cache_dtype: torch.dtype, reason: str) -> PackedQuantKvCapability:
    return PackedQuantKvCapability(
        kv_cache_dtype=kv_cache_dtype,
        main_kv_packed=False,
        main_kv_scale_dtype=None,
        indexer_quant=False,
        indexer_kv_dtype=kv_cache_dtype,
        indexer_scale_dtype=None,
        unsupported_reason=reason,
    )


def resolve_kv_capability(
    *,
    kv_cache_dtype: torch.dtype,
    is_arch35: bool,
    soc_name: str,
    requested: Optional[str] = None,
    mla_preprocess_enabled: bool = False,
    fia_nz_enabled: bool = False,
) -> PackedQuantKvCapability:
    capability = _resolve_dtype_capability(
        kv_cache_dtype=kv_cache_dtype,
        is_arch35=is_arch35,
        soc_name=soc_name,
        requested=requested,
    )
    if not capability.main_kv_packed:
        return capability
    return _reject_packed_kv_conflicts(
        capability=capability,
        is_arch35=is_arch35,
        mla_preprocess_enabled=mla_preprocess_enabled,
        fia_nz_enabled=fia_nz_enabled,
    )


def _resolve_dtype_capability(
    *,
    kv_cache_dtype: torch.dtype,
    is_arch35: bool,
    soc_name: str,
    requested: Optional[str],
) -> PackedQuantKvCapability:
    if kv_cache_dtype == torch.float8_e4m3fn:
        if is_arch35:
            return PackedQuantKvCapability(
                kv_cache_dtype=kv_cache_dtype,
                main_kv_packed=True,
                main_kv_scale_dtype=torch.float32,
                indexer_quant=True,
                indexer_kv_dtype=kv_cache_dtype,
                indexer_scale_dtype=torch.float32,
            )
        requested = requested or "fp8_e4m3"
        return _rejected(
            kv_cache_dtype=kv_cache_dtype,
            reason=(
                f"--kv-cache-dtype {requested} resolves to a {kv_cache_dtype} KV "
                f"cache, which is not supported on this NPU ({soc_name}): FP8 KV "
                "cache storage requires an Ascend arch35 NPU. Supported "
                "--kv-cache-dtype values on this device: "
                f"{_NPU_NON_ARCH35_KV_CACHE_DTYPES}."
            ),
        )
    if kv_cache_dtype == torch.int8:
        if is_arch35:
            requested = requested or "int8"
            return _rejected(
                kv_cache_dtype=kv_cache_dtype,
                reason=(
                    f"--kv-cache-dtype {requested} is not supported on an Ascend "
                    f"arch35 NPU ({soc_name}); use fp8_e4m3 for a packed KV cache "
                    "there. Supported --kv-cache-dtype values on this device: "
                    f"{_NPU_ARCH35_KV_CACHE_DTYPES}."
                ),
            )
        return PackedQuantKvCapability(
            kv_cache_dtype=kv_cache_dtype,
            main_kv_packed=True,
            main_kv_scale_dtype=torch.float32,
            # The indexer K cache is a separate axis: it stays unquantized here
            # and keeps the dtype the sparse attention op reads queries in.
            indexer_quant=False,
            indexer_kv_dtype=_UNQUANTIZED_INDEXER_KV_DTYPE,
            indexer_scale_dtype=None,
        )
    return PackedQuantKvCapability(
        kv_cache_dtype=kv_cache_dtype,
        main_kv_packed=False,
        main_kv_scale_dtype=None,
        indexer_quant=False,
        indexer_kv_dtype=kv_cache_dtype,
        indexer_scale_dtype=None,
    )


def _reject_packed_kv_conflicts(
    *,
    capability: PackedQuantKvCapability,
    is_arch35: bool,
    mla_preprocess_enabled: bool,
    fia_nz_enabled: bool,
) -> PackedQuantKvCapability:
    """Startup rejections for the write paths that cannot emit packed records."""
    if fia_nz_enabled:
        return _rejected(
            kv_cache_dtype=capability.kv_cache_dtype,
            reason=(
                f"A packed {capability.kv_cache_dtype} KV cache only supports the "
                "PA_BSND layout, so it cannot be combined with SGLANG_USE_FIA_NZ. "
                "Unset SGLANG_USE_FIA_NZ, or pick an unquantized --kv-cache-dtype."
            ),
        )
    # MLAProlog writes packed INT8 as well as FP8 records. Weight compatibility
    # and legacy preprocess paths are checked by NPUFusedMLAPreprocess, where
    # the loaded projection methods are available.
    return capability


@functools.lru_cache(maxsize=1)
def _get_npu_soc_name() -> str:
    try:
        return str(torch.npu.get_device_name())
    except Exception:
        logger.debug("Failed to query the NPU SoC name", exc_info=True)
        return "unknown SoC"


def resolve_npu_kv_capability(
    kv_cache_dtype: torch.dtype, requested: Optional[str] = None
) -> PackedQuantKvCapability:
    from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
        is_fia_nz,
        is_mla_preprocess_enabled,
    )
    from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

    return resolve_kv_capability(
        kv_cache_dtype=kv_cache_dtype,
        is_arch35=is_npu_arch35(),
        soc_name=_get_npu_soc_name(),
        requested=requested,
        mla_preprocess_enabled=is_mla_preprocess_enabled(),
        fia_nz_enabled=is_fia_nz(),
    )
