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

_NPU_NON_ARCH35_KV_CACHE_DTYPES = "bf16, bfloat16, or auto with a non-FP8 checkpoint"


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


def resolve_kv_capability(
    *,
    kv_cache_dtype: torch.dtype,
    is_arch35: bool,
    soc_name: str,
    requested: Optional[str] = None,
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
        return PackedQuantKvCapability(
            kv_cache_dtype=kv_cache_dtype,
            main_kv_packed=False,
            main_kv_scale_dtype=None,
            indexer_quant=False,
            indexer_kv_dtype=kv_cache_dtype,
            indexer_scale_dtype=None,
            unsupported_reason=(
                f"--kv-cache-dtype {requested} resolves to a {kv_cache_dtype} KV "
                f"cache, which is not supported on this NPU ({soc_name}): FP8 KV "
                "cache storage requires an Ascend arch35 NPU. Supported "
                "--kv-cache-dtype values on this device: "
                f"{_NPU_NON_ARCH35_KV_CACHE_DTYPES}."
            ),
        )
    return PackedQuantKvCapability(
        kv_cache_dtype=kv_cache_dtype,
        main_kv_packed=False,
        main_kv_scale_dtype=None,
        indexer_quant=False,
        indexer_kv_dtype=kv_cache_dtype,
        indexer_scale_dtype=None,
    )


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
    from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

    return resolve_kv_capability(
        kv_cache_dtype=kv_cache_dtype,
        is_arch35=is_npu_arch35(),
        soc_name=_get_npu_soc_name(),
        requested=requested,
    )
