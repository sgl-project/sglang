"""KV cache storage capability of the DSA KV path on Ascend NPUs.

The configurator and the device pool read the layout decisions from here
instead of re-deriving them from the dtype and the SoC; the attention backend
follows the device pool.
Pools without a DSA indexer (dense MLA, ModelSlim KV layouts) keep their own
layouts and do not consult it.
"""

import functools
import logging
from typing import Optional

import msgspec
import torch

logger = logging.getLogger(__name__)

_NPU_NON_ARCH35_KV_CACHE_DTYPES = "bf16, bfloat16, or auto with a non-FP8 checkpoint"


class PackedQuantKvCapability(msgspec.Struct, frozen=True, kw_only=True):
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
        return PackedQuantKvCapability(
            kv_cache_dtype=kv_cache_dtype,
            main_kv_packed=False,
            main_kv_scale_dtype=None,
            indexer_quant=False,
            indexer_kv_dtype=kv_cache_dtype,
            indexer_scale_dtype=None,
            unsupported_reason=(
                f"DSA models cannot use a {kv_cache_dtype} KV cache on this NPU "
                f"({soc_name}): the packed FP8 DSA KV record requires an Ascend "
                "arch35 NPU. Supported --kv-cache-dtype values for DSA models on "
                f"this device: {_NPU_NON_ARCH35_KV_CACHE_DTYPES}."
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


def resolve_npu_kv_capability(kv_cache_dtype: torch.dtype) -> PackedQuantKvCapability:
    from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

    return resolve_kv_capability(
        kv_cache_dtype=kv_cache_dtype,
        is_arch35=is_npu_arch35(),
        soc_name=_get_npu_soc_name(),
    )
