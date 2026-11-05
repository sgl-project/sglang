"""KV cache storage capability of the DSA KV path on Ascend NPUs.

The configurator, the pool sizer, the device pool and the attention backend
read the layout decisions from here instead of re-deriving them from the dtype
and the SoC.
"""

import functools
import logging
from dataclasses import dataclass, replace
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_NPU_NON_ARCH35_KV_CACHE_DTYPES = (
    "bf16, bfloat16, int8, or auto with a non-FP8 checkpoint"
)
_NPU_ARCH35_KV_CACHE_DTYPES = "bf16, bfloat16, fp8_e4m3, or auto"

_NPU_NON_ARCH35_INDEXER_KV_DTYPES = "auto, bf16, or int8"
_NPU_ARCH35_INDEXER_KV_DTYPES = "auto, bf16, or fp8_e4m3"

# The vendor op that reads a quantized INT8 indexer K cache. It is registered by
# the CANN custom_transformer OPP package plus the custom_ops wheel, both of
# which the official NPU image installs at build time.
_INT8_INDEXER_OP = "torch.ops.custom.npu_quant_lightning_indexer"

# The vendor op's only registered head count; dsa_npu_indexer pads up to it.
_INT8_INDEXER_MAX_HEADS = 64

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
    requested_indexer_kv_dtype: Optional[str] = None,
    int8_indexer_op_available: bool = False,
    index_n_heads: Optional[int] = None,
    prefill_cp_enabled: bool = False,
    dcp_size: int = 1,
    mla_preprocess_enabled: bool = False,
    fia_nz_enabled: bool = False,
) -> PackedQuantKvCapability:
    capability = _resolve_dtype_capability(
        kv_cache_dtype=kv_cache_dtype,
        is_arch35=is_arch35,
        soc_name=soc_name,
        requested=requested,
    )
    if not capability.supported:
        return capability
    capability = _resolve_indexer_capability(
        capability=capability,
        is_arch35=is_arch35,
        soc_name=soc_name,
        requested=requested_indexer_kv_dtype,
        int8_indexer_op_available=int8_indexer_op_available,
    )
    if capability.supported and capability.indexer_kv_dtype == torch.int8:
        capability = _reject_int8_indexer_conflicts(
            capability=capability,
            index_n_heads=index_n_heads,
            prefill_cp_enabled=prefill_cp_enabled,
            dcp_size=dcp_size,
        )
    if not capability.supported or not capability.main_kv_packed:
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
            # The indexer K cache is a separate axis: an int8 main KV leaves it
            # unquantized, and only --dsa-indexer-kv-dtype moves it.
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


def _resolve_indexer_capability(
    *,
    capability: PackedQuantKvCapability,
    is_arch35: bool,
    soc_name: str,
    requested: Optional[str],
    int8_indexer_op_available: bool,
) -> PackedQuantKvCapability:
    kv_cache_dtype = capability.kv_cache_dtype
    if requested is None or requested == "auto":
        # auto follows --kv-cache-dtype, which _resolve_dtype_capability has
        # already answered: arch35 FP8 quantizes, everything else does not.
        return capability
    if requested == "bf16":
        return _unquantized_indexer(capability)
    if requested == "int8":
        if is_arch35:
            return _rejected(
                kv_cache_dtype=kv_cache_dtype,
                reason=(
                    "--dsa-indexer-kv-dtype int8 is not supported on an Ascend "
                    f"arch35 NPU ({soc_name}); use fp8_e4m3 for a quantized "
                    "indexer K cache there. Supported "
                    "--dsa-indexer-kv-dtype values on this device: "
                    f"{_NPU_ARCH35_INDEXER_KV_DTYPES}."
                ),
            )
        if not int8_indexer_op_available:
            return _rejected(
                kv_cache_dtype=kv_cache_dtype,
                reason=(
                    "--dsa-indexer-kv-dtype int8 needs the vendor operator "
                    f"{_INT8_INDEXER_OP} and its metadata operator. "
                    "Install the CANN custom_transformer OPP package and the "
                    "custom_ops wheel (both ship in the official NPU image), "
                    "or pass --dsa-indexer-kv-dtype bf16."
                ),
            )
        return replace(
            capability,
            indexer_quant=True,
            indexer_kv_dtype=torch.int8,
            # The vendor op only registers an FP16 indexer scale; the arch35
            # FP8 path is the one that uses FP32.
            indexer_scale_dtype=torch.float16,
        )
    if requested == "fp8_e4m3":
        if not is_arch35:
            return _rejected(
                kv_cache_dtype=kv_cache_dtype,
                reason=(
                    "--dsa-indexer-kv-dtype fp8_e4m3 is not supported on this "
                    f"NPU ({soc_name}): the quantized indexer operator only "
                    "registers an INT8 indexer K cache outside arch35. "
                    "Supported --dsa-indexer-kv-dtype values on this device: "
                    f"{_NPU_NON_ARCH35_INDEXER_KV_DTYPES}."
                ),
            )
        return replace(
            capability,
            indexer_quant=True,
            indexer_kv_dtype=torch.float8_e4m3fn,
            indexer_scale_dtype=torch.float32,
        )
    supported = (
        _NPU_ARCH35_INDEXER_KV_DTYPES
        if is_arch35
        else _NPU_NON_ARCH35_INDEXER_KV_DTYPES
    )
    return _rejected(
        kv_cache_dtype=kv_cache_dtype,
        reason=(
            f"--dsa-indexer-kv-dtype {requested} is not a known indexer K "
            f"cache dtype. Supported values on this device ({soc_name}): "
            f"{supported}."
        ),
    )


def _unquantized_indexer(
    capability: PackedQuantKvCapability,
) -> PackedQuantKvCapability:
    if not capability.indexer_quant and capability.indexer_kv_dtype == torch.bfloat16:
        return capability
    return replace(
        capability,
        indexer_quant=False,
        indexer_kv_dtype=_UNQUANTIZED_INDEXER_KV_DTYPE,
        indexer_scale_dtype=None,
    )


def _reject_int8_indexer_conflicts(
    *,
    capability: PackedQuantKvCapability,
    index_n_heads: Optional[int],
    prefill_cp_enabled: bool,
    dcp_size: int,
) -> PackedQuantKvCapability:
    if index_n_heads is not None and index_n_heads > _INT8_INDEXER_MAX_HEADS:
        return _rejected(
            kv_cache_dtype=capability.kv_cache_dtype,
            reason=(
                f"--dsa-indexer-kv-dtype int8 supports at most "
                f"{_INT8_INDEXER_MAX_HEADS} indexer heads: {_INT8_INDEXER_OP} "
                f"only registers {_INT8_INDEXER_MAX_HEADS} heads and fewer are "
                f"padded up to it, but this model has index_n_heads="
                f"{index_n_heads}. Pass --dsa-indexer-kv-dtype bf16."
            ),
        )
    if prefill_cp_enabled or dcp_size > 1:
        return _rejected(
            kv_cache_dtype=capability.kv_cache_dtype,
            reason=(
                "--dsa-indexer-kv-dtype int8 requires attention CP and DCP size one. "
                "Use --dsa-indexer-kv-dtype bf16 for the context-parallel indexer."
            ),
        )
    return capability


def _int8_indexer_op_available() -> bool:
    # Both vendor entry points are required by eager and graph execution.
    try:
        import custom_ops  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.ops.custom, "npu_quant_lightning_indexer") and hasattr(
        torch.ops.custom, "npu_quant_lightning_indexer_metadata"
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
    kv_cache_dtype: torch.dtype,
    requested: Optional[str] = None,
    *,
    is_draft_worker: bool = False,
) -> PackedQuantKvCapability:
    from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
        is_fia_nz,
        is_mla_preprocess_enabled,
    )
    from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
    from sglang.srt.runtime_context import get_model, get_parallel

    # One read point for the indexer switch: the sizer, the pool, the backend
    # and the dtype pipeline all come through here, and a call site that read
    # it on its own could size a buffer the pool does not allocate.
    requested_indexer_kv_dtype = get_model().dsa_indexer_kv_dtype
    # The INT8 indexer is hardware-validated for the target model only,
    # so a draft resolves it as auto and keeps the unquantized indexer.
    if is_draft_worker and requested_indexer_kv_dtype == "int8":
        requested_indexer_kv_dtype = "auto"
    # Only an int8 indexer reads these, so other requests never probe them.
    int8_requested = requested_indexer_kv_dtype == "int8"
    return resolve_kv_capability(
        kv_cache_dtype=kv_cache_dtype,
        is_arch35=is_npu_arch35(),
        soc_name=_get_npu_soc_name(),
        requested=requested,
        requested_indexer_kv_dtype=requested_indexer_kv_dtype,
        int8_indexer_op_available=int8_requested and _int8_indexer_op_available(),
        index_n_heads=_dsa_index_n_heads() if int8_requested else None,
        prefill_cp_enabled=int8_requested and _dsa_prefill_cp_enabled(),
        dcp_size=get_parallel().attn_dcp_size if int8_requested else 1,
        mla_preprocess_enabled=is_mla_preprocess_enabled(),
        fia_nz_enabled=is_fia_nz(),
    )


def _dsa_index_n_heads() -> Optional[int]:
    from sglang.srt.configs.model_config import get_dsa_index_n_heads, is_deepseek_dsa
    from sglang.srt.runtime_context import process_model_config

    hf_config = process_model_config().hf_config
    if not is_deepseek_dsa(hf_config):
        return None
    return get_dsa_index_n_heads(hf_config)


def _dsa_prefill_cp_enabled() -> bool:
    from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp

    return is_dsa_enable_prefill_cp()
