import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.mem_cache.platform_kv_cache_dtype import (
    registered_kv_cache_storage_dtype,
)
from sglang.srt.platforms import current_platform
from sglang.srt.utils import is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

logger = logging.getLogger(__name__)

_is_hip = is_hip()

TORCH_DTYPE_TO_KV_CACHE_STR = {
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.float8_e4m3fnuz: "fp8_e4m3",
    torch.float8_e5m2: "fp8_e5m2",
    torch.bfloat16: "bf16",
}


def configure_kv_cache_dtype(
    *,
    server_args_kv_cache_dtype: str,
    quant_config: Optional["QuantizationConfig"],
    model_dtype: torch.dtype,
    is_draft_worker: bool,
    is_dflash: bool,
    speculative_draft_attention_backend: str,
    speculative_draft_kv_cache_dtype: Optional[str] = None,
) -> tuple[str, torch.dtype]:
    """(label, torch dtype) of a runner's KV cache; a pool in the compute
    dtype is labelled "auto", which backends gate FP8 casts and descales on."""
    if is_draft_worker and speculative_draft_kv_cache_dtype is not None:
        server_args_kv_cache_dtype = speculative_draft_kv_cache_dtype
    kv_cache_dtype_str = server_args_kv_cache_dtype
    if server_args_kv_cache_dtype == "auto":
        kv_cache_quant_algo = getattr(quant_config, "kv_cache_quant_algo", None)
        if (
            isinstance(kv_cache_quant_algo, str)
            and kv_cache_quant_algo.upper() == "FP8"
        ):
            kv_cache_dtype = fp8_dtype if _is_hip else torch.float8_e4m3fn
            kv_cache_dtype_str = TORCH_DTYPE_TO_KV_CACHE_STR[kv_cache_dtype]
        else:
            kv_cache_dtype = model_dtype
    elif server_args_kv_cache_dtype == "fp8_e5m2":
        if current_platform.is_cpu():
            raise ValueError("--kv-cache-dtype fp8_e5m2 is not supported on CPU.")
        if _is_hip:  # Using natively supported format
            kv_cache_dtype = fp8_dtype
        else:
            kv_cache_dtype = torch.float8_e5m2
    elif server_args_kv_cache_dtype == "fp8_e4m3":
        if _is_hip:  # Using natively supported format
            kv_cache_dtype = fp8_dtype
        else:
            kv_cache_dtype = torch.float8_e4m3fn
    elif server_args_kv_cache_dtype == "mxfp8":
        kv_cache_dtype = torch.float8_e4m3fn
    elif server_args_kv_cache_dtype in ("bf16", "bfloat16"):
        kv_cache_dtype = torch.bfloat16
    elif server_args_kv_cache_dtype in ("nvfp4", "fp4_mx_block16"):
        if hasattr(torch, "float4_e2m1fn_x2"):
            kv_cache_dtype = torch.float4_e2m1fn_x2
            logger.warning(
                "%s KV Cache might lead to an accuracy drop!",
                server_args_kv_cache_dtype.upper(),
            )
        else:
            raise ValueError(
                f"--kv-cache-dtype={server_args_kv_cache_dtype} requires "
                "torch.float4_e2m1fn_x2 support. Please use PyTorch 2.8.0+ "
                "with CUDA 12.8+."
            )
    elif (
        storage_dtype := registered_kv_cache_storage_dtype(server_args_kv_cache_dtype)
    ) is not None:
        kv_cache_dtype = storage_dtype
    else:
        raise ValueError(f"Unsupported kv_cache_dtype: {server_args_kv_cache_dtype}.")

    # DFLASH: fa4 draft attention can't read the target's fp8 KV (needs K.dtype == Q.dtype),
    # so give the fa4 draft its own compute-dtype KV. fp8-capable backends keep the target dtype.
    if (
        is_draft_worker
        and is_dflash
        and speculative_draft_attention_backend == "fa4"
        and kv_cache_dtype != model_dtype
    ):
        logger.info(
            "DFLASH fa4 draft: overriding KV cache dtype %s -> %s "
            "(fa4 needs K.dtype == Q.dtype; cannot read the target's quantized KV).",
            kv_cache_dtype,
            model_dtype,
        )
        kv_cache_dtype = model_dtype
        # "auto" is the tag for an unquantized pool; backends gate descale on it.
        kv_cache_dtype_str = "auto"

    return kv_cache_dtype_str, kv_cache_dtype
