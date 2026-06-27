import logging

import torch
from torch import nn

from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.srt.utils import is_hip

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
    model: nn.Module,
    model_dtype: torch.dtype,
) -> tuple[str, torch.dtype]:
    if server_args_kv_cache_dtype == "auto":
        quant_config = getattr(model, "quant_config", None)
        kv_cache_quant_algo = getattr(quant_config, "kv_cache_quant_algo", None)
        if (
            isinstance(kv_cache_quant_algo, str)
            and kv_cache_quant_algo.upper() == "FP8"
        ):
            if _is_hip:
                kv_cache_dtype = fp8_dtype
                server_args_kv_cache_dtype = TORCH_DTYPE_TO_KV_CACHE_STR[kv_cache_dtype]
            else:
                kv_cache_dtype = torch.float8_e4m3fn
                server_args_kv_cache_dtype = TORCH_DTYPE_TO_KV_CACHE_STR[kv_cache_dtype]
        else:
            kv_cache_dtype = model_dtype
    elif server_args_kv_cache_dtype == "fp8_e5m2":
        if _is_hip:  # Using natively supported format
            kv_cache_dtype = fp8_dtype
        else:
            kv_cache_dtype = torch.float8_e5m2
    elif server_args_kv_cache_dtype == "fp8_e4m3":
        if _is_hip:  # Using natively supported format
            kv_cache_dtype = fp8_dtype
        else:
            kv_cache_dtype = torch.float8_e4m3fn
    elif server_args_kv_cache_dtype in ("bf16", "bfloat16"):
        kv_cache_dtype = torch.bfloat16
    elif server_args_kv_cache_dtype == "fp4_e2m1":
        if hasattr(torch, "float4_e2m1fn_x2"):
            kv_cache_dtype = torch.float4_e2m1fn_x2
            logger.warning(f"FP4 (E2M1) KV Cache might lead to a accuracy drop!")
        else:
            logger.warning(
                f"--kv-cache-dtype falls back to 'auto' because this torch version does not support torch.float4_e2m1fn_x2"
            )
            kv_cache_dtype = model_dtype
    else:
        raise ValueError(f"Unsupported kv_cache_dtype: {server_args_kv_cache_dtype}.")
    return server_args_kv_cache_dtype, kv_cache_dtype
