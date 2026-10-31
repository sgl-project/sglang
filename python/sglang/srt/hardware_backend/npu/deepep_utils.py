"""Public DeepEP buffer options for the opt-in NPU all-to-all workaround."""

import inspect
from typing import Optional


def alltoall_buffer_kwargs(
    buffer_cls,
    *,
    enabled: bool,
    normal_enabled: bool,
    use_zbal: bool,
    deep_use_mode: Optional[str],
) -> dict:
    # ZBAL and low-latency-only buffers retain their own implementation. The
    # default path must also work with older DeepEP versions without strategies.
    if not enabled or use_zbal or not normal_enabled:
        return {}
    if "normal_strategy" not in inspect.signature(buffer_cls).parameters:
        raise ValueError(
            "SGLANG_USE_NPU_DEEPEP_ALLTOALL requires a DeepEP Buffer with the "
            "public normal_strategy parameter"
        )
    # Vendor Buffer applies DEEP_USE_MODE after its constructor arguments,
    # overriding both normal and low-latency strategies. Do not silently ignore
    # an explicit SGLang selection or change the low-latency implementation.
    if deep_use_mode is not None:
        raise ValueError(
            "Unset DEEP_USE_MODE when using SGLANG_USE_NPU_DEEPEP_ALLTOALL; "
            "the vendor environment overrides normal_strategy"
        )
    return {"normal_strategy": "alltoall"}
