from __future__ import annotations

from dataclasses import dataclass

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import (
    get_device_sm,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    is_xpu,
)


@dataclass
class AttentionCapabilities:
    """
    This class is used to store the capabilities of an attention backend to validate the runtime
    configuration and raise errors if the configuration is invalid or not supported.
    In the future, all backends should be registered here instead of the current server arguments validation.
    This is to avoid the server args validation from becoming too complex and difficult to maintain.
    """

    # Supported hardware, subset of ["cuda", "hip", "cpu", "xpu", "npu", "hpu"]
    hardware: list[str]
    # List of supported major versions of SM capability if using Nvidia GPU
    sm_capability_major: list[int] | None = None
    # Whether the backend supports page size > 1
    # If set as a list of int, it means the backend only supports the page sizes in the list
    allowed_page_sizes_gt_1: bool | list[int] = False
    # Cuda graph support
    cuda_graph: bool = False
    # Speculative decoding support (whether topk = 1 or topk > 1)
    spec: bool = False
    # Speculative decoding support for topk > 1
    spec_topk_gt_1: bool = False
    # Sliding window support (SWA)
    sliding_window: bool = False
    # MLA support
    mla: bool = False
    # DP attention support. This depends on the MLA support.
    dp_attention: bool = False
    # Chunked prefix cache support. This depends on the MLA support.
    chunked_prefix_cache: bool = False
    # Supported kv cache dtypes (< bf16 precision)
    kv_cache_dtype: list[str] | None = None
    # Deterministic
    deterministic: bool = False


ATTN_BACKEND_CAPS: dict[str, AttentionCapabilities] = {
    "trtllm_mha": AttentionCapabilities(
        hardware=["cuda"],
        sm_capability_major=[10],
        allowed_page_sizes_gt_1=[16, 32, 64],
        cuda_graph=True,
        spec=True,
        spec_topk_gt_1=False,
        sliding_window=True,
        mla=False,
        dp_attention=False,
        chunked_prefix_cache=False,
        kv_cache_dtype=["fp8_e4m3", "bfloat16"],
        deterministic=False,
    ),
}


def _current_hardware() -> str:
    if is_cuda():
        return "cuda"
    if is_hip():
        return "hip"
    if is_xpu():
        return "xpu"
    if is_npu():
        return "npu"
    if is_cpu():
        return "cpu"
    raise RuntimeError(
        "Unknown hardware platform. Please check if your device environment is properly configured."
    )


def _validate_single_backend(
    backend_name: str,
    role: str,
    server_args: ServerArgs,
    use_mla: bool,
    sliding_window_size: int | None,
):
    if backend_name is None:
        return

    # TODO(brayden): Enable validation for additional backends.
    if backend_name != "trtllm_mha":
        return

    if backend_name not in ATTN_BACKEND_CAPS:
        raise KeyError(
            f"Attention backend '{backend_name}' is not registered. "
            f"Available backends are: {list(ATTN_BACKEND_CAPS.keys())}"
        )

    caps = ATTN_BACKEND_CAPS[backend_name]

    # Hardware checks
    hw = _current_hardware()
    if hw not in caps.hardware:
        raise RuntimeError(f"{backend_name} is not supported on {hw}.")

    # CUDA SM checks.
    if hw == "cuda" and caps.sm_capability_major is not None:
        major = get_device_sm() // 10
        if major not in caps.sm_capability_major:
            raise RuntimeError(
                f"{backend_name} is not supported on CUDA with SM{major}."
            )

    if server_args.page_size and server_args.page_size > 1:
        if isinstance(caps.allowed_page_sizes_gt_1, list):
            if server_args.page_size not in caps.allowed_page_sizes_gt_1:
                raise RuntimeError(
                    f"Page size {server_args.page_size} is not supported for {backend_name}. "
                    f"It should be one of {caps.allowed_page_sizes_gt_1}."
                )
        elif not caps.allowed_page_sizes_gt_1:
            raise RuntimeError(f"Page size > 1 is not supported for {backend_name}.")

    # CUDA graph checks
    if hw == "cuda" and not server_args.disable_cuda_graph:
        if not caps.cuda_graph:
            raise RuntimeError(f"{backend_name} does not support CUDA graph.")

    # Speculative decoding (decode role)
    if server_args.speculative_algorithm is not None and role == "decode":
        if not caps.spec:
            raise RuntimeError(f"{backend_name} does not support speculative decoding.")
        if getattr(server_args, "speculative_eagle_topk", 1) > 1:
            if not caps.spec_topk_gt_1:
                raise RuntimeError(
                    f"{backend_name} does not support speculative decoding with topk > 1."
                )

    # Sliding window
    if sliding_window_size is not None and sliding_window_size > 0:
        if not caps.sliding_window:
            raise RuntimeError(
                f"{backend_name} does not support sliding window attention."
            )

    # MLA usage
    if use_mla and not caps.mla:
        raise RuntimeError(f"{backend_name} does not support MLA models.")

    # DP Attention
    if server_args.enable_dp_attention and not caps.dp_attention:
        raise RuntimeError(f"{backend_name} does not support DP attention.")

    # Chunked prefix cache
    if not server_args.disable_chunked_prefix_cache and not caps.chunked_prefix_cache:
        raise RuntimeError(f"{backend_name} does not support chunked prefix cache.")

    # KV cache dtype validation
    # Policy:
    # - Always allow 'auto' and bfloat16 as a baseline that all backends support.
    # - For custom dtypes (e.g., fp8_*), require backend to explicitly list support.
    req_dtype = server_args.kv_cache_dtype
    if req_dtype == "bf16":
        req_dtype = "bfloat16"
    if req_dtype != "auto":
        if caps.kv_cache_dtype is None:
            raise RuntimeError(
                f"{backend_name} does not support custom kv cache dtype '{server_args.kv_cache_dtype}' (only 'auto' or bfloat16)."
            )
        if req_dtype not in caps.kv_cache_dtype:
            raise RuntimeError(
                f"{backend_name} does not support kv cache dtype '{server_args.kv_cache_dtype}'. Supported: {sorted(caps.kv_cache_dtype)}."
            )


def validate_attention_backends(
    server_args: ServerArgs,
    use_mla: bool,
    sliding_window_size: int | None,
):
    prefill_backend, decode_backend = server_args.get_attention_backends()
    _validate_single_backend(
        prefill_backend,
        role="prefill",
        server_args=server_args,
        use_mla=use_mla,
        sliding_window_size=sliding_window_size,
    )
    _validate_single_backend(
        decode_backend,
        role="decode",
        server_args=server_args,
        use_mla=use_mla,
        sliding_window_size=sliding_window_size,
    )
