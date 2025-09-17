from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_cpu, is_cuda, is_hip, is_npu, is_xpu


@dataclass
class AttentionFeatureList:
    # Supported hardware, should be a subset of ['cuda', 'hip', 'cpu', 'xpu', 'npu', 'hpu']
    hardware: List[str]
    # List of supported major versions of SM capability if using Nvidia GPU
    sm_capability_major: Optional[List[int]] = None
    # Whether the backend supports page size > 1
    # If set as a list of int, it means the backend only supports the page sizes in the list
    page_size_gt_1: Union[bool, List[int]] = False
    # Cuda graph support
    cuda_graph: bool = False
    # Speculative decoding support (whether topk = 1 or topk > 1)
    # Ref: https://docs.sglang.ai/advanced_features/speculative_decoding.html
    spec: bool = False
    # Speculative decoding support for topk > 1
    spec_topk_gt_1: bool = False
    # Sliding window support(SWA)
    sliding_window: bool = False
    # MLA(Multi-head Latent Attention) support
    mla: bool = False
    # Dp attention support, only targeted MLA models
    # Ref: https://docs.sglang.ai/basic_usage/deepseek.html#data-parallelism-attention
    dp_attention: bool = False
    # Chunked prefix cache support, only targeted MLA models
    # Ref: https://github.com/sgl-project/sglang/pull/5113
    chunked_prefix_cache: bool = False
    # Support of kv cache dtypes with precision lower than 16 bits
    kv_cache_dtype: Optional[List[str]] = None


# TODO: add more backends
SUPPORT_MATRIX = {
    # Common
    "triton": AttentionFeatureList(
        hardware=["cuda", "hip", "cpu", "xpu", "npu", "hpu"]
    ),
    "torch_native": AttentionFeatureList(
        hardware=["cuda", "hip", "cpu", "xpu", "npu", "hpu"]
    ),
    # NVIDIA specific
    "cutlass_mla": AttentionFeatureList(hardware=["cuda"]),
    "fa3": AttentionFeatureList(
        hardware=["cuda"],
        sm_capability_major=[8, 9],
        page_size_gt_1=True,
        cuda_graph=True,
        spec=True,
        spec_topk_gt_1=True,
        mla=True,
        sliding_window=True,
        dp_attention=True,
        chunked_prefix_cache=True,
        kv_cache_dtype=["fp8_e4m3"],
    ),
    "flashinfer": AttentionFeatureList(hardware=["cuda"]),
    "flashmla": AttentionFeatureList(hardware=["cuda"]),
    "trtllm_mla": AttentionFeatureList(hardware=["cuda"]),
    "trtllm_mha": AttentionFeatureList(hardware=["cuda"]),
    "dual_chunk_flash_attn": AttentionFeatureList(hardware=["cuda"]),
    # AMD specific
    "aiter": AttentionFeatureList(hardware=["hip"]),
    "wave": AttentionFeatureList(hardware=["hip"]),
    # Other platforms
    "intel_amx": AttentionFeatureList(hardware=["cpu"]),
    "ascend": AttentionFeatureList(hardware=["npu"]),
}


def check_attention_backend_support(
    server_args: ServerArgs,
    use_mla: bool = False,
    sliding_window_size: Optional[int] = None,
):
    """
    This function is mainly for banning unsupported or unvalidated combination of attention backend and features.
    When calling this function, we assume that the attention backend is already set by model_specific_adjustment in model_runner.py.
    """
    # TODO: Add some extra functions for checking, e.g. topk > 1 and page size > 1 ? They can be fetched from server_args.py or model_runner.py.
    attention_backend = server_args.attention_backend
    assert (
        attention_backend in SUPPORT_MATRIX
    ), f"Attention backend {attention_backend} is not supported."

    feature_support_list = SUPPORT_MATRIX[attention_backend]

    # Check hardware
    if is_cuda():
        assert (
            "cuda" in feature_support_list.hardware
        ), f"{attention_backend} is not supported on CUDA."
        assert (
            feature_support_list.sm_capability_major is not None
        ), f"{attention_backend} is not supported on CUDA."
        cuda_capability = torch.cuda.get_device_capability()
        assert (
            cuda_capability[0] in feature_support_list.sm_capability_major
        ), f"{attention_backend} is not supported on CUDA with capability SM{cuda_capability}"
    elif is_hip():
        assert (
            "hip" in feature_support_list.hardware
        ), f"{attention_backend} is not supported on HIP."
    elif is_xpu():
        assert (
            "xpu" in feature_support_list.hardware
        ), f"{attention_backend} is not supported on XPU."
    elif is_npu():
        assert (
            "npu" in feature_support_list.hardware
        ), f"{attention_backend} is not supported on NPU."
    elif is_cpu():
        assert (
            "cpu" in feature_support_list.hardware
        ), f"{attention_backend} is not supported on CPU."
    else:
        raise ValueError(f"Unsupported hardware: {attention_backend}")

    # Check page size
    page_size = server_args.page_size
    if page_size > 1:
        if isinstance(feature_support_list.page_size_gt_1, list):
            assert (
                page_size in feature_support_list.page_size_gt_1
            ), f"Page size {page_size} is not supported for {attention_backend}. It should be one of {feature_list.page_size_gt_1}."
        else:
            assert (
                feature_support_list.page_size_gt_1
            ), f"Page size > 1 is not supported for {attention_backend}."

    # Check cuda graph
    if is_cuda() and not server_args.disable_cuda_graph:
        assert (
            feature_support_list.cuda_graph
        ), f"{attention_backend} doesn't support cuda graph on CUDA."

    # Check speculative decoding
    if server_args.speculative_algorithm is not None:
        assert (
            feature_support_list.spec
        ), f"{attention_backend} doesn't support speculative decoding."
        if server_args.speculative_eagle_topk > 1:
            assert (
                feature_support_list.spec_topk_gt_1
            ), f"{attention_backend} doesn't support speculative decoding with topk > 1."

    # Check sliding window
    if sliding_window_size is not None and sliding_window_size > 0:
        assert (
            feature_support_list.sliding_window
        ), f"{attention_backend} doesn't support sliding window."

    # Check MLA
    if use_mla:
        assert feature_support_list.mla, f"{attention_backend} doesn't support MLA."

    # Check dp attention support
    if server_args.enable_dp_attention:
        assert use_mla, f"DP attention is only enabled for MLA models."
        assert (
            feature_support_list.dp_attention
        ), f"{attention_backend} doesn't support DP attention."

    # Check chunked prefix cache support
    if server_args.enable_chunked_prefix_cache:
        assert use_mla, f"Chunked prefix cache is only enabled for MLA models."
        assert (
            feature_support_list.chunked_prefix_cache
        ), f"{attention_backend} doesn't support chunked prefix cache."

    # Check kv cache dtype support
    if server_args.kv_cache_dtype != "auto":
        assert (
            feature_support_list.kv_cache_dtype is not None
        ), f"{attention_backend} doesn't support kv cache dtype with precision lower than 16 bits."
        assert (
            server_args.kv_cache_dtype in feature_support_list.kv_cache_dtype
        ), f"{attention_backend} doesn't support kv cache dtype {server_args.kv_cache_dtype}."
