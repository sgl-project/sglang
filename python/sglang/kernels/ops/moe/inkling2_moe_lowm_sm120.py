import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import cache_once, load_jit

LOWM_MAX_TOKENS = 64
LOWM_FUSED_ROUTE_MAX_TOKENS = 16
_NUM_EXPERTS = 128
_TOP_K = 8


def _require_sm12x() -> None:
    if not current_arch().is_sm(12):
        raise RuntimeError("inkling2_moe_lowm_sm120 requires NVIDIA SM12x")


@cache_once
def _combine_module():
    _require_sm12x()
    return load_jit(
        "inkling2_moe_combine_lowm_sm120",
        cuda_files=["inkling2/moe_lowm_sm120.cuh"],
        cuda_wrappers=[
            ("moe_combine_bf16", "inkling2_moe_lowm_sm120::moe_combine_bf16"),
        ],
    )


@cache_once
def _route_module():
    _require_sm12x()
    return load_jit(
        "inkling2_moe_route_lowm_sm120",
        cuda_files=["inkling2/moe_route_lowm_sm120.cuh"],
        cuda_wrappers=[
            (
                "route_topk_meta_lowm",
                "inkling2_moe_route_sm120::route_topk_meta_lowm",
            ),
            (
                "route_metadata_lowm",
                "inkling2_moe_route_sm120::route_metadata_lowm",
            ),
        ],
    )


@cache_once
def _quant_module(hidden_size: int):
    _require_sm12x()
    return load_jit(
        "inkling2_moe_quant_lowm_sm120",
        str(hidden_size),
        cuda_files=["inkling2/moe_nvfp4_quant_lowm_sm120.cuh"],
        cuda_wrappers=[
            (
                "nvfp4_quant_dedup_lowm",
                f"inkling2_moe_nvfp4_quant_sm120::nvfp4_quant_dedup_lowm<{hidden_size}>",
            ),
        ],
    )


def can_use_lowm_moe(tokens, experts, top_k, hidden_size, intermediate_size, device):
    return (
        0 < tokens <= LOWM_MAX_TOKENS
        and experts == _NUM_EXPERTS
        and top_k == _TOP_K
        and hidden_size % 128 == 0
        and intermediate_size % 128 == 0
        and hidden_size <= 4096
        and intermediate_size <= 4096
        and device.type == "cuda"
        and current_arch().is_sm(12)
    )


def can_use_lowm_mxfp8(tokens):
    return current_arch().is_sm(12, 0) or tokens == 1


def route_topk_meta_lowm(partials, bias, route_scale, global_scale=None):
    _, tokens, num_experts = partials.shape
    if num_experts != _NUM_EXPERTS:
        raise ValueError(
            f"route_topk_meta_lowm requires {_NUM_EXPERTS} experts, got {num_experts}"
        )
    n = tokens * _TOP_K
    dev = partials.device
    ids = torch.empty((tokens, _TOP_K), dtype=torch.int32, device=dev)
    weights = torch.empty((tokens, _TOP_K), dtype=torch.float32, device=dev)
    src2dst = torch.empty(n, dtype=torch.int32, device=dev)
    dst2src = torch.empty(n, dtype=torch.int32, device=dev)
    gammas = torch.empty(n, dtype=torch.float32, device=dev)
    offsets = (
        torch.zeros(num_experts + 1, dtype=torch.int32, device=dev)
        if tokens == 0
        else torch.empty(num_experts + 1, dtype=torch.int32, device=dev)
    )
    if tokens:
        _route_module().route_topk_meta_lowm(
            partials.contiguous(),
            bias.contiguous() if bias is not None else None,
            global_scale,
            ids,
            weights,
            src2dst,
            dst2src,
            gammas,
            offsets,
            float(route_scale),
        )
    return ids, weights, src2dst, dst2src, gammas, offsets


def route_metadata_lowm(ids, weights, num_experts=_NUM_EXPERTS):
    """Build expert-order routing maps; out-of-range expert IDs emit -1 in src2dst."""
    if num_experts != _NUM_EXPERTS:
        raise ValueError(
            f"route_metadata_lowm requires {_NUM_EXPERTS} experts, got {num_experts}"
        )
    flat_ids = ids.reshape(-1).contiguous()
    flat_w = weights.reshape(-1)
    if flat_w.dtype != torch.float32:
        flat_w = flat_w.float()
    flat_w = flat_w.contiguous()
    n = flat_ids.numel()
    dev = flat_ids.device
    src2dst = torch.empty(n, dtype=torch.int32, device=dev)
    dst2src = torch.empty(n, dtype=torch.int32, device=dev)
    gammas = torch.empty(n, dtype=torch.float32, device=dev)
    offsets = (
        torch.zeros(num_experts + 1, dtype=torch.int32, device=dev)
        if n == 0
        else torch.empty(num_experts + 1, dtype=torch.int32, device=dev)
    )
    if n:
        _route_module().route_metadata_lowm(
            flat_ids, flat_w, src2dst, dst2src, gammas, offsets
        )
    return src2dst, dst2src, gammas, offsets


def nvfp4_quant_dedup_lowm(x, ids, src2dst, offsets):
    """Quantize each token once and scatter to valid expert slots (skipping -1 sentinels)."""
    tokens, width = x.shape
    rows = ids.numel()
    experts = offsets.numel() - 1
    if experts != _NUM_EXPERTS:
        raise ValueError(
            f"nvfp4_quant_dedup_lowm requires {_NUM_EXPERTS} experts, got {experts}"
        )
    scale_rows = ((rows + 127 * experts + 127) // 128) * 128
    values = torch.empty((rows, width // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty((scale_rows, width // 16), dtype=torch.uint8, device=x.device)
    amax = torch.empty(rows, dtype=torch.float32, device=x.device)
    if tokens:
        _quant_module(width).nvfp4_quant_dedup_lowm(
            x, ids.reshape(-1).contiguous(), src2dst, offsets, values, scales, amax
        )
    return values, scales, amax


def moe_combine_bf16(down_bf16, src2dst, top_k=_TOP_K):
    """Sum routed BF16 rows per token in FP32, skipping -1 or out-of-range src2dst slots."""
    if top_k != _TOP_K:
        raise ValueError(f"moe_combine_bf16 requires top_k == {_TOP_K}, got {top_k}")
    map_len = src2dst.numel()
    if map_len % top_k != 0:
        raise ValueError(
            f"moe_combine_bf16 src2dst length {map_len} must be divisible by {top_k}"
        )
    if down_bf16.ndim != 2 or down_bf16.shape[0] != map_len:
        raise ValueError(
            f"moe_combine_bf16 down_bf16 must have shape ({map_len}, H), "
            f"got {tuple(down_bf16.shape)}"
        )
    tokens = map_len // top_k
    out = torch.empty(
        (tokens, down_bf16.shape[1]), dtype=torch.bfloat16, device=down_bf16.device
    )
    if tokens:
        _combine_module().moe_combine_bf16(down_bf16, src2dst, out, top_k)
    return out
