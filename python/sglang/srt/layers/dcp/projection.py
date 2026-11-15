"""Replicated output weights for the LSE-only DCP merge (``ag_delay_reduce``).

Gathered after weight post-processing and before KV-cache sizing and graph
capture. Prefill keeps the original TP shard.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch


def prepare_dcp_output_projections(
    *,
    model: torch.nn.Module,
    group,
    is_mla: bool,
    decode_attention_backend: Optional[str] = None,
    num_kv_heads: Optional[int] = None,
) -> int:
    if not is_mla and decode_attention_backend != "triton":
        raise ValueError(
            "--dcp-comm-backend ag_delay_reduce needs the Triton decode backend for GQA, "
            f"got {decode_attention_backend!r}"
        )

    prepared = 0
    for attention, proj in _dcp_attention_projections(model, group.world_size):
        if not is_mla:
            _check_kv_head_layout(attention, num_kv_heads)
        proj.dcp_replica = _gather_output_projection(proj, group)
        proj.dcp_replica_input_size = proj.input_size_per_partition * group.world_size
        proj._derived_weight_cache_error = (
            "Online weight updates are not supported with DCP ag_delay_reduce: captured "
            "graphs retain DCP-replicated output weights. Restart with the "
            "updated checkpoint, or use --dcp-comm-backend ag_rs."
        )
        if is_mla:
            _gather_mla_value_weights(attention, group)
        prepared += 1
    if prepared == 0:
        raise ValueError(
            "--dcp-comm-backend ag_delay_reduce found no supported MLA/GQA output projections"
        )
    return prepared


def _dcp_attention_projections(
    model: torch.nn.Module, dcp_size: int
) -> Iterator[tuple[torch.nn.Module, torch.nn.Module]]:
    """Yield (attention, o_proj) pairs whose decode output spans a DCP group."""
    from sglang.srt.layers.linear import RowParallelLinear
    from sglang.srt.layers.radix_attention import RadixAttention

    # Snapshot the modules: attaching a replica changes the module tree.
    for attention in list(model.modules()):
        proj = getattr(attention, "o_proj", None)
        if not isinstance(proj, RowParallelLinear):
            continue
        core = getattr(attention, "attn", None) or getattr(attention, "attn_mqa", None)
        if not isinstance(core, RadixAttention):
            continue
        # Hybrid models disable DCP per layer (K3 keeps KDA layers unsharded).
        if getattr(attention, "dcp_size", dcp_size) == 1:
            continue
        if hasattr(attention, "prepare_attention_output_gate"):
            # Such a gate may fuse a head-sharded projection or mix across
            # heads; either needs its own full-head path.
            raise ValueError(
                "--dcp-comm-backend ag_delay_reduce does not support model-specific MLA "
                "output gates"
            )
        yield attention, proj


def _check_kv_head_layout(attention: torch.nn.Module, expected_num_kv_heads: int):
    local_kv_heads = attention.attn.tp_k_head_num
    if local_kv_heads != expected_num_kv_heads:
        raise ValueError(
            "--dcp-comm-backend ag_delay_reduce requires GQA K/V heads replicated within "
            f"each DCP group: layer has {local_kv_heads} heads, but the KV "
            f"cache expects {expected_num_kv_heads}."
        )


def _gather_output_projection(proj: torch.nn.Module, group) -> torch.nn.Module:
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    if isinstance(proj.quant_method, UnquantizedLinearMethod):
        replica = torch.nn.Module()
        replica.register_buffer(
            "weight",
            group.all_gather(proj.weight.data.contiguous(), dim=1),
            persistent=False,
        )
        replica.quant_method = proj.quant_method
        return replica
    if is_block_fp8_linear(proj):
        return gather_block_fp8_linear(proj, group, dim=1)
    raise ValueError(
        "--dcp-comm-backend ag_delay_reduce output projection requires an unquantized or "
        f"128x128 block-FP8 weight, got {type(proj.quant_method).__name__}"
    )


def _gather_mla_value_weights(attention: torch.nn.Module, group):
    weights = attention.w_vc
    if weights is None:
        raise ValueError(
            "--dcp-comm-backend ag_delay_reduce requires prepared MLA V weights"
        )
    if attention.use_deep_gemm_bmm:
        # DeepGEMM keeps head-local block scales, so gather raw bytes and
        # scales separately instead of dequantizing.
        attention._dcp_w_vc = group.all_gather(
            weights.contiguous().view(torch.uint8), dim=0
        ).view(weights.dtype)
        attention._dcp_w_scale_v = group.all_gather(
            attention.w_scale_v.contiguous(), dim=0
        )
        return
    if weights.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        # Each rank may hold a different per-tensor scale; apply it before the
        # gather so peers read correctly scaled weights.
        weights = weights.to(torch.bfloat16) * attention.w_scale
    if weights.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(
            "--dcp-comm-backend ag_delay_reduce requires BF16/FP16 or FP8 MLA V weights, "
            f"got {weights.dtype}"
        )
    attention._dcp_w_vc = group.all_gather(weights.contiguous(), dim=0)


def is_block_fp8_linear(layer: torch.nn.Module) -> bool:
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    method = layer.quant_method
    return (
        isinstance(method, Fp8LinearMethod)
        and not method.use_mxfp8
        and not method.use_marlin
        and method.weight_block_size == [128, 128]
        and layer.weight.dtype == torch.float8_e4m3fn
        # An aiter-preshuffled weight is no longer row-major, so it cannot be
        # gathered along either dim.
        and not getattr(layer, "aiter_bpreshuffled", False)
    )


def gather_block_fp8_linear(
    layer: torch.nn.Module, group, dim: int = 0
) -> torch.nn.Module:
    """Gather a weight shard without requantizing processed FP8 weights.

    ``dim=0`` gathers output rows (query replication), ``dim=1`` input
    columns (DCP-group output projection).
    """
    from sglang.srt.layers.quantization.fp8_utils import (
        inverse_transform_scale_ue8m0,
        transform_scale_ue8m0,
    )

    scale = layer.weight_scale_inv.data
    packed_ue8m0 = scale.dtype == torch.int32  # DeepGEMM layout
    if packed_ue8m0:
        # Re-pack rather than requantize: that pass is not idempotent (a block
        # whose amax sits on a power-of-two boundary halves its scale), so the
        # replica would not match the shards bit for bit.
        scale = inverse_transform_scale_ue8m0(scale, mn=layer.weight.shape[0])
    if dim == 1:
        if layer.weight.shape[1] % 128:
            raise ValueError(
                "DCP FP8 projection shards must align to 128-wide K blocks, got "
                f"{layer.weight.shape[1]}"
            )
        if packed_ue8m0:
            # UE8M0 packs four K blocks per int32, so the inverse returns a
            # K-padded scale; trim before concatenation makes it interior.
            scale = scale[:, : layer.weight.shape[1] // 128]
    replica = torch.nn.Module()
    # NCCL has no FP8 arithmetic, but an all-gather is a byte copy.
    weight = group.all_gather(layer.weight.data.view(torch.uint8), dim=dim)
    replica.weight = torch.nn.Parameter(
        weight.view(layer.weight.dtype), requires_grad=False
    )
    scale = group.all_gather(scale.contiguous(), dim=dim)
    if packed_ue8m0:
        scale = transform_scale_ue8m0(scale, mn=replica.weight.shape[0])
    replica.weight_scale_inv = torch.nn.Parameter(scale, requires_grad=False)
    replica.weight_scale_inv.format_ue8m0 = packed_ue8m0 or getattr(
        layer.weight_scale_inv, "format_ue8m0", False
    )
    # What Fp8LinearMethod's post-load step reads off a LinearBase; copied so
    # the replica picks the same kernel (UE8M0 requant, MXFP8 view).
    replica.orig_dtype = getattr(layer, "orig_dtype", None)
    replica.deepgemm_mxfp8_input = getattr(layer, "deepgemm_mxfp8_input", False)
    replica.keep_plain_weight_layout = getattr(layer, "keep_plain_weight_layout", False)
    replica.prefix = f"{getattr(layer, 'prefix', 'projection')}.dcp_replica"
    replica.quant_method = layer.quant_method
    layer.quant_method.process_weights_after_loading(replica)
    return replica
