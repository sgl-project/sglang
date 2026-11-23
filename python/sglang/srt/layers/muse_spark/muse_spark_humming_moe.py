"""Muse Spark NVFP4 routed experts on Humming's W4A16 GEMMs.

Humming keeps NVFP4 as the checkpoint stores it (E2M1 weights, an E4M3 scale
per 16, one FP32 scale per expert) and its INDEXED GEMM reads routes aligned in
expert blocks as Marlin's does, so only the two expert GEMMs and the route
alignment differ from the Marlin path; routing, the activation and the
post-expert-norm finalize are shared.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

import msgspec
import torch
from torch import nn

from sglang.kernels.arch import current_arch
from sglang.srt.layers.muse_spark.muse_spark_humming_tuning import (
    TuningTable,
    block_m_for,
    ep_tuning_tables,
    ep_valid_shape_m,
)
from sglang.srt.runtime_context import get_platform

if TYPE_CHECKING:
    from humming.schema import HummingWeightSchema

# FP32 accumulation: Humming's F16 accumulation is lower precision than Marlin's.
_COMPUTE_CONFIG = json.dumps({"use_f16_accum": False, "gemm_type": "indexed"})
# Humming's MoE padding, as in SGLang's prepare_humming_moe_layer.
_PAD_N, _PAD_K = 256, 128
_SUBLAYERS = ("w13", "w2")


class HummingTuning(msgspec.Struct, frozen=True):
    """Each expert GEMM's tuning table, and the JSON Humming's forward takes."""

    tables: dict[str, TuningTable]
    serialized: dict[str, str]

    @classmethod
    def from_tables(cls, tables: dict[str, TuningTable]) -> HummingTuning:
        return cls(
            tables=tables,
            serialized={sub: json.dumps(table) for sub, table in tables.items()},
        )


@torch.no_grad()
def prepare_humming_experts(layer: nn.Module) -> HummingTuning:
    """Move the checkpoint's NVFP4 experts into Humming's layout, one copy kept,
    and return the tuning tables the expert GEMMs run with."""
    from humming.config import GemmType
    from humming.layer import HummingMethod
    from humming.schema import HummingInputSchema

    inter, hidden = layer.intermediate_size_per_partition, layer.hidden_size
    for sublayer, native, (size_n, size_k), n_stacks in (
        ("w13", "gate_up_proj", (2 * inter, hidden), [inter, inter]),
        ("w2", "down_proj", (hidden, inter), [hidden]),
    ):
        schema, tensors = _convert_checkpoint_tensors(
            layer, sublayer=sublayer, native=native, n_stacks=n_stacks, size_k=size_k
        )
        meta = HummingMethod.prepare_layer_meta(
            layer=layer,
            shape_n=size_n,
            shape_k=size_k,
            weight_schema=schema,
            input_schema=HummingInputSchema(),
            num_experts=layer.num_local_experts,
            pad_n_to_multiple=_PAD_N,
            pad_k_to_multiple=_PAD_K,
            torch_dtype=layer.params_dtype,
            sublayer_name=sublayer,
            tensors=tensors,
        )
        # Registered only now: preparing the meta may replace converted tensors.
        _replace_checkpoint_tensors(
            layer, sublayer=sublayer, native=native, tensors=tensors
        )
        # With nothing to pad, Humming's pad step only clones every tensor, a
        # transient full copy of the sublayer.
        HummingMethod.transform_humming_layer(
            layer,
            sublayer_name=sublayer,
            already_padded=meta.pad_shape_n == 0 and meta.pad_shape_k == 0,
        )

    tables = ep_tuning_tables(
        *(
            HummingMethod.get_default_tuning_configs(
                layer=layer,
                use_f16_accum=False,
                gemm_type=GemmType.INDEXED,
                sublayer_name=sublayer,
            )
            for sublayer in _SUBLAYERS
        ),
        # The decode tiles were timed on H200 and serve SM90 parts with its 132
        # SMs; other SM90 parts, such as H20, keep Humming's own rows.
        h200_class=get_platform().is_sm90 and current_arch().num_sms == 132,
    )
    return HummingTuning.from_tables(dict(zip(_SUBLAYERS, tables)))


def _convert_checkpoint_tensors(
    layer: nn.Module, *, sublayer: str, native: str, n_stacks: list[int], size_k: int
) -> tuple[HummingWeightSchema, dict[str, torch.Tensor]]:
    """Reinterpret one projection's checkpoint tensors under Humming's schema.

    The conversion only views the checkpoint storage; nothing is copied.
    """
    from humming import dtypes
    from humming.config import WeightScale2Type
    from humming.schema.modelopt import ModeloptNvfp4WeightSchema

    scale_2 = getattr(layer, f"{native}_weight_scale_2")
    if sublayer == "w13":
        # One FP32 scale per expert, the gate's, as on the Marlin path: a
        # gate/up pair makes the schema store it as a BF16 per-channel scale.
        scale_2 = scale_2[:, 0]
    schema, tensors = ModeloptNvfp4WeightSchema().convert_humming(
        tensors={
            "weight": getattr(layer, native),
            "weight_scale": getattr(layer, f"{native}_weight_scale"),
            "weight_scale_2": scale_2.contiguous(),
        },
        shape_n_stacks=n_stacks,
        shape_k_stacks=[size_k],
        param_dtype=layer.params_dtype,
        num_experts=layer.num_local_experts,
    )
    if (
        schema.weight_scale_2_type != WeightScale2Type.TENSOR
        or schema.bs_dtype != dtypes.float8e4m3
    ):
        raise ValueError(
            f"Humming would requantize the {sublayer} NVFP4 scales ({schema}); "
            "expected E4M3 group scales and one FP32 scale per expert."
        )
    return schema, tensors


def _replace_checkpoint_tensors(
    layer: nn.Module, *, sublayer: str, native: str, tensors: dict[str, torch.Tensor]
) -> None:
    for suffix in ("", "_weight_scale", "_weight_scale_2"):
        delattr(layer, native + suffix)
    for name, tensor in tensors.items():
        setattr(layer, f"{sublayer}_{name}", nn.Parameter(tensor, requires_grad=False))


def humming_expert_outputs(
    weights: nn.Module,
    *,
    tuning: HummingTuning,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: Optional[torch.Tensor],
    num_local_experts: int,
    num_global_experts: int,
    intermediate_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-route outputs [tokens, top_k, hidden] and local expert ids [tokens, top_k].

    weights holds the tensors and metas prepare_humming_experts left on the
    layer. Rows of routes to other ranks are left undefined; the finalize skips
    them by their local id of -1.
    """
    from humming.layer import HummingMethod

    from sglang.kernels.ops.activation.activation import silu_and_mul

    hidden_states = hidden_states.contiguous()
    num_tokens, hidden = hidden_states.shape
    top_k = topk_ids.shape[1]
    valid_shape_m = ep_valid_shape_m(
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        num_global_experts=num_global_experts,
    )
    local_ids, alignments = _align_per_gemm(
        tuning,
        topk_ids=topk_ids,
        valid_shape_m=valid_shape_m,
        expert_map=expert_map,
        num_local_experts=num_local_experts,
    )
    routes = num_tokens * top_k
    # GEMM 2 writes over GEMM 1's output, dead once activated, as Marlin's
    # cache13 does.
    gate_up_down = hidden_states.new_empty(
        (routes * max(2 * intermediate_size, hidden),)
    )
    gate_up = gate_up_down[: routes * 2 * intermediate_size].view(
        routes, 2 * intermediate_size
    )
    down = gate_up_down[: routes * hidden].view(routes, hidden)
    activated = hidden_states.new_empty((routes, intermediate_size))
    HummingMethod.forward_layer(
        layer=weights,
        inputs=hidden_states,
        outputs=gate_up,
        top_k=top_k,
        sublayer_name="w13",
        tuning_config=tuning.serialized["w13"],
        valid_shape_m=valid_shape_m,
        compute_config=_COMPUTE_CONFIG,
        **alignments["w13"],
    )
    # Local routes only; SGLang's Humming runner activates every route.
    silu_and_mul(gate_up, activated, expert_ids=local_ids.view(-1), expert_step=1)
    HummingMethod.forward_layer(
        layer=weights,
        inputs=activated,
        outputs=down,
        top_k=1,
        sublayer_name="w2",
        tuning_config=tuning.serialized["w2"],
        valid_shape_m=valid_shape_m,
        compute_config=_COMPUTE_CONFIG,
        **alignments["w2"],
    )
    return down.view(num_tokens, top_k, hidden), local_ids


def _align_per_gemm(
    tuning: HummingTuning,
    *,
    topk_ids: torch.Tensor,
    valid_shape_m: int,
    expert_map: Optional[torch.Tensor],
    num_local_experts: int,
) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
    # Each indexed GEMM walks the routes in blocks of its own tile's M, and the
    # two GEMMs' rows can pick different block M.
    by_block_m = {}
    alignments = {}
    for sublayer in _SUBLAYERS:
        block_m = block_m_for(tuning.tables[sublayer], valid_shape_m)
        if block_m not in by_block_m:
            by_block_m[block_m] = _align_routes(
                topk_ids=topk_ids,
                expert_map=expert_map,
                num_local_experts=num_local_experts,
                block_m=block_m,
            )
        local_ids, sorted_ids, block_experts, num_tokens_padded = by_block_m[block_m]
        alignments[sublayer] = dict(
            sorted_ids=sorted_ids,
            expert_ids=block_experts,
            num_tokens_padded=num_tokens_padded,
        )
    return local_ids, alignments


def _align_routes(
    *,
    topk_ids: torch.Tensor,
    expert_map: Optional[torch.Tensor],
    num_local_experts: int,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from sglang.kernels.ops.moe.moe_align_ep_local import (
        ep_local_align_supported,
        moe_align_ep_local,
    )
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        _map_topk_ids_to_local_experts,
    )

    if expert_map is not None and ep_local_align_supported(
        num_routes=topk_ids.numel(), num_local_experts=num_local_experts
    ):
        return moe_align_ep_local(
            topk_ids=topk_ids,
            expert_map=expert_map,
            num_local_experts=num_local_experts,
            block_size=block_m,
        )
    local_ids = topk_ids.to(torch.int32)
    if expert_map is not None:
        local_ids = _map_topk_ids_to_local_experts(local_ids, expert_map)
    local_ids = local_ids.contiguous()
    sorted_ids, block_experts, num_tokens_padded = moe_align_block_size(
        topk_ids=local_ids,
        block_size=block_m,
        num_experts=num_local_experts,
        ignore_invalid_expert=True,
    )
    return local_ids, sorted_ids, block_experts, num_tokens_padded


__all__ = ["HummingTuning", "humming_expert_outputs", "prepare_humming_experts"]
