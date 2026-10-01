from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import msgspec
from torch import nn

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig


def _get_loop_num(hf_config: Any) -> int:
    # Nanbeige uses num_loops; IQuestLoopCoder uses loop_num.
    return int(getattr(hf_config, "loop_num", getattr(hf_config, "num_loops", 1)) or 1)


def compute_attention_layer_info(layer_model: Any) -> tuple[int, bool]:
    """Count supported attention layers and detect MHA companions for graph gates."""
    attention_layer_count = 0
    has_mha_companion_layers = False

    layers = layer_model.layers
    if isinstance(layers, nn.ModuleDict):
        layers = layers.values()
    for layer in layers:
        attn_layer = None
        mha_companion_layer = None
        if hasattr(layer, "self_attn"):
            if hasattr(layer.self_attn, "attn"):
                attn_layer = layer.self_attn.attn
            elif hasattr(layer.self_attn, "attn_mqa"):
                # For DeepSeek model
                attn_layer = layer.self_attn.attn_mqa
                if hasattr(layer.self_attn, "attn_mha"):
                    mha_companion_layer = layer.self_attn.attn_mha
        # For hybrid model
        elif hasattr(layer, "attn"):
            inner = layer.attn
            # Inkling wraps RadixAttention inside a InklingAttention module
            # (layer.attn.attn); descend to the inner RadixAttention that BCG
            # needs. Other hybrid models put RadixAttention at layer.attn.
            attn_layer = inner.attn if hasattr(inner, "attn") else inner
        elif hasattr(layer, "linear_attn"):
            if hasattr(layer.linear_attn, "attn"):
                attn_layer = layer.linear_attn.attn
            else:
                attn_layer = layer.linear_attn
        # For InternVL model
        elif hasattr(layer, "attention"):
            if hasattr(layer.attention, "attn"):
                attn_layer = layer.attention.attn
        # For NemotronH and similar hybrid models using 'mixer' attribute
        elif hasattr(layer, "mixer"):
            if hasattr(layer.mixer, "attn"):
                attn_layer = layer.mixer.attn
            elif hasattr(layer, "_forward_mamba"):
                # Mamba layer with graph support
                attn_layer = layer

        if isinstance(attn_layer, nn.ModuleList):
            # Loop models have one attention module for each execution of a block.
            attention_layer_count += sum(attn is not None for attn in attn_layer)
            if len(attn_layer) and mha_companion_layer is not None:
                has_mha_companion_layers = True
        elif attn_layer is not None:
            attention_layer_count += 1
            has_mha_companion_layers |= mha_companion_layer is not None

    return attention_layer_count, has_mha_companion_layers


class _PPLayerRange(msgspec.Struct, frozen=True, kw_only=True):
    start_layer: int
    end_layer: int


class ModelLayerInfo(msgspec.Struct, frozen=True, kw_only=True):
    start_layer: int
    end_layer: int
    num_effective_layers: int
    # Global ids of the layers this runner owns; None when the model has no split.
    swa_attention_layer_ids: Optional[list[int]] = None
    full_attention_layer_ids: Optional[list[int]] = None
    # Owns the single block at layer_id == draft_model_idx, not a [start, end) slice.
    is_hybrid_swa_mtp_draft: bool = False


def resolve_layer_indices(
    *,
    model: Any,
    model_config: ModelConfig,
    is_draft_worker: bool,
    draft_model_idx: Optional[int] = None,
) -> ModelLayerInfo:
    # For MTP models like DeepSeek-V3 or GLM-4.5, the MTP layer(s) are used separately as draft
    # models for speculative decoding. In those cases, `num_nextn_predict_layers` is used to
    # determine the number of layers.
    model_num_layers = _compute_model_num_layers(
        model=model, model_config=model_config, is_draft_worker=is_draft_worker
    )
    pp_range = _resolve_pp_layer_range(model=model, model_num_layers=model_num_layers)
    num_effective_layers = pp_range.end_layer - pp_range.start_layer

    # For LoopCoder models, each loop has its own layer_id, so we need to multiply by loop_num
    loop_num = _get_loop_num(model_config.hf_config)
    if loop_num > 1:
        num_effective_layers = num_effective_layers * loop_num

    is_hybrid_swa_mtp_draft = (
        is_draft_worker
        and draft_model_idx is not None
        and model_config.is_hybrid_swa
        and getattr(model, "mtp_layer_id_is_depth", False)
    )
    owned_layers = (
        range(draft_model_idx, draft_model_idx + 1)
        if is_hybrid_swa_mtp_draft
        else range(pp_range.start_layer, pp_range.end_layer)
    )
    swa_attention_layer_ids, full_attention_layer_ids = (
        _resolve_local_hybrid_swa_layer_ids(
            model_config=model_config, owned_layers=owned_layers
        )
    )

    return ModelLayerInfo(
        start_layer=pp_range.start_layer,
        end_layer=pp_range.end_layer,
        num_effective_layers=num_effective_layers,
        swa_attention_layer_ids=swa_attention_layer_ids,
        full_attention_layer_ids=full_attention_layer_ids,
        is_hybrid_swa_mtp_draft=is_hybrid_swa_mtp_draft,
    )


def _resolve_local_hybrid_swa_layer_ids(
    *,
    model_config: ModelConfig,
    owned_layers: range,
) -> tuple[Optional[list[int]], Optional[list[int]]]:
    if model_config.swa_attention_layer_ids is None:
        return None, None
    return (
        [i for i in model_config.swa_attention_layer_ids if i in owned_layers],
        [i for i in model_config.full_attention_layer_ids if i in owned_layers],
    )


def _compute_model_num_layers(
    *,
    model: Any,
    model_config: ModelConfig,
    is_draft_worker: bool,
) -> int:
    # Some EAGLE3 drafts (e.g. nvidia/Kimi-K2.5-Thinking-Eagle3) carry the full DeepSeek-V3
    # config schema and explicitly set `num_nextn_predict_layers: 0`. Treat that the same as
    # the field being absent — otherwise the draft worker takes the MTP branch below with
    # model_num_layers=0, sizing the draft KV pool to zero and producing an IndexError on
    # the first forward (`set_mla_kv_buffer` -> `self.kv_buffer[layer_id - self.start_layer]`).
    _nnpl = model_config.num_nextn_predict_layers
    model_has_mtp_layers = _nnpl is not None and _nnpl > 0
    model_num_layers = (
        getattr(model, "num_stages", model_config.num_nextn_predict_layers)
        if is_draft_worker and model_has_mtp_layers
        else max(
            model_config.num_hidden_layers,
            model_config.num_attention_layers,
        )
    )
    if model_config.hf_config.architectures[0] == "MiMoV2MTP":
        model_num_layers = 1
    elif model_config.hf_config.architectures[0] == "Step3p5MTP":
        model_num_layers = 1
    elif (
        model_config.hf_config.architectures[0] == "InklingForConditionalGenerationMTP"
    ):
        model_num_layers = 1
    return model_num_layers


def _resolve_pp_layer_range(*, model: Any, model_num_layers: int) -> _PPLayerRange:
    return _PPLayerRange(
        start_layer=getattr(model, "start_layer", 0),
        end_layer=getattr(model, "end_layer", model_num_layers),
    )
