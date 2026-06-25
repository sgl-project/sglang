from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


@dataclass(frozen=True, slots=True, kw_only=True)
class _PPLayerRange:
    start_layer: int
    end_layer: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelLayerInfo:
    start_layer: int
    end_layer: int
    num_effective_layers: int


def resolve_layer_indices(
    *,
    model: Any,
    model_config: ModelConfig,
    is_draft_worker: bool,
    spec_algorithm: SpeculativeAlgorithm,
) -> ModelLayerInfo:
    # For MTP models like DeepSeek-V3 or GLM-4.5, the MTP layer(s) are used separately as draft
    # models for speculative decoding. In those cases, `num_nextn_predict_layers` is used to
    # determine the number of layers.
    model_num_layers = _compute_model_num_layers(
        model_config=model_config, is_draft_worker=is_draft_worker
    )
    _nnpl = model_config.num_nextn_predict_layers
    model_has_mtp_layers = _nnpl is not None and _nnpl > 0
    pp_range = _resolve_pp_layer_range(model=model, model_num_layers=model_num_layers)
    num_effective_layers = pp_range.end_layer - pp_range.start_layer

    # For LoopCoder models, each loop has its own layer_id, so we need to multiply by loop_num
    loop_num = getattr(model_config.hf_config, "loop_num", 1)
    if loop_num > 1:
        num_effective_layers = num_effective_layers * loop_num

    _assert_pp_mtp_compat(
        model_has_mtp_layers=model_has_mtp_layers,
        spec_algorithm=spec_algorithm,
        num_effective_layers=num_effective_layers,
        model_num_layers=model_num_layers,
    )

    return ModelLayerInfo(
        start_layer=pp_range.start_layer,
        end_layer=pp_range.end_layer,
        num_effective_layers=num_effective_layers,
    )


def _compute_model_num_layers(
    *,
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
        model_config.num_nextn_predict_layers
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
    return model_num_layers


def _resolve_pp_layer_range(*, model: Any, model_num_layers: int) -> _PPLayerRange:
    return _PPLayerRange(
        start_layer=getattr(model, "start_layer", 0),
        end_layer=getattr(model, "end_layer", model_num_layers),
    )


def _assert_pp_mtp_compat(
    *,
    model_has_mtp_layers: bool,
    spec_algorithm: SpeculativeAlgorithm,
    num_effective_layers: int,
    model_num_layers: int,
) -> None:
    assert (
        (not model_has_mtp_layers)
        or (spec_algorithm.is_none())
        or (
            (not spec_algorithm.is_none())
            and (num_effective_layers == model_num_layers)
        )
    ), "PP is not compatible with MTP models."


def adjust_hybrid_swa_layer_ids(
    *,
    model_config: ModelConfig,
    start_layer: int,
    end_layer: int,
    is_hybrid_swa: bool,
) -> None:
    if not is_hybrid_swa:
        return

    if model_config.is_deepseek_v4_arch:
        return

    full_attention_layer_ids = [
        layer_idx
        for layer_idx in range(start_layer, end_layer + 1)
        if hasattr(model_config, "full_attention_layer_ids")
        and layer_idx in model_config.full_attention_layer_ids
    ]
    swa_attention_layer_ids = [
        layer_idx
        for layer_idx in range(start_layer, end_layer + 1)
        if hasattr(model_config, "swa_attention_layer_ids")
        and layer_idx in model_config.swa_attention_layer_ids
    ]
    model_config.swa_attention_layer_ids = swa_attention_layer_ids
    model_config.full_attention_layer_ids = full_attention_layer_ids
