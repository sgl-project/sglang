from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


@dataclass(frozen=True, slots=True, kw_only=True)
class PPLayerRange:
    start_layer: int
    end_layer: int


def compute_model_num_layers(
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


def resolve_pp_layer_range(*, model: Any, model_num_layers: int) -> PPLayerRange:
    return PPLayerRange(
        start_layer=getattr(model, "start_layer", 0),
        end_layer=getattr(model, "end_layer", model_num_layers),
    )


def assert_pp_mtp_compat(
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
