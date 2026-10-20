"""Resolve collective preferences before constructing target and draft models."""

import json

from sglang.srt.arg_groups.model_override_base import model_config_of

_RSV_ARCHITECTURES = {
    "BailingMoELinearForCausalLM",
    "BailingMoeV2_5ForCausalLM",
    "LongcatFlashForCausalLM",
    "LongcatFlashForCausalLMNextN",
    "Step3VLForConditionalGeneration",
}
_AR_ARCHITECTURES = {
    "KimiK3ForConditionalGeneration",
    "KimiK3LinearForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3ForSequenceClassification",
    "Qwen3Model",
    "Qwen3ForRewardModel",
    "MossVLForCausalLM",
    "MossVLForConditionalGeneration",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpForCausalLMMTP",
}


def model_boundary_reduction(config):
    """One default for a model's dense and MoE layers, including its MTP."""
    architectures = set(getattr(config, "architectures", None) or ())
    if architectures & _RSV_ARCHITECTURES:
        return "rsv"
    if architectures & _AR_ARCHITECTURES:
        return "ar"
    # MiniCPM-V 4.5 constructs Qwen3 from its top-level config.
    if (
        getattr(config, "model_type", None) == "minicpmv"
        and str(getattr(config, "version", "")) == "4.5"
    ):
        return "ar"
    # Wrappers carry their actual language backbone in the nested HF config.
    # Match Qwen3 specifically: Qwen3-MoE/Next/3.5 fall through to rs+rsv.
    if getattr(config, "model_type", None) in ("qwen3", "qwen3_vl_text"):
        return "ar"
    get_text = getattr(config, "get_text_config", None)
    if get_text is not None:
        text = get_text()
        if text is not config:
            return model_boundary_reduction(text)
    for name in ("text_config", "llm_config", "language_config"):
        text = getattr(config, name, None)
        if text is not None and text is not config:
            return model_boundary_reduction(text)
    return "rs+rsv"


def resolve_boundary_reduction(view):
    """Resolve auto once, after speculative paths/revisions have been resolved."""
    requested = view.boundary_reduction
    dummy = view.model_path.lower() in ("none", "dummy")
    if requested != "auto":
        target = draft = requested
    else:
        target = draft = (
            "rs+rsv"
            if dummy
            else model_boundary_reduction(model_config_of(view).hf_config)
        )
        draft_path = view.speculative_draft_model_path
        if (
            not dummy
            and view.speculative_algorithm is not None
            and draft_path is not None
            and draft_path != view.model_path
        ):
            from sglang.srt.utils.hf_transformers_utils import get_config

            kwargs = {}
            if view.decrypted_draft_config_file:
                kwargs["_configuration_file"] = view.decrypted_draft_config_file
            config = get_config(
                draft_path,
                trust_remote_code=view.trust_remote_code,
                revision=view.speculative_draft_model_revision,
                model_override_args=json.loads(view.json_model_override_args),
                **kwargs,
            )
            draft = model_boundary_reduction(config)
    return {
        "boundary_reduction": target,
        "speculative_boundary_reduction": draft,
    }
