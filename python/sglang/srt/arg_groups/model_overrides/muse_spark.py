"""Sampling overrides for Muse Spark."""

from sglang.srt.arg_groups.model_override_base import register_model_sampling_override


@register_model_sampling_override(
    "MuseSparkForConditionalGeneration", "MuseSpark1xForCausalLM"
)
def _muse_spark_sampling_overrides(sampling_params: dict) -> dict:
    # Transformers uses zero to disable top-k; SGLang uses -1.
    if sampling_params.get("top_k") == 0:
        return {"top_k": -1}
    return {}
