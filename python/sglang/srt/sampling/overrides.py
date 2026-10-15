"""Model-specific sampling aliases applied before SamplingParams normalization."""

from typing import Any, Sequence


def apply_model_sampling_overrides(
    sampling_params: dict[str, Any], architectures: Sequence[str] | None
) -> dict[str, Any]:
    if sampling_params.get("top_k") == 0 and any(
        architecture in ("MuseSparkForConditionalGeneration", "MuseSpark1xForCausalLM")
        for architecture in architectures or ()
    ):
        # Transformers uses zero to disable top-k; SGLang uses -1.
        return {**sampling_params, "top_k": -1}
    return sampling_params
