# SPDX-License-Identifier: Apache-2.0
# Speculators field semantics follow vLLM's transformers_utils/configs/speculators
# (Copyright contributors to the vLLM project); no vLLM runtime dependency.
"""Normalize the data-only Speculators DSpark schema before HF architecture lookup."""

from numbers import Integral
from typing import Any


def normalize_speculators_dspark_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a native DSpark config; applying this twice is a no-op.

    Speculators counts query rows in block_size. SGLang stores the number of
    proposed tokens there, and its target layer IDs refer to decoder outputs
    (HF auxiliary hidden states include the embedding at index zero).
    """
    if config.get("speculators_model_type") != "dspark":
        return config
    if config.get("_sglang_speculators_dspark_normalized", False):
        return config
    layers = config.get("transformer_layer_config")
    if not isinstance(layers, dict):
        raise ValueError("Speculators DSpark requires transformer_layer_config object.")
    if layers.get("model_type") not in (None, "qwen3") or "Qwen3OmniDSparkModel" in (
        config.get("architectures") or []
    ):
        raise ValueError(
            "SGLang Speculators DSpark currently supports the Qwen3 dense backbone only."
        )
    use_aux = config.get(
        "use_aux_hidden_state", layers.get("use_aux_hidden_state", True)
    )
    if use_aux is False:
        raise ValueError(
            "SGLang Speculators DSpark requires use_aux_hidden_state=True; "
            "the current context-KV path requires trained fc projection weights."
        )
    normalized = dict(layers)
    normalized["model_type"] = "qwen3"
    normalized["architectures"] = ["Qwen3DSparkModel"]
    aux = config.get("aux_hidden_state_layer_ids")
    if (
        not isinstance(aux, (list, tuple))
        or not aux
        or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 1 for i in aux)
    ):
        raise ValueError(
            "Speculators DSpark requires non-empty positive aux_hidden_state_layer_ids."
        )
    normalized["eagle_aux_hidden_state_layer_ids"] = list(aux)
    normalized["target_layer_ids"] = [int(i) - 1 for i in aux]
    sample_from_anchor = config.get("sample_from_anchor", False)
    if not isinstance(sample_from_anchor, bool):
        raise ValueError("Speculators DSpark sample_from_anchor must be a boolean.")
    normalized["sample_from_anchor"] = sample_from_anchor
    block_size = config.get("block_size")
    if isinstance(block_size, bool) or not isinstance(block_size, Integral):
        raise ValueError("Speculators DSpark block_size must be an integer.")
    gamma = int(block_size) - int(not sample_from_anchor)
    if gamma < 1:
        raise ValueError(
            "Speculators DSpark block_size must provide at least one draft token."
        )
    normalized["block_size"] = gamma
    normalized["speculators_block_size"] = int(block_size)
    # Speculators' confidence defaults differ from legacy SGLang DSpark.
    normalized.setdefault("enable_confidence_head", False)
    normalized.setdefault("confidence_head_with_markov", False)
    for name in (
        "draft_vocab_size",
        "target_vocab_size",
        "target_hidden_size",
        "mask_token_id",
        "markov_rank",
        "markov_head_type",
        "markov_topk",
        "dspark_draft_topk",
        "markov_bias_topk",
        "logit_scale",
        "enable_confidence_head",
        "confidence_head_with_markov",
        "use_aux_hidden_state",
    ):
        if config.get(name) is not None:
            normalized[name] = config[name]
    normalized["speculators_model_type"] = "dspark"
    normalized["_sglang_speculators_dspark_normalized"] = True
    return normalized
