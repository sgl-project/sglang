from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    register_model_override,
    resolving_view,
)
from sglang.srt.configs.nemotron_decision import is_nemotron_decision
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.device import get_device_memory_capacity


def apply_nemotron_decision_prefill_defaults(
    server_args: Any, model_config: Any
) -> None:
    if not is_nemotron_decision(model_config.hf_config):
        return

    from sglang.srt.arg_groups.overrides import declare_resolution

    # Bidirectional attention needs the whole prompt at once, so reserve
    # activation memory for a full read before sizing the KV pool.
    updates = {"chunked_prefill_size": -1}
    if server_args.max_prefill_tokens is None:
        gpu_mem = get_device_memory_capacity(resolving_view(server_args).device)
        # Smaller GPUs need a smaller activation reserve for a complete read.
        budget = 16384 if gpu_mem is not None and gpu_mem < 60 * 1024 else 32768
        updates["max_prefill_tokens"] = min(
            budget,
            model_config.context_len,
            model_config.nemotron_decision_config["max_input_tokens"],
        )
    declare_resolution(server_args, "nemotron_decision_prefill_defaults", **updates)


@register_model_override("NemotronForDecision")
def _nemotron_decision_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if get_platform().is_hip:
        # The default ROCm backend ignores encoder-only attention.
        raise ValueError("Nemotron Decision is not supported on ROCm.")
    if hf_config.is_causal or hf_config.sliding_window is not None:
        raise ValueError("Nemotron Decision requires full bidirectional attention")
    if cfg.dllm_algorithm is not None or cfg.speculative_algorithm is not None:
        raise ValueError(
            "Nemotron Decision scores each question in one bidirectional read; "
            "diffusion generation and speculative decoding are not supported"
        )
    return {}
