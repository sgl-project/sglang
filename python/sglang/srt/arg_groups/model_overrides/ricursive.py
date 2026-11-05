"""Config-time override declarations for ricursive."""

from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    is_attention_backend_not_set,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform


@_register_for("RicursiveForCausalLM")
def _ricursive_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    # On a tiny random Ricursive checkpoint, flashinfer's prefill logprobs match HF
    # with no window at all. Drop this once its prefill applies per-layer windows.
    if "flashinfer" in (
        cfg.attention_backend,
        cfg.prefill_attention_backend,
        cfg.decode_attention_backend,
    ):
        raise ValueError(
            "Ricursive's sliding attention layers can't use "
            "--attention-backend flashinfer: it drops the window in prefill."
        )
    overrides: Dict[str, Any] = {}
    if not get_platform().is_cuda:
        return overrides
    if get_platform().is_sm100:
        # trtllm_mha doesn't support deterministic inference, which an on-policy
        # target turns on after this runs.
        deterministic = (
            cfg.enable_deterministic_inference or cfg.rl_on_policy_target is not None
        )
        default_backend = "fa4" if deterministic else "trtllm_mha"
    elif get_platform().is_sm90:
        default_backend = "fa3"
    else:
        default_backend = "triton"
    if is_attention_backend_not_set(cfg):
        overrides["attention_backend"] = default_backend
    elif cfg.attention_backend is None:
        if cfg.prefill_attention_backend is None:
            overrides["prefill_attention_backend"] = default_backend
        if cfg.decode_attention_backend is None:
            overrides["decode_attention_backend"] = default_backend
    return overrides
