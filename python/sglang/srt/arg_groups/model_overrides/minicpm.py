"""Config-time override declarations for minicpm.

Architectures: MiniCPMForCausalLM, MiniCPMSALAForCausalLM.
"""

from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    is_attention_backend_not_set,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform


@_register_for("MiniCPMForCausalLM", "MiniCPMSALAForCausalLM")
def _minicpm_sala_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if cfg.enable_dp_attention:
        raise ValueError("MiniCPM does not support DP attention")
    has_sparse_attention = getattr(hf_config, "has_minicpm_sparse_attention", False)
    has_hybrid_attention = has_sparse_attention or getattr(
        hf_config, "has_lightning_layers", False
    )
    overrides: Dict[str, Any] = {}
    if has_hybrid_attention:
        if cfg.enable_hierarchical_cache:
            raise ValueError("MiniCPM SALA does not support hierarchical cache")
        overrides["disable_radix_cache"] = True
    if envs.SGLANG_MINICPM_FORCE_DENSE.get():
        dense_backends = {
            "minicpm_flashattn": ("fa4" if get_platform().is_blackwell else "fa3"),
            "minicpm_flashinfer": "flashinfer",
        }
        # Literal keys keep the written-field set statically derivable; a loop
        # variable hides it from the census in test_chain_read_ratchet.py.
        dense_attention = dense_backends.get(cfg.attention_backend)
        if dense_attention is not None:
            overrides["attention_backend"] = dense_attention
        dense_prefill = dense_backends.get(cfg.prefill_attention_backend)
        if dense_prefill is not None:
            overrides["prefill_attention_backend"] = dense_prefill
        dense_decode = dense_backends.get(cfg.decode_attention_backend)
        if dense_decode is not None:
            overrides["decode_attention_backend"] = dense_decode
    elif has_sparse_attention:
        uses_sparse_backend = is_attention_backend_not_set(cfg) or any(
            backend in ("minicpm_flashattn", "minicpm_flashinfer")
            for backend in (
                cfg.attention_backend,
                cfg.prefill_attention_backend,
                cfg.decode_attention_backend,
            )
        )
        if uses_sparse_backend and cfg.disaggregation_mode != "null":
            raise ValueError(
                "MiniCPM sparse attention does not support PD disaggregation"
            )
        if is_attention_backend_not_set(cfg):
            overrides["attention_backend"] = (
                "minicpm_flashinfer"
                if get_platform().is_blackwell
                else "minicpm_flashattn"
            )
    return overrides
