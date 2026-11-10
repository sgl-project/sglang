"""Configuration for the Ascend DSA DCP KV layout."""

from sglang.srt.environ import envs


def npu_dcp_interleave_size(page_size: int, *, use_dsa: bool) -> int:
    if use_dsa and envs.SGLANG_NPU_DSA_DCP_TOKEN_INTERLEAVE.get():
        return 1
    return page_size


def validate_dsa_token_dcp_config(server_args) -> None:
    if not envs.SGLANG_NPU_DSA_DCP_TOKEN_INTERLEAVE.get():
        return
    from sglang.srt.arg_groups.model_override_base import (
        model_config_of,
        resolving_view,
    )
    from sglang.srt.configs.model_config import is_deepseek_dsa

    if not is_deepseek_dsa(model_config_of(server_args).hf_config):
        return
    cfg = resolving_view(server_args)
    if cfg.enable_unified_memory:
        raise ValueError("NPU DSA token DCP requires a static KV pool.")
    if cfg.disaggregation_mode != "null" and (
        cfg.disaggregation_transfer_backend != "mooncake"
        or cfg.enable_prefill_cp
        or (cfg.dcp_size == 1) != (cfg.disaggregation_mode == "prefill")
    ):
        raise ValueError(
            "NPU DSA token DCP PD requires Mooncake, prefill CP=1, "
            "and prefill DCP1 -> decode DCP-N (N > 1)."
        )
