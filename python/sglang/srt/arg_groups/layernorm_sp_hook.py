from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

from sglang.srt.arg_groups.overrides import model_config_of, resolving_view
from sglang.srt.runtime_context import attn_dp_enabled_of

logger = logging.getLogger(__name__)


def handle_layernorm_sp(server_args: ServerArgs) -> None:
    """Validate --enable-layernorm-sp against the resolved parallelism config.

    Runs in the resolution pipeline rather than in the layers so a model that
    does not use layer boundaries rejects the flag instead of ignoring it.
    """
    cfg = resolving_view(server_args)
    if not cfg.enable_layernorm_sp:
        return
    architectures = model_config_of(server_args).hf_config.architectures
    architecture = architectures[0] if architectures else None
    validate_layernorm_sp(
        architecture=architecture,
        tp_size=cfg.tp_size,
        ep_size=cfg.ep_size,
        pp_size=cfg.pp_size,
        attn_cp_size=cfg.attn_cp_size,
        attn_dp_enabled=attn_dp_enabled_of(cfg),
        speculative_algorithm=cfg.speculative_algorithm,
    )


def validate_layernorm_sp(
    *,
    architecture: Optional[str],
    tp_size: int,
    ep_size: int,
    pp_size: int,
    attn_cp_size: int,
    attn_dp_enabled: bool,
    speculative_algorithm: Optional[str],
) -> None:
    """Fail loud for unsupported / incompatible configs. Callers gate on the flag."""
    from sglang.srt.layers.layernorm_sp import SP_SUPPORTED_ARCHITECTURES

    if architecture not in SP_SUPPORTED_ARCHITECTURES:
        raise ValueError(
            "--enable-layernorm-sp is only supported for "
            f"{sorted(SP_SUPPORTED_ARCHITECTURES)}; got {architecture}."
        )
    if tp_size <= 1:
        raise ValueError(
            "--enable-layernorm-sp requires tp_size > 1: there is no sequence to "
            "shard across a single TP rank."
        )
    if architecture and architecture.startswith("Qwen4Exp"):
        if ep_size != tp_size:
            raise ValueError(
                "--enable-layernorm-sp requires ep_size == tp_size for Qwen4Exp; "
                f"got ep_size={ep_size}, tp_size={tp_size}."
            )
        if pp_size != 1:
            raise ValueError(
                "--enable-layernorm-sp requires pp_size == 1 for Qwen4Exp; "
                f"got pp_size={pp_size}."
            )
        if attn_cp_size != 1:
            raise ValueError(
                "--enable-layernorm-sp requires attn_cp_size == 1 for Qwen4Exp; "
                f"got attn_cp_size={attn_cp_size}."
            )
    if attn_dp_enabled:
        raise ValueError(
            "--enable-layernorm-sp is not compatible with attention DP: "
            "SP shards the sequence across the full TP group, which under DP "
            "attention spans data-parallel groups holding different sequences."
        )
    if speculative_algorithm is not None:
        raise ValueError(
            "--enable-layernorm-sp is not compatible with speculative decoding "
            "(EAGLE/EAGLE3): the captured aux hidden states would be "
            "sequence-sharded."
        )
