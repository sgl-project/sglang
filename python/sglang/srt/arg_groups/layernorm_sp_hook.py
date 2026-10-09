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
    validate_layernorm_sp(
        architecture=architectures[0] if architectures else None,
        tp_size=cfg.tp_size,
        attn_dp_enabled=attn_dp_enabled_of(cfg),
        speculative_algorithm=cfg.speculative_algorithm,
        pp_size=cfg.pp_size,
    )


def validate_layernorm_sp(
    *,
    architecture: Optional[str],
    tp_size: int,
    attn_dp_enabled: bool,
    speculative_algorithm: Optional[str],
    pp_size: int = 1,
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
    if pp_size > 1:
        raise ValueError(
            "--enable-layernorm-sp is not compatible with pipeline parallelism: "
            "only the first rank's stack starts the sequence-sharded region, so "
            "the next rank would receive a stream it does not expect."
        )
