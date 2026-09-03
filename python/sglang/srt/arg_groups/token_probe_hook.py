from __future__ import annotations

import os
from typing import Any

from sglang.srt.arg_groups.overrides import model_config_of, resolving_view


def validate_token_probe(server_args: Any) -> None:
    cfg = resolving_view(server_args)
    if cfg.probe_ckpt is None:
        return

    architectures = model_config_of(server_args).hf_config.architectures or []
    if "BailingMoeV3ForCausalLM" not in architectures:
        raise ValueError(
            "--probe-ckpt currently supports only Bailing V3 MoE "
            "(BailingMoeV3ForCausalLM); got "
            f"architectures={architectures!r}."
        )
    if cfg.pp_size != 1:
        raise ValueError("--probe-ckpt does not support pipeline parallelism")
    if cfg.attn_cp_size != 1 or cfg.dcp_size != 1:
        raise ValueError("--probe-ckpt does not support context parallelism")

    if cfg.speculative_algorithm is None:
        return

    draft_path = cfg.speculative_draft_model_path
    same_checkpoint = draft_path is None
    if draft_path is not None:
        if os.path.exists(draft_path) or os.path.exists(cfg.model_path):
            same_checkpoint = os.path.realpath(draft_path) == os.path.realpath(
                cfg.model_path
            )
        else:
            same_checkpoint = draft_path.rstrip("/") == cfg.model_path.rstrip("/")

    if (
        cfg.speculative_algorithm != "EAGLE"
        or cfg.speculative_eagle_topk != 1
        or not same_checkpoint
    ):
        raise ValueError(
            "--probe-ckpt supports speculative decoding only with bundled "
            "MTP/NEXTN from the target checkpoint; got "
            f"algorithm={cfg.speculative_algorithm!r}, "
            f"topk={cfg.speculative_eagle_topk!r}, "
            f"draft_model={draft_path!r}."
        )
