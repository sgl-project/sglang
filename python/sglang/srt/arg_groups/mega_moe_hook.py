from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolving_view,
)

logger = logging.getLogger(__name__)


def handle_mega_moe(server_args: ServerArgs) -> None:
    handle_moe_runner_backend_alias(server_args)
    handle_flashinfer_megamoe(server_args)


def handle_moe_runner_backend_alias(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    if cfg.moe_runner_backend != "megamoe":
        return

    if cfg.moe_a2a_backend not in ("none", "megamoe"):
        logger.warning(
            "--moe-runner-backend megamoe is an alias for "
            "--moe-a2a-backend megamoe; overriding "
            "--moe-a2a-backend %s.",
            cfg.moe_a2a_backend,
        )
    declare_resolution(
        server_args,
        "handle_moe_runner_backend_alias",
        moe_runner_backend="auto",
        moe_a2a_backend="megamoe",
    )


def handle_flashinfer_megamoe(server_args: ServerArgs) -> None:
    """Bind the FlashInfer MegaMoE runner to SGLang's fused MegaMoE path."""
    cfg = resolving_view(server_args)
    if cfg.moe_runner_backend != "flashinfer_megamoe":
        return

    if cfg.enable_two_batch_overlap:
        raise ValueError(
            "--moe-runner-backend flashinfer_megamoe does not support "
            "--enable-two-batch-overlap yet; disable TBO for this backend."
        )
    if cfg.moe_a2a_backend not in ("none", "megamoe"):
        raise ValueError(
            "--moe-runner-backend flashinfer_megamoe owns dispatch and combine; "
            "it cannot be combined with --moe-a2a-backend "
            f"{cfg.moe_a2a_backend!r}."
        )
    if cfg.enable_eplb or cfg.ep_num_redundant_experts != 0:
        raise ValueError(
            "FlashInfer MegaMoE currently requires an even, non-replicated "
            "expert placement; disable EPLB and redundant experts."
        )
    if cfg.flashinfer_megamoe_max_num_tokens <= 0:
        raise ValueError("--flashinfer-megamoe-max-num-tokens must be positive")

    if not cfg.disable_shared_experts_fusion:
        logger.warning(
            "FlashInfer MegaMoE computes the shared expert separately from the "
            "routed MegaMoE kernel; enabling --disable-shared-experts-fusion."
        )
    declare_resolution(
        server_args,
        "handle_flashinfer_megamoe",
        moe_a2a_backend="megamoe",
        disable_shared_experts_fusion=True,
    )
