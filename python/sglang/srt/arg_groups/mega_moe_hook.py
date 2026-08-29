from __future__ import annotations

import logging
import os
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
    handle_w4a4_mxfp4_megamoe_env(server_args)


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


def handle_w4a4_mxfp4_megamoe_env(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    if not cfg.enable_w4a4_mxfp4_megamoe:
        return

    os.environ["DG_USE_FP4_ACTS"] = "1"
    os.environ["DG_USE_MXF4_KIND"] = "1"
