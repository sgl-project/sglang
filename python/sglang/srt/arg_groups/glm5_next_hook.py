from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def apply_glm5_next_spec_backend_defaults(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    platform = get_platform()

    if (
        cfg.speculative_algorithm is None
        or cfg.linear_attn_verify_backend is not None
        or not (
            platform.is_sm100
            # SM90: measured end-to-end win with the ReplaySSM ring, mixed without it.
            or (platform.is_sm90 and cfg.enable_linear_replayssm_spec)
        )
    ):
        return

    declare_resolution(
        server_args,
        "apply_glm5_next_spec_backend_defaults",
        linear_attn_verify_backend="nv_cutedsl",
    )
    logger.info(
        "GLM-5-Next with speculative decoding: defaulting "
        "--linear-attn-verify-backend to nv_cutedsl."
    )
