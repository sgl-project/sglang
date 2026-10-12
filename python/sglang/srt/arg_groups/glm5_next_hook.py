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

    if (
        cfg.speculative_algorithm is None
        or cfg.linear_attn_verify_backend is not None
        or not get_platform().is_sm100
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
