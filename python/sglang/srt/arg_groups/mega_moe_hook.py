from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

from sglang.srt.arg_groups.overrides import declare_resolution

logger = logging.getLogger(__name__)


def handle_mega_moe(server_args: ServerArgs) -> None:
    handle_moe_runner_backend_alias(server_args)
    handle_w4a4_mxfp4_megamoe_env(server_args)
    handle_rocm_megamoe(server_args)


def handle_moe_runner_backend_alias(server_args: ServerArgs) -> None:
    if server_args.moe_runner_backend != "megamoe":
        return

    if server_args.moe_a2a_backend not in ("none", "megamoe"):
        logger.warning(
            "--moe-runner-backend megamoe is an alias for "
            "--moe-a2a-backend megamoe; overriding "
            "--moe-a2a-backend %s.",
            server_args.moe_a2a_backend,
        )
    declare_resolution(
        server_args,
        "handle_moe_runner_backend_alias",
        moe_runner_backend="auto",
        moe_a2a_backend="megamoe",
    )


def handle_w4a4_mxfp4_megamoe_env(server_args: ServerArgs) -> None:
    if not server_args.enable_w4a4_mxfp4_megamoe:
        return

    os.environ["DG_USE_FP4_ACTS"] = "1"
    os.environ["DG_USE_MXF4_KIND"] = "1"


def handle_rocm_megamoe(server_args: ServerArgs) -> None:
    if server_args.moe_a2a_backend != "megamoe":
        return

    from sglang.srt.environ import envs
    from sglang.srt.utils import is_hip

    if not is_hip():
        return

    mtpr = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    if mtpr <= 0 or (mtpr & (mtpr - 1)) != 0:
        raise ValueError(
            "MegaMoE on ROCm requires "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK to be a "
            f"positive power of two (AITER MegaMoEV2 P2P wire format), got {mtpr}"
        )
    if not envs.SGLANG_USE_AITER.get():
        raise ValueError(
            "MegaMoE on ROCm requires SGLANG_USE_AITER=1 and an AITER build "
            "that exports aiter.ops.flydsl.kernels.mega_moe.MegaMoEV2 "
            "(ROCm/aiter#4439)."
        )
    logger.info(
        "MegaMoE on ROCm: using AITER FlyDSL MegaMoEV2 (mtpr=%s, ep_size=%s).",
        mtpr,
        server_args.ep_size,
    )
