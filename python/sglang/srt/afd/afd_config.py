"""AFD role definition and server-argument resolution."""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Optional

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    model_config_of,
    resolving_view,
)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class AFDRole(Enum):
    ATTN = "attn"
    FFN = "ffn"

    def is_attn(self) -> bool:
        return self == AFDRole.ATTN

    def is_ffn(self) -> bool:
        return self == AFDRole.FFN


def afd_role_from_args(server_args: ServerArgs) -> Optional[AFDRole]:
    if server_args.afd_role is None:
        return None
    return AFDRole(server_args.afd_role)


def is_afd_attn_moe_scattered() -> bool:
    from sglang.srt.runtime_context import get_disagg

    server_args = get_disagg()
    role = afd_role_from_args(server_args)
    return (
        role is not None
        and role.is_attn()
        and server_args.afd_ffn_moe_backend == "deepep"
    )


def prepare_afd_ffn_moe_backend(server_args: ServerArgs) -> None:
    """Separate AFD's logical expert layout from the local MoE backend."""
    cfg = resolving_view(server_args)
    role = afd_role_from_args(server_args)
    if role is None:
        return

    requested_backend = cfg.moe_a2a_backend
    if role.is_attn() and requested_backend == "afd":
        requested_backend = cfg.afd_ffn_moe_backend
    declare_resolution(
        server_args,
        "prepare_afd_ffn_moe_backend",
        afd_ffn_moe_backend=requested_backend,
    )
    if role.is_attn():
        declare_resolution(
            server_args, "prepare_afd_ffn_moe_backend", moe_a2a_backend="afd"
        )


def check_afd_server_args(server_args: ServerArgs) -> None:
    """Resolve AFD roles and reject unsupported execution combinations."""
    cfg = resolving_view(server_args)
    role = afd_role_from_args(server_args)
    if role is None:
        if cfg.moe_a2a_backend == "afd":
            raise ValueError("--moe-a2a-backend afd requires --afd-role attn")
        return

    prepare_afd_ffn_moe_backend(server_args)
    _check_common_args(server_args)
    _configure_cuda_graph(server_args, role)
    if role.is_attn():
        _check_attn_args(server_args)
    else:
        _check_ffn_args(server_args)
    logger.info("AFD enabled: role=%s backend=%s", role.value, cfg.afd_backend)


def _check_common_args(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    unsupported = {
        "pp_size": (cfg.pp_size, 1),
        "speculative_algorithm": (cfg.speculative_algorithm, None),
        "enable_torch_compile": (cfg.enable_torch_compile, False),
        "enable_eplb": (cfg.enable_eplb, False),
        "enable_waterfill": (cfg.enable_waterfill, False),
        "enable_two_batch_overlap": (cfg.enable_two_batch_overlap, False),
        "enable_single_batch_overlap": (cfg.enable_single_batch_overlap, False),
        "enable_pdmux": (cfg.enable_pdmux, False),
        "attn_cp_size": (cfg.attn_cp_size, 1),
        "enable_prefill_cp": (cfg.enable_prefill_cp, False),
        "enforce_shared_experts_fusion": (
            cfg.enforce_shared_experts_fusion,
            False,
        ),
    }
    for name, (value, allowed) in unsupported.items():
        if value != allowed:
            raise ValueError(
                f"AFD does not support {name}={value!r} yet (requires {allowed!r})"
            )

    if not cfg.disable_shared_experts_fusion:
        logger.info("AFD: disabling shared-expert fusion")
        declare_resolution(
            server_args, "_check_common_args", disable_shared_experts_fusion=True
        )


def _configure_cuda_graph(server_args: ServerArgs, role: AFDRole) -> None:
    cfg = resolving_view(server_args)
    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase

    config = cfg.cuda_graph_config
    if config.prefill.backend != Backend.DISABLED:
        logger.info("AFD: disabling prefill CUDA graph")
        config = with_phase(config, Phase.PREFILL, backend=Backend.DISABLED)

    if role.is_attn():
        if config.decode.backend not in (
            Backend.FULL,
            Backend.BREAKABLE,
            Backend.DISABLED,
        ):
            raise ValueError(
                "AFD attn instance requires decode CUDA graph backend 'full', "
                f"'breakable', or 'disabled'; got {config.decode.backend!r}"
            )
        if config.decode.backend != Backend.DISABLED and cfg.debug_cuda_graph:
            raise ValueError("AFD attn instance does not support --debug-cuda-graph")
    elif config.decode.backend not in (Backend.FULL, Backend.DISABLED):
        raise ValueError(
            "AFD ffn instance requires decode CUDA graph backend 'full' or "
            f"'disabled'; got {config.decode.backend!r}"
        )

    if config is not cfg.cuda_graph_config:
        declare_resolution(
            server_args, "_configure_cuda_graph", cuda_graph_config=config
        )


_DEFAULT_NON_STANDARD_TOPK_RUNNERS = (
    "flashinfer_trtllm",
    "experimental_sgl_trtllm",
    "flashinfer_mxfp4",
    "triton_kernel",
)

_TRITON_INCOMPATIBLE_QUANTS = (
    "nvfp4_online",
    "mxfp8",
    "modelopt_fp4",
    "modelopt_mixed",
)


def _normalize_ffn_runner_backend(server_args: ServerArgs) -> None:
    """Ensure the F FULL runner can consume A's materialized routing."""
    cfg = resolving_view(server_args)
    runner_backend = cfg.moe_runner_backend
    if runner_backend not in _DEFAULT_NON_STANDARD_TOPK_RUNNERS:
        return

    if runner_backend == "flashinfer_trtllm" and cfg.quantization in (
        "modelopt_fp4",
        "nvfp4_online",
    ):
        # The FP4 runner switches to its routed kernel when given STANDARD topk.
        return

    if (
        runner_backend == "flashinfer_mxfp4"
        and model_config_of(server_args).is_fp4_experts
    ):
        return

    if cfg.quantization in _TRITON_INCOMPATIBLE_QUANTS:
        raise ValueError(
            "AFD ffn instance with --moe-a2a-backend none requires a runner "
            "that accepts externally routed STANDARD topk, but "
            f"moe_runner_backend={runner_backend!r} is incompatible and quantization "
            f"{cfg.quantization!r} cannot fall back to triton. Pass a "
            "STANDARD-compatible --moe-runner-backend explicitly."
        )

    logger.info(
        "AFD ffn role: overriding moe_runner_backend %r -> 'triton' "
        "to consume externally routed STANDARD topk",
        runner_backend,
    )
    declare_resolution(
        server_args, "_normalize_ffn_runner_backend", moe_runner_backend="triton"
    )


def _check_attn_args(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    if cfg.disaggregation_mode not in ("null", "decode"):
        raise ValueError(
            "AFD attn instances support disaggregation_mode 'null' or 'decode'; "
            f"got {cfg.disaggregation_mode!r}"
        )
    if cfg.afd_ffn_addr is None:
        raise ValueError("--afd-ffn-addr is required for --afd-role attn")
    if cfg.moe_a2a_backend != "afd":
        raise ValueError(
            "AFD attn instance requires the internal MoE backend 'afd'; "
            f"got {cfg.moe_a2a_backend!r}"
        )
    if cfg.afd_ffn_moe_backend not in ("none", "deepep"):
        raise ValueError(
            "AFD supports F-internal --moe-a2a-backend none or deepep; "
            f"got {cfg.afd_ffn_moe_backend!r}"
        )
    if cfg.ep_size != 1 or cfg.moe_dp_size != 1:
        raise ValueError(
            "AFD attn instance requires --ep-size 1 and --moe-dp-size 1; "
            "configure expert parallelism only on the ffn instance"
        )
    # FULL uses the native DP/TP collectives for expert inputs and outputs.
    if cfg.tp_size > 1 and not (cfg.enable_dp_attention and cfg.dp_size == cfg.tp_size):
        raise ValueError(
            "AFD attn instance requires attn_tp_size == 1: with --tp > 1, pass "
            "--enable-dp-attention --dp-size equal to --tp-size"
        )


def _check_ffn_args(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    if cfg.disaggregation_mode != "null":
        raise ValueError(
            "AFD ffn instances require disaggregation_mode 'null'; "
            f"got {cfg.disaggregation_mode!r}"
        )
    if cfg.afd_ffn_addr is not None:
        raise ValueError(
            "--afd-ffn-addr is an attn-side arg pointing at the ffn "
            "instance; the ffn instance itself only needs --afd-role ffn"
        )
    if cfg.moe_a2a_backend not in ("none", "deepep"):
        raise ValueError(
            "AFD ffn instance supports --moe-a2a-backend none or deepep; "
            f"got {cfg.moe_a2a_backend!r}"
        )
    if cfg.moe_a2a_backend == "none":
        _normalize_ffn_runner_backend(server_args)
    # Minimal parameter surface: the ffn instance has no attention, so
    # attention-only parallel knobs would silently misconfigure its
    # parallel groups and memory pools.
    if cfg.enable_dp_attention or cfg.dp_size > 1:
        raise ValueError(
            "AFD ffn instance must not use --enable-dp-attention/--dp-size; "
            "it only accepts tp/ep parallel args"
        )
    if not cfg.skip_server_warmup:
        # The ffn event loop never serves generation requests, so HTTP warmup
        # would block forever waiting for a completion.
        logger.info("AFD ffn role: forcing skip_server_warmup=True")
        declare_resolution(server_args, "_check_ffn_args", skip_server_warmup=True)
