"""AFD role definition and server-args validation.

The role is derived from ``ServerArgs.afd_role`` so every process spawned
with the same args agrees on it without extra plumbing.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Optional

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    model_config_of,
    resolving_view,
)
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_DEFAULT_COMM_SM_BUDGETS = {
    "nccl": 8,
    "nvshmem": 2,
    "direct_deepep": 24,
    "direct_megamoe": 24,
}

_DIRECT_AFD_BACKENDS = frozenset({"direct_deepep", "direct_megamoe"})


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


def is_direct_afd_backend(backend: str) -> bool:
    return backend in _DIRECT_AFD_BACKENDS


def requires_afd_mega_moe_weights() -> bool:
    from sglang.srt.runtime_context import get_disagg

    server_args = get_disagg()
    role = afd_role_from_args(server_args)
    return (
        role is not None
        and role.is_ffn()
        and server_args.afd_backend == "direct_megamoe"
    )


def resolve_overlap_sm_partition(
    *,
    total_num_sms: int,
    num_microbatches: int,
    overlap_sm_budget: int,
    direct_deepep_communication_num_sms_per_lane: Optional[int] = None,
) -> tuple[int, int]:
    communication_num_sms = overlap_sm_budget
    if direct_deepep_communication_num_sms_per_lane is not None:
        # The compute lane is between its dispatch and combine, so only the
        # other lanes can own communication CTAs concurrently.
        communication_num_sms = (
            num_microbatches - 1
        ) * direct_deepep_communication_num_sms_per_lane

    compute_num_sms = total_num_sms - communication_num_sms
    if compute_num_sms <= 0 or compute_num_sms % 2 != 0:
        raise RuntimeError(
            "AFD overlap requires a positive, even compute SM count: "
            f"total={total_num_sms}, reserved={communication_num_sms}, "
            f"compute={compute_num_sms}"
        )
    return compute_num_sms, communication_num_sms


def normalize_afd_ffn_speculative_args(server_args: ServerArgs) -> None:
    """Ignore speculative decoding arguments on the expert-only F instance."""
    role = afd_role_from_args(server_args)
    if role is None or not role.is_ffn():
        return

    declare_resolution(
        server_args,
        "normalize_afd_ffn_speculative_args",
        speculative_algorithm=None,
        speculative_draft_model_path=None,
        speculative_num_steps=None,
        speculative_eagle_topk=None,
        speculative_num_draft_tokens=None,
    )


def prepare_afd_ffn_moe_backend(server_args: ServerArgs) -> None:
    """Separate AFD's logical expert layout from the local MoE backend."""
    cfg = resolving_view(server_args)
    role = afd_role_from_args(server_args)
    if role is None:
        return

    if is_direct_afd_backend(cfg.afd_backend):
        declare_resolution(
            server_args,
            "prepare_afd_ffn_moe_backend",
            afd_ffn_moe_backend="deepep",
            moe_a2a_backend="afd" if role.is_attn() else "deepep",
        )
        if role.is_ffn():
            if cfg.moe_runner_backend == "auto":
                declare_resolution(
                    server_args,
                    "prepare_afd_ffn_moe_backend",
                    moe_runner_backend=(
                        "flashinfer_cutedsl"
                        if cfg.quantization == "modelopt_fp4"
                        else "deep_gemm"
                    ),
                )
            # Direct owns decode. This mode only selects the native dispatcher
            # used by the staged prefill fallback.
            declare_resolution(
                server_args,
                "prepare_afd_ffn_moe_backend",
                deepep_mode="low_latency"
                if cfg.moe_runner_backend == "flashinfer_cutedsl"
                else "auto",
            )
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


def is_afd_attn_moe_scattered() -> bool:
    from sglang.srt.runtime_context import get_disagg

    server_args = get_disagg()
    role = afd_role_from_args(server_args)
    return (
        role is not None
        and role.is_attn()
        and server_args.afd_ffn_moe_backend == "deepep"
    )


def check_afd_server_args(server_args: ServerArgs) -> None:
    """Validate AFD and declare its resolved configuration before publishing.

    No-op when AFD is disabled. Raises ValueError on unsupported combinations
    (fail-fast at startup instead of hanging at runtime).
    """
    cfg = resolving_view(server_args)
    role = afd_role_from_args(server_args)
    afd_decode_num_microbatches = cfg.afd_decode_num_microbatches
    if cfg.afd_direct_deepep_disable_hybrid and (
        role is not AFDRole.FFN or cfg.afd_backend != "direct_deepep"
    ):
        raise ValueError(
            "--afd-direct-deepep-disable-hybrid requires --afd-role ffn "
            "and --afd-backend direct_deepep"
        )
    expert_alignment = envs.SGLANG_AFD_DIRECT_DEEPEP_EXPERT_ALIGNMENT.get()
    if expert_alignment not in (16, 32, 64, 128):
        raise ValueError(
            "SGLANG_AFD_DIRECT_DEEPEP_EXPERT_ALIGNMENT must be 16, 32, 64, or 128"
        )
    if expert_alignment != 128 and (
        role is not AFDRole.FFN or cfg.afd_backend != "direct_deepep"
    ):
        raise ValueError(
            "SGLANG_AFD_DIRECT_DEEPEP_EXPERT_ALIGNMENT requires --afd-role ffn "
            "and --afd-backend direct_deepep"
        )
    _check_comm_sm_budget_args(server_args, role)
    if role is None:
        if is_direct_afd_backend(cfg.afd_backend):
            raise ValueError(f"--afd-backend {cfg.afd_backend} requires --afd-role")
        if afd_decode_num_microbatches != 1:
            raise ValueError("--afd-decode-num-microbatches requires --afd-role attn")
        if cfg.afd_attn_instances != 1:
            raise ValueError("--afd-attn-instances requires --afd-role ffn")
        if cfg.moe_a2a_backend == "afd":
            raise ValueError("--moe-a2a-backend afd requires --afd-role attn")
        return

    prepare_afd_ffn_moe_backend(server_args)
    _check_common_args(server_args)
    _check_direct_args(server_args, role)
    _configure_cuda_graph(server_args, role)
    _check_decode_microbatch_args(server_args, role, afd_decode_num_microbatches)
    if role.is_attn():
        _check_attn_args(server_args)
        endpoint = f"ffn_addr={cfg.afd_ffn_addr}"
    else:
        _check_ffn_args(server_args)
        endpoint = f"api_port={cfg.port}"

    logger.info(
        "AFD enabled: role=%s backend=%s %s",
        role.value,
        cfg.afd_backend,
        endpoint,
    )


def finalize_afd_speculative_args(server_args: ServerArgs) -> None:
    """Validate AFD after speculative arguments have been normalized."""
    role = afd_role_from_args(server_args)
    if role is not None:
        cfg = resolving_view(server_args)
        if (
            role.is_attn()
            and cfg.speculative_algorithm == "DSPARK"
            and cfg.speculative_moe_runner_backend == "auto"
            and model_config_of(server_args).is_fp4_experts
        ):
            declare_resolution(
                server_args,
                "finalize_afd_speculative_args",
                speculative_moe_runner_backend="deep_gemm",
            )
        _check_speculative_args(server_args, role)


def _check_comm_sm_budget_args(
    server_args: ServerArgs, role: Optional[AFDRole]
) -> None:
    cfg = resolving_view(server_args)
    budget = cfg.afd_comm_sm_budget
    if role is None:
        if budget is not None:
            raise ValueError("--afd-comm-sm-budget requires --afd-role ffn")
        return
    if role.is_attn():
        declare_resolution(
            server_args, "_check_comm_sm_budget_args", afd_comm_sm_budget=None
        )
        return
    if budget is None:
        declare_resolution(
            server_args,
            "_check_comm_sm_budget_args",
            afd_comm_sm_budget=_DEFAULT_COMM_SM_BUDGETS[cfg.afd_backend],
        )
    elif budget <= 0:
        raise ValueError("--afd-comm-sm-budget must be positive")


def _check_decode_microbatch_args(
    server_args: ServerArgs,
    role: AFDRole,
    num_microbatches: int,
) -> None:
    cfg = resolving_view(server_args)
    if num_microbatches == 1:
        return
    if not role.is_attn():
        raise ValueError("--afd-decode-num-microbatches is an attn-side argument")
    if num_microbatches not in (2, 3):
        raise ValueError(
            "--afd-decode-num-microbatches currently supports only 1, 2, or 3"
        )
    if not cfg.enable_dp_attention or cfg.dp_size <= 1:
        raise ValueError(
            "--afd-decode-num-microbatches greater than 1 requires DP attention with "
            "--dp-size greater than 1"
        )
    if cfg.tp_size != cfg.dp_size:
        raise ValueError(
            "--afd-decode-num-microbatches greater than 1 currently requires "
            "attention TP size 1 "
            "(--tp must equal --dp-size)"
        )
    if cfg.enable_lora:
        raise ValueError(
            "--afd-decode-num-microbatches greater than 1 currently does not support LoRA"
        )


def _check_common_args(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    unsupported = {
        "pp_size": (cfg.pp_size, 1),
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


def _check_speculative_args(server_args: ServerArgs, role: AFDRole) -> None:
    cfg = resolving_view(server_args)
    algorithm = cfg.speculative_algorithm
    if algorithm is None:
        return
    if not role.is_attn() or algorithm not in ("EAGLE", "DSPARK"):
        raise ValueError(
            "AFD speculative decoding requires an attn instance with "
            f"speculative_algorithm in {{'EAGLE', 'DSPARK'}}; got role={role.value!r}, "
            f"speculative_algorithm={algorithm!r}"
        )

    topk = cfg.speculative_eagle_topk
    if topk != 1:
        raise ValueError(
            f"AFD speculative_algorithm={algorithm!r} requires "
            f"--speculative-eagle-topk 1; got {topk!r}"
        )

    unsupported = {
        "speculative_adaptive": (
            cfg.speculative_adaptive,
            False,
        ),
        "enable_multi_layer_eagle": (
            cfg.enable_multi_layer_eagle,
            False,
        ),
        "speculative_skip_dp_mlp_sync": (
            cfg.speculative_skip_dp_mlp_sync,
            False,
        ),
        "enable_torch_compile": (
            cfg.enable_torch_compile,
            False,
        ),
        "decoupled_spec_role": (
            cfg.decoupled_spec_role,
            "null",
        ),
    }
    for name, (value, allowed) in unsupported.items():
        if value != allowed:
            raise ValueError(
                f"AFD {algorithm} does not support "
                f"{name}={value!r} yet (requires {allowed!r})"
            )

    from sglang.srt.model_executor.cuda_graph_config import Backend

    if cfg.cuda_graph_config.decode.backend != Backend.FULL:
        raise ValueError(
            f"AFD {algorithm} attn instance requires decode CUDA graph backend 'full'; "
            f"got {cfg.cuda_graph_config.decode.backend!r}"
        )

    draft_moe_backend = cfg.speculative_moe_a2a_backend
    if draft_moe_backend is None:
        declare_resolution(
            server_args, "_check_speculative_args", speculative_moe_a2a_backend="none"
        )
    elif draft_moe_backend != "none":
        raise ValueError(
            f"AFD {algorithm} currently requires draft-local "
            f"--speculative-moe-a2a-backend none; got {draft_moe_backend!r}"
        )


def get_afd_decode_num_tokens_per_bs() -> int:
    """Return the fixed target-forward rows per request negotiated by AFD."""
    from sglang.srt.runtime_context import get_disagg, get_spec

    role = afd_role_from_args(get_disagg())
    spec = get_spec()
    if (
        role is not None
        and role.is_attn()
        and spec.speculative_algorithm in ("EAGLE", "DSPARK")
    ):
        num_tokens_per_bs = spec.speculative_num_draft_tokens
        if not isinstance(num_tokens_per_bs, int) or num_tokens_per_bs <= 0:
            raise RuntimeError(
                "AFD speculative args were not validated before building the handshake"
            )
        return num_tokens_per_bs
    return 1


def _configure_cuda_graph(server_args: ServerArgs, role: AFDRole) -> None:
    cfg = resolving_view(server_args)
    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase

    config = cfg.cuda_graph_config
    if config.prefill.backend != Backend.DISABLED:
        logger.info("AFD: disabling prefill CUDA graph")
        config = with_phase(config, Phase.PREFILL, backend=Backend.DISABLED)

    if (
        is_direct_afd_backend(cfg.afd_backend)
        and config.decode.backend != Backend.DISABLED
        and config.decode.backend != Backend.FULL
    ):
        logger.info(
            "AFD direct backend: switching %s decode CUDA graph backend "
            "from %s to full",
            role.value,
            config.decode.backend,
        )
        config = with_phase(config, Phase.DECODE, backend=Backend.FULL)

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


def _check_direct_args(server_args: ServerArgs, role: AFDRole) -> None:
    cfg = resolving_view(server_args)
    if not is_direct_afd_backend(cfg.afd_backend):
        return
    if cfg.afd_backend == "direct_megamoe":
        if cfg.ep_num_redundant_experts != 0:
            raise ValueError("AFD Direct MegaMoE does not support redundant experts")
        if cfg.init_expert_location != "trivial":
            raise ValueError("AFD Direct MegaMoE requires trivial expert placement")
    if role.is_ffn():
        supported_runners = ("deep_gemm", "flashinfer_cutedsl")
        if cfg.moe_runner_backend not in supported_runners:
            raise ValueError(
                f"AFD {cfg.afd_backend} requires --moe-runner-backend "
                f"{' or '.join(supported_runners)} on "
                f"the ffn instance, got {cfg.moe_runner_backend!r}"
            )
        if cfg.moe_runner_backend == "flashinfer_cutedsl":
            declare_resolution(
                server_args, "_check_direct_args", deepep_mode="low_latency"
            )
        else:
            declare_resolution(server_args, "_check_direct_args", deepep_mode="normal")
    if cfg.enable_torch_compile:
        raise ValueError(
            "AFD direct backends do not support --enable-torch-compile yet"
        )
    if role.is_ffn() and (cfg.ep_size != cfg.tp_size or cfg.moe_dp_size != 1):
        raise ValueError(
            "AFD direct ffn instances require pure expert parallelism: "
            "--ep-size must equal --tp-size and --moe-dp-size must be 1"
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
    if cfg.afd_attn_instances != 1:
        raise ValueError(
            "--afd-attn-instances is an ffn-side arg declaring how many "
            "attn instances the ffn instance serves"
        )
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
    # FULL gathers expert inputs and combines ffn shard outputs through the
    # attn instance's native DP/TP collectives. Additional attn ranks can
    # contribute zero remote routed output to the same collective.
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
    if cfg.afd_attn_instances < 1:
        raise ValueError("--afd-attn-instances must be at least 1")
    if (
        cfg.afd_attn_instances > 1
        and not is_direct_afd_backend(cfg.afd_backend)
        and envs.SGLANG_AFD_STAGED_COHORT_JOIN_TIMEOUT_US.get() < 0
    ):
        raise ValueError(
            "SGLANG_AFD_STAGED_COHORT_JOIN_TIMEOUT_US must be non-negative"
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
