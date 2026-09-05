# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for context- and decode-context parallelism."""

from __future__ import annotations

import logging
import os
from typing import Any

from sglang.srt.arg_groups.overrides import (
    _data_parallelism_defaults,
    _dp_lm_head_validation,
    _tp_lm_head_all_to_all_default,
    declare_resolution,
    model_config_of,
    resolved_view,
    resolving_view,
    run_post_process_pass,
    should_report_expert_balancedness,
)
from sglang.srt.connector import ConnectorType
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import parse_connector_type

logger = logging.getLogger(__name__)


def handle_context_parallelism(server_args: Any):

    cfg = resolving_view(server_args)
    if parse_connector_type(cfg.model_path) != ConnectorType.INSTANCE:
        model_config = model_config_of(server_args)
        hf_config = model_config.hf_config
        model_arch = hf_config.architectures[0]
        platform = get_platform()
        if (
            cfg.enable_prefill_cp
            and model_arch == "DeepseekV32ForCausalLM"
            and cfg.cp_strategy == "zigzag"
            and not (platform.is_hip or platform.is_npu)
        ):
            raise ValueError(
                "DeepSeek V3.2 prefill CP does not support --cp-strategy "
                "zigzag; use interleave."
            )
        if cfg.enable_prefill_cp and model_arch in (
            "MiMoV2ForCausalLM",
            "MiMoV2FlashForCausalLM",
        ):
            if cfg.cp_strategy != "zigzag":
                raise ValueError(
                    "MiMo V2 prefill CP only supports --cp-strategy zigzag."
                )
            if (
                model_config.is_multimodal
                and not cfg.language_only
                and not cfg.language_model_only
            ):
                raise ValueError(
                    "MiMo V2 prefill CP only supports text inference; add "
                    "--language-only."
                )

    if cfg.enable_prefill_cp and cfg.cp_strategy is None:
        raise ValueError(
            "--cp-strategy must be set when --enable-prefill-cp is enabled."
        )

    if cfg.enable_prefill_context_parallel and cfg.enable_dsa_prefill_context_parallel:
        raise ValueError(
            "--enable-prefill-context-parallel and "
            "--enable-nsa-prefill-context-parallel are mutually "
            "exclusive. Use --enable-nsa-prefill-context-parallel for "
            "DeepSeek V3.2 (NSA) models and "
            "--enable-prefill-context-parallel for MLA-based models "
            "(DeepSeek V3/R1, Kimi K2.5) or MHA/GQA-based models."
        )

    view = resolved_view(server_args)
    if view.attn_cp_size > 1:
        # The tp_size is the world size, not the real tensor parallel size
        assert cfg.tp_size % view.attn_cp_size == 0, (
            "tp_size must be divisible by attn_cp_size"
        )
        assert cfg.tp_size % (cfg.dp_size * view.attn_cp_size) == 0, (
            "tp_size must be divisible by dp_size * attn_cp_size"
        )

        assert not cfg.enable_aiter_allreduce_fusion, (
            "Aiter allreduce fusion is not supported with context parallelism"
        )

    if cfg.moe_dp_size > 1:
        # The tp_size is the world size, not the real tensor parallel size
        assert cfg.tp_size % cfg.moe_dp_size == 0, (
            "tp_size must be divisible by moe_dp_size"
        )
        assert view.ep_size * cfg.moe_dp_size <= cfg.tp_size, (
            "ep_size * moe_dp_size must be less than or equal to tp_size"
        )
        assert cfg.pp_size == 1, "PP is not supported with context parallelism"

        if view.ep_size > 1:
            assert view.ep_size * cfg.moe_dp_size == cfg.tp_size, (
                "ep_size * moe_dp_size must be equal to tp_size"
            )

        assert not cfg.enable_aiter_allreduce_fusion, (
            "Aiter allreduce fusion is not supported with context parallelism"
        )

    if view.attn_cp_size != cfg.moe_dp_size:
        assert cfg.moe_dp_size == 1, (
            "attn_cp_size != moe_dp_size is only supported when moe_dp_size == 1"
        )

    from sglang.srt.layers.cp.base import init_cp_strategy

    init_cp_strategy(
        enable_prefill_cp=bool(cfg.enable_prefill_cp),
        cp_size=cfg.attn_cp_size,
        cp_strategy=cfg.cp_strategy,
    )


def handle_dcp_validation(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.dcp_size < 1:
        raise ValueError(
            "Decode context parallel size (--dcp-size / "
            "--decode-context-parallel-size) must be >= 1, but got "
            f"dcp_size={cfg.dcp_size}."
        )
    if cfg.dcp_comm_backend in ("a2a", "fi_a2a") and cfg.dcp_size <= 1:
        raise ValueError(
            f"--dcp-comm-backend {cfg.dcp_comm_backend} only affects the "
            "decode context-parallel attention reduction and therefore "
            "requires --dcp-size / --decode-context-parallel-size > 1, but "
            f"got dcp_size={cfg.dcp_size}."
        )
    if cfg.dcp_comm_backend == "fi_a2a" and not get_platform().is_cuda:
        raise ValueError(
            "--dcp-comm-backend fi_a2a delegates the exchange to FlashInfer's "
            "MNNVL All-to-All kernel, which requires an NVIDIA CUDA platform "
            "with SM90+ and MNNVL fabric memory (e.g. GB200 NVL72). The "
            "authoritative fabric probe runs at model-runner init; use 'a2a' "
            "or 'ag_rs' on clusters without MNNVL."
        )
    if cfg.dcp_replicate_q_proj:
        if cfg.dcp_size <= 1:
            raise ValueError("--dcp-replicate-q-proj requires --dcp-size > 1.")
        if cfg.dcp_comm_backend not in ("a2a", "fi_a2a"):
            raise ValueError(
                "--dcp-replicate-q-proj only applies to the a2a/fi_a2a DCP "
                "communication backend (it removes the head-dim Q all-gather); "
                f"got --dcp-comm-backend={cfg.dcp_comm_backend}."
            )


def handle_data_parallelism(server_args: Any):
    # The dp_size==1 resets moved to the resolution pipeline
    # (arg_groups/overrides.py: _data_parallelism_defaults).
    from sglang.srt.arg_groups.cuda_graph_hook import (
        generate_prefill_cuda_graph_batch_sizes,
    )

    cfg = resolving_view(server_args)

    run_post_process_pass(server_args, _data_parallelism_defaults)

    if cfg.mm_enable_dp_encoder:
        if cfg.tp_size == 1:
            logger.warning(
                "--mm-enable-dp-encoder is enabled with TP=1, so the encoder "
                "has no data-parallel work to distribute. Disable it unless "
                "you need to validate this configuration."
            )
        else:
            logger.info(
                "--mm-enable-dp-encoder is enabled across TP=%d. It replicates "
                "the vision encoder and distributes image work across ranks; "
                "this is most useful when high-resolution or multi-image ViT "
                "prefill is a material part of TTFT. Measure against the default "
                "for small-image workloads because replication and aggregation "
                "can increase memory use and overhead.",
                cfg.tp_size,
            )

    if resolved_view(server_args).enable_dp_attention:
        declare_resolution(
            server_args,
            "_handle_data_parallelism",
            schedule_conservativeness=cfg.schedule_conservativeness * 0.3,
        )
        assert cfg.tp_size % cfg.dp_size == 0
        original_chunked_prefill_size = cfg.chunked_prefill_size
        declare_resolution(
            server_args,
            "_handle_data_parallelism",
            chunked_prefill_size=cfg.chunked_prefill_size // cfg.dp_size,
        )
        logger.warning(
            f"DP attention is enabled. chunked prefill size is adjusted "
            f"from {original_chunked_prefill_size} to {cfg.chunked_prefill_size}."
        )

        # The prefill CUDA graph max_bs was derived from the pre-DP-division
        # chunked_prefill_size in _handle_gpu_memory_settings (which runs
        # before this handler). Re-clamp it (and the captured shape list) to
        # the per-DP-rank chunked_prefill_size so breakable CUDA graph
        # capture never exceeds the MoE all-to-all's max_num_tokens budget,
        # which is also sized from the DP-adjusted chunked_prefill_size.
        prefill_cfg = cfg.cuda_graph_config.prefill
        if (
            prefill_cfg.backend != Backend.DISABLED
            and prefill_cfg.max_bs is not None
            and prefill_cfg.max_bs > cfg.chunked_prefill_size
            and (Phase.PREFILL, "max_bs") not in server_args._cuda_graph_config_locked
        ):
            clamped = {"max_bs": cfg.chunked_prefill_size}
            if (Phase.PREFILL, "bs") not in server_args._cuda_graph_config_locked:
                clamped["bs"] = generate_prefill_cuda_graph_batch_sizes(
                    clamped["max_bs"]
                )
            declare_resolution(
                server_args,
                "_handle_data_parallelism",
                cuda_graph_config=with_phase(
                    cfg.cuda_graph_config, Phase.PREFILL, **clamped
                ),
            )

    # Resolve the phase-aware TP LM-head default before validating the
    # resulting DP/TP LM-head configuration.

    run_post_process_pass(server_args, _tp_lm_head_all_to_all_default)
    run_post_process_pass(server_args, _dp_lm_head_validation)


def handle_dwdp(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.dwdp_size <= 1:
        return

    assert cfg.dwdp_size >= 2, (
        f"dwdp_size must be >= 2 when enabled, got {cfg.dwdp_size}"
    )
    assert cfg.dwdp_size == cfg.tp_size, (
        f"dwdp_size ({cfg.dwdp_size}) must equal tp_size ({cfg.tp_size})"
    )
    assert cfg.disaggregation_mode in (
        "null",
        "prefill",
    ), "DWDP requires --disaggregation-mode null or prefill"
    assert not cfg.enable_eplb, (
        "EPLB dynamic migration conflicts with static DWDP partitioning"
    )
    assert cfg.speculative_algorithm is None, (
        "DWDP does not support speculative decoding (MTP/draft workers)"
    )
    assert cfg.pp_size == 1, "DWDP requires pp_size == 1"
    assert not cfg.enable_two_batch_overlap, (
        "DWDP's prefetch event protocol does not support two-batch overlap"
    )

    if cfg.disaggregation_mode == "null":
        logger.warning(
            "DWDP with --disaggregation-mode null: decode steps re-fetch all "
            "remote expert weights every step, which is slow. DWDP is "
            "recommended only with --disaggregation-mode prefill."
        )

    declare_resolution(
        server_args,
        "_handle_dwdp",
        dp_size=cfg.dwdp_size,
    )
    declare_resolution(
        server_args,
        "_handle_dwdp",
        enable_dp_attention=True,
    )
    declare_resolution(
        server_args, "_handle_dwdp", enable_dp_attention_local_control_broadcast=True
    )
    declare_resolution(
        server_args,
        "_handle_dwdp",
        enable_dp_lm_head=True,
    )
    declare_resolution(
        server_args,
        "_handle_dwdp",
        moe_dense_tp_size=1,
    )
    declare_resolution(
        server_args,
        "_handle_dwdp",
        ep_size=cfg.dwdp_size,
    )
    declare_resolution(
        server_args,
        "_handle_dwdp",
        moe_dp_size=1,
    )
    declare_resolution(
        server_args,
        "_handle_dwdp",
        moe_a2a_backend="none",
    )

    envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.set(True)

    declare_resolution(
        server_args,
        "_handle_dwdp",
        disable_cuda_graph=True,
    )

    logger.info(
        f"DWDP enabled: dwdp_size={cfg.dwdp_size}, "
        f"auto-forced dp_size={cfg.dp_size}, ep_size={cfg.dwdp_size}, "
        f"moe_dense_tp_size=1, moe_a2a_backend=none, "
        f"dp_attention_local_control_broadcast=True, "
        f"enable_dp_lm_head=True, SCHEDULER_SKIP_ALL_GATHER=True, "
        f"disable_cuda_graph=True"
    )


def handle_elastic_ep(server_args: Any):
    from sglang.srt.arg_groups.validation_hook import validate_ib_devices

    cfg = resolving_view(server_args)
    if cfg.elastic_ep_rejoin:
        if cfg.ep_join_mode is None:
            logger.warning(
                "--elastic-ep-rejoin is deprecated, use --elastic-ep-join-mode recover instead."
            )
            declare_resolution(
                server_args,
                "_handle_elastic_ep",
                ep_join_mode="recover",
            )
        else:
            assert cfg.ep_join_mode == "recover", (
                "--elastic-ep-rejoin (deprecated) conflicts with "
                f"--elastic-ep-join-mode {cfg.ep_join_mode}."
            )
    if cfg.elastic_ep_backend is not None:
        if cfg.enable_eplb:
            if cfg.eplb_algorithm == "auto":
                declare_resolution(
                    server_args,
                    "_handle_elastic_ep",
                    eplb_algorithm="elasticity_aware",
                )
            assert cfg.eplb_algorithm in [
                "elasticity_aware",
                "elasticity_aware_hierarchical",
            ], (
                "Elastic EP requires eplb_algorithm to be set to 'auto' or 'elasticity_aware(_hierarchical)'."
            )

        assert cfg.pp_size == 1, "PP size should be set to 1 under elastic EP"

        if cfg.elastic_ep_backend == "mooncake":
            declare_resolution(
                server_args,
                "_handle_elastic_ep",
                mooncake_ib_device=validate_ib_devices(cfg.mooncake_ib_device),
            )
    if cfg.ep_join_mode is not None:
        assert cfg.elastic_ep_backend is not None, (
            "--elastic-ep-join-mode requires --elastic-ep-backend to be set."
        )
        if cfg.ep_join_mode == "scale":
            assert cfg.node_rank == 1, (
                "Elastic EP scale-up requires one joining TP group at "
                f"--node-rank 1 (got {cfg.node_rank})."
            )
            assert cfg.ep_join_rank_offset > 0, (
                "Elastic EP scale joiners require "
                "--elastic-ep-join-rank-offset set to the current "
                "effective EP size."
            )
    if cfg.ep_join_rank_offset != 0:
        assert cfg.ep_join_mode == "scale", (
            "--elastic-ep-join-rank-offset is only valid with "
            "--elastic-ep-join-mode scale."
        )
        assert cfg.ep_join_rank_offset >= 0, "elastic EP join rank offset must be >= 0."
    if cfg.max_ep_size is not None:
        assert cfg.elastic_ep_backend is not None, (
            "--max-ep-size requires --elastic-ep-backend to be set."
        )
        assert cfg.max_ep_size > 0, "--max-ep-size must be a positive integer."

    scaling_active = (
        cfg.elastic_ep_backend is not None
        and cfg.max_ep_size is not None
        and cfg.max_ep_size > cfg.tp_size
    )
    if cfg.elastic_ep_initial_size is not None:
        assert scaling_active, (
            "--elastic-ep-initial-size is only valid for an Elastic EP "
            "deployment with --max-ep-size larger than its local TP size."
        )
    if scaling_active:
        resolved = resolved_view(server_args)
        assert cfg.elastic_ep_scale_timeout > 0, (
            "--elastic-ep-scale-timeout must be greater than zero."
        )
        assert cfg.tokenizer_worker_num == 1, (
            "Elastic EP runtime scale-up currently requires --tokenizer-worker-num 1."
        )
        assert not cfg.use_ray, (
            "Elastic EP runtime scale-up does not support --use-ray."
        )
        assert not cfg.enable_elastic_expert_backup, (
            "Elastic EP runtime scale-up does not support "
            "--enable-elastic-expert-backup."
        )
        declare_resolution(
            server_args,
            "_handle_elastic_ep",
            enable_dp_attention_local_control_broadcast=True,
        )
        if cfg.ep_join_mode == "scale":
            assert cfg.elastic_ep_initial_size is not None, (
                "Elastic EP scale joiners require --elastic-ep-initial-size "
                "set to the primary deployment's launch-time EP size."
            )
            assert cfg.elastic_ep_initial_size <= cfg.ep_join_rank_offset, (
                "--elastic-ep-initial-size cannot exceed the current EP size "
                f"(initial={cfg.elastic_ep_initial_size}, "
                f"current={cfg.ep_join_rank_offset})."
            )
            join_target = cfg.ep_join_rank_offset + cfg.tp_size
            assert join_target <= cfg.max_ep_size, (
                "Elastic EP joining group exceeds --max-ep-size "
                f"(join_target={join_target}, max_ep_size={cfg.max_ep_size})."
            )
            if cfg.tp_size == 1:
                assert cfg.moe_dense_tp_size == 1, (
                    "A single-rank Elastic EP joining group requires "
                    "--moe-dense-tp-size 1."
                )
        else:
            if cfg.elastic_ep_initial_size is None:
                declare_resolution(
                    server_args,
                    "_handle_elastic_ep",
                    elastic_ep_initial_size=cfg.tp_size,
                )
            assert cfg.elastic_ep_initial_size == cfg.tp_size, (
                "The primary --elastic-ep-initial-size must equal its "
                f"launch-time TP size ({cfg.tp_size})."
            )
        assert cfg.elastic_ep_initial_size > 0
        assert cfg.load_balance_method == "round_robin", (
            "Elastic EP scale-up requires --load-balance-method round_robin; "
            "load-aware methods "
            "require global-rank load snapshots after scale "
            f"(got {cfg.load_balance_method})."
        )
        assert cfg.elastic_ep_backend == "mooncake", (
            "Elastic EP runtime scale-up requires --elastic-ep-backend "
            f"mooncake (got elastic_ep_backend={cfg.elastic_ep_backend})."
        )
        assert cfg.pp_size == 1, (
            "Elastic EP scale-up requires --pp-size 1 "
            f"(got pp_size={cfg.pp_size}); WORLD must not span PP stages."
        )

        decode_cuda_graph_disabled = (
            cfg.cuda_graph_config.decode.backend == Backend.DISABLED
        )
        prefill_cuda_graph_disabled = (
            cfg.cuda_graph_config.prefill.backend == Backend.DISABLED
        )
        assert decode_cuda_graph_disabled and prefill_cuda_graph_disabled, (
            "Elastic EP runtime scale-up requires decode and prefill CUDA "
            "graphs to be disabled."
        )
        assert resolved.enable_dp_attention, (
            "Elastic EP scale-up requires --enable-dp-attention; without it "
            "the TP group is not equivalent to WORLD and the post-scale "
            "collective path is invalid."
        )
        assert resolved.enable_dp_lm_head, (
            "Elastic EP scale-up requires --enable-dp-lm-head so output "
            "projection does not depend on the joining group's TP size."
        )
        assert resolved.attn_cp_size == 1, (
            "Elastic EP scale-up requires --attn-cp-size 1 "
            f"(got attn_cp_size={resolved.attn_cp_size})."
        )
        assert cfg.moe_dp_size == 1, (
            "Elastic EP scale-up requires --moe-dp-size 1 "
            f"(got moe_dp_size={cfg.moe_dp_size})."
        )
        assert resolved.ep_size == cfg.tp_size, (
            "Elastic EP scale-up requires ep_size == tp_size "
            f"(got ep_size={resolved.ep_size}, tp_size={cfg.tp_size}); EP, TP "
            "and the attention DP group must all coincide with WORLD."
        )
        assert cfg.dp_size == cfg.tp_size, (
            "Elastic EP scale-up requires dp_size == tp_size "
            f"(got dp_size={cfg.dp_size}, tp_size={cfg.tp_size})."
        )
        assert resolved.moe_a2a_backend == "nixl", (
            "Elastic EP scale-up requires --moe-a2a-backend nixl "
            f"(got moe_a2a_backend={resolved.moe_a2a_backend})."
        )


def handle_eplb_and_dispatch(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.enable_eplb and (cfg.expert_distribution_recorder_mode is None):
        declare_resolution(
            server_args,
            "_handle_eplb_and_dispatch",
            expert_distribution_recorder_mode="stat",
        )
        logger.warning(
            "EPLB is enabled. The expert_distribution_recorder_mode is automatically set."
        )

    # Without an a2a backend all EP ranks run the MoE over the same tokens and
    # sum their partial outputs, so the pick has to agree across ranks.
    needs_rank_invariant_dispatch = resolved_view(server_args).moe_a2a_backend == "none"

    if (cfg.enable_eplb or (cfg.init_expert_location != "trivial")) and (
        cfg.ep_dispatch_algorithm is None
    ):
        declare_resolution(
            server_args,
            "_handle_eplb_and_dispatch",
            ep_dispatch_algorithm=(
                "dynamic" if needs_rank_invariant_dispatch else "static"
            ),
        )

    # `dynamic` / `fake` switch to the row-index pick; `static` reads a
    # per-rank table and `lp` samples inside its kernel.
    if needs_rank_invariant_dispatch and cfg.ep_dispatch_algorithm in (
        "static",
        "lp",
    ):
        raise ValueError(
            f"--ep-dispatch-algorithm {cfg.ep_dispatch_algorithm} picks a "
            "different physical replica per rank, which only holds up when an "
            "a2a backend routes each token to a single rank. Use "
            "--ep-dispatch-algorithm dynamic with --moe-a2a-backend none."
        )

    if cfg.enable_eplb and cfg.ep_join_mode != "scale":
        assert resolved_view(server_args).ep_size > 1


def handle_platform_cp_compatibility(server_args: Any):
    cfg = resolving_view(server_args)
    platform = get_platform()
    is_protected_platform = platform.is_hip or platform.is_npu
    if not is_protected_platform:
        if (
            cfg.enable_prefill_context_parallel
            or cfg.enable_dsa_prefill_context_parallel
        ):
            raise ValueError(
                "Legacy prefill context-parallel options are supported only "
                "by protected HIP or Ascend NPU paths. Use "
                "--enable-prefill-cp with --cp-strategy."
            )
        return

    legacy_mode_to_strategy = {
        "in-seq-split": "zigzag",
        "round-robin-split": "interleave",
    }

    if cfg.enable_prefill_context_parallel or cfg.enable_dsa_prefill_context_parallel:
        declare_resolution(
            server_args,
            "_handle_platform_cp_compatibility",
            enable_prefill_cp=True,
        )

    if cfg.enable_prefill_context_parallel and cfg.cp_strategy is None:
        declare_resolution(
            server_args,
            "_handle_platform_cp_compatibility",
            cp_strategy=legacy_mode_to_strategy[cfg.prefill_cp_mode],
        )
    if cfg.enable_dsa_prefill_context_parallel and cfg.cp_strategy is None:
        declare_resolution(
            server_args,
            "_handle_platform_cp_compatibility",
            cp_strategy=legacy_mode_to_strategy[cfg.dsa_prefill_cp_mode],
        )


def handle_legacy_cp_runtime_compatibility(server_args: Any):
    """Project canonical CP settings for runtime consumers removed by PR3."""
    cfg = resolving_view(server_args)

    if cfg.enable_prefill_context_parallel and cfg.enable_dsa_prefill_context_parallel:
        return

    if not cfg.enable_prefill_cp or cfg.cp_strategy is None:
        return

    strategy_to_legacy_mode = {
        "zigzag": "in-seq-split",
        "interleave": "round-robin-split",
    }
    mode = strategy_to_legacy_mode[cfg.cp_strategy]
    use_dsa_legacy_aliases = cfg.enable_dsa_prefill_context_parallel or getattr(
        resolved_view(server_args), "attention_backend", None
    ) in ("dsa", "dsv4")
    if use_dsa_legacy_aliases:
        declare_resolution(
            server_args,
            "_handle_legacy_cp_runtime_compatibility",
            enable_dsa_prefill_context_parallel=True,
        )
        declare_resolution(
            server_args,
            "_handle_legacy_cp_runtime_compatibility",
            enable_prefill_context_parallel=False,
        )
    else:
        declare_resolution(
            server_args,
            "_handle_legacy_cp_runtime_compatibility",
            enable_prefill_context_parallel=True,
        )
    declare_resolution(
        server_args,
        "_handle_legacy_cp_runtime_compatibility",
        dsa_prefill_cp_mode=mode,
    )
    declare_resolution(
        server_args,
        "_handle_legacy_cp_runtime_compatibility",
        prefill_cp_mode=mode,
    )


def handle_expert_distribution_metrics(server_args: Any):
    cfg = resolving_view(server_args)
    if "SGLANG_ENABLE_EPLB_BALANCEDNESS_METRIC" in os.environ:
        raise ValueError(
            "SGLANG_ENABLE_EPLB_BALANCEDNESS_METRIC is no longer supported. Use "
            "--expert-balancedness-report-mode with one of: off, server_log, "
            "prometheus, both."
        )

    if should_report_expert_balancedness(server_args) and (
        cfg.expert_distribution_recorder_mode is None
    ):
        declare_resolution(
            server_args,
            "_handle_expert_distribution_metrics",
            expert_distribution_recorder_mode="stat",
        )

    if cfg.expert_distribution_recorder_buffer_size is None:
        if (x := cfg.eplb_rebalance_num_iterations) is not None:
            declare_resolution(
                server_args,
                "_handle_expert_distribution_metrics",
                expert_distribution_recorder_buffer_size=x,
            )
        elif cfg.expert_distribution_recorder_mode is not None:
            declare_resolution(
                server_args,
                "_handle_expert_distribution_metrics",
                expert_distribution_recorder_buffer_size=1000,
            )
