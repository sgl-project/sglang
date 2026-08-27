# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for context- and decode-context parallelism."""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.arg_groups.overrides import (
    resolved_view,
    resolving_view,
)
from sglang.srt.connector import ConnectorType
from sglang.srt.environ import envs
from sglang.srt.utils.common import is_cuda, parse_connector_type

logger = logging.getLogger(__name__)


def handle_context_parallelism(server_args: Any):
    cfg = resolving_view(server_args)
    if parse_connector_type(cfg.model_path) != ConnectorType.INSTANCE:
        from sglang.srt.configs.model_config import is_deepseek_dsa
        from sglang.srt.layers.cp.utils import CP_V2_DEFAULT_MODEL_CLASSES

        model_config = server_args.get_model_config()
        hf_config = model_config.hf_config
        model_arch = hf_config.architectures[0]
        if model_arch in CP_V2_DEFAULT_MODEL_CLASSES:
            is_dsa_default_model = is_deepseek_dsa(hf_config)
            # DSA CP-v2 currently supports only the interleave strategy.
            enable_default_cp_v2 = not is_dsa_default_model or (
                cfg.enable_prefill_cp and cfg.cp_strategy == "interleave"
            )
            if enable_default_cp_v2 and not envs.SGLANG_ENABLE_CP_V2.is_set():
                envs.SGLANG_ENABLE_CP_V2.set(True)

        if (
            cfg.enable_prefill_cp
            and model_arch in ("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM")
            and envs.SGLANG_ENABLE_CP_V2.get()
        ):
            if cfg.cp_strategy != "zigzag":
                raise ValueError("MiMo V2 CP-v2 only supports --cp-strategy zigzag.")
            if (
                model_config.is_multimodal
                and not cfg.language_only
                and not cfg.language_model_only
            ):
                raise ValueError(
                    "MiMo V2 CP-v2 only supports text inference; add "
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
        assert (
            cfg.tp_size % view.attn_cp_size == 0
        ), "tp_size must be divisible by attn_cp_size"
        assert (
            cfg.tp_size % (cfg.dp_size * view.attn_cp_size) == 0
        ), "tp_size must be divisible by dp_size * attn_cp_size"

        assert (
            not cfg.enable_aiter_allreduce_fusion
        ), "Aiter allreduce fusion is not supported with context parallelism"

    if cfg.moe_dp_size > 1:
        # The tp_size is the world size, not the real tensor parallel size
        assert (
            cfg.tp_size % cfg.moe_dp_size == 0
        ), "tp_size must be divisible by moe_dp_size"
        assert (
            view.ep_size * cfg.moe_dp_size <= cfg.tp_size
        ), "ep_size * moe_dp_size must be less than or equal to tp_size"
        assert cfg.pp_size == 1, "PP is not supported with context parallelism"

        if view.ep_size > 1:
            assert (
                view.ep_size * cfg.moe_dp_size == cfg.tp_size
            ), "ep_size * moe_dp_size must be equal to tp_size"

        assert (
            not cfg.enable_aiter_allreduce_fusion
        ), "Aiter allreduce fusion is not supported with context parallelism"

    if view.attn_cp_size != cfg.moe_dp_size:
        assert (
            cfg.moe_dp_size == 1
        ), "attn_cp_size != moe_dp_size is only supported when moe_dp_size == 1"

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
    if cfg.dcp_comm_backend == "fi_a2a" and not is_cuda():
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
