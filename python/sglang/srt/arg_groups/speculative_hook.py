from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Optional

from sglang.srt.arg_groups.overrides import (
    _speculative_moe_runner_default,
    attention_backends_of,
    declare_direct_writes,
    declare_resolution,
    model_config_of,
    resolved_view,
    resolving_view,
    run_post_process_pass,
)
from sglang.srt.runtime_context import get_platform

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _disable_overlap_schedule_for_cpu(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    if cfg.device != "cpu" or cfg.disable_overlap_schedule:
        return

    declare_resolution(
        server_args,
        "_disable_overlap_schedule_for_cpu",
        disable_overlap_schedule=True,
    )
    logger.warning(
        "Overlap schedule is not implemented for speculative decoding on CPU."
    )


def _resolve_speculative_algorithm_alias(
    speculative_algorithm: Optional[str],
    speculative_draft_model_path: Optional[str],
    trust_remote_code: bool = False,
    kwargs: Optional[dict] = {},
) -> Optional[str]:
    """Resolve CLI speculative algorithm; NEXTN/EAGLE may become FROZEN_KV_MTP for Gemma4 assistant drafts."""

    is_gemma4_draft = False
    if speculative_draft_model_path:
        from sglang.srt.utils.hf_transformers_utils import get_config

        cfg = get_config(
            speculative_draft_model_path, trust_remote_code=trust_remote_code, **kwargs
        )
        draft_archs = getattr(cfg, "architectures", None) or []
        is_gemma4_draft = any(
            arch in ("Gemma4AssistantForCausalLM", "Gemma4UnifiedAssistantForCausalLM")
            for arch in draft_archs
        )

    if speculative_algorithm == "EAGLE3" and is_gemma4_draft:
        raise ValueError(
            "Gemma4AssistantForCausalLM draft requires "
            "--speculative-algorithm NEXTN or EAGLE; EAGLE3 is "
            "not supported for this draft architecture."
        )

    if speculative_algorithm == "NEXTN" or speculative_algorithm == "EAGLE":
        if is_gemma4_draft:
            logger.info(
                "Detected Gemma4AssistantForCausalLM draft; "
                f"promoting --speculative-algorithm {speculative_algorithm} to FROZEN_KV_MTP."
            )
            return "FROZEN_KV_MTP"
        return "EAGLE"

    return speculative_algorithm


def handle_speculative_decoding(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    if (
        cfg.speculative_draft_model_path is not None
        and cfg.speculative_draft_model_revision is None
    ):
        declare_resolution(
            server_args,
            "handle_speculative_decoding",
            speculative_draft_model_revision="main",
        )

    # Moved to the resolution pipeline (arg_groups/overrides.py:
    # _speculative_moe_runner_default), invoked here at its legacy slot.

    run_post_process_pass(server_args, _speculative_moe_runner_default)

    if cfg.speculative_algorithm is not None:
        declare_resolution(
            server_args,
            "handle_speculative_decoding",
            speculative_algorithm=cfg.speculative_algorithm.upper(),
        )

    # Removal notice for the retired env var; raw os.getenv on purpose -- the
    # Envs descriptor is gone. Drop this check after one release.
    if os.getenv("SGLANG_ENABLE_SPEC_V2") is not None:
        logger.warning(
            "SGLANG_ENABLE_SPEC_V2 has been removed: speculative decoding "
            "always runs the V2 worker. Use --disable-overlap-schedule to "
            "select the non-overlap (synchronous) path."
        )

    kwargs = {}

    override_config_file = cfg.decrypted_draft_config_file
    if override_config_file and override_config_file.strip():
        kwargs["_configuration_file"] = override_config_file.strip()

    declare_resolution(
        server_args,
        "handle_speculative_decoding",
        speculative_algorithm=_resolve_speculative_algorithm_alias(
            cfg.speculative_algorithm,
            cfg.speculative_draft_model_path,
            trust_remote_code=cfg.trust_remote_code,
            kwargs=kwargs,
        ),
    )

    # Validate --speculative-draft-window-size once, regardless of algorithm.
    # Consumed by DFLASH (compact draft KV cache) and Llama EAGLE-3 (drafter attention SWA).
    if cfg.speculative_draft_window_size is not None:
        window_size = int(cfg.speculative_draft_window_size)
        if window_size <= 0:
            raise ValueError(
                f"--speculative-draft-window-size must be positive, got {window_size}."
            )
        declare_resolution(
            server_args,
            "handle_speculative_decoding",
            speculative_draft_window_size=window_size,
        )
        if cfg.speculative_algorithm not in ("EAGLE3", "DFLASH"):
            logger.warning(
                "--speculative-draft-window-size has no effect with "
                "speculative_algorithm=%s (honored by Llama EAGLE-3 and DFLASH only).",
                cfg.speculative_algorithm,
            )

    algo = None
    if cfg.speculative_algorithm is not None:
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang.srt.speculative.spec_registry import CustomSpecAlgo

        algo = SpeculativeAlgorithm.from_string(cfg.speculative_algorithm)

        # TODO: move the per-algorithm validation below into spec module hooks.
        if isinstance(algo, CustomSpecAlgo) and algo.validate_server_args is not None:
            declare_direct_writes(
                server_args,
                "handle_speculative_decoding.custom_validate",
                algo.validate_server_args,
            )

    if cfg.speculative_skip_dp_mlp_sync:
        assert cfg.speculative_algorithm == "EAGLE", (
            "--speculative-skip-dp-mlp-sync is only supported with "
            f"speculative_algorithm == EAGLE, got {cfg.speculative_algorithm}."
        )

    if cfg.speculative_adaptive:
        _maybe_disable_adaptive(server_args)
        if cfg.speculative_adaptive:
            _init_adaptive_speculative_params(server_args)

    if algo is not None:
        # A registered algorithm's callback lives outside this tree and sets
        # fields on the record, so the writes are captured around the call.
        declare_direct_writes(
            server_args,
            "handle_speculative_decoding.custom_algo",
            algo.handle_server_args,
        )


def _handle_dflash(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)

    if not (cfg.device.startswith("cuda") or cfg.device == "npu"):
        raise ValueError(
            "DFLASH speculative decoding only supports CUDA and NPU devices."
        )

    if resolved_view(server_args).enable_dp_attention:
        raise ValueError(
            "Currently DFLASH speculative decoding does not support dp attention."
        )

    if cfg.pp_size != 1:
        raise ValueError(
            "Currently DFLASH speculative decoding only supports pp_size == 1."
        )

    if cfg.speculative_draft_model_path is None:
        raise ValueError(
            "DFLASH speculative decoding requires setting --speculative-draft-model-path."
        )

    # DFLASH does not use EAGLE-style `num_steps`/`topk`, but those fields still
    # affect generic scheduler/KV-cache accounting (buffer sizing, KV freeing,
    # RoPE reservation). Force them to 1 to avoid surprising memory behavior.
    #
    # For DFlash, the natural unit is `block_size` (verify window length).
    if cfg.speculative_num_steps is None:
        declare_resolution(
            server_args,
            "_handle_dflash",
            speculative_num_steps=1,
        )
    elif int(cfg.speculative_num_steps) != 1:
        logger.warning(
            "DFLASH only supports speculative_num_steps == 1; overriding speculative_num_steps=%s to 1.",
            cfg.speculative_num_steps,
        )
        declare_resolution(
            server_args,
            "_handle_dflash",
            speculative_num_steps=1,
        )

    if cfg.speculative_eagle_topk is None:
        declare_resolution(
            server_args,
            "_handle_dflash",
            speculative_eagle_topk=1,
        )
    elif int(cfg.speculative_eagle_topk) != 1:
        logger.warning(
            "DFLASH only supports speculative_eagle_topk == 1; overriding speculative_eagle_topk=%s to 1.",
            cfg.speculative_eagle_topk,
        )
        declare_resolution(
            server_args,
            "_handle_dflash",
            speculative_eagle_topk=1,
        )

    if cfg.speculative_dflash_block_size is not None:
        if int(cfg.speculative_dflash_block_size) <= 0:
            raise ValueError(
                "DFLASH requires --speculative-dflash-block-size to be positive, "
                f"got {cfg.speculative_dflash_block_size}."
            )
        if cfg.speculative_num_draft_tokens is not None and int(
            cfg.speculative_num_draft_tokens
        ) != int(cfg.speculative_dflash_block_size):
            raise ValueError(
                "Both --speculative-num-draft-tokens and --speculative-dflash-block-size are set "
                "but they differ. For DFLASH they must match. "
                f"speculative_num_draft_tokens={cfg.speculative_num_draft_tokens}, "
                f"speculative_dflash_block_size={cfg.speculative_dflash_block_size}."
            )
        declare_resolution(
            server_args,
            "_handle_dflash",
            speculative_num_draft_tokens=int(cfg.speculative_dflash_block_size),
        )

    if cfg.speculative_num_draft_tokens is None:
        from sglang.srt.speculative.dflash_utils import (
            parse_dflash_draft_config,
        )

        model_override_args = json.loads(cfg.json_model_override_args)
        inferred_block_size = None
        try:
            from sglang.srt.utils.hf_transformers_utils import get_config

            draft_hf_config = get_config(
                cfg.speculative_draft_model_path,
                trust_remote_code=cfg.trust_remote_code,
                revision=cfg.speculative_draft_model_revision,
                model_override_args=model_override_args,
            )
            inferred_block_size = parse_dflash_draft_config(
                draft_hf_config=draft_hf_config
            ).resolve_block_size(default=None)
        except Exception as e:
            logger.warning(
                "Failed to infer DFLASH block_size from draft model config; "
                "defaulting speculative_num_draft_tokens to 16. Error: %s",
                e,
            )

        if inferred_block_size is None:
            inferred_block_size = 16
            logger.warning(
                "speculative_num_draft_tokens is not set; defaulting to %d for DFLASH.",
                inferred_block_size,
            )
        declare_resolution(
            server_args,
            "_handle_dflash",
            speculative_num_draft_tokens=inferred_block_size,
        )

    if cfg.speculative_draft_window_size is not None:
        draft_tokens = int(cfg.speculative_num_draft_tokens)
        if cfg.speculative_draft_window_size < draft_tokens:
            raise ValueError(
                "--speculative-draft-window-size must be >= "
                "--speculative-num-draft-tokens (block_size). "
                f"window_size={cfg.speculative_draft_window_size}, block_size={draft_tokens}."
            )

    _resolve_dflash_draft_attention_backend(server_args)

    if cfg.max_running_requests is None:
        declare_resolution(
            server_args,
            "_handle_dflash",
            max_running_requests=48,
        )
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )


def _handle_uno(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)

    if not cfg.device.startswith("cuda"):
        raise ValueError("UNO only supports CUDA.")
    if cfg.speculative_draft_model_path is not None:
        raise ValueError(
            "UNO reuses the target model and does not accept "
            "--speculative-draft-model-path."
        )
    if cfg.uno_lora_path is None:
        raise ValueError("UNO requires --uno-lora-path.")
    if cfg.enable_deterministic_inference:
        raise ValueError(
            "UNO does not support --enable-deterministic-inference because its "
            "sampling path does not use per-request seeds."
        )
    if cfg.enable_strict_thinking:
        raise ValueError(
            "UNO does not support --enable-strict-thinking because it requires "
            "grammar decoding."
        )

    verify_width = cfg.speculative_num_draft_tokens
    if verify_width is None or int(verify_width) < 1:
        raise ValueError(
            "UNO requires --speculative-num-draft-tokens to be a positive "
            "integer denoting the linear width or tree verify width Q."
        )
    verify_width = int(verify_width)
    declare_resolution(
        server_args,
        "_handle_uno",
        speculative_num_draft_tokens=verify_width,
    )

    candidate_top_k = (
        1 if cfg.speculative_eagle_topk is None else int(cfg.speculative_eagle_topk)
    )
    if candidate_top_k < 1:
        raise ValueError(
            "UNO requires --speculative-eagle-topk to be at least 1, "
            f"got {candidate_top_k}."
        )

    if candidate_top_k > 1:
        if cfg.speculative_num_steps is None:
            raise ValueError(
                "UNO tree mode requires --speculative-num-steps so its draft "
                "width F can be derived as speculative_num_steps + 1."
            )
        speculative_num_steps = int(cfg.speculative_num_steps)
        if speculative_num_steps < 1:
            raise ValueError(
                "UNO tree mode requires --speculative-num-steps to be positive, "
                f"got {speculative_num_steps}."
            )

        draft_width = speculative_num_steps + 1
        if verify_width < draft_width:
            raise ValueError(
                f"UNO tree mode requires Q >= F; got Q={verify_width}, F={draft_width}."
            )
        if verify_width > 128:
            raise ValueError(
                "UNO tree mode currently supports at most Q=128 verify nodes, "
                f"got Q={verify_width}."
            )
        frontier_slots = verify_width * candidate_top_k
        if frontier_slots > 2048:
            raise ValueError(
                "UNO tree mode currently supports Q*K <= 2048; got "
                f"Q*K={verify_width}*{candidate_top_k}={frontier_slots}."
            )

        tree_capacity = 1
        nodes_at_depth = 1
        for _ in range(speculative_num_steps):
            nodes_at_depth *= candidate_top_k
            tree_capacity += nodes_at_depth
            if tree_capacity >= verify_width:
                break
        if verify_width > tree_capacity:
            raise ValueError(
                "UNO tree mode cannot build the requested Q from F and K: "
                f"Q={verify_width} exceeds capacity={tree_capacity} for "
                f"F={draft_width}, K={candidate_top_k}."
            )

        parent_width = candidate_top_k * max(speculative_num_steps - 1, 0) + 1
        if verify_width - 1 > parent_width:
            raise ValueError(
                "UNO tree mode cannot represent the requested Q in EAGLE's "
                "parent-list ABI: "
                f"Q-1={verify_width - 1} exceeds "
                f"K*(F-2)+1={parent_width} for "
                f"F={draft_width}, K={candidate_top_k}."
            )

        if cfg.enable_pdmux:
            raise ValueError("UNO tree mode does not yet support PDMux.")
        if cfg.enable_two_batch_overlap:
            raise ValueError("UNO tree mode does not yet support two-batch overlap.")
        if (
            cfg.speculative_accept_threshold_single != 1.0
            or cfg.speculative_accept_threshold_acc != 1.0
        ):
            raise ValueError(
                "UNO tree mode reuses EAGLE target-only sampling and requires "
                "both speculative accept thresholds to be 1.0."
            )
        declare_resolution(
            server_args,
            "_handle_uno",
            speculative_num_steps=speculative_num_steps,
            speculative_eagle_topk=candidate_top_k,
        )
    else:
        for field in ("speculative_num_steps", "speculative_eagle_topk"):
            old_value = getattr(cfg, field)
            if old_value not in (None, 1):
                logger.warning("UNO uses %s=1; overriding %s.", field, old_value)
        declare_resolution(
            server_args,
            "_handle_uno",
            speculative_num_steps=1,
            speculative_eagle_topk=1,
        )

    if (cfg.tp_size, cfg.pp_size) != (1, 1):
        raise ValueError("UNO requires TP=PP=1.")
    if cfg.enable_dp_attention or cfg.attn_cp_size != 1:
        raise ValueError("UNO does not support DP attention or context parallelism.")
    if cfg.enable_lora or cfg.lora_paths:
        raise ValueError("UNO does not support public Multi-LoRA serving.")
    declare_resolution(
        server_args,
        "_handle_uno",
        enable_lora_overlap_loading=False,
        lora_strict_loading=True,
    )

    if cfg.speculative_use_rejection_sampling:
        raise ValueError(
            "UNO manages its own stochastic verification and does not use "
            "--speculative-use-rejection-sampling."
        )
    if cfg.enable_mixed_chunk:
        declare_resolution(
            server_args,
            "_handle_uno",
            enable_mixed_chunk=False,
        )
        logger.warning(
            "Mixed chunked prefill is disabled for UNO speculative decoding."
        )

    prefill_backend, decode_backend = attention_backends_of(resolved_view(server_args))
    if (prefill_backend, decode_backend) != ("fa3", "fa3"):
        raise ValueError(
            "UNO requires FA3 for both prefill and decode attention; "
            f"got prefill={prefill_backend!r}, decode={decode_backend!r}."
        )


def _target_checkpoint_bundles_dspark_draft(server_args: ServerArgs) -> bool:
    from sglang.srt.speculative.dspark_components.dspark_config import (
        checkpoint_bundles_dspark_draft,
    )

    return checkpoint_bundles_dspark_draft(model_config_of(server_args).hf_config)


def _handle_dspark(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    _is_npu = cfg.device.startswith("npu")
    if not cfg.device.startswith(("cuda", "npu")):
        raise ValueError(
            "DSpark speculative decoding only supports CUDA or NPU device."
        )

    # dp_size==1 with dp_attention is a degenerate flag under DSV4 CP; skip DP-only checks.
    if cfg.enable_dp_attention and cfg.dp_size > 1:
        if not cfg.enable_dp_lm_head:
            raise ValueError("DSpark with dp attention requires --enable-dp-lm-head.")
        if not _is_npu and cfg.moe_a2a_backend not in ("none", "megamoe"):
            raise ValueError(
                "DSpark with dp attention supports moe_a2a_backend 'none' "
                "(built-in TP MoE) or 'megamoe', got "
                f"{cfg.moe_a2a_backend!r}."
            )
        if not _is_npu and cfg.moe_a2a_backend != "none":
            from sglang.srt.speculative.ragged_verify import (
                RaggedVerifyMode,
                read_ragged_verify_mode,
            )

            if read_ragged_verify_mode() is not RaggedVerifyMode.STATIC:
                raise ValueError(
                    "DSpark with dp attention + "
                    f"moe_a2a_backend={cfg.moe_a2a_backend!r} requires "
                    "SGLANG_RAGGED_VERIFY_MODE=static."
                )
        if cfg.attn_cp_size > 1:
            raise ValueError(
                "DSpark with dp attention does not support context parallel "
                f"(attn_cp_size={cfg.attn_cp_size})."
            )
        if (
            not _is_npu
            and cfg.speculative_moe_a2a_backend is not None
            and cfg.speculative_moe_a2a_backend != cfg.moe_a2a_backend
        ):
            raise ValueError(
                "DSpark ignores --speculative-moe-a2a-backend; with dp attention it "
                f"must match the target moe_a2a_backend={cfg.moe_a2a_backend!r} "
                f"(got {cfg.speculative_moe_a2a_backend!r})."
            )

    if cfg.pp_size != 1:
        raise ValueError(
            "Currently DSpark speculative decoding only supports pp_size == 1."
        )

    if cfg.speculative_draft_model_path is None:
        if _target_checkpoint_bundles_dspark_draft(server_args):
            declare_resolution(
                server_args,
                "_handle_dspark",
                speculative_draft_model_path=cfg.model_path,
            )
            declare_resolution(
                server_args,
                "_handle_dspark",
                speculative_draft_model_revision=cfg.revision,
            )
            logger.info(
                "DSpark draft weights are bundled in the target checkpoint; "
                "defaulting --speculative-draft-model-path to --model-path (%s).",
                cfg.model_path,
            )
        else:
            raise ValueError(
                "DSpark dense speculative decoding requires setting "
                "--speculative-draft-model-path."
            )

    if cfg.speculative_num_steps is None:
        declare_resolution(
            server_args,
            "_handle_dspark",
            speculative_num_steps=1,
        )
    elif int(cfg.speculative_num_steps) != 1:
        logger.warning(
            "DSpark only supports speculative_num_steps == 1; overriding speculative_num_steps=%s to 1.",
            cfg.speculative_num_steps,
        )
        declare_resolution(
            server_args,
            "_handle_dspark",
            speculative_num_steps=1,
        )

    if cfg.speculative_eagle_topk is None:
        declare_resolution(
            server_args,
            "_handle_dspark",
            speculative_eagle_topk=1,
        )
    elif int(cfg.speculative_eagle_topk) != 1:
        logger.warning(
            "DSpark only supports speculative_eagle_topk == 1; overriding speculative_eagle_topk=%s to 1.",
            cfg.speculative_eagle_topk,
        )
        declare_resolution(
            server_args,
            "_handle_dspark",
            speculative_eagle_topk=1,
        )

    from sglang.srt.speculative.dspark_components.dspark_config import (
        DEFAULT_DSPARK_GAMMA,
        read_draft_checkpoint_config,
    )

    draft_config = None
    try:
        draft_config = read_draft_checkpoint_config(server_args=server_args)
    except Exception as e:
        logger.warning(
            "Failed to read DSpark draft config; preserving explicit/default "
            "gamma resolution. Error: %s",
            e,
        )

    gamma: Optional[int] = None
    if cfg.speculative_dspark_block_size is not None:
        if int(cfg.speculative_dspark_block_size) <= 0:
            raise ValueError(
                "DSpark requires --speculative-dspark-block-size to be positive, "
                f"got {cfg.speculative_dspark_block_size}."
            )
        gamma = int(cfg.speculative_dspark_block_size)
    else:
        if draft_config is not None:
            gamma = draft_config.resolve_gamma(default=None)
        if gamma is None and cfg.speculative_num_draft_tokens is None:
            gamma = DEFAULT_DSPARK_GAMMA
            logger.warning(
                "DSpark gamma is not set; defaulting to %d.",
                gamma,
            )

    if gamma is not None:
        verify_window = int(gamma) + 1
        if (
            cfg.speculative_num_draft_tokens is not None
            and int(cfg.speculative_num_draft_tokens) != verify_window
        ):
            raise ValueError(
                "DSpark speculative_num_draft_tokens must equal gamma + 1 "
                f"(= {verify_window} for gamma={gamma}), but got "
                f"speculative_num_draft_tokens={cfg.speculative_num_draft_tokens}."
            )
        declare_resolution(
            server_args,
            "_handle_dspark",
            speculative_num_draft_tokens=verify_window,
        )

    if cfg.speculative_num_draft_tokens is None:
        raise ValueError(
            "DSpark could not resolve speculative_num_draft_tokens; set "
            "--speculative-dspark-block-size (= gamma)."
        )
    if int(cfg.speculative_num_draft_tokens) < 2:
        raise ValueError(
            "DSpark speculative_num_draft_tokens must be >= 2 (= gamma + 1), "
            f"got {cfg.speculative_num_draft_tokens}."
        )

    if cfg.max_running_requests is None:
        declare_resolution(
            server_args,
            "_handle_dspark",
            max_running_requests=48,
        )
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )

    ragged_mode = read_ragged_verify_mode()
    if (
        cfg.speculative_dspark_align_verify_tokens_to_graph_tier
        and ragged_mode is not RaggedVerifyMode.COMPACT
    ):
        logger.warning(
            "--speculative-dspark-align-verify-tokens-to-graph-tier only takes "
            "effect with SGLANG_RAGGED_VERIFY_MODE=compact (got %r); it will be "
            "a no-op.",
            ragged_mode.value,
        )
    if cfg.speculative_dspark_sps_table_path and ragged_mode is RaggedVerifyMode.STATIC:
        logger.warning(
            "--speculative-dspark-sps-table-path feeds the ragged-verify budget "
            "scheduler, which is off under SGLANG_RAGGED_VERIFY_MODE=static; it "
            "will be a no-op."
        )


def _resolve_dflash_draft_attention_backend(server_args: ServerArgs) -> None:
    """Resolve `speculative_draft_attention_backend` to a final, supported value.

    Consumed by ModelRunner's `is_draft_worker` override (one backend for all
    draft modes).
    """
    cfg = resolving_view(server_args)

    supported_draft_backends = (
        "flashinfer",
        "fa3",
        "fa4",
        "triton",
        "trtllm_mha",
        "ascend",
    )
    # Use triton on ROCm (no FlashInfer), flashinfer on CUDA.
    fallback_backend = "triton" if get_platform().is_hip else "flashinfer"

    draft_backend = cfg.speculative_draft_attention_backend
    if draft_backend is None:
        draft_backend, _ = attention_backends_of(resolved_view(server_args))
    if draft_backend is None:
        draft_backend = fallback_backend
    elif draft_backend == "trtllm_mha":
        from sglang.srt.speculative.dflash_utils import get_dflash_layer_types
        from sglang.srt.utils.hf_transformers_utils import get_config

        draft_hf_config = get_config(
            cfg.speculative_draft_model_path,
            trust_remote_code=cfg.trust_remote_code,
            revision=cfg.speculative_draft_model_revision,
            model_override_args=json.loads(cfg.json_model_override_args),
        )
        draft_text_config = (
            getattr(draft_hf_config, "text_config", None) or draft_hf_config
        )
        layer_types = get_dflash_layer_types(draft_hf_config)
        num_layers = getattr(draft_text_config, "num_hidden_layers", None)
        all_sliding = (
            layer_types
            and len(layer_types) == num_layers
            and set(layer_types) == {"sliding_attention"}
        )
        all_causal = getattr(draft_text_config, "is_causal", False) is True
        if not (all_sliding or all_causal):
            logger.warning(
                "DFLASH only enables 'trtllm_mha' when all layers use sliding "
                "attention or the draft is explicitly causal; got "
                "layer_types=%r, is_causal=%r. "
                "Falling back to '%s'.",
                layer_types,
                getattr(draft_text_config, "is_causal", None),
                fallback_backend,
            )
            draft_backend = fallback_backend
    elif draft_backend not in supported_draft_backends:
        logger.warning(
            "DFLASH draft worker only supports attention_backend in %s for now, "
            "but got %r. Falling back to '%s'.",
            supported_draft_backends,
            draft_backend,
            fallback_backend,
        )
        draft_backend = fallback_backend
    # FIXME: avoid overriding server args directly; pass the resolved draft
    # backend to the draft worker explicitly instead.
    declare_resolution(
        server_args,
        "_resolve_dflash_draft_attention_backend",
        speculative_draft_attention_backend=draft_backend,
    )


def _handle_frozen_kv_mtp(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    if cfg.max_running_requests is None:
        declare_resolution(
            server_args,
            "_handle_frozen_kv_mtp",
            max_running_requests=48,
        )
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    if cfg.enable_mixed_chunk:
        declare_resolution(
            server_args,
            "_handle_frozen_kv_mtp",
            enable_mixed_chunk=False,
        )
        logger.warning(
            "Mixed chunked prefill is disabled because of using "
            "Frozen-KV MTP speculative decoding."
        )


def _handle_eagle_family(server_args: ServerArgs) -> None:

    cfg = resolving_view(server_args)

    if (
        cfg.speculative_algorithm == "STANDALONE"
        and resolved_view(server_args).enable_dp_attention
    ):
        # TODO: support dp attention for standalone speculative decoding
        raise ValueError(
            "Currently standalone speculative decoding does not support dp attention."
        )

    if cfg.max_running_requests is None:
        declare_resolution(
            server_args,
            "_handle_eagle_family",
            max_running_requests=48,
        )
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    _disable_overlap_schedule_for_cpu(server_args)

    if resolved_view(server_args).disable_overlap_schedule:
        logger.warning(
            "Non-overlap (synchronous) spec v2 is used for eagle/eagle3/standalone "
            "speculative decoding."
        )

    # Mixed steps degrade running requests to a plain 1-token decode.
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    algo = SpeculativeAlgorithm.from_string(cfg.speculative_algorithm)
    if cfg.enable_mixed_chunk and not algo.supports_mixed_chunk():
        declare_resolution(
            server_args,
            "_handle_eagle_family",
            enable_mixed_chunk=False,
        )
        logger.warning(
            "Mixed chunked prefill is disabled: %s speculative decoding does "
            "not support it.",
            cfg.speculative_algorithm,
        )

    model_arch = model_config_of(server_args).hf_config.architectures[0]
    if model_arch in [
        "DeepseekV32ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV4ForCausalLM",
        "Glm4MoeForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "GlmMoeDsaForCausalLM",
        "BailingMoeForCausalLM",
        "BailingMoeV2ForCausalLM",
        "BailingMoeV2_5ForCausalLM",
        "MistralLarge3ForCausalLM",
        "PixtralForConditionalGeneration",
        "HYV3ForCausalLM",
    ]:
        if cfg.speculative_draft_model_path is None:
            declare_resolution(
                server_args,
                "_handle_eagle_family",
                speculative_draft_model_path=cfg.model_path,
            )
            declare_resolution(
                server_args,
                "_handle_eagle_family",
                speculative_draft_model_revision=cfg.revision,
            )
        else:
            if model_arch not in [
                "MistralLarge3ForCausalLM",
                "PixtralForConditionalGeneration",
            ]:
                logger.warning(
                    "DeepSeek MTP does not require setting speculative_draft_model_path."
                )

    if not cfg.speculative_adaptive and cfg.speculative_num_steps is None:
        assert (
            cfg.speculative_eagle_topk is None
            and cfg.speculative_num_draft_tokens is None
        )

        steps, topk, draft_tokens = _auto_choose_speculative_params(
            server_args, model_arch
        )
        declare_resolution(
            server_args,
            "_handle_eagle_family.auto_params",
            speculative_num_steps=steps,
            speculative_eagle_topk=topk,
            speculative_num_draft_tokens=draft_tokens,
        )

    if "trtllm_mha" in attention_backends_of(resolved_view(server_args)):
        if cfg.speculative_eagle_topk > 1:
            raise ValueError(
                "trtllm_mha backend only supports topk = 1 for speculative decoding."
            )

    if cfg.speculative_use_rejection_sampling:
        # Resolved alias by now: NEXTN -> EAGLE, Gemma4 draft -> FROZEN_KV_MTP.
        # Only the EAGLE/EAGLE3 draft workers emit a target-vocab proposal that
        # the rejection-sampling kernel consumes; everything else (STANDALONE,
        # FROZEN_KV_MTP, NGRAM, DFLASH) is unsupported.
        if cfg.speculative_algorithm not in ("EAGLE", "EAGLE3"):
            raise NotImplementedError(
                "--speculative-use-rejection-sampling is only supported for "
                "EAGLE / EAGLE3 / NEXTN, not "
                f"speculative_algorithm={cfg.speculative_algorithm}."
            )
        if cfg.speculative_eagle_topk != 1:
            raise ValueError(
                "--speculative-use-rejection-sampling requires --speculative-eagle-topk=1."
            )
        if (
            cfg.speculative_accept_threshold_single != 1.0
            or cfg.speculative_accept_threshold_acc != 1.0
        ):
            raise ValueError(
                "--speculative-use-rejection-sampling is incompatible with "
                "--speculative-accept-threshold-single / "
                "--speculative-accept-threshold-acc; rejection sampling ignores "
                "the accept thresholds."
            )
        if cfg.enable_deterministic_inference:
            raise ValueError(
                "--speculative-use-rejection-sampling is incompatible with "
                "--enable-deterministic-inference; the sampling kernel draws "
                "coins from the global RNG and is not batch-invariant."
            )

        if (
            resolved_view(server_args).enable_multi_layer_eagle
            and cfg.speculative_eagle_topk != 1
        ):
            raise ValueError(
                "--speculative-use-rejection-sampling with multi-layer EAGLE "
                "(--enable-multi-layer-eagle) requires --speculative-eagle-topk 1; "
                "rejection sampling is only implemented for the linear (topk=1) chain."
            )
        logger.info(
            "Rejection sampling is enabled for speculative decoding "
            "(speculative_use_rejection_sampling=True)."
        )

    if (
        cfg.speculative_eagle_topk == 1
        and cfg.speculative_num_draft_tokens != cfg.speculative_num_steps + 1
    ):
        logger.warning(
            "speculative_num_draft_tokens is adjusted to speculative_num_steps + 1 when speculative_eagle_topk == 1"
        )
        declare_resolution(
            server_args,
            "_handle_eagle_family",
            speculative_num_draft_tokens=cfg.speculative_num_steps + 1,
        )

    # topk > 1 + page_size > 1 needs the two-pass cascade draft-decode (shared prefix
    # pass + per-branch expand pass with prefix-tail dup). Only these backends implement
    # it; flashmla / trtllm_mla / cutlass_mla can't express the per-branch tree, so reject.
    _PAGE_TREE_SPEC_BACKENDS = ("flashinfer", "fa3", "triton")
    view = resolved_view(server_args)
    if (
        cfg.speculative_eagle_topk > 1
        and view.page_size > 1
        and view.attention_backend not in _PAGE_TREE_SPEC_BACKENDS
    ):
        raise ValueError(
            f"speculative_eagle_topk > 1 with page_size > 1 is only supported on "
            f"{_PAGE_TREE_SPEC_BACKENDS}; got attention_backend="
            f"{view.attention_backend!r}. Use page_size == 1 or one of those backends."
        )


def _handle_ngram(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    if cfg.device not in ("cuda", "cpu"):
        raise ValueError(
            "Ngram speculative decoding only supports CUDA or CPU devices."
        )

    _disable_overlap_schedule_for_cpu(server_args)

    if cfg.max_running_requests is None:
        declare_resolution(
            server_args,
            "_handle_ngram",
            max_running_requests=48,
        )
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    declare_resolution(
        server_args,
        "_handle_ngram",
        enable_mixed_chunk=False,
    )
    declare_resolution(
        server_args,
        "_handle_ngram",
        speculative_eagle_topk=cfg.speculative_ngram_max_bfs_breadth,
    )
    if cfg.speculative_num_draft_tokens is None:
        declare_resolution(
            server_args,
            "_handle_ngram",
            speculative_num_draft_tokens=12,
        )
        logger.warning(
            "speculative_num_draft_tokens is set to 12 by default for ngram speculative decoding. "
            "You can override this by explicitly setting --speculative-num-draft-tokens."
        )
    if cfg.speculative_num_steps is None:
        declare_resolution(
            server_args,
            "_handle_ngram",
            speculative_num_steps=cfg.speculative_num_draft_tokens
            // cfg.speculative_eagle_topk,
        )
    if cfg.speculative_ngram_external_corpus_path is not None:
        if cfg.speculative_ngram_external_sam_budget <= 0:
            raise ValueError(
                "--speculative-ngram-external-sam-budget must be positive when "
                "--speculative-ngram-external-corpus-path is set."
            )
        if cfg.speculative_ngram_external_corpus_max_tokens <= 0:
            raise ValueError(
                "--speculative-ngram-external-corpus-max-tokens must be positive when "
                "--speculative-ngram-external-corpus-path is set."
            )
        if (
            cfg.speculative_ngram_external_sam_budget
            > cfg.speculative_num_draft_tokens - 1
        ):
            raise ValueError(
                "speculative_ngram_external_sam_budget must be less than or equal to "
                f"speculative_num_draft_tokens - 1 ({cfg.speculative_num_draft_tokens - 1})."
            )
    logger.warning(
        "The mixed chunked prefill are disabled because of "
        "using ngram speculative decoding."
    )

    view = resolved_view(server_args)
    if (
        cfg.speculative_eagle_topk > 1
        and view.page_size > 1
        and view.attention_backend != "flashinfer"
    ):
        raise ValueError(
            f"speculative_eagle_topk({cfg.speculative_eagle_topk}) > 1 "
            f"with page_size({view.page_size}) > 1 is unstable "
            "and produces incorrect results for paged attention backends. "
            "This combination is only supported for the 'flashinfer' backend."
        )
    if view.enable_dp_attention:
        # TODO: support dp attention for ngram speculative decoding
        raise ValueError(
            "Currently ngram speculative decoding does not support dp attention."
        )


def _maybe_disable_adaptive(server_args: ServerArgs) -> None:
    from sglang.srt.speculative.adaptive_spec_params import (
        adaptive_unsupported_reason,
    )

    reason = adaptive_unsupported_reason(server_args)
    if reason is not None:
        logger.warning(
            f"speculative_adaptive disabled: {reason}. "
            "Falling back to static speculative params."
        )
        declare_resolution(
            server_args,
            "_maybe_disable_adaptive",
            speculative_adaptive=False,
        )


def _init_adaptive_speculative_params(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    from sglang.srt.speculative.adaptive_spec_params import (
        resolve_candidate_steps_from_config,
    )

    candidate_steps = resolve_candidate_steps_from_config(
        cfg_path=cfg.speculative_adaptive_config,
    )

    if cfg.speculative_eagle_topk is None:
        declare_resolution(
            server_args,
            "_init_adaptive_speculative_params",
            speculative_eagle_topk=1,
        )

    if cfg.speculative_num_steps is None:
        declare_resolution(
            server_args,
            "_init_adaptive_speculative_params",
            speculative_num_steps=candidate_steps[len(candidate_steps) // 2],
        )

    if cfg.speculative_num_steps not in candidate_steps:
        raise ValueError(
            f"--speculative-num-steps={cfg.speculative_num_steps} "
            f"is not in the adaptive config candidate_steps {candidate_steps}. "
            "Pass one of those values."
        )

    declare_resolution(
        server_args,
        "_init_adaptive_speculative_params",
        speculative_num_draft_tokens=cfg.speculative_num_steps + 1,
    )


def _auto_choose_speculative_params(server_args: ServerArgs, model_arch: str) -> tuple:
    """
    Automatically choose the parameters for speculative decoding.

    You can tune them on your own models and prompts with scripts/playground/bench_speculative.py
    """
    cfg = resolving_view(server_args)
    if cfg.speculative_algorithm == "STANDALONE":
        return (3, 1, 4)
    if model_arch in ["LlamaForCausalLM"]:
        return (5, 4, 8)
    elif model_arch in [
        "DeepseekV32ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV2ForCausalLM",
        "GptOssForCausalLM",
        "Glm4MoeForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "GlmMoeDsaForCausalLM",
        "BailingMoeForCausalLM",
        "BailingMoeV2ForCausalLM",
        "BailingMoeV2_5ForCausalLM",
        "MistralLarge3ForCausalLM",
        "PixtralForConditionalGeneration",
        "MiMoV2ForCausalLM",
        "MiMoV2FlashForCausalLM",
    ]:
        return (3, 1, 4)
    elif model_arch in ["Grok1ForCausalLM", "Grok1VForCausalLM"]:
        return (5, 4, 8)
    else:
        return (3, 1, 4)
