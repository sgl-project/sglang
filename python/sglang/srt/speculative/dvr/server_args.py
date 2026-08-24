"""Server contracts for decode-verify-rollback speculative decoding."""

from __future__ import annotations

import json
import logging

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

DVR_SPECULATIVE_ALGORITHM = "DECODE_VERIFY_ROLLBACK"
DVR_EAGLE_SPECULATIVE_ALGORITHM = "DECODE_VERIFY_ROLLBACK_EAGLE"
DVR_DFLASH_SPECULATIVE_ALGORITHM = "DECODE_VERIFY_ROLLBACK_DFLASH"
_DVR_FULL_ATTENTION_BACKENDS = {"triton", "fa3"}


def handle_dvr_defaults(server_args):
    algorithm = server_args.speculative_algorithm.upper()
    server_args.speculative_algorithm = algorithm

    # Adaptive spec mutates the fixed chain before DVR's algorithm hook runs.
    if server_args.speculative_adaptive:
        raise ValueError(
            "DVR uses a fixed verify chain and does not support adaptive spec."
        )

    if server_args.grammar_backend not in (None, "none"):
        raise ValueError("DVR does not support grammar-constrained decoding.")
    server_args.grammar_backend = "none"

    # Deterministic target prefill/verify disables custom all-reduce later in
    # ServerArgs. Preserve the user's original choice for provisional draft
    # graphs without exposing another CLI option.
    server_args.dvr_enable_draft_custom_all_reduce = (
        not server_args.disable_custom_all_reduce
    )

    if server_args.enable_prefill_only_deterministic_inference:
        raise ValueError(
            "DVR makes target prefill and verify deterministic; use "
            "--enable-deterministic-inference instead of "
            "--enable-prefill-only-deterministic-inference. DVR restores the "
            "normal fast path only inside provisional draft execution."
        )

    # DVR only permits provisional draft decode to be non-deterministic. Target
    # prefill and verify define the output contract and must use the ordinary
    # deterministic-inference configuration from server initialization onward.
    if not server_args.enable_deterministic_inference:
        logger.warning("Deterministic inference is enabled for DVR target execution.")
        server_args.enable_deterministic_inference = True

    if not _is_dvr_gated_linear_state_model(server_args):
        return

    if server_args.page_size is None:
        server_args.page_size = FLA_CHUNK_SIZE
    elif server_args.page_size != FLA_CHUNK_SIZE:
        raise ValueError(
            "DVR gated linear-state models require page_size == "
            f"FLA_CHUNK_SIZE == {FLA_CHUNK_SIZE}, got {server_args.page_size}."
        )

    if server_args.mamba_track_interval != FLA_CHUNK_SIZE:
        logger.warning(
            "DVR for gated linear-state models requires mamba_track_interval "
            "to match FLA_CHUNK_SIZE=%s. Larger intervals may still be "
            "chunk-size multiples, but the current extra_buffer prefill path "
            "keeps only one tracked checkpoint and can miss the first "
            "prefill's last chunk boundary. Setting --mamba-track-interval %s.",
            FLA_CHUNK_SIZE,
            FLA_CHUNK_SIZE,
        )
        server_args.mamba_track_interval = FLA_CHUNK_SIZE

    if server_args.mamba_ssm_dtype != "float32":
        logger.warning(
            "DVR for gated linear-state models requires fp32 recurrent "
            "states. Setting --mamba-ssm-dtype float32."
        )
        server_args.mamba_ssm_dtype = "float32"


def _handle_dvr_speculative_decoding(server_args):
    algorithm = server_args.speculative_algorithm
    is_self_draft = algorithm == DVR_SPECULATIVE_ALGORITHM
    is_eagle_draft = algorithm == DVR_EAGLE_SPECULATIVE_ALGORITHM
    is_dflash_draft = algorithm == DVR_DFLASH_SPECULATIVE_ALGORITHM

    from sglang.srt.arg_groups.overrides import resolved_view

    view = resolved_view(server_args)
    if not server_args.device.startswith("cuda"):
        raise ValueError("DVR currently only supports CUDA device.")
    if is_hip():
        raise ValueError("DVR currently supports NVIDIA CUDA, not ROCm/HIP.")
    if server_args.pp_size != 1:
        raise ValueError("DVR currently requires pipeline parallel size one.")
    if view.enable_dp_attention:
        raise ValueError("DVR currently does not support DP attention.")
    if server_args.enable_pdmux:
        raise ValueError("DVR currently does not support PDMux attention backends.")
    if server_args.disaggregation_mode != "null":
        raise ValueError("DVR currently does not support disaggregation mode.")
    from sglang.srt.platforms import current_platform

    if current_platform.is_out_of_tree():
        raise ValueError("DVR requires SGLang's CUDA graph runner.")
    if server_args.enable_custom_logit_processor:
        raise ValueError("DVR does not support custom logit processors.")
    if server_args.enable_return_hidden_states:
        raise ValueError("DVR does not return user-requested hidden states.")
    if server_args.decoupled_spec_role != "null":
        raise ValueError("DVR does not support decoupled speculative execution.")
    if server_args.speculative_token_map is not None:
        raise ValueError(
            "DVR request-local rejection sampling requires target-vocabulary "
            "proposal "
            "distributions and does not support --speculative-token-map."
        )
    if is_self_draft:
        if server_args.speculative_draft_model_path is not None:
            raise ValueError("DVR self draft does not use a draft model path.")
        if server_args.speculative_attention_mode != "prefill":
            raise ValueError(
                "DVR target verify requires --speculative-attention-mode prefill."
            )
    if is_eagle_draft:
        if server_args.speculative_draft_model_path is None:
            raise ValueError(
                "DVR EAGLE requires setting --speculative-draft-model-path."
            )
        if server_args.max_running_requests is None:
            # Match upstream EAGLE so request-indexed FutureMap storage is
            # bounded before memory-pool profiling.
            server_args.max_running_requests = 48
    if is_dflash_draft:
        _handle_dvr_dflash_args(server_args)
    if server_args.speculative_num_draft_tokens is None:
        server_args.speculative_num_draft_tokens = 2 if is_eagle_draft else 16

    uses_gated_linear_state = _is_dvr_gated_linear_state_model(server_args)
    if (
        uses_gated_linear_state
        and not view.disable_radix_cache
        and view.mamba_radix_cache_strategy != "extra_buffer"
    ):
        raise ValueError(
            "DVR gated linear-state Radix caching requires the resolved "
            "--mamba-radix-cache-strategy extra_buffer."
        )
    if uses_gated_linear_state and server_args.enable_two_batch_overlap:
        raise ValueError(
            "DVR gated linear-state models do not support two-batch overlap."
        )
    if uses_gated_linear_state and server_args.enable_page_major_kv_layout:
        raise ValueError(
            "DVR gated linear-state verify requires contiguous recurrent-state "
            "storage and does not support --enable-page-major-kv-layout."
        )
    if uses_gated_linear_state and server_args.enable_linear_replayssm:
        raise ValueError(
            "DVR gated linear-state rollback does not support "
            "--enable-linear-replayssm."
        )
    if uses_gated_linear_state and server_args.enable_streaming_session:
        raise ValueError(
            "DVR gated linear-state models do not yet support streaming sessions."
        )
    if uses_gated_linear_state and server_args.enable_int8_mamba_checkpoint:
        raise ValueError(
            "DVR requires exact recurrent checkpoints and does not support "
            "--enable-int8-mamba-checkpoint."
        )
    for phase, backend in (
        ("prefill", view.prefill_attention_backend or view.attention_backend),
        ("decode", view.decode_attention_backend or view.attention_backend),
    ):
        if backend not in _DVR_FULL_ATTENTION_BACKENDS:
            raise ValueError(
                "DVR currently supports only Triton and FA3 full-attention "
                f"backends, got effective {phase} backend {backend}."
            )
    linear_prefill_backend = (
        view.linear_attn_prefill_backend or view.linear_attn_backend
    )
    if uses_gated_linear_state and linear_prefill_backend != "triton":
        raise ValueError(
            "DVR GDN verify requires --linear-attn-prefill-backend triton "
            "because the selected backend must export exact chunk-boundary states."
        )

    if server_args.speculative_num_steps is None:
        server_args.speculative_num_steps = server_args.speculative_num_draft_tokens - 1
    elif (
        server_args.speculative_num_draft_tokens
        != server_args.speculative_num_steps + 1
    ):
        raise ValueError(
            "DVR chain mode requires speculative_num_draft_tokens == "
            "speculative_num_steps + 1."
        )

    if server_args.speculative_num_draft_tokens < 2:
        raise ValueError(
            "DVR requires speculative_num_draft_tokens >= 2 because chain mode "
            "needs at least one draft step."
        )
    if (
        server_args.speculative_num_draft_tokens > FLA_CHUNK_SIZE
        and uses_gated_linear_state
    ):
        raise ValueError(
            "DVR currently commits at most one FLA chunk boundary per verify. "
            f"Please set --speculative-num-draft-tokens <= {FLA_CHUNK_SIZE}."
        )

    if server_args.speculative_eagle_topk is None:
        server_args.speculative_eagle_topk = 1
    elif server_args.speculative_eagle_topk != 1:
        raise ValueError("DVR currently supports only chain mode with topk == 1.")
    # DVR uses request-local rejection sampling. The one-root short-prompt
    # sentinel is the only target-only iteration and is selected by the worker.
    server_args.speculative_use_rejection_sampling = True
    if (
        server_args.speculative_accept_threshold_single != 1.0
        or server_args.speculative_accept_threshold_acc != 1.0
    ):
        raise ValueError(
            "DVR rejection sampling does not use speculative acceptance "
            "thresholds; both thresholds must remain 1.0."
        )
    if is_eagle_draft and view.enable_multi_layer_eagle:
        raise NotImplementedError(
            "DVR rejection sampling does not support multi-layer EAGLE."
        )


def _handle_dvr_dflash_args(server_args):
    if server_args.speculative_draft_model_path is None:
        raise ValueError("DVR DFlash requires setting --speculative-draft-model-path.")

    block_size = server_args.speculative_dflash_block_size
    if block_size is not None:
        block_size = int(block_size)
        if block_size <= 0:
            raise ValueError(
                "DVR DFlash requires --speculative-dflash-block-size to be "
                f"positive, got {block_size}."
            )
        if (
            server_args.speculative_num_draft_tokens is not None
            and int(server_args.speculative_num_draft_tokens) != block_size
        ):
            raise ValueError(
                "--speculative-num-draft-tokens and "
                "--speculative-dflash-block-size must match for DVR DFlash."
            )
        server_args.speculative_num_draft_tokens = block_size

    if server_args.speculative_num_draft_tokens is None:
        from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
        from sglang.srt.utils.hf_transformers_utils import get_config

        try:
            draft_config = get_config(
                server_args.speculative_draft_model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.speculative_draft_model_revision,
                model_override_args=json.loads(server_args.json_model_override_args),
            )
            block_size = parse_dflash_draft_config(
                draft_hf_config=draft_config
            ).resolve_block_size(default=None)
        except Exception as e:
            logger.warning(
                "Failed to infer DVR DFlash block size; defaulting to 16: %s", e
            )
            block_size = None
        server_args.speculative_num_draft_tokens = block_size or 16

    if (
        server_args.speculative_draft_window_size is not None
        and server_args.speculative_draft_window_size
        < server_args.speculative_num_draft_tokens
    ):
        raise ValueError(
            "--speculative-draft-window-size must be at least the DVR DFlash "
            "block size."
        )

    from sglang.srt.arg_groups.speculative_hook import (
        _resolve_dflash_draft_attention_backend,
    )

    _resolve_dflash_draft_attention_backend(server_args)
    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
    if server_args.enable_mixed_chunk:
        server_args.enable_mixed_chunk = False
        logger.warning(
            "Mixed chunked prefill is disabled for the DVR DFlash draft worker."
        )


def handle_dvr_cuda_graph_config(server_args):
    """Apply DVR constraints after the generic CUDA graph config is resolved."""

    if _is_dvr_gated_linear_state_model(server_args):
        prefill_graph = server_args.cuda_graph_config.prefill
        if prefill_graph.backend != Backend.DISABLED:
            if (Phase.PREFILL, "backend") in server_args._cuda_graph_config_locked:
                raise ValueError(
                    "DVR gated linear-state prefill is incompatible with prefill "
                    "CUDA graphs; set cuda_graph_config[prefill].backend='disabled'."
                )
            logger.warning(
                "Prefill CUDA graph is disabled for DVR gated linear-state models."
            )
            prefill_graph.backend = Backend.DISABLED

    if server_args.speculative_algorithm == DVR_SPECULATIVE_ALGORITHM:
        decode_graph = server_args.cuda_graph_config.decode
        if (
            decode_graph.backend == Backend.DISABLED
            or server_args.disable_draft_cuda_graph
        ):
            raise ValueError(
                "DVR self-draft requires draft CUDA "
                "graphs. Remove --disable-cuda-graph/--disable-draft-cuda-graph "
                "or use a non-self-draft DVR mode."
            )
        if server_args.max_running_requests is None:
            server_args.max_running_requests = decode_graph.max_bs
        elif (
            decode_graph.max_bs is not None
            and server_args.max_running_requests > decode_graph.max_bs
        ):
            raise ValueError(
                "DVR self-draft has no eager fallback, so "
                "--max-running-requests must not exceed the decode CUDA graph "
                f"max_bs ({decode_graph.max_bs})."
            )


def _is_dvr_gated_linear_state_model(server_args):
    from sglang.srt.configs import JetNemotronConfig, Qwen3NextConfig
    from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_config

    hf_config = server_args.get_model_config().hf_config
    registered = get_linear_attn_config(hf_config)
    if registered is not None:
        backend = registered[0].backend_class_name
        raise ValueError(
            "DVR does not yet install its state adapter for registered "
            f"linear-state backend {backend!r}."
        )

    # Qwen3.5/InternS2 text configs inherit Qwen3NextConfig; JetVLM unwraps
    # to JetNemotronConfig. Match ModelRunner's capability check without
    # materializing cache params before tensor-parallel groups are initialized.
    text_config = hf_config.get_text_config()
    if isinstance(text_config, Qwen3NextConfig | JetNemotronConfig):
        return bool(text_config.linear_layer_ids)
    if hasattr(type(text_config), "mamba2_cache_params") or getattr(
        text_config, "linear_attn_config", None
    ):
        raise ValueError(
            "DVR currently supports GDN or pure-attention models, not linear-state "
            f"config {type(text_config).__name__}."
        )
    return False
