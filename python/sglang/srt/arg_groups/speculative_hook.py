from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


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
    if (
        server_args.speculative_draft_model_path is not None
        and server_args.speculative_draft_model_revision is None
    ):
        server_args.speculative_draft_model_revision = "main"

    if server_args.speculative_moe_runner_backend is None:
        server_args.speculative_moe_runner_backend = server_args.moe_runner_backend

    if server_args.speculative_algorithm is not None:
        server_args.speculative_algorithm = server_args.speculative_algorithm.upper()

    # Removal notice for the retired env var; raw os.getenv on purpose -- the
    # Envs descriptor is gone. Drop this check after one release.
    if os.getenv("SGLANG_ENABLE_SPEC_V2") is not None:
        logger.warning(
            "SGLANG_ENABLE_SPEC_V2 has been removed: speculative decoding "
            "always runs the V2 worker. Use --disable-overlap-schedule to "
            "select the non-overlap (synchronous) path."
        )

    kwargs = {}

    override_config_file = server_args.decrypted_draft_config_file
    if override_config_file and override_config_file.strip():
        kwargs["_configuration_file"] = override_config_file.strip()

    server_args.speculative_algorithm = _resolve_speculative_algorithm_alias(
        server_args.speculative_algorithm,
        server_args.speculative_draft_model_path,
        trust_remote_code=server_args.trust_remote_code,
        kwargs=kwargs,
    )

    # Validate --speculative-draft-window-size once, regardless of algorithm.
    # Consumed by DFLASH (compact draft KV cache) and Llama EAGLE-3 (drafter attention SWA).
    if server_args.speculative_draft_window_size is not None:
        window_size = int(server_args.speculative_draft_window_size)
        if window_size <= 0:
            raise ValueError(
                f"--speculative-draft-window-size must be positive, got {window_size}."
            )
        server_args.speculative_draft_window_size = window_size
        if server_args.speculative_algorithm not in ("EAGLE3", "DFLASH"):
            logger.warning(
                "--speculative-draft-window-size has no effect with "
                "speculative_algorithm=%s (honored by Llama EAGLE-3 and DFLASH only).",
                server_args.speculative_algorithm,
            )

    algo = None
    if server_args.speculative_algorithm is not None:
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang.srt.speculative.spec_registry import CustomSpecAlgo

        algo = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)

        # TODO: move the per-algorithm validation below into spec module hooks.
        if isinstance(algo, CustomSpecAlgo) and algo.validate_server_args is not None:
            algo.validate_server_args(server_args)

    if server_args.speculative_skip_dp_mlp_sync:
        assert server_args.speculative_algorithm == "EAGLE", (
            "--speculative-skip-dp-mlp-sync is only supported with "
            f"speculative_algorithm == EAGLE, got {server_args.speculative_algorithm}."
        )

    if server_args.speculative_adaptive:
        _maybe_disable_adaptive(server_args)
        if server_args.speculative_adaptive:
            _init_adaptive_speculative_params(server_args)

    # Decoupled spec is a role, not an algorithm. Run it before the per-algorithm
    # dispatch so its role-level defaults win (see _handle_decoupled_spec).
    if server_args.decoupled_spec_role != "null":
        _handle_decoupled_spec(server_args)

    if algo is not None:
        algo.handle_server_args(server_args)


def _handle_decoupled_spec(server_args: ServerArgs) -> None:
    """Validate and normalize the decoupled-spec role (verifier / drafter) args.

    Decoupled spec is a role orthogonal to the speculative algorithm. Runs before
    the per-algorithm dispatch so the role's choices win: max_running_requests
    defaults to 64 (vs the per-algo 48) and topk is pinned to 1 before an algorithm
    handler could grow it. Algorithm handlers only fill unset values, so they keep
    the role's choices.
    """
    is_drafter = server_args.is_decoupled_drafter()
    role = "drafter" if is_drafter else "verifier"

    # No data parallelism: the IPC mesh addresses one engine per role-rank; a
    # second (dp) axis is not modeled in the rank/routing yet. To support it, key
    # the mesh on (dp_rank, role_rank) and route each request by its dp owner.
    if server_args.enable_dp_attention:
        raise ValueError(
            "decoupled speculative decoding does not support dp attention."
        )
    if server_args.dp_size != 1:
        raise ValueError(
            f"decoupled {role} requires dp_size == 1, got {server_args.dp_size}."
        )

    if (
        server_args.decoupled_spec_bind_endpoint is None
        or server_args.decoupled_spec_connect_endpoints is None
        or server_args.decoupled_spec_rank is None
    ):
        raise ValueError(
            "--decoupled-spec-bind-endpoint, "
            "--decoupled-spec-connect-endpoints, and "
            "--decoupled-spec-rank are required for decoupled speculative decoding."
        )
    if int(server_args.decoupled_spec_rank) < 0:
        raise ValueError("--decoupled-spec-rank must be non-negative.")

    # page_size == 1: the drafter rolls back diverged KV at token granularity;
    # paged rollback isn't implemented. To support page_size > 1, add page-aligned
    # drafter rollback.
    if server_args.page_size is not None and server_args.page_size > 1:
        raise ValueError(
            "decoupled speculative decoding requires page_size == 1, "
            f"got {server_args.page_size}."
        )

    # The drafter runs many perpetual requests, so default its running cap higher.
    if server_args.max_running_requests is None:
        server_args.max_running_requests = 64
        logger.warning(
            "max_running_requests is reset to 64 for decoupled speculative "
            "decoding; override with --max-running-requests."
        )

    # Synchronous scheduler: the cross-process verify/commit handshake observes
    # each step before starting the next, which the overlap pipeline may break.
    # To support overlap, have the decoupled mixin relay per-step results as futures.
    if not server_args.disable_overlap_schedule:
        server_args.disable_overlap_schedule = True
        logger.warning(
            "Overlap scheduler is disabled for decoupled speculative decoding."
        )

    # Drafter only.
    # - Radix cache off: rollback frees the diverged KV tail with a direct
    #   token_to_kv_pool_allocator.free that isn't radix ref-count aware, so a
    #   shared (COW) page could be freed wrongly. The codebase supports radix
    #   (colocated spec uses it); to keep it on, make rollback free only the
    #   request's private tail (never inserted in the tree) under lock_ref.
    # - Mamba no_buffer: the extra_buffer cross-request SSM-state prefix cache isn't
    #   validated for decoupled yet; set explicitly because the mamba handler ran
    #   earlier in __post_init__, so disabling radix here doesn't undo it.
    if is_drafter:
        if not server_args.disable_radix_cache:
            server_args.disable_radix_cache = True
            logger.warning("Radix cache is disabled for the decoupled drafter.")
        if server_args.mamba_radix_cache_strategy != "no_buffer":
            server_args.mamba_radix_cache_strategy = "no_buffer"
            logger.warning(
                "mamba_radix_cache_strategy is set to 'no_buffer' for the "
                "decoupled drafter."
            )

    # Mixed chunked prefill is unsupported by speculative decoding in general
    # (check_server_args asserts it), not only decoupled.
    if server_args.enable_mixed_chunk:
        server_args.enable_mixed_chunk = False
        logger.warning(
            "Mixed chunked prefill is disabled for decoupled speculative decoding."
        )

    # Decoupled runs a real draft algorithm with an explicit speculation depth.
    if server_args.speculative_algorithm is None:
        raise ValueError(
            "decoupled speculative decoding requires --speculative-algorithm "
            "to be set (the draft technique the engines run, e.g. STANDALONE)."
        )
    if server_args.speculative_num_steps is None:
        raise ValueError(
            "decoupled speculative decoding requires speculative_num_steps to be set."
        )

    # topk == 1 (chain only): the IPC protocol streams linear tokens; tree drafting
    # (topk > 1) needs a tree-structured protocol + tree verify. Pinned to 1.
    if (
        server_args.speculative_eagle_topk is not None
        and server_args.speculative_eagle_topk != 1
    ):
        raise ValueError(
            "decoupled speculative decoding only supports speculative_eagle_topk "
            f"== 1, got {server_args.speculative_eagle_topk}."
        )
    server_args.speculative_eagle_topk = 1

    # Chain speculation verifies num_steps + 1 tokens per window.
    expected_num_draft_tokens = int(server_args.speculative_num_steps) + 1
    if server_args.speculative_num_draft_tokens != expected_num_draft_tokens:
        logger.warning(
            "speculative_num_draft_tokens is adjusted to speculative_num_steps + 1 "
            "for decoupled speculative decoding."
        )
        server_args.speculative_num_draft_tokens = expected_num_draft_tokens


def _handle_dflash(server_args: ServerArgs) -> None:
    if server_args.enable_dp_attention:
        raise ValueError(
            "Currently DFLASH speculative decoding does not support dp attention."
        )

    if server_args.pp_size != 1:
        raise ValueError(
            "Currently DFLASH speculative decoding only supports pp_size == 1."
        )

    if server_args.speculative_draft_model_path is None:
        raise ValueError(
            "DFLASH speculative decoding requires setting --speculative-draft-model-path."
        )

    # DFLASH does not use EAGLE-style `num_steps`/`topk`, but those fields still
    # affect generic scheduler/KV-cache accounting (buffer sizing, KV freeing,
    # RoPE reservation). Force them to 1 to avoid surprising memory behavior.
    #
    # For DFlash, the natural unit is `block_size` (verify window length).
    if server_args.speculative_num_steps is None:
        server_args.speculative_num_steps = 1
    elif int(server_args.speculative_num_steps) != 1:
        logger.warning(
            "DFLASH only supports speculative_num_steps == 1; overriding speculative_num_steps=%s to 1.",
            server_args.speculative_num_steps,
        )
        server_args.speculative_num_steps = 1

    if server_args.speculative_eagle_topk is None:
        server_args.speculative_eagle_topk = 1
    elif int(server_args.speculative_eagle_topk) != 1:
        logger.warning(
            "DFLASH only supports speculative_eagle_topk == 1; overriding speculative_eagle_topk=%s to 1.",
            server_args.speculative_eagle_topk,
        )
        server_args.speculative_eagle_topk = 1

    if server_args.speculative_dflash_block_size is not None:
        if int(server_args.speculative_dflash_block_size) <= 0:
            raise ValueError(
                "DFLASH requires --speculative-dflash-block-size to be positive, "
                f"got {server_args.speculative_dflash_block_size}."
            )
        if server_args.speculative_num_draft_tokens is not None and int(
            server_args.speculative_num_draft_tokens
        ) != int(server_args.speculative_dflash_block_size):
            raise ValueError(
                "Both --speculative-num-draft-tokens and --speculative-dflash-block-size are set "
                "but they differ. For DFLASH they must match. "
                f"speculative_num_draft_tokens={server_args.speculative_num_draft_tokens}, "
                f"speculative_dflash_block_size={server_args.speculative_dflash_block_size}."
            )
        server_args.speculative_num_draft_tokens = int(
            server_args.speculative_dflash_block_size
        )

    if server_args.speculative_num_draft_tokens is None:
        from sglang.srt.speculative.dflash_utils import (
            parse_dflash_draft_config,
        )

        model_override_args = json.loads(server_args.json_model_override_args)
        inferred_block_size = None
        try:
            from sglang.srt.utils.hf_transformers_utils import get_config

            draft_hf_config = get_config(
                server_args.speculative_draft_model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.speculative_draft_model_revision,
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
        server_args.speculative_num_draft_tokens = inferred_block_size

    if server_args.speculative_draft_window_size is not None:
        draft_tokens = int(server_args.speculative_num_draft_tokens)
        if server_args.speculative_draft_window_size < draft_tokens:
            raise ValueError(
                "--speculative-draft-window-size must be >= "
                "--speculative-num-draft-tokens (block_size). "
                f"window_size={server_args.speculative_draft_window_size}, block_size={draft_tokens}."
            )

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    if server_args.enable_mixed_chunk:
        server_args.enable_mixed_chunk = False
        logger.warning(
            "Mixed chunked prefill is disabled because of using dflash speculative decoding."
        )


def _handle_frozen_kv_mtp(server_args: ServerArgs) -> None:
    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    if server_args.enable_mixed_chunk:
        server_args.enable_mixed_chunk = False
        logger.warning(
            "Mixed chunked prefill is disabled because of using "
            "Frozen-KV MTP speculative decoding."
        )


def _handle_eagle_family(server_args: ServerArgs) -> None:
    if (
        server_args.speculative_algorithm == "STANDALONE"
        and server_args.enable_dp_attention
    ):
        # TODO: support dp attention for standalone speculative decoding
        raise ValueError(
            "Currently standalone speculative decoding does not support dp attention."
        )

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    if server_args.disable_overlap_schedule:
        logger.warning(
            "Non-overlap (synchronous) spec v2 is used for eagle/eagle3/standalone "
            "speculative decoding."
        )

    if server_args.enable_mixed_chunk:
        server_args.enable_mixed_chunk = False
        logger.warning(
            "Mixed chunked prefill is disabled because of using "
            "eagle speculative decoding."
        )

    model_arch = server_args.get_model_config().hf_config.architectures[0]
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
        if server_args.speculative_draft_model_path is None:
            server_args.speculative_draft_model_path = server_args.model_path
            server_args.speculative_draft_model_revision = server_args.revision
        else:
            if model_arch not in [
                "MistralLarge3ForCausalLM",
                "PixtralForConditionalGeneration",
            ]:
                logger.warning(
                    "DeepSeek MTP does not require setting speculative_draft_model_path."
                )

    if (
        not server_args.speculative_adaptive
        and server_args.speculative_num_steps is None
    ):
        assert (
            server_args.speculative_eagle_topk is None
            and server_args.speculative_num_draft_tokens is None
        )

        (
            server_args.speculative_num_steps,
            server_args.speculative_eagle_topk,
            server_args.speculative_num_draft_tokens,
        ) = _auto_choose_speculative_params(server_args, model_arch)

    if (
        server_args.attention_backend == "trtllm_mha"
        or server_args.decode_attention_backend == "trtllm_mha"
        or server_args.prefill_attention_backend == "trtllm_mha"
    ):
        if server_args.speculative_eagle_topk > 1:
            raise ValueError(
                "trtllm_mha backend only supports topk = 1 for speculative decoding."
            )

    if server_args.speculative_use_rejection_sampling:
        # Resolved alias by now: NEXTN -> EAGLE, Gemma4 draft -> FROZEN_KV_MTP.
        # Only the EAGLE/EAGLE3 draft workers emit a target-vocab proposal that
        # the rejection-sampling kernel consumes; everything else (STANDALONE,
        # FROZEN_KV_MTP, NGRAM, DFLASH) is unsupported.
        if server_args.speculative_algorithm not in ("EAGLE", "EAGLE3"):
            raise NotImplementedError(
                "--speculative-use-rejection-sampling is only supported for "
                "EAGLE / EAGLE3 / NEXTN, not "
                f"speculative_algorithm={server_args.speculative_algorithm}."
            )
        if server_args.speculative_eagle_topk != 1:
            raise ValueError(
                "--speculative-use-rejection-sampling requires --speculative-eagle-topk=1."
            )
        if (
            server_args.speculative_accept_threshold_single != 1.0
            or server_args.speculative_accept_threshold_acc != 1.0
        ):
            raise ValueError(
                "--speculative-use-rejection-sampling is incompatible with "
                "--speculative-accept-threshold-single / "
                "--speculative-accept-threshold-acc; rejection sampling ignores "
                "the accept thresholds."
            )
        if server_args.enable_deterministic_inference:
            raise ValueError(
                "--speculative-use-rejection-sampling is incompatible with "
                "--enable-deterministic-inference; the sampling kernel draws "
                "coins from the global RNG and is not batch-invariant."
            )
        if server_args.enable_multi_layer_eagle:
            raise NotImplementedError(
                "--speculative-use-rejection-sampling is not supported with "
                "multi-layer EAGLE (--enable-multi-layer-eagle)."
            )
        logger.info(
            "Rejection sampling is enabled for speculative decoding "
            "(speculative_use_rejection_sampling=True)."
        )

    if (
        server_args.speculative_eagle_topk == 1
        and server_args.speculative_num_draft_tokens
        != server_args.speculative_num_steps + 1
    ):
        logger.warning(
            "speculative_num_draft_tokens is adjusted to speculative_num_steps + 1 when speculative_eagle_topk == 1"
        )
        server_args.speculative_num_draft_tokens = server_args.speculative_num_steps + 1

    # topk > 1 + page_size > 1 needs the two-pass cascade draft-decode (shared prefix
    # pass + per-branch expand pass with prefix-tail dup). Only these backends implement
    # it; flashmla / trtllm_mla / cutlass_mla can't express the per-branch tree, so reject.
    _PAGE_TREE_SPEC_BACKENDS = ("flashinfer", "fa3", "triton")
    if (
        server_args.speculative_eagle_topk > 1
        and server_args.page_size > 1
        and server_args.attention_backend not in _PAGE_TREE_SPEC_BACKENDS
    ):
        raise ValueError(
            f"speculative_eagle_topk > 1 with page_size > 1 is only supported on "
            f"{_PAGE_TREE_SPEC_BACKENDS}; got attention_backend="
            f"{server_args.attention_backend!r}. Use page_size == 1 or one of those backends."
        )


def _handle_ngram(server_args: ServerArgs) -> None:
    if not server_args.device.startswith("cuda"):
        raise ValueError("Ngram speculative decoding only supports CUDA device.")

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    server_args.enable_mixed_chunk = False
    server_args.speculative_eagle_topk = server_args.speculative_ngram_max_bfs_breadth
    if server_args.speculative_num_draft_tokens is None:
        server_args.speculative_num_draft_tokens = 12
        logger.warning(
            "speculative_num_draft_tokens is set to 12 by default for ngram speculative decoding. "
            "You can override this by explicitly setting --speculative-num-draft-tokens."
        )
    if server_args.speculative_num_steps is None:
        server_args.speculative_num_steps = (
            server_args.speculative_num_draft_tokens
            // server_args.speculative_eagle_topk
        )
    if server_args.speculative_ngram_external_corpus_path is not None:
        if server_args.speculative_ngram_external_sam_budget <= 0:
            raise ValueError(
                "--speculative-ngram-external-sam-budget must be positive when "
                "--speculative-ngram-external-corpus-path is set."
            )
        if server_args.speculative_ngram_external_corpus_max_tokens <= 0:
            raise ValueError(
                "--speculative-ngram-external-corpus-max-tokens must be positive when "
                "--speculative-ngram-external-corpus-path is set."
            )
        if (
            server_args.speculative_ngram_external_sam_budget
            > server_args.speculative_num_draft_tokens - 1
        ):
            raise ValueError(
                "speculative_ngram_external_sam_budget must be less than or equal to "
                f"speculative_num_draft_tokens - 1 ({server_args.speculative_num_draft_tokens - 1})."
            )
    logger.warning(
        "The mixed chunked prefill are disabled because of "
        "using ngram speculative decoding."
    )

    if (
        server_args.speculative_eagle_topk > 1
        and server_args.page_size > 1
        and server_args.attention_backend != "flashinfer"
    ):
        raise ValueError(
            f"speculative_eagle_topk({server_args.speculative_eagle_topk}) > 1 "
            f"with page_size({server_args.page_size}) > 1 is unstable "
            "and produces incorrect results for paged attention backends. "
            "This combination is only supported for the 'flashinfer' backend."
        )
    if server_args.enable_dp_attention:
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
        server_args.speculative_adaptive = False


def _init_adaptive_speculative_params(server_args: ServerArgs) -> None:
    from sglang.srt.speculative.adaptive_spec_params import (
        resolve_candidate_steps_from_config,
    )

    candidate_steps = resolve_candidate_steps_from_config(
        cfg_path=server_args.speculative_adaptive_config,
    )

    if server_args.speculative_eagle_topk is None:
        server_args.speculative_eagle_topk = 1

    if server_args.speculative_num_steps is None:
        server_args.speculative_num_steps = candidate_steps[len(candidate_steps) // 2]

    if server_args.speculative_num_steps not in candidate_steps:
        raise ValueError(
            f"--speculative-num-steps={server_args.speculative_num_steps} "
            f"is not in the adaptive config candidate_steps {candidate_steps}. "
            "Pass one of those values."
        )

    server_args.speculative_num_draft_tokens = server_args.speculative_num_steps + 1


def _auto_choose_speculative_params(server_args: ServerArgs, model_arch: str) -> tuple:
    """
    Automatically choose the parameters for speculative decoding.

    You can tune them on your own models and prompts with scripts/playground/bench_speculative.py
    """
    if server_args.speculative_algorithm == "STANDALONE":
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
