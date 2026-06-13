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


def handle_speculative_decoding(server_args: "ServerArgs") -> None:
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

    if algo is not None:
        algo.handle_server_args(server_args)


def _maybe_disable_adaptive(server_args: "ServerArgs") -> None:
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


def _init_adaptive_speculative_params(server_args: "ServerArgs") -> None:
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
