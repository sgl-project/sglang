from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _disable_overlap_schedule_for_cpu(server_args: ServerArgs) -> None:
    if server_args.device != "cpu" or server_args.disable_overlap_schedule:
        return

    server_args.disable_overlap_schedule = True
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
    if (
        server_args.speculative_draft_model_path is not None
        and server_args.speculative_draft_model_revision is None
    ):
        server_args.speculative_draft_model_revision = "main"

    # Moved to the resolution pipeline (arg_groups/overrides.py:
    # _speculative_moe_runner_default), invoked here at its legacy slot.
    from sglang.srt.arg_groups.overrides import (
        _speculative_moe_runner_default,
        run_post_process_pass,
    )

    run_post_process_pass(server_args, _speculative_moe_runner_default)

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


def _handle_dflash(server_args: ServerArgs) -> None:
    from sglang.srt.arg_groups.overrides import resolved_view

    if not (server_args.device.startswith("cuda") or server_args.device == "npu"):
        raise ValueError(
            "DFLASH speculative decoding only supports CUDA and NPU devices."
        )

    if resolved_view(server_args).enable_dp_attention:
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

    _resolve_dflash_widths(server_args)

    if server_args.speculative_draft_window_size is not None:
        block_size = int(server_args.speculative_dflash_block_size)
        if server_args.speculative_draft_window_size < block_size:
            raise ValueError(
                "--speculative-draft-window-size must be >= "
                "--speculative-dflash-block-size (the draft block width). "
                f"window_size={server_args.speculative_draft_window_size}, block_size={block_size}."
            )

    _resolve_dflash_draft_attention_backend(server_args)

    _validate_dflash_tree_admission(server_args)

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


# Target attention backends that can express a *tree*-shaped TARGET_VERIFY: they either
# consume `spec_info.custom_mask` + `mask_indptr` directly (triton, flashinfer) or rebuild a
# compacted page table from it (fa3 cascade attention). Deliberately a positive list, not the
# complement of `_DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS` -- that set means "can express a
# *linear* verify with its built-in causal path, so skip building a mask", and it contains
# exactly the mask-capable backends, so complementing it would reject triton (the SM100
# default for hybrid-GDN models).
_DFLASH_TREE_VERIFY_BACKENDS = ("flashinfer", "triton", "fa3")

# Backends that silently compute a causal chain over a tree layout instead of raising: their
# metadata is scalar-uniform with no per-node visibility, so a wrong answer is the failure mode.
_DFLASH_TREE_SILENTLY_WRONG_BACKENDS = ("trtllm_mha",)

_DFLASH_TREE_WIDTH_FLAG = "--speculative-dflash-tree-width"


def _load_dflash_draft_config(server_args: ServerArgs, *, required: bool):
    """Parse `dflash_config` out of the draft checkpoint, or return None if unreadable.

    `required=True` raises instead of returning None: tree-width admission validates
    `tree_width <= selector_top_k` and "checkpoint has a selector" from this config and
    lives only here (the worker does not re-check), so a swallowed read would turn both
    checks into silent passes.
    """
    from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
    from sglang.srt.utils.hf_transformers_utils import get_config

    try:
        draft_hf_config = get_config(
            server_args.speculative_draft_model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.speculative_draft_model_revision,
            model_override_args=json.loads(server_args.json_model_override_args),
        )
        return parse_dflash_draft_config(draft_hf_config=draft_hf_config)
    except Exception as e:
        if required:
            raise ValueError(
                f"{_DFLASH_TREE_WIDTH_FLAG} > 1 requires reading dflash_config from the draft "
                f"checkpoint at {server_args.speculative_draft_model_path!r} to validate the "
                "beam width against selector_top_k, but the config could not be parsed. "
                f"Fix the draft checkpoint or drop {_DFLASH_TREE_WIDTH_FLAG}. Error: {e}"
            ) from e
        logger.warning(
            "Failed to read DFLASH dflash_config from the draft model config; "
            "block_size falls back to the built-in default. Error: %s",
            e,
        )
        return None


def _resolve_dflash_tree_width(server_args: ServerArgs) -> int:
    """The beam width kept per draft depth. 1 reproduces today's single-path chain."""
    if server_args.speculative_eagle_topk is not None:
        raise ValueError(
            "--speculative-eagle-topk is EAGLE-only and has no meaning for DFLASH, which "
            "drafts a whole block in parallel rather than expanding one token at a time. "
            f"Use {_DFLASH_TREE_WIDTH_FLAG} to widen the DFLASH draft tree instead. Got "
            f"--speculative-eagle-topk={server_args.speculative_eagle_topk}."
        )

    if server_args.speculative_dflash_tree_width is None:
        return 1

    tree_width = int(server_args.speculative_dflash_tree_width)
    if tree_width < 1:
        raise ValueError(
            f"{_DFLASH_TREE_WIDTH_FLAG} must be >= 1 (1 = single-path chain), "
            f"got {tree_width}."
        )
    return tree_width


_DFLASH_DEFAULT_BLOCK_SIZE = 16


def _resolve_dflash_block_size(
    server_args: ServerArgs, *, tree_width: int, draft_config
) -> int:
    """The draft block width, from (in order) the explicit flag, the
    --speculative-num-draft-tokens alias, the draft checkpoint, or a hardcoded default."""
    explicit = server_args.speculative_dflash_block_size
    alias = server_args.speculative_num_draft_tokens

    if alias is not None and tree_width > 1:
        raise ValueError(
            "--speculative-num-draft-tokens is the *verify* window width, and with "
            f"{_DFLASH_TREE_WIDTH_FLAG} > 1 DFLASH derives it as "
            "1 + (block_size - 1) * tree_width, so setting it directly is ambiguous. "
            "Pass the draft block width as --speculative-dflash-block-size instead. Got "
            f"--speculative-num-draft-tokens={alias}, "
            f"{_DFLASH_TREE_WIDTH_FLAG}={tree_width}."
        )

    if explicit is not None:
        if int(explicit) <= 0:
            raise ValueError(
                "DFLASH requires --speculative-dflash-block-size to be positive, "
                f"got {explicit}."
            )
        if alias is not None and int(alias) != int(explicit):
            raise ValueError(
                "Both --speculative-num-draft-tokens and --speculative-dflash-block-size are set "
                "but they differ. At tree width 1 they mean the same thing and must match. "
                f"speculative_num_draft_tokens={alias}, "
                f"speculative_dflash_block_size={explicit}."
            )
        return int(explicit)

    if alias is not None:
        return int(alias)

    inferred = (
        None if draft_config is None else draft_config.resolve_block_size(default=None)
    )
    if inferred is None:
        logger.warning(
            "DFLASH block_size is not set and could not be inferred from the draft "
            "checkpoint; defaulting to %d.",
            _DFLASH_DEFAULT_BLOCK_SIZE,
        )
        return _DFLASH_DEFAULT_BLOCK_SIZE
    return int(inferred)


def _resolve_dflash_widths(server_args: ServerArgs) -> None:
    """Split the one width DFLASH used to carry into three, and write them back.

    Post-conditions, relied on by the worker and by generic KV / scheduler accounting:

      speculative_dflash_block_size  = block_size  (draft block width, never None)
      speculative_num_draft_tokens   = 1 + (block_size - 1) * tree_width  (verify width)
      speculative_eagle_topk         = tree_width

    `speculative_eagle_topk` is not a user-facing knob for DFLASH (it is rejected above); it
    is reused as the carrier for "is this verify a tree" because two correctness-critical
    gates already read it -- `conv_window_dedup_enabled` (dense conv windows, which tree
    ancestors need) and the GDN backend's tree-kernel dispatch.
    """
    tree_width = _resolve_dflash_tree_width(server_args)

    # W > 1 must read the draft config to validate the beam against selector_top_k; W == 1
    # only needs it when block_size has no explicit source, so the common single-path launch
    # keeps its current (config-free) startup path.
    needs_config = tree_width > 1 or (
        server_args.speculative_dflash_block_size is None
        and server_args.speculative_num_draft_tokens is None
    )
    draft_config = (
        _load_dflash_draft_config(server_args, required=tree_width > 1)
        if needs_config
        else None
    )

    block_size = _resolve_dflash_block_size(
        server_args, tree_width=tree_width, draft_config=draft_config
    )

    if tree_width > 1:
        _validate_dflash_tree_selector(
            draft_config=draft_config, tree_width=tree_width, block_size=block_size
        )

    server_args.speculative_dflash_block_size = block_size
    server_args.speculative_num_draft_tokens = 1 + (block_size - 1) * tree_width
    server_args.speculative_eagle_topk = tree_width


def _validate_dflash_tree_selector(
    *, draft_config, tree_width: int, block_size: int
) -> None:
    """The tree is a beam over the candidate selector's transition lattice, so the checkpoint
    must have a selector and the beam cannot be wider than the lattice's candidate axis."""
    if not draft_config.selector_rank or not draft_config.selector_top_k:
        raise ValueError(
            f"{_DFLASH_TREE_WIDTH_FLAG} > 1 requires a DFlash 2 draft checkpoint with a "
            "candidate selector (dflash_config.selector_rank / selector_top_k); the tree is "
            "built from the selector's transition lattice. This checkpoint has "
            f"selector_rank={draft_config.selector_rank}, "
            f"selector_top_k={draft_config.selector_top_k}."
        )
    if tree_width > int(draft_config.selector_top_k):
        raise ValueError(
            f"{_DFLASH_TREE_WIDTH_FLAG} must be <= the draft checkpoint's "
            f"dflash_config.selector_top_k ({draft_config.selector_top_k}): draft depth 1 has "
            "only selector_top_k candidates under the root, so a wider beam cannot fill the "
            f"fixed tree shape. Got {_DFLASH_TREE_WIDTH_FLAG}={tree_width}."
        )
    if block_size < 2:
        raise ValueError(
            f"{_DFLASH_TREE_WIDTH_FLAG} > 1 needs at least one drafted depth, i.e. "
            f"--speculative-dflash-block-size >= 2, got {block_size}."
        )


def _dflash_effective_target_backends(server_args: ServerArgs) -> dict[str, str]:
    """The target-side attention backends that a DFLASH verify forward can land on."""
    from sglang.srt.arg_groups.overrides import resolved_view

    view = resolved_view(server_args)
    candidates = {
        "--attention-backend": view.attention_backend,
        "--prefill-attention-backend": view.prefill_attention_backend,
        "--decode-attention-backend": view.decode_attention_backend,
    }
    return {flag: name for flag, name in candidates.items() if name is not None}


def _validate_dflash_tree_backends(server_args: ServerArgs, *, tree_width: int) -> None:
    from sglang.srt.arg_groups.overrides import resolved_view

    for flag, backend in _dflash_effective_target_backends(server_args).items():
        if backend in _DFLASH_TREE_VERIFY_BACKENDS:
            continue
        if backend in _DFLASH_TREE_SILENTLY_WRONG_BACKENDS:
            raise ValueError(
                f"{flag}={backend!r} cannot verify a tree-shaped DFLASH draft: it has no "
                "custom-mask path and its attention metadata carries no per-node visibility, "
                "so it would silently verify the tree as a causal chain and return wrong "
                f"tokens rather than failing. Use one of {_DFLASH_TREE_VERIFY_BACKENDS} with "
                f"{_DFLASH_TREE_WIDTH_FLAG}={tree_width}."
            )
        raise ValueError(
            f"{_DFLASH_TREE_WIDTH_FLAG} > 1 needs a target attention backend that consumes a "
            f"custom tree mask; only {_DFLASH_TREE_VERIFY_BACKENDS} do. Got {flag}={backend!r}."
        )

    # DFLASH cannot borrow EAGLE's page-tree gate: that check lives in the EAGLE branch of
    # this hook, and DFLASH dispatches to _handle_dflash instead, so it never runs here.
    page_size = int(resolved_view(server_args).page_size)
    if page_size > 1:
        raise ValueError(
            f"{_DFLASH_TREE_WIDTH_FLAG} > 1 is only implemented at --page-size 1; a paged "
            "tree verify needs the two-pass cascade draft-decode that DFLASH does not "
            f"implement. Got --page-size {page_size}."
        )


def _validate_dflash_tree_admission(server_args: ServerArgs) -> None:
    """Reject configurations that are structurally incompatible with a tree-shaped verify.

    Only reached with tree_width > 1; every branch raises rather than degrading, because a
    silent fallback to chain verify would look like "the tree bought us nothing" in the
    acceptance-length measurement this feature exists to produce.
    """
    tree_width = int(server_args.speculative_eagle_topk)
    if tree_width <= 1:
        return

    _validate_dflash_tree_backends(server_args, tree_width=tree_width)

    if server_args.enable_linear_replayssm_spec:
        raise ValueError(
            "--enable-linear-replayssm-spec is structurally incompatible with "
            f"{_DFLASH_TREE_WIDTH_FLAG} > 1: it skips the per-step intermediate SSM states a "
            "tree needs to restart each node from its parent, and its chunked verify kernel "
            "uses a strictly-lower causal mask. Drop one of the two. (This is not caught by "
            "the flag's own topk check, which runs before the speculative hook assigns topk.)"
        )

    if not (server_args.disable_cuda_graph or server_args.disable_decode_cuda_graph):
        raise ValueError(
            f"{_DFLASH_TREE_WIDTH_FLAG} > 1 requires --disable-decode-cuda-graph for now: the "
            "captured DFLASH verify buffers carry neither the tree custom mask nor the "
            "retrieve_* tree links, and a zero-initialized retrieve_parent_token replays as a "
            "star-shaped tree without warning."
        )

    from sglang.srt.environ import envs

    if (
        envs.SGLANG_SIMULATE_ACC_LEN.get() > 0
        and envs.SGLANG_SIMULATE_ACC_TOKEN_MODE.get() == "real-draft-token"
    ):
        raise ValueError(
            "SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token forces an accept length along a "
            f"linear chain and cannot address tree nodes, so it is invalid with "
            f"{_DFLASH_TREE_WIDTH_FLAG}={tree_width}. Use SGLANG_SIMULATE_ACC_TOKEN_MODE=fixed "
            "or unset SGLANG_SIMULATE_ACC_LEN."
        )



def _target_checkpoint_bundles_dspark_draft(server_args: ServerArgs) -> bool:
    from sglang.srt.speculative.dspark_components.dspark_config import (
        checkpoint_bundles_dspark_draft,
    )

    return checkpoint_bundles_dspark_draft(server_args.get_model_config().hf_config)


def _handle_dspark(server_args: ServerArgs) -> None:
    _is_npu = server_args.device.startswith("npu")
    if not server_args.device.startswith(("cuda", "npu")):
        raise ValueError(
            "DSpark speculative decoding only supports CUDA or NPU device."
        )

    # dp_size==1 with dp_attention is a degenerate flag under DSV4 CP; skip DP-only checks.
    if server_args.enable_dp_attention and server_args.dp_size > 1:
        if not server_args.enable_dp_lm_head:
            raise ValueError("DSpark with dp attention requires --enable-dp-lm-head.")
        if not _is_npu and server_args.moe_a2a_backend not in ("none", "megamoe"):
            raise ValueError(
                "DSpark with dp attention supports moe_a2a_backend 'none' "
                "(built-in TP MoE) or 'megamoe', got "
                f"{server_args.moe_a2a_backend!r}."
            )
        if not _is_npu and server_args.moe_a2a_backend != "none":
            from sglang.srt.speculative.ragged_verify import (
                RaggedVerifyMode,
                read_ragged_verify_mode,
            )

            if read_ragged_verify_mode() is not RaggedVerifyMode.STATIC:
                raise ValueError(
                    "DSpark with dp attention + "
                    f"moe_a2a_backend={server_args.moe_a2a_backend!r} requires "
                    "SGLANG_RAGGED_VERIFY_MODE=static."
                )
        if server_args.attn_cp_size > 1:
            raise ValueError(
                "DSpark with dp attention does not support context parallel "
                f"(attn_cp_size={server_args.attn_cp_size})."
            )
        if (
            not _is_npu
            and server_args.speculative_moe_a2a_backend is not None
            and server_args.speculative_moe_a2a_backend != server_args.moe_a2a_backend
        ):
            raise ValueError(
                "DSpark ignores --speculative-moe-a2a-backend; with dp attention it "
                f"must match the target moe_a2a_backend={server_args.moe_a2a_backend!r} "
                f"(got {server_args.speculative_moe_a2a_backend!r})."
            )

    if server_args.pp_size != 1:
        raise ValueError(
            "Currently DSpark speculative decoding only supports pp_size == 1."
        )

    if server_args.speculative_draft_model_path is None:
        if _target_checkpoint_bundles_dspark_draft(server_args):
            server_args.speculative_draft_model_path = server_args.model_path
            server_args.speculative_draft_model_revision = server_args.revision
            logger.info(
                "DSpark draft weights are bundled in the target checkpoint; "
                "defaulting --speculative-draft-model-path to --model-path (%s).",
                server_args.model_path,
            )
        else:
            raise ValueError(
                "DSpark dense speculative decoding requires setting "
                "--speculative-draft-model-path."
            )

    if server_args.speculative_num_steps is None:
        server_args.speculative_num_steps = 1
    elif int(server_args.speculative_num_steps) != 1:
        logger.warning(
            "DSpark only supports speculative_num_steps == 1; overriding speculative_num_steps=%s to 1.",
            server_args.speculative_num_steps,
        )
        server_args.speculative_num_steps = 1

    if server_args.speculative_eagle_topk is None:
        server_args.speculative_eagle_topk = 1
    elif int(server_args.speculative_eagle_topk) != 1:
        logger.warning(
            "DSpark only supports speculative_eagle_topk == 1; overriding speculative_eagle_topk=%s to 1.",
            server_args.speculative_eagle_topk,
        )
        server_args.speculative_eagle_topk = 1

    gamma: Optional[int] = None
    if server_args.speculative_dspark_block_size is not None:
        if int(server_args.speculative_dspark_block_size) <= 0:
            raise ValueError(
                "DSpark requires --speculative-dspark-block-size to be positive, "
                f"got {server_args.speculative_dspark_block_size}."
            )
        gamma = int(server_args.speculative_dspark_block_size)
    else:
        from sglang.srt.speculative.dspark_components.dspark_config import (
            DEFAULT_DSPARK_GAMMA,
            read_draft_checkpoint_gamma,
        )

        try:
            gamma = read_draft_checkpoint_gamma(server_args=server_args)
        except Exception as e:
            logger.warning(
                "Failed to read DSpark gamma from draft model config; "
                "cannot cross-check --speculative-num-draft-tokens. Error: %s",
                e,
            )
        if gamma is None and server_args.speculative_num_draft_tokens is None:
            gamma = DEFAULT_DSPARK_GAMMA
            logger.warning(
                "DSpark gamma is not set; defaulting to %d.",
                gamma,
            )

    if gamma is not None:
        verify_window = int(gamma) + 1
        if (
            server_args.speculative_num_draft_tokens is not None
            and int(server_args.speculative_num_draft_tokens) != verify_window
        ):
            raise ValueError(
                "DSpark speculative_num_draft_tokens must equal gamma + 1 "
                f"(= {verify_window} for gamma={gamma}), but got "
                f"speculative_num_draft_tokens={server_args.speculative_num_draft_tokens}."
            )
        server_args.speculative_num_draft_tokens = verify_window

    if server_args.speculative_num_draft_tokens is None:
        raise ValueError(
            "DSpark could not resolve speculative_num_draft_tokens; set "
            "--speculative-dspark-block-size (= gamma)."
        )
    if int(server_args.speculative_num_draft_tokens) < 2:
        raise ValueError(
            "DSpark speculative_num_draft_tokens must be >= 2 (= gamma + 1), "
            f"got {server_args.speculative_num_draft_tokens}."
        )

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    if server_args.enable_mixed_chunk:
        server_args.enable_mixed_chunk = False
        logger.warning(
            "Mixed chunked prefill is disabled because of using dspark speculative decoding."
        )

    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )

    ragged_mode = read_ragged_verify_mode()
    if (
        server_args.speculative_dspark_align_verify_tokens_to_graph_tier
        and ragged_mode is not RaggedVerifyMode.COMPACT
    ):
        logger.warning(
            "--speculative-dspark-align-verify-tokens-to-graph-tier only takes "
            "effect with SGLANG_RAGGED_VERIFY_MODE=compact (got %r); it will be "
            "a no-op.",
            ragged_mode.value,
        )
    if (
        server_args.speculative_dspark_sps_table_path
        and ragged_mode is RaggedVerifyMode.STATIC
    ):
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
    from sglang.srt.utils import is_hip

    supported_draft_backends = (
        "flashinfer",
        "fa3",
        "fa4",
        "triton",
        "trtllm_mha",
        "ascend",
    )
    # Use triton on ROCm (no FlashInfer), flashinfer on CUDA.
    fallback_backend = "triton" if is_hip() else "flashinfer"

    draft_backend = server_args.speculative_draft_attention_backend
    if draft_backend is None:
        from sglang.srt.arg_groups.overrides import (
            attention_backends_of,
            resolved_view,
        )

        draft_backend, _ = attention_backends_of(resolved_view(server_args))
    if draft_backend is None:
        draft_backend = fallback_backend
    elif draft_backend == "trtllm_mha":
        from sglang.srt.speculative.dflash_utils import get_dflash_layer_types
        from sglang.srt.utils.hf_transformers_utils import get_config

        draft_hf_config = get_config(
            server_args.speculative_draft_model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.speculative_draft_model_revision,
            model_override_args=json.loads(server_args.json_model_override_args),
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
    server_args.speculative_draft_attention_backend = draft_backend


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
    from sglang.srt.arg_groups.overrides import (
        attention_backends_of,
        resolved_view,
    )

    if (
        server_args.speculative_algorithm == "STANDALONE"
        and resolved_view(server_args).enable_dp_attention
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

    _disable_overlap_schedule_for_cpu(server_args)

    if resolved_view(server_args).disable_overlap_schedule:
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

    if "trtllm_mha" in attention_backends_of(resolved_view(server_args)):
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

        from sglang.srt.arg_groups.overrides import resolved_view

        if (
            resolved_view(server_args).enable_multi_layer_eagle
            and server_args.speculative_eagle_topk != 1
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
    view = resolved_view(server_args)
    if (
        server_args.speculative_eagle_topk > 1
        and view.page_size > 1
        and view.attention_backend not in _PAGE_TREE_SPEC_BACKENDS
    ):
        raise ValueError(
            f"speculative_eagle_topk > 1 with page_size > 1 is only supported on "
            f"{_PAGE_TREE_SPEC_BACKENDS}; got attention_backend="
            f"{view.attention_backend!r}. Use page_size == 1 or one of those backends."
        )


def _handle_ngram(server_args: ServerArgs) -> None:
    if server_args.device not in ("cuda", "cpu"):
        raise ValueError(
            "Ngram speculative decoding only supports CUDA or CPU devices."
        )

    _disable_overlap_schedule_for_cpu(server_args)

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

    from sglang.srt.arg_groups.overrides import resolved_view

    view = resolved_view(server_args)
    if (
        server_args.speculative_eagle_topk > 1
        and view.page_size > 1
        and view.attention_backend != "flashinfer"
    ):
        raise ValueError(
            f"speculative_eagle_topk({server_args.speculative_eagle_topk}) > 1 "
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
