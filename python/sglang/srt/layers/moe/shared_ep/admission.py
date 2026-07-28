"""Release admission checks for the SharedEP backend."""

from sglang.srt.environ import envs
from sglang.srt.layers.moe.shared_ep.lanes import (
    SharedEpLaneProtocol,
    compute_shared_ep_lane_protocol,
)
from sglang.srt.layers.moe.shared_ep.profiles import (
    RELEASE_MAX_TOKENS_PER_RANK,
)
from sglang.srt.layers.moe.utils import get_shared_ep_prefill_backend
from sglang.srt.model_executor.cuda_graph_config import Backend

_SUPPORTED_MODEL_ARCHITECTURES = frozenset(
    {
        "DeepseekV4ForCausalLM",
        "GlmMoeDsaForCausalLM",
    }
)


def _resolved_server_args(server_args):
    from sglang.srt.arg_groups.overrides import resolved_view

    return resolved_view(server_args)


def _validate_model_architecture(server_args) -> str | None:
    if str(server_args.model_path).lower() in ("dummy", "none"):
        return None
    architecture = server_args.get_model_config().hf_config.architectures[0]
    if architecture not in _SUPPORTED_MODEL_ARCHITECTURES:
        raise ValueError(
            "SharedEP initial release supports GLM-5.2 and "
            "DeepSeek-V4 Flash/Pro only; got "
            f"{architecture}"
        )
    return architecture


def _validate_shared_ep_speculative(
    view,
    *,
    prefill_backend,
) -> None:
    algorithm = getattr(view, "speculative_algorithm", None)
    speculative_fields = {
        "speculative_num_steps": getattr(view, "speculative_num_steps", None),
        "speculative_eagle_topk": getattr(view, "speculative_eagle_topk", None),
        "speculative_num_draft_tokens": getattr(
            view, "speculative_num_draft_tokens", None
        ),
        "enable_multi_layer_eagle": getattr(view, "enable_multi_layer_eagle", False),
    }
    if algorithm is None:
        if any(
            value is not None and value is not False
            for value in speculative_fields.values()
        ):
            raise ValueError(
                "SharedEP MTP settings require --speculative-algorithm NEXTN: "
                f"{speculative_fields}."
            )
        return

    algorithm = str(algorithm).upper()
    if algorithm not in ("NEXTN", "EAGLE"):
        raise ValueError(
            "SharedEP speculative release supports only linear NEXTN/MTP "
            f"(resolved algorithm EAGLE), got {algorithm}."
        )
    topk = speculative_fields["speculative_eagle_topk"]
    steps = speculative_fields["speculative_num_steps"]
    draft_tokens = speculative_fields["speculative_num_draft_tokens"]
    if topk != 1:
        raise ValueError(
            f"SharedEP NEXTN/MTP requires --speculative-eagle-topk 1, got {topk}."
        )
    if not isinstance(steps, int) or steps < 1:
        raise ValueError(
            "SharedEP NEXTN/MTP requires a positive --speculative-num-steps."
        )
    if not isinstance(draft_tokens, int) or draft_tokens != steps + 1:
        raise ValueError(
            "SharedEP NEXTN/MTP requires --speculative-num-draft-tokens "
            f"to equal steps + 1, got steps={steps}, draft_tokens={draft_tokens}."
        )
    if speculative_fields["enable_multi_layer_eagle"]:
        raise ValueError("SharedEP does not support multi-layer EAGLE.")
    if getattr(view, "speculative_adaptive", False):
        raise ValueError("SharedEP does not support adaptive speculative decoding.")

    expected_a2a = "mori" if prefill_backend.is_aiter() else "deepep"
    if getattr(view, "speculative_moe_a2a_backend", None) != expected_a2a:
        raise ValueError(
            "SharedEP NEXTN/MTP requires the materialized draft fallback "
            f"--speculative-moe-a2a-backend {expected_a2a}."
        )
    if getattr(view, "speculative_moe_runner_backend", None) != (prefill_backend.value):
        raise ValueError(
            "SharedEP NEXTN/MTP requires "
            f"--speculative-moe-runner-backend {prefill_backend.value}."
        )


def get_shared_ep_lane_protocol(server_args) -> SharedEpLaneProtocol:
    """Public pure helper used by admission tests and runtime construction."""

    return compute_shared_ep_lane_protocol(_resolved_server_args(server_args))


def validate_shared_ep_server_args(server_args) -> None:
    view = _resolved_server_args(server_args)
    if view.nnodes != 1:
        raise ValueError(
            f"SharedEP initial release is same-host only, got {view.nnodes} nodes."
        )
    if view.ep_size != 8:
        raise ValueError(f"SharedEP release requires EP8, got EP{view.ep_size}.")
    if view.dp_size != 8:
        raise ValueError(f"SharedEP release requires DP8, got DP{view.dp_size}.")
    if not view.enable_dp_attention:
        raise ValueError("SharedEP release requires --enable-dp-attention.")
    max_running_requests = RELEASE_MAX_TOKENS_PER_RANK * view.dp_size
    if view.max_running_requests is None:
        server_args.override(
            "validate_shared_ep_server_args",
            max_running_requests=max_running_requests,
        )
    elif view.max_running_requests > max_running_requests:
        raise ValueError(
            "SharedEP supports --max-running-requests "
            f"{max_running_requests} or lower, got "
            f"{view.max_running_requests}."
        )
    if view.enable_lora or view.lora_paths:
        raise ValueError("SharedEP release does not support LoRA.")
    if getattr(view, "enable_pdmux", False):
        raise ValueError(
            "SharedEP does not support PD-Multiplexing; its concurrent stream "
            "index is not part of the admitted lane protocol."
        )

    architecture = _validate_model_architecture(server_args)
    prefill_backend = get_shared_ep_prefill_backend()
    if prefill_backend.is_aiter() and not envs.SGLANG_USE_AITER.get():
        envs.SGLANG_USE_AITER.set(True)
    if view.moe_runner_backend == "auto":
        server_args.override(
            "validate_shared_ep_server_args",
            moe_runner_backend=prefill_backend.value,
        )
    elif view.moe_runner_backend != prefill_backend.value:
        raise ValueError(
            "SharedEP requires --moe-runner-backend "
            f"{prefill_backend.value} for its composite decode and prefill path "
            "on this platform."
        )

    if view.enable_single_batch_overlap:
        raise ValueError(
            "SharedEP does not support SBO/--enable-single-batch-overlap; "
            "the lane protocol covers TBO subbatches only."
        )
    if (
        view.enable_two_batch_overlap
        and architecture is not None
        and architecture != "DeepseekV4ForCausalLM"
    ):
        raise ValueError("SharedEP TBO is currently admitted only for DeepSeek-V4.")

    _validate_shared_ep_speculative(view, prefill_backend=prefill_backend)
    # Compute after all routing checks. This is also the fixed memory-resource
    # cap: every admitted lane owns disjoint VMM and epoch objects.
    compute_shared_ep_lane_protocol(view)

    decode = view.cuda_graph_config.decode
    if decode.backend == Backend.DISABLED:
        return
    capture_sizes = list(decode.bs or ())
    if decode.max_bs is not None:
        capture_sizes.append(decode.max_bs)
    largest_capture = max(capture_sizes, default=0)
    if largest_capture > RELEASE_MAX_TOKENS_PER_RANK:
        raise ValueError(
            "SharedEP decode CUDA Graph capacity exceeded: "
            f"{largest_capture} > {RELEASE_MAX_TOKENS_PER_RANK} local "
            "tokens per rank. Lower --cuda-graph-max-bs-decode and "
            "--cuda-graph-bs-decode, or disable decode CUDA Graph."
        )
