"""Release admission checks for the SharedEP backend."""

from sglang.srt.layers.moe.shared_ep.profiles import (
    RELEASE_MAX_PREFILL_TOKENS_PER_RANK,
    RELEASE_MAX_TOKENS_PER_RANK,
)
from sglang.srt.model_executor.cuda_graph_config import Backend


def validate_shared_ep_server_args(server_args) -> None:
    if server_args.dp_size != 8:
        raise ValueError(f"SharedEP release requires DP8, got DP{server_args.dp_size}.")
    if not server_args.enable_dp_attention:
        raise ValueError("SharedEP release requires --enable-dp-attention.")
    max_running_requests = RELEASE_MAX_TOKENS_PER_RANK * server_args.dp_size
    if server_args.max_running_requests is None:
        server_args.override(
            "validate_shared_ep_server_args",
            max_running_requests=max_running_requests,
        )
    elif server_args.max_running_requests > max_running_requests:
        raise ValueError(
            "SharedEP supports --max-running-requests "
            f"{max_running_requests} or lower, got "
            f"{server_args.max_running_requests}."
        )
    if server_args.enable_lora or server_args.lora_paths:
        raise ValueError("SharedEP release does not support LoRA.")

    prefill_chunk = server_args.chunked_prefill_size
    if (
        prefill_chunk is None
        or prefill_chunk <= 0
        or prefill_chunk > RELEASE_MAX_PREFILL_TOKENS_PER_RANK
    ):
        raise ValueError(
            "SharedEP requires the DP-adjusted --chunked-prefill-size to be "
            f"in [1, {RELEASE_MAX_PREFILL_TOKENS_PER_RANK}], got "
            f"{prefill_chunk}."
        )

    if server_args.moe_runner_backend not in ("auto", "triton"):
        raise ValueError("SharedEP requires --moe-runner-backend auto or triton.")

    for enabled, option in (
        (server_args.enable_two_batch_overlap, "--enable-two-batch-overlap"),
        (server_args.enable_single_batch_overlap, "--enable-single-batch-overlap"),
    ):
        if enabled:
            raise ValueError(
                f"SharedEP does not support {option}; its phase-local epoch "
                "states are not staged for overlapping batches."
            )

    if server_args.speculative_algorithm is not None:
        raise ValueError(
            "SharedEP initial release does not support speculative decoding."
        )

    if server_args.cuda_graph_config.prefill.backend != Backend.DISABLED:
        raise ValueError(
            "SharedEP prefill CUDA Graph is not supported; disable prefill "
            "CUDA Graph so the pull-cache consumer is selected at runtime."
        )

    decode = server_args.cuda_graph_config.decode
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
