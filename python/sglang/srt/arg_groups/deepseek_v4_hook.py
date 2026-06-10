import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def apply_deepseek_v4_defaults(server_args: "ServerArgs", model_arch: str) -> None:
    """Apply DeepSeek V4 model-specific server arg defaults and constraints."""
    from sglang.srt.environ import envs
    from sglang.srt.server_args import ServerArgs

    # V4 is a DSA architecture; mark it so `is_dsa_enable_prefill_cp()` resolves
    # correctly downstream (the V3.2 branch in server_args.__post_init__ sets
    # this flag for V3.2, but V4 takes the v4-specific hook path and would
    # otherwise stay False — leaving V4's CP code paths silently disabled).
    server_args._is_dsa_model_arch = True
    server_args.attention_backend = "dsv4"
    server_args.page_size = 256
    logger.info(
        f"Use dsv4 attention backend for {model_arch}, setting page_size to 256."
    )

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 256
        logger.warning(
            f"Setting max_running_requests to {server_args.max_running_requests} for {model_arch}."
        )

    if server_args.kv_cache_dtype == "auto":
        server_args.kv_cache_dtype = "fp8_e4m3"
        logger.warning(
            f"Setting KV cache dtype to {server_args.kv_cache_dtype} for {model_arch}."
        )
    assert server_args.kv_cache_dtype in [
        "fp8_e4m3"
    ], f"{server_args.kv_cache_dtype} is not supported for {model_arch}"

    if server_args.speculative_algorithm is not None:
        assert (
            server_args.speculative_algorithm == "EAGLE"
        ), f"Only EAGLE speculative algorithm is supported for {model_arch}"
        assert (
            server_args.speculative_eagle_topk == 1
        ), f"Only EAGLE speculative algorithm with topk == 1 is supported for {model_arch}"

        if not envs.SGLANG_ENABLE_SPEC_V2.get():
            envs.SGLANG_ENABLE_SPEC_V2.set(True)
            logger.warning("Spec v2 is enabled for EAGLE speculative decoding.")

    if server_args.swa_full_tokens_ratio == ServerArgs.swa_full_tokens_ratio:
        server_args.swa_full_tokens_ratio = 0.1
        logger.info(
            f"Setting swa_full_tokens_ratio to {server_args.swa_full_tokens_ratio} for {model_arch}."
        )


def validate_deepseek_v4_cp(server_args: "ServerArgs") -> None:
    """Validate DeepSeek V4 context-parallel configuration.

    V4 only has interleave-mode kernel paths today (FlashMLA reindex,
    bf16-KV → fp8-cache split etc.). V4 also requires the DP-attention
    flag to remain ON throughout init — `_handle_data_parallelism` clears
    `enable_dp_attention` when `dp_size == 1`, so we restore it here for
    the V4 CP path (this matches the pre-refactor behaviour).
    """
    if not server_args.enable_prefill_cp:
        return

    if server_args.cp_strategy != "interleave":
        raise ValueError(
            "DeepSeek V4 only supports --cp-strategy interleave "
            "(formerly --dsa-prefill-cp-mode round-robin-split); "
            f"got cp_strategy={server_args.cp_strategy!r}."
        )

    server_args.enable_dsa_prefill_context_parallel = True
    server_args.dsa_prefill_cp_mode = "round-robin-split"
    # V4 CP requires `is_dp_attention_enabled()` to be True downstream
    # (model body and dsv4 backend branch on it). The dp_size=1 +
    # enable_dp_attention=True layout is what V4 CP was always written
    # against; clearing the flag silently produces wrong results.
    server_args.enable_dp_attention = True
    server_args.moe_dense_tp_size = 1
    # If the user didn't pin attn_cp_size, derive from tp/dp.
    if server_args.attn_cp_size == 1:
        server_args.attn_cp_size = server_args.tp_size // max(server_args.dp_size, 1)
    assert (
        server_args.tp_size <= 8
    ), "Context parallel only supports single machine (tp_size <= 8). Cross-machine CP has precision issues."
    logger.warning(
        "Enabled DeepSeek V4 CP (interleave): "
        f"dp_size={server_args.dp_size}, enable_dp_attention={server_args.enable_dp_attention}, "
        f"moe_dense_tp_size={server_args.moe_dense_tp_size}, "
        f"attn_cp_size={server_args.attn_cp_size}, ep_size={server_args.ep_size}, "
        f"tp_size={server_args.tp_size}"
    )
