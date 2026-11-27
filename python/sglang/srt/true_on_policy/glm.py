"""Server-start checks for GLM's TOP forward (BF16 or block-FP8 checkpoints)."""

from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import (
    get_deepep_mode,
    is_tbo_enabled,
)
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.true_on_policy.moe_ep import validate_ep_runtime

# The trainer's BF16 index selection calls the FlashInfer DSA top-k, which reads these in its own process.
DSA_SELECTION_ENVS = {
    "SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC": True,
    "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK": "small",
}
_DSA_SERVING_ENVS = {
    "SGLANG_DISABLE_DSA_INDEXER_FUSION": True,
    "SGLANG_DSA_FUSE_TOPK": False,
    **DSA_SELECTION_ENVS,
    "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD": 0,
}


def validate_envs(expected):
    for name, value in expected.items():
        if getattr(envs, name).get() != value:
            raise NotImplementedError(f"[top] GLM requires {name}={value!r}")


def validate_index_layout(config):
    """Serving and the trainer must pick the same index keys: serving's per-layer index sharing (indexer_types,
    cli_factor or index_topk_pattern first) must equal the index_topk_freq/offset rule the trainer runs, and
    neither side pools keys. Shared by serving admission and the trainer's GLM config check."""
    from sglang.srt.configs.model_config import dsa_layer_skips_topk

    freq = getattr(config, "index_topk_freq", 1) or 1
    offset = getattr(config, "index_skip_topk_offset", 0) or 0
    for layer in range(config.num_hidden_layers):
        trainer = freq > 1 and max(layer + 1 - offset, 0) % freq != 0
        if dsa_layer_skips_topk(config, layer) != trainer:
            raise NotImplementedError(
                f"[top] GLM layer {layer} index sharing differs from the index_topk_freq={freq}/"
                f"index_skip_topk_offset={offset} rule the trainer runs (indexer_types, cli_factor "
                "or index_topk_pattern)"
            )
    if (getattr(config, "index_kpool", None) or 1) != 1:
        raise NotImplementedError(
            "[top] GLM index key pooling (index_kpool > 1) is not admitted"
        )


def validate_glm(config, quant_config):
    from sglang.srt.true_on_policy import fp8

    if config.model_type != "glm_moe_dsa":
        raise NotImplementedError("[top] GLM support requires a glm_moe_dsa checkpoint")
    fp8.validate_quant_config(quant_config)
    args = get_server_args()
    if fp8.enabled():
        fp8.validate_device()
        if args.kv_cache_dtype != "fp8_e4m3":
            raise NotImplementedError("[top] GLM FP8 requires an FP8 E4M3 KV cache")
        if config.index_n_heads % 16 or config.hidden_size % 128:
            raise NotImplementedError(
                "[top] GLM FP8 requires 16-aligned index heads and a 128-aligned hidden size"
            )
    elif args.kv_cache_dtype not in ("bf16", "bfloat16"):
        raise NotImplementedError("[top] BF16 GLM requires a BF16 KV cache")
    # DeepEP transport, DeepGEMM experts and one expert domain; the op registry admits the EP degree.
    validate_ep_runtime()
    if (
        args.dsa_prefill_backend != "flashmla_sparse"
        # The paged FP8 decode (flashmla_kv) plans its split-KV schedule over the whole batch.
        or args.dsa_decode_backend != "flashmla_sparse"
    ):
        raise NotImplementedError(
            "[top] GLM requires sparse FlashMLA on prefill and decode"
        )
    if args.dsa_topk_backend != "flashinfer" or args.page_size != 64:
        raise NotImplementedError(
            "[top] GLM requires FlashInfer top-k and 64-token cache pages"
        )
    # Short prefills would take dense MHA (the KV-length threshold) while decode stays sparse, and the fused
    # top-k + page transform leaves no step for TOP's ascending slot order.
    validate_envs(_DSA_SERVING_ENVS)
    for name, value in (
        ("index_head_dim", 128),
        ("kv_lora_rank", 512),
        ("qk_rope_head_dim", 64),
    ):
        if getattr(config, name, None) != value:
            raise NotImplementedError(f"[top] GLM requires {name}={value}")
    if getattr(config, "index_k_norm_type", "layer") != "layer":
        raise NotImplementedError("[top] GLM index keys require LayerNorm")
    if any(
        getattr(config, name, 0) for name in ("index_init_tokens", "index_local_tokens")
    ):
        raise NotImplementedError("[top] GLM has no forced index-prefix/local tokens")
    validate_index_layout(config)
    if (
        config.hidden_act != "silu"
        or config.scoring_func != "sigmoid"
        or not config.norm_topk_prob
    ):
        raise NotImplementedError(
            "[top] GLM requires SwiGLU and normalized sigmoid routing"
        )
    if config.hidden_size % 128 or config.moe_intermediate_size % 128:
        raise NotImplementedError("[top] GLM expert dimensions must be 128-aligned")
    parallel = get_parallel()
    if config.n_routed_experts % parallel.moe_ep_size:
        raise NotImplementedError(
            "[top] GLM EP must divide the checkpoint expert count"
        )
    mode = get_deepep_mode().value
    if mode not in ("auto", "normal"):
        raise NotImplementedError("[top] GLM prefill requires normal DeepEP")
    if mode == "auto" and config.hidden_size not in (
        2048,
        2560,
        4096,
        5120,
        6144,
        7168,
    ):
        raise NotImplementedError(
            "[top] GLM hidden width requires normal DeepEP, not LL/auto"
        )
    if mode == "normal" and not args.disable_cuda_graph:
        raise NotImplementedError(
            "[top] normal receipt validation is eager; select auto for decode graphs or explicitly disable capture"
        )
    if is_tbo_enabled():
        raise NotImplementedError("[top] GLM forward does not admit TBO")
    # SBO reorders the shared-expert and down-GEMM program; hisparse changes which KV the indexer sees.
    if (
        args.enable_single_batch_overlap
        or envs.SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO.get()
        or args.enable_hisparse
    ):
        raise NotImplementedError(
            "[top] GLM forward does not admit single-batch overlap or hisparse"
        )
    from sglang.srt.true_on_policy.mtp import validate_mtp

    validate_mtp(config, args, family="GLM")
    # DeepEP serving leaves shared experts unfused by default too; fused, the shared expert is an FP32-weighted combine
    # slot instead of the BF16 add after the combine that the slot transport and the trainer's shared MLP run.
    if args.enable_waterfill or args.enforce_shared_experts_fusion:
        raise NotImplementedError(
            "[top] GLM shared experts remain outside routed slots"
        )
