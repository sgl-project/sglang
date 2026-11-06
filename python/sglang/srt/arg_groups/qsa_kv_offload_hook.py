from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sglang.srt.arg_groups.model_override_base import model_config_of, resolving_view

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


def validate_qsa_kv_offload(server_args: ServerArgs) -> None:
    """Validate --enable-qsa-kv-offload constraints before the model loads."""
    cfg = resolving_view(server_args)
    if cfg.enable_qsa_kv_offload:
        check_qsa_kv_offload_args(cfg, model_config_of(server_args).hf_config)


def check_qsa_kv_offload_args(cfg: Any, hf_config: Any) -> None:
    """The offloaded pool keeps only device rows on the GPU, so every reader of
    the full-attention K/V that addresses logical slots directly is refused."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.qsa.config import parse_qsa_profile
    from sglang.srt.mem_cache.qsa_kv_offload import compute_offload_layout
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.utils import is_hip

    profile = parse_qsa_profile(hf_config)
    if profile is None:
        raise ValueError(
            "--enable-qsa-kv-offload needs a model with the compressed QSA "
            "indexer (Qwen4-Exp)."
        )
    if cfg.qsa_kv_offload_host_size <= 0:
        raise ValueError(
            "--enable-qsa-kv-offload needs --qsa-kv-offload-host-size (GB)."
        )
    if cfg.max_running_requests is None:
        raise ValueError(
            "--enable-qsa-kv-offload sizes the GPU cache per request slot; set "
            "--max-running-requests."
        )
    if (cfg.chunked_prefill_size or 0) <= 0:
        raise ValueError(
            "--enable-qsa-kv-offload stages each extend in a scratch sized by "
            "--chunked-prefill-size; enable chunked prefill."
        )
    speculative = (cfg.speculative_algorithm or "NONE").upper() != "NONE"
    unsupported = {
        "--enable-hierarchical-cache": cfg.enable_hierarchical_cache,
        "--enable-hisparse": cfg.enable_hisparse,
        "--enable-lmcache": cfg.enable_lmcache,
        "--enable-mixed-chunk": cfg.enable_mixed_chunk,
        "--enable-dp-attention": cfg.enable_dp_attention,
        "--enable-prefill-cp": cfg.enable_prefill_cp,
        "--attn-cp-size > 1": (cfg.attn_cp_size or 1) > 1,
        "--dcp-size > 1": (cfg.dcp_size or 1) > 1,
        "--disaggregation-mode": cfg.disaggregation_mode != "null",
        "--pp-size > 1": (cfg.pp_size or 1) > 1,
        "--speculative-algorithm NGRAM": speculative
        and cfg.speculative_algorithm.upper() == "NGRAM",
        "--speculative-eagle-topk > 1": speculative
        and (cfg.speculative_eagle_topk or 1) != 1,
    }
    for flag, enabled in unsupported.items():
        if enabled:
            raise ValueError(f"--enable-qsa-kv-offload does not support {flag}.")

    graphs = cfg.cuda_graph_config
    # Same condition as QSAKVHostOffload.use_fused: ROCm runs the torch resolve.
    torch_resolve = envs.SGLANG_USE_QSA_KV_OFFLOAD_TORCH_RESOLVE.get() or is_hip()
    if torch_resolve and graphs.decode.backend != Backend.DISABLED:
        raise ValueError(
            "The QSA K/V offload resolves eagerly on ROCm and with "
            "SGLANG_USE_QSA_KV_OFFLOAD_TORCH_RESOLVE=1; add --disable-cuda-graph."
        )
    if graphs.prefill.backend != Backend.DISABLED:
        raise ValueError(
            "--enable-qsa-kv-offload gathers the prefill context from the host "
            "outside CUDA graphs; add --disable-prefill-cuda-graph."
        )
    # The same layout checks the pool runs, before the weights load.
    compute_offload_layout(
        compress_ratio=profile.compress_ratio,
        token_topk=profile.budget,
        hot_blocks=cfg.qsa_kv_offload_hot_blocks,
        num_request_slots=cfg.max_running_requests + 1,
        rows_per_request=(cfg.speculative_num_draft_tokens or 1) if speculative else 1,
        max_extend_tokens=max(cfg.chunked_prefill_size, cfg.max_prefill_tokens or 0),
        page_size=cfg.page_size,
    )
