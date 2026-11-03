from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.arg_groups.model_override_base import model_config_of, resolving_view

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


def validate_qsa_kv_offload(server_args: ServerArgs) -> None:
    """Validate --enable-qsa-kv-offload constraints.

    The offloaded pool keeps only device rows on the GPU, so every reader of the
    full-attention K/V that addresses logical slots directly is refused.
    """
    cfg = resolving_view(server_args)
    if not cfg.enable_qsa_kv_offload:
        return

    from sglang.srt.layers.attention.qsa.config import parse_qsa_profile

    hf_config = model_config_of(server_args).hf_config
    if parse_qsa_profile(hf_config) is None:
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
    unsupported = {
        "--enable-hierarchical-cache": cfg.enable_hierarchical_cache,
        "--enable-hisparse": cfg.enable_hisparse,
        "--enable-mixed-chunk": cfg.enable_mixed_chunk,
        "--enable-dp-attention": cfg.enable_dp_attention,
        "--enable-prefill-cp": cfg.enable_prefill_cp,
        "--attn-cp-size > 1": (cfg.attn_cp_size or 1) > 1,
        "--dcp-size > 1": (cfg.dcp_size or 1) > 1,
        "--disaggregation-mode": cfg.disaggregation_mode != "null",
        "--pp-size > 1": (cfg.pp_size or 1) > 1,
        "--speculative-algorithm NGRAM": (cfg.speculative_algorithm or "").upper()
        == "NGRAM",
    }
    for flag, enabled in unsupported.items():
        if enabled:
            raise ValueError(f"--enable-qsa-kv-offload does not support {flag}.")

    from sglang.srt.model_executor.cuda_graph_config import Backend

    graphs = cfg.cuda_graph_config
    eager = (graphs.decode.backend, graphs.prefill.backend) == (
        Backend.DISABLED,
        Backend.DISABLED,
    )
    if not eager:
        raise ValueError(
            "--enable-qsa-kv-offload resolves the hot cache eagerly; add "
            "--disable-cuda-graph."
        )
    if (
        cfg.speculative_algorithm is not None
        and cfg.speculative_algorithm.upper() != "NONE"
        and (cfg.speculative_eagle_topk or 1) != 1
    ):
        raise ValueError(
            "--enable-qsa-kv-offload supports chain speculation only; use "
            "--speculative-eagle-topk 1."
        )
