"""The explicitly supported, static packed-DSA DCP storage contract."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedDSADCPLayout:
    page_size: int = 64
    bytes_per_token: int = 656


SM120_DSA_LAYOUT = PackedDSADCPLayout()


def uses_sm120_dsa_dcp(kernel_config, dcp_size: int) -> bool:
    return dcp_size > 1 and (
        kernel_config.dsa_prefill_backend == "flashinfer_sparse_mla"
        and kernel_config.dsa_decode_backend == "flashinfer_sparse_mla"
    )


def validate_sm120_dsa_dcp(cfg, hf_config, sm_major: int) -> None:
    """Gate after backend and cache-dtype resolution, before allocating pools."""
    selected = {cfg.dsa_prefill_backend, cfg.dsa_decode_backend}
    if cfg.dcp_size <= 1 or "flashinfer_sparse_mla" not in selected:
        return
    from sglang.srt.configs.model_config import get_dsa_index_kpool, get_dsa_index_topk

    unsupported = []
    if selected != {"flashinfer_sparse_mla"}:
        unsupported.append("mixed attention backends")
    if sm_major != 12 or hf_config.architectures != ["GlmMoeDsaForCausalLM"]:
        unsupported.append("model/platform other than GLM DSA on SM120/SM121")
    if cfg.kv_cache_dtype != "fp8_e4m3" or cfg.page_size != 64:
        unsupported.append("cache other than FP8 with physical page size 64")
    if cfg.dcp_size not in (2, 4, 8):
        unsupported.append("DCP size outside 2/4/8")
    if cfg.speculative_algorithm is not None:
        unsupported.append("speculative decoding (chain MTP is a separate integration)")
    for flag in (
        "enable_hisparse",
        "enable_hierarchical_cache",
        "enable_lmcache",
        "enable_prefill_cp",
        "enable_unified_memory",
        "enable_two_batch_overlap",
        "enable_mixed_chunk",
        "enable_dp_attention",
        "enable_page_major_kv_layout",
        "enable_unified_cache_external_linker",
    ):
        if getattr(cfg, flag):
            unsupported.append(flag)
    if cfg.pp_size != 1:
        unsupported.append(
            "pipeline parallelism (PP validation is a separate integration)"
        )
    if cfg.disaggregation_mode != "null":
        unsupported.append("PD disaggregation")
    if cfg.dcp_comm_backend != "ag_rs" or cfg.dcp_replicate_q_proj:
        unsupported.append("DCP communication other than ordinary ag_rs")
    if get_dsa_index_kpool(hf_config) != 1 or get_dsa_index_topk(hf_config) != 2048:
        unsupported.append("pooled/compressed Index-K or top-k other than 2048")
    if (hf_config.kv_lora_rank, hf_config.qk_rope_head_dim) != (512, 64):
        unsupported.append("latent/RoPE dimensions other than 512/64")
    if unsupported:
        raise ValueError(
            "SM120 packed DSA DCP does not support: " + ", ".join(unsupported)
        )


def localize_sparse_indices(indices, translator, dcp_size: int, dcp_rank: int):
    """Global replicated Index-K ids -> compact owner-local latent-KV ids.

    Translation is performed exactly once, through main's read-address door.
    Negative padding never reaches the translator.
    """
    valid = (indices >= 0) & (indices % dcp_size == dcp_rank)
    lengths = valid.sum(-1, dtype=torch.int32)
    order = (~valid).to(torch.int32).argsort(dim=-1, stable=True)
    widened = indices.gather(-1, order)
    local = translator.translate_dcp_read_ids(widened.clamp_min(0))
    active = torch.arange(indices.shape[-1], device=indices.device) < lengths[:, None]
    return torch.where(active, local, -1).to(torch.int32), lengths
