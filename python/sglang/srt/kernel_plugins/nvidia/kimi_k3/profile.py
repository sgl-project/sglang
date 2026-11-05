# SPDX-License-Identifier: Apache-2.0
"""Explicit experimental native-MLA profile for the full K3 decode executor."""

import os


def selected_profile() -> str:
    selected = {
        value.strip()
        for value in os.getenv("SGLANG_NVIDIA_MEGAKERNEL_SLOTS", "").split(",")
    }
    profiles = selected & {"k3_native_mla", "k3_full_decode"}
    if len(profiles) > 1:
        raise ValueError("select one K3 decode profile: native MLA or full decode")
    return next(iter(profiles), "community")


def native_mla_enabled() -> bool:
    return selected_profile() == "k3_native_mla"


def full_decode_enabled() -> bool:
    return selected_profile() == "k3_full_decode"


def native_config_reasons(cfg) -> list[str]:
    """Admission for the new profile, independent of measured GPU qualification."""
    reasons = []
    context = int(cfg.context_length or 0)
    if not 1 <= context <= 1_048_576:
        reasons.append("extended decode profile requires context in [1, 1048576]")
    if (cfg.dcp_size or 1) not in (1, 8):
        reasons.append("extended decode profile supports DCP1 or DCP8 with TP8")
    if (getattr(cfg, "nnodes", 1) or 1) != 1:
        reasons.append("extended decode profile currently requires one node")
    if (cfg.attn_cp_size or 1) != 1 or cfg.enable_prefill_cp:
        reasons.append(
            "extended decode profile does not support prefill context parallelism"
        )
    # Replicated Q is a stock-path optimization, not a change to the original
    # sharded parameters. The extended providers validate and project those
    # twelve-head shards, then gather Q through their explicit DCP path; they
    # never consume or disable the stock path's full-head replica buffers.
    # KV_OFFLOAD is accepted for the HiCache path whose restore barriers the
    # runtime adapter calls. Other offload implementations are not implied.
    if any(
        getattr(cfg, name, False)
        for name in (
            "enable_lmcache",
            "enable_flexkv",
            "disaggregation_decode_enable_offload_kvcache",
        )
    ):
        reasons.append("extended decode profile accepts HiCache offload only")
    return reasons
