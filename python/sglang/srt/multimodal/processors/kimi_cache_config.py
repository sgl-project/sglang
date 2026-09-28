"""Compatibility checks for Kimi wide image cache identities."""

from typing import Optional, Sequence

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.environ import envs

KIMI_WIDE_PAD_ARCHITECTURES = (
    "KimiK3ForConditionalGeneration",
    "KimiK25ForConditionalGeneration",
)


def uses_kimi_wide_image_pads(
    architectures: Optional[Sequence[str]], model_impl: ModelImpl
) -> bool:
    return model_impl == ModelImpl.SGLANG and any(
        architecture in KIMI_WIDE_PAD_ARCHITECTURES
        for architecture in architectures or ()
    )


def validate_kimi_wide_pad_config(server_args, *, hicache_storage_backend=None):
    """Refuse cache consumers that store token IDs narrower than int64."""
    if envs.SGLANG_MM_SKIP_COMPUTE_HASH.get():
        return

    unsupported = []
    if server_args.kv_events_config is not None:
        unsupported.append("--kv-events-config")
    if (
        hicache_storage_backend is not None
        or server_args.hicache_storage_backend is not None
    ):
        unsupported.append("--hicache-storage-backend")
    if server_args.disaggregation_mode in ("prefill", "decode"):
        unsupported.append(f"--disaggregation-mode={server_args.disaggregation_mode}")
    if unsupported:
        raise ValueError(
            "Kimi wide image pad IDs require int64 cache keys and conflict with "
            + ", ".join(unsupported)
            + ". Disable these options when serving Kimi image inputs."
        )
