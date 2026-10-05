"""CPU-only sizing and capability checks for compact Mamba2 verify records."""

from dataclasses import dataclass


def mamba2_spec_replay_enabled(cfg, model_type):
    """Dispatch the shared spec-replay flag to the supported Mamba2 family."""
    return cfg.enable_linear_replayssm_spec and model_type == "nemotron_h"


def validate_mamba2_spec_replay(cfg, model_type, *, is_cuda, resolved=False):
    if not cfg.enable_linear_replayssm_spec:
        return
    prefix = "--enable-linear-replayssm-spec for Mamba2 "
    if not is_cuda or model_type != "nemotron_h":
        raise ValueError(prefix + "requires a CUDA NemotronH Mamba2 target model.")
    if cfg.mamba_backend not in (("flashinfer",) if resolved else (None, "flashinfer")):
        raise ValueError(prefix + "requires --mamba-backend flashinfer.")
    if cfg.mamba_ssm_dtype != "float16":
        raise ValueError(prefix + "requires --mamba-ssm-dtype float16.")
    if (cfg.speculative_algorithm or "").upper() not in ("EAGLE", "NEXTN"):
        raise ValueError(
            prefix + "requires the EAGLE/NEXTN accepted-state commit worker."
        )
    if cfg.speculative_eagle_topk not in ((1,) if resolved else (None, 1)):
        raise ValueError(
            prefix + "requires a linear chain (--speculative-eagle-topk 1)."
        )
    width = cfg.speculative_num_draft_tokens
    if (resolved or width is not None) and width != 4:
        raise ValueError(
            prefix + "currently requires four speculative verify positions."
        )
    if cfg.disaggregation_mode != "null":
        raise ValueError(prefix + "does not yet support PD disaggregation.")
    if cfg.enable_unified_memory:
        raise ValueError(prefix + "does not yet support unified memory pools.")
    if getattr(cfg, "enable_page_major_kv_layout", False):
        raise ValueError(prefix + "requires contiguous layer-major Mamba pools.")
    if cfg.enable_linear_replayssm:
        raise ValueError(prefix + "cannot be combined with --enable-linear-replayssm.")
    if getattr(cfg, "speculative_adaptive", False):
        raise ValueError(prefix + "does not yet support adaptive verify widths.")
    if getattr(cfg, "enable_int8_mamba_checkpoint", False):
        raise ValueError(prefix + "does not yet support int8 checkpoint pools.")


@dataclass(frozen=True)
class Mamba2ReplaySizing:
    """Exact local-layer allocation costs, including physical conv scratch."""

    persistent_per_slot: int
    record_per_slot: int
    conv_per_row: int
    parameters: int

    @classmethod
    def from_params(cls, params, *, layers, width, activation_bytes):
        shape = params.shape
        heads, dim, dstate = shape.temporal
        conv_dim = shape.conv[0][0]
        groups, remainder = divmod(conv_dim - heads * dim, 2 * dstate)
        if len(shape.conv) != 1 or groups < 1 or remainder or heads % groups:
            raise ValueError("Unsupported Mamba2 replay state/conv geometry")
        persistent = (
            heads * dim * dstate * params.dtype.temporal.itemsize
            + conv_dim * (shape.conv_kernel - 1) * params.dtype.conv.itemsize
        )
        if (dim, dstate, activation_bytes, params.dtype.temporal.itemsize) != (
            64,
            128,
            2,
            2,
        ):
            raise ValueError(
                "Mamba2 replay requires head_dim64/state_dim128 and BF16 inputs/FP16 state"
            )
        # FlashInfer 0.7: max_window >= verify width and ring = window + width.
        # Eager mode uses the smallest legal ring (2 * width), with p=start=0.
        # x/B/processed dt and two int32 values are PHYSICAL-SLOT indexed.
        record = (
            2 * width * ((heads * dim + groups * dstate) * activation_bytes + heads * 4)
            + 2 * 4
        )
        conv_steps = (
            width * (shape.conv_kernel - 1)
            if shape.disable_conv_window_dedup
            else width + shape.conv_kernel - 2
        )
        return cls(
            persistent * layers,
            record * layers,
            conv_dim * conv_steps * params.dtype.conv.itemsize * layers,
            (8 + heads * 4 + 11 * 8) * layers,  # seeds, A vectors, pointer tables
        )

    def bytes_for(self, slots, request_cap, slots_per_request):
        rows = min(request_cap, slots // slots_per_request) + 1
        return (
            (slots + 1) * (self.persistent_per_slot + self.record_per_slot)
            + rows * self.conv_per_row
            + self.parameters
        )

    def solve(self, budget_bytes, request_cap, slots_per_request):
        # Exact integer solve also handles saturation at the configured cap.
        lo, hi = 0, max(0, int(budget_bytes) // self.persistent_per_slot)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.bytes_for(mid, request_cap, slots_per_request) <= budget_bytes:
                lo = mid
            else:
                hi = mid - 1
        return lo
