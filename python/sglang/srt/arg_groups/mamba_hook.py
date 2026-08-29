# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the Mamba / linear-attention backends."""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.arg_groups.overrides import (
    resolving_view,
)
from sglang.srt.utils.common import (
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_musa,
    is_npu,
    is_sm100_supported,
    is_xpu,
)

logger = logging.getLogger(__name__)


def handle_mamba_backend(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.mamba_cache_philox_rounds < 0:
        raise ValueError("--mamba-cache-philox-rounds must be non-negative.")

    if cfg.mamba_max_states_per_path == 0 or cfg.mamba_max_states_per_path < -1:
        raise ValueError(
            "--mamba-max-states-per-path must be -1 (unlimited) or a positive "
            f"integer, got {cfg.mamba_max_states_per_path}."
        )

    if cfg.enable_mamba_cache_stochastic_rounding:
        if cfg.mamba_ssm_dtype != "float16":
            raise ValueError(
                "Stochastic rounding for the Mamba SSM cache requires "
                f"--mamba-ssm-dtype float16, got {cfg.mamba_ssm_dtype!r}. "
                "Run with --mamba-ssm-dtype float16 or disable "
                "--enable-mamba-cache-stochastic-rounding."
            )
        if not is_cuda():
            raise ValueError(
                "Stochastic rounding for the Mamba SSM cache is only "
                "supported on NVIDIA CUDA platforms. Disable "
                "--enable-mamba-cache-stochastic-rounding on this platform."
            )
        if cfg.mamba_backend == "triton" and not is_sm100_supported():
            raise ValueError(
                "Stochastic rounding for the Mamba SSM cache with "
                "--mamba-backend triton requires SM100 with CUDA >= 12.8 "
                "because it uses the cvt.rs.f16x2.f32 PTX instruction. On "
                "H100/SM90, run with --mamba-backend flashinfer "
                "--mamba-ssm-dtype float16, or disable "
                "--enable-mamba-cache-stochastic-rounding."
            )

    if cfg.mamba_backend == "flashinfer":
        flashinfer_error = (
            "FlashInfer mamba module not available, please check the "
            "FlashInfer installation."
        )
        if cfg.enable_mamba_cache_stochastic_rounding:
            flashinfer_error += (
                " Stochastic rounding with --mamba-backend flashinfer "
                "requires FlashInfer Mamba and --mamba-ssm-dtype float16."
            )
        if is_flashinfer_available():
            try:
                import flashinfer.mamba  # noqa: F401

                logger.info("Successfully imported FlashInfer mamba module")
            except (ImportError, AttributeError):
                raise ValueError(flashinfer_error)
        else:
            raise ValueError(flashinfer_error)


def handle_int8_mamba_checkpoint(server_args: Any):
    # The int8 mamba checkpoint pool is only wired into the built-in
    # MambaRadixCache. The host-offload path (enabled by
    # --enable-hierarchical-cache) and custom radix-cache backends are NOT
    # int8-aware: they would read int8 checkpoint slots as bf16 active slots
    # (wrong pool / out-of-range). Reject the combination up front rather than
    # silently corrupting state.
    cfg = resolving_view(server_args)
    if not cfg.enable_int8_mamba_checkpoint:
        return
    if cfg.enable_hierarchical_cache:
        raise ValueError(
            "--enable-int8-mamba-checkpoint is not supported together with "
            "--enable-hierarchical-cache: the host-offload path "
            "is not int8-aware. Disable one of them."
        )
    if cfg.radix_cache_backend is not None:
        raise ValueError(
            "--enable-int8-mamba-checkpoint only supports the built-in mamba "
            f"radix cache; --radix-cache-backend={cfg.radix_cache_backend!r} "
            "is not int8-aware. Omit --radix-cache-backend."
        )


def validate_mamba_extra_buffer(view, model_arch: str, *, mamba_cache_chunk_size_of):
    from sglang.srt.arg_groups.overrides import (
        requires_short_conv_track_limits,
        supports_mamba_cache_extra_buffer,
    )

    assert supports_mamba_cache_extra_buffer(
        view, model_arch
    ), f"extra_buffer is not supported for {model_arch}; use no_buffer."
    if requires_short_conv_track_limits(model_arch):
        # The extend snapshot's data-dependent row count is the symptom; the
        # cause is that no prefill graph backend this model can reach fixes the
        # request axis. Only Full pads it (PhaseConfig.full_prefill_max_req);
        # Breakable / TcPiecewise bake the capture batch's request count, and
        # PrefillCudaGraphRunner.capture_prepare builds ONE synthetic request,
        # so a bs=3 extend replays a graph shaped for bs=1. Full is out of reach
        # anyway: ShortConvAttnBackend.init_forward_metadata_out_graph returns
        # decode-shaped metadata (MambaAttnBackendBase._replay_metadata).
        assert view.cuda_graph_backend_prefill in (None, "disabled"), (
            f"extra_buffer for {model_arch} is not supported together with a "
            "prefill CUDA graph: the captured extend path has no fixed request "
            "axis, so the track snapshot (and every other per-request tensor) "
            "freezes at the capture batch's one request. Use "
            "--mamba-radix-cache-strategy no_buffer, or "
            "--cuda-graph-backend-prefill disabled."
        )
        # The snapshot has to land on the accepted step, and the decode graph
        # runner drops its mamba-track buffers outright when a spec algorithm is
        # set, so it would silently never fire.
        assert view.speculative_algorithm is None, (
            f"extra_buffer for {model_arch} does not support speculative "
            "decoding; use --mamba-radix-cache-strategy no_buffer."
        )
    assert (
        is_cuda() or is_musa() or is_npu() or is_hip() or is_xpu()
    ), "extra_buffer needs CUDA/MUSA/NPU/ROCm/XPU (FLA)."
    if view.mamba_radix_cache_strategy == "extra_buffer_lazy":
        # The PD-disagg decode pool is not wired for lazy slots.
        assert view.disaggregation_mode == "null", (
            "extra_buffer_lazy unsupported under PD disaggregation; use "
            "--mamba-radix-cache-strategy extra_buffer."
        )
        # eagle/ngram/dspark/dflash all verify through
        # prepare_mamba_track_for_verify (lazy plan wired); dflash gained
        # the hook in DFlashVerifyInput.prepare_for_verify.
    if view.speculative_num_draft_tokens is not None:
        assert view.mamba_track_interval >= view.speculative_num_draft_tokens
    if view.page_size is not None:
        assert view.mamba_track_interval % view.page_size == 0
        # Called here and not passed in: `mamba_cache_chunk_size` derives from
        # `page_size`, which resolution writes after this validator runs, so
        # evaluating it at the call site raises on the unresolved `None`.
        mamba_cache_chunk_size = mamba_cache_chunk_size_of()
        assert mamba_cache_chunk_size is not None

        if (
            view.chunked_prefill_size is not None
            and 0 < view.chunked_prefill_size < mamba_cache_chunk_size
        ):
            logger.warning(
                "Mamba radix extra-buffer is enabled with chunked_prefill_size=%s "
                "smaller than mamba_cache_chunk_size=%s. This can make "
                "mamba_track_mask false for unfinished chunked-prefill handoff "
                "and skip Mamba state checkpoints.",
                view.chunked_prefill_size,
                mamba_cache_chunk_size,
            )


def validate_mamba_no_buffer(view, model_arch: str):
    assert view.page_size in (1, None), "no_buffer only supports page_size=1."
    assert (
        view.disable_overlap_schedule
    ), "no_buffer do not support overlap schedule. Try to set disable_overlap_schedule=True."
    assert (
        view.attention_backend != "trtllm_mha"
    ), "no_buffer do not support trtllm_mha attention backend."
