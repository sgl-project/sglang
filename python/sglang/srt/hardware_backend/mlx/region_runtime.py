"""Runtime requirements of the optional exported MLX computation region."""

from sglang.srt.hardware_backend.mps.runtime import validate_mps_runtime


def validate_mlx_region_runtime() -> None:
    """Validate MLX interoperability without tightening eager Torch MPS."""
    import torch

    from sglang.srt.hardware_backend.mlx.runtime import _validate_runtime

    validate_mps_runtime()
    # Reuse the runtime pair supported by the existing Torch/MLX bridge.
    try:
        _validate_runtime()
    except RuntimeError as exc:
        raise RuntimeError(
            str(exc).replace("SGLANG_USE_MLX", "SGLANG_ENABLE_MLX_WHOLE_REGION")
        ) from exc
    if not callable(getattr(torch.mps, "compile_shader", None)):
        raise RuntimeError(
            "The exported MLX region requires torch.mps.compile_shader "
            "for committing K/V into Torch-owned buffers"
        )
