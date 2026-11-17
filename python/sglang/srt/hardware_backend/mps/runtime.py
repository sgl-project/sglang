"""Runtime validation for SGLang's Apple MPS backend."""

import torch
from packaging.version import Version


def validate_mps_runtime() -> None:
    """Validate the Torch runtime required by the standard MPS model path."""
    if Version(torch.__version__) < Version("2.13.0"):
        raise RuntimeError(
            "The standard SGLang MPS model path requires Torch >= 2.13.0; "
            f"found Torch {torch.__version__}; reinstall with the "
            "srt_mps extra"
        )

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "The SGLang MPS backend requires an available PyTorch MPS device"
        )
    for memory_api in ("recommended_max_memory", "driver_allocated_memory"):
        if not callable(getattr(torch.mps, memory_api, None)):
            raise RuntimeError(
                f"The SGLang MPS backend requires torch.mps.{memory_api} from "
                "Torch >= 2.13.0"
            )
