from __future__ import annotations

import importlib
import logging
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

_FLASHINFER_KERNEL_MODULE = (
    "flashinfer.quantization.kernels.cutile.mxfp8_grouped_quantize_cutile"
)
_FLASHINFER_KERNEL_NAME = "mxfp8_grouped_quantize_cutile"
# Import the lightweight cuTile availability check before the kernel module. The
# kernel imports `cuda.tile` at module load, so probing it directly can fail in
# environments where FlashInfer is installed without cuTile support.
_FLASHINFER_CUTILE_MODULE = "flashinfer.cutile"
_FLASHINFER_AVAILABILITY_NAME = "is_cuda_tile_available"

logger = logging.getLogger(__name__)


class Mxfp8GroupedQuantBackend(Enum):
    """Enum for MXFP8 grouped quantization backend selection."""

    AUTO = "auto"
    FLASHINFER = "flashinfer"
    NATIVE = "native"

    def is_auto(self) -> bool:
        return self == Mxfp8GroupedQuantBackend.AUTO

    def is_flashinfer(self) -> bool:
        return self == Mxfp8GroupedQuantBackend.FLASHINFER

    def is_native(self) -> bool:
        return self == Mxfp8GroupedQuantBackend.NATIVE


MXFP8_GROUPED_QUANT_BACKEND: Mxfp8GroupedQuantBackend | None = None


def initialize_mxfp8_grouped_quant_config(server_args: ServerArgs) -> None:
    """Initialize MXFP8 grouped quantization configuration from server args."""
    global MXFP8_GROUPED_QUANT_BACKEND
    MXFP8_GROUPED_QUANT_BACKEND = Mxfp8GroupedQuantBackend(
        server_args.mxfp8_grouped_quant_backend
    )


def get_mxfp8_grouped_quant_backend() -> Mxfp8GroupedQuantBackend:
    """Get the current MXFP8 grouped quantization backend."""
    global MXFP8_GROUPED_QUANT_BACKEND
    if MXFP8_GROUPED_QUANT_BACKEND is None:
        MXFP8_GROUPED_QUANT_BACKEND = Mxfp8GroupedQuantBackend.AUTO
    return MXFP8_GROUPED_QUANT_BACKEND


def _missing_flashinfer_error(detail: str) -> RuntimeError:
    return RuntimeError(
        "FlashInfer MXFP8 grouped quantization was requested by "
        "`--mxfp8-grouped-quant-backend=flashinfer`, but the FlashInfer cuTile "
        f"backend is unavailable: {detail}. Install a FlashInfer build that "
        f"provides `{_FLASHINFER_KERNEL_MODULE}.{_FLASHINFER_KERNEL_NAME}` and "
        f"gates on `{_FLASHINFER_CUTILE_MODULE}.{_FLASHINFER_AVAILABILITY_NAME}` "
        "(the `cuda.tile` dependency), or use "
        "`--mxfp8-grouped-quant-backend=native`."
    )


def _is_cuda_tile_available() -> bool:
    try:
        module = importlib.import_module(_FLASHINFER_CUTILE_MODULE)
    except Exception:
        return False

    checker = getattr(module, _FLASHINFER_AVAILABILITY_NAME, None)
    if checker is None:
        return False

    try:
        return bool(checker())
    except Exception:
        return False


def is_flashinfer_mxfp8_grouped_quant_available() -> bool:
    if not _is_cuda_tile_available():
        return False

    try:
        module = importlib.import_module(_FLASHINFER_KERNEL_MODULE)
    except Exception:
        return False

    return getattr(module, _FLASHINFER_KERNEL_NAME, None) is not None


@lru_cache(maxsize=1)
def _load_flashinfer_mxfp8_grouped_quant() -> Callable[..., None]:
    if not _is_cuda_tile_available():
        raise _missing_flashinfer_error(
            f"`{_FLASHINFER_CUTILE_MODULE}.{_FLASHINFER_AVAILABILITY_NAME}` "
            "reports `cuda.tile` is not importable"
        )

    try:
        module = importlib.import_module(_FLASHINFER_KERNEL_MODULE)
    except Exception as exc:
        raise _missing_flashinfer_error(
            f"failed to import `{_FLASHINFER_KERNEL_MODULE}`"
        ) from exc

    kernel = getattr(module, _FLASHINFER_KERNEL_NAME, None)
    if kernel is None:
        raise _missing_flashinfer_error(f"`{_FLASHINFER_KERNEL_NAME}` is missing")

    return kernel


def _flashinfer_mxfp8_grouped_quant(
    input: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    quant_output: torch.Tensor,
    scale_factor: torch.Tensor,
) -> None:
    kernel = _load_flashinfer_mxfp8_grouped_quant()
    kernel(
        input,
        problem_sizes,
        expert_offsets,
        blockscale_offsets,
        quant_output,
        scale_factor,
    )


def _native_mxfp8_grouped_quant(
    input: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    quant_output: torch.Tensor,
    scale_factor: torch.Tensor,
) -> None:
    from sgl_kernel import es_sm100_mxfp8_blockscaled_grouped_quant

    es_sm100_mxfp8_blockscaled_grouped_quant(
        input,
        problem_sizes,
        expert_offsets,
        blockscale_offsets,
        quant_output,
        scale_factor,
    )


@lru_cache(maxsize=1)
def _resolve_grouped_quant_impl(
    backend: Mxfp8GroupedQuantBackend,
) -> Callable[..., None]:
    """Resolve the configured backend to a concrete implementation.

    Cached so resolution (and any FlashInfer or cuda.tile import) happens once on
    the first call rather than at config init, and only for workloads that
    actually use MXFP8 grouped quantization. `auto` falls back to the native
    kernel when FlashInfer is unavailable or unusable; an explicit `flashinfer`
    request raises a clear error instead.
    """
    if backend.is_flashinfer():
        _load_flashinfer_mxfp8_grouped_quant()
        return _flashinfer_mxfp8_grouped_quant

    if backend.is_auto() and is_flashinfer_mxfp8_grouped_quant_available():
        return _flashinfer_mxfp8_grouped_quant

    return _native_mxfp8_grouped_quant


def mxfp8_grouped_quant(
    input: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    quant_output: torch.Tensor,
    scale_factor: torch.Tensor,
) -> None:
    impl = _resolve_grouped_quant_impl(get_mxfp8_grouped_quant_backend())
    impl(
        input,
        problem_sizes,
        expert_offsets,
        blockscale_offsets,
        quant_output,
        scale_factor,
    )
