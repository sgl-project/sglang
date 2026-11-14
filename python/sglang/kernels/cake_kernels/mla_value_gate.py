"""Cake MLA value projection + sigmoid gate through FlashInfer's public API."""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@lru_cache(maxsize=None)
def _device_supported(device_index: int) -> bool:
    import torch

    # SM100 and SM103: the architectures FlashInfer ships this program for.
    return (
        torch.cuda.get_device_capability(device_index) in ((10, 0), (10, 3))
        and find_spec("flashinfer.cake_mla_value_gate") is not None
    )


def supports_mla_value_gate(w_uv: torch.Tensor) -> bool:
    """The device and FlashInfer build can prepare a runner for ``w_uv``."""
    return w_uv.is_cuda and _device_supported(w_uv.device.index)


def prepare_mla_value_gate(
    w_uv: torch.Tensor, *, max_rows: int, rows_per_request: int = 1
) -> Any:
    """Bind ``w_uv`` (BF16 [H, 512, 128], strides (65536, 1, 512)) outside CUDA
    graph capture and return FlashInfer's runner; ``runner.launch(latent, gate,
    valid_lengths, out)`` is graph-capturable. The sigmoid is FP32, as in MlaValue.
    """
    from flashinfer.cake_mla_value_gate import prepare_mla_value_gate as prepare

    return prepare(
        w_uv,
        max_rows=max_rows,
        sigmoid_precision="fp32",
        rows_per_request=rows_per_request,
        enable_pdl=True,
    )
