"""Spatial / green-context stream helpers."""

from __future__ import annotations

from typing import Optional

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import FormatSignature, KernelBackend, KernelSpec

register_kernel(
    KernelSpec(
        op="spatial.get_sm_available",
        backend=KernelBackend.AOT,
        target="sgl_kernel.spatial:get_sm_available",
        format_signature=FormatSignature(
            description="number of SMs available on device"
        ),
        description="Query available SM count (sgl_kernel wheel).",
    )
)
register_kernel(
    KernelSpec(
        op="spatial.create_greenctx_stream_by_value",
        backend=KernelBackend.AOT,
        target="sgl_kernel.spatial:create_greenctx_stream_by_value",
        format_signature=FormatSignature(
            description="create two green-context streams partitioned by SM count"
        ),
        description="Green-context stream creation (sgl_kernel wheel).",
    )
)


def get_sm_available(device_id: Optional[int] = None) -> int:
    """Return the number of SMs available on ``device_id``."""
    return get_kernel("spatial.get_sm_available", KernelBackend.AOT)(device_id)


def create_greenctx_stream_by_value(
    SM_a: int, SM_b: int, device_id: Optional[int] = None
):
    """Create two green-context streams partitioned by ``SM_a`` / ``SM_b``."""
    return get_kernel("spatial.create_greenctx_stream_by_value", KernelBackend.AOT)(
        SM_a, SM_b, device_id
    )


__all__ = ["get_sm_available", "create_greenctx_stream_by_value"]
