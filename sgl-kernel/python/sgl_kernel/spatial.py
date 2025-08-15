import torch
from torch.cuda.streams import ExternalStream

# Load the spatial extension only when this module is imported
try:
    from . import spatial_ops  # noqa: F401  # triggers TORCH extension registration
except Exception as _e:  # pragma: no cover
    # Defer failure until functions are called, but give informative error on import issues
    _spatial_import_error = _e
else:
    _spatial_import_error = None


def create_greenctx_stream_by_value(
    SM_a: int, SM_b: int, device_id: int = None
) -> tuple[ExternalStream, ExternalStream]:
    """
    Create two streams for greenctx.
    Args:
        sm_A (int): The SM of stream A.
        sm_B (int): The weight of stream B.
        device_id (int): The device id.
    Returns:
        tuple[ExternalStream, ExternalStream]: The two streams.
    """
    if _spatial_import_error is not None:
        raise ImportError(
            "Failed to load sgl_kernel.spatial_ops extension. "
            "Ensure CUDA Toolkit >= 12.4 and the project is built with spatial_ops enabled."
        ) from _spatial_import_error
    if torch.version.cuda < "12.4":
        raise RuntimeError(
            "Green Contexts feature requires CUDA Toolkit 12.4 or newer."
        )

    if device_id is None:
        device_id = torch.cuda.current_device()

    res = torch.ops.sgl_kernel.create_greenctx_stream_by_value(SM_a, SM_b, device_id)

    stream_a = ExternalStream(
        stream_ptr=res[0], device=torch.device(f"cuda:{device_id}")
    )
    stream_b = ExternalStream(
        stream_ptr=res[1], device=torch.device(f"cuda:{device_id}")
    )

    return stream_a, stream_b


def get_sm_available(device_id: int = None) -> int:
    """
    Get the SMs available on the device.
    Args:
        device_id (int): The device id.
    Returns:
        int: The SMs available.
    """
    if _spatial_import_error is not None:
        raise ImportError(
            "Failed to load sgl_kernel.spatial_ops extension. "
            "Ensure CUDA Toolkit >= 12.4 and the project is built with spatial_ops enabled."
        ) from _spatial_import_error
    if device_id is None:
        device_id = torch.cuda.current_device()

    device_props = torch.cuda.get_device_properties(device_id)

    # Get the number of Streaming Multiprocessors (SMs)
    sm_count = device_props.multi_processor_count

    return sm_count
