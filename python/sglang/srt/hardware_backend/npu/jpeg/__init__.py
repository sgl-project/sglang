"""SGLang's byte-oriented entry point for NPU JPEG decoding."""

import logging

import torch

from sglang.srt.hardware_backend.npu.jpeg_header import prepare_jpeg

logger = logging.getLogger(__name__)
# Lower end of the measured crossover for square JPEGs with four CPU threads.
SMALL_IMAGE_WARNING_PIXELS = 360 * 360


def decode_jpeg(
    data: bytes | torch.Tensor, device: str | torch.device = "npu"
) -> torch.Tensor | None:
    """Decode bytes or a one-dimensional CPU uint8 tensor to contiguous CHW RGB.

    Unsupported images and devices return None for PIL fallback. The default
    device is the current thread's NPU. Execution errors propagate.
    """
    prepared = prepare_jpeg(data)
    if prepared is None:
        return None
    data, info = prepared

    # Only qualified JPEGs need the optional CANN and Torch NPU runtimes.
    from sglang.srt.hardware_backend.npu.dvpp import Dvpp
    from sglang.srt.hardware_backend.npu.jpeg._cache import cached_decoders

    target = torch.device(device)
    if target.type != "npu":
        raise ValueError("NPU JPEG decoding requires an NPU device")
    device_index = (
        target.index if target.index is not None else torch.npu.current_device()
    )
    if not Dvpp.supports_device(device_index):
        return None
    if info.width * info.height <= SMALL_IMAGE_WARNING_PIXELS:
        logger.warning_once(
            "NPU JPEG decoding may be slower than CPU/PIL for images "
            "with up to 129,600 pixels (360x360). "
            "Actual performance depends on CPU configuration and workload."
        )
    with torch.npu.device(device_index):
        return cached_decoders.get(device_index)._decode_prepared_jpeg(data, info)
