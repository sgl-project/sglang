"""Intel XPU hardware video decode policy for :class:`VideoDecoderWrapper`.

torchcodec loads the Intel plugin (PyPI ``torchcodec-xpu``) through its
``torchcodec.backends`` entry point, so ``VideoDecoder(..., device="xpu")`` is
the whole integration. This module decides whether that device is used, how
many hardware decodes may run at once, and when to fall back to the CPU; the
wrapper keeps its existing RuntimeError -> CPU fallback.
"""

import importlib.util
import logging
import threading
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# One decoder per codec engine; Arc Pro B60/B70 expose two (xe fdinfo drm-engine-capacity-vcs).
XPU_DECODERS_PER_CARD = 2

_dense_video_seen = False
_decode_slots: Optional[threading.BoundedSemaphore] = None
_decode_slots_lock = threading.Lock()
_fallback_warned = False


@lru_cache(maxsize=1)
def xpu_video_decode_device() -> Optional[str]:
    # Bare "xpu" is the current device, where the HF fast image processor already runs.
    from sglang.srt.environ import envs
    from sglang.srt.utils.common import is_xpu

    if (
        _dense_video_seen
        or not is_xpu()
        or not envs.SGLANG_ENABLE_XPU_VIDEO_HW_DECODE.get()
    ):
        return None
    if importlib.util.find_spec("torchcodec_xpu") is None:
        logger.info(
            "torchcodec-xpu is not installed; video inputs decode on the CPU. "
            "Install torchcodec-xpu to use the XPU media engine."
        )
        return None
    return "xpu"


def xpu_decoders_per_video() -> int:
    return min(XPU_DECODERS_PER_CARD, _max_concurrent_decodes())


def xpu_decode_slot() -> threading.BoundedSemaphore:
    # WORKAROUND(torchcodec-xpu): intel/torchlib-xpu#161; see SGLANG_XPU_VIDEO_MAX_CONCURRENT_DECODES.
    global _decode_slots
    with _decode_slots_lock:
        if _decode_slots is None:
            _decode_slots = threading.BoundedSemaphore(_max_concurrent_decodes())
    return _decode_slots


def note_video_frames(num_frames: int) -> bool:
    # WORKAROUND(torchcodec-xpu): intel/torchlib-xpu#162. Process-wide and sticky,
    # since a dense video stalls the hardware decodes of every video in the process.
    global _dense_video_seen
    from sglang.srt.environ import envs

    if _dense_video_seen:
        return True
    limit = envs.SGLANG_XPU_VIDEO_HW_MAX_FRAMES.get()
    if num_frames <= limit:
        return False
    _dense_video_seen = True
    xpu_video_decode_device.cache_clear()
    # Nothing reuses the cached frame blocks after the switch, and xe evicts
    # instead of failing allocations, so torch would never trim them.
    import torch

    torch.xpu.empty_cache()
    logger.info(
        "A video samples %d frames (> SGLANG_XPU_VIDEO_HW_MAX_FRAMES=%d); video "
        "inputs decode on the CPU from now on in this process.",
        num_frames,
        limit,
    )
    return True


def warn_once_on_cpu_fallback(decoder) -> None:
    # The plugin falls back silently by default (CPU_FALLBACK=on).
    global _fallback_warned
    if _fallback_warned:
        return
    status = decoder.cpu_fallback
    if status:
        _fallback_warned = True
        logger.warning(
            "torchcodec-xpu fell back to CPU decoding (%s); video inputs are "
            "not using the XPU media engine.",
            status,
        )


def _max_concurrent_decodes() -> int:
    from sglang.srt.environ import envs

    return max(1, envs.SGLANG_XPU_VIDEO_MAX_CONCURRENT_DECODES.get())
