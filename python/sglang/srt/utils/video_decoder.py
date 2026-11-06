"""Unified video decoder: torchcodec preferred, decord as fallback."""

import logging
import os
from contextlib import nullcontext

import numpy as np

logger = logging.getLogger(__name__)

try:
    from torchcodec.decoders import VideoDecoder

    _BACKEND = "torchcodec"
except (ImportError, RuntimeError):
    _BACKEND = "decord"


_cuda_backend_enabled: bool | None = None


def _try_cuda_backend() -> bool:
    """Try to enable torchcodec CUDA backend. Caches result after first call."""
    global _cuda_backend_enabled
    if _cuda_backend_enabled is not None:
        return _cuda_backend_enabled
    try:
        from torchcodec.decoders import set_cuda_backend

        set_cuda_backend("beta")
        _cuda_backend_enabled = True
    except Exception:
        _cuda_backend_enabled = False
    return _cuda_backend_enabled


def _try_xpu_backend() -> bool:
    # Lazy import: keeps torch/XPU out of the module import path.
    from sglang.srt.utils.video_decoder_xpu import xpu_video_decode_device

    return xpu_video_decode_device() is not None


def _is_cpu_engine() -> bool:
    # Lazy import to avoid circular dependency issues and unnecessary imports on module load
    from sglang.srt.utils.common import is_cpu

    return is_cpu()


def _as_output(data):
    return data if data.device.type != "cpu" else data.numpy()


class VideoDecoderWrapper:
    """Unified video decoder that uses torchcodec when available, decord as fallback.

    All frames are returned in NHWC uint8 format: numpy on the CPU, a torch
    tensor on an accelerator (CUDA or XPU).
    """

    def __init__(self, source, device: str = "cpu", num_decode_threads: int = 0):
        """source: file path (str) or video bytes.
        device: "cpu", "cuda" or "xpu[:N]". GPU decoding only supported with
            torchcodec; XPU additionally needs the torchcodec-xpu plugin.
        num_decode_threads: number of parallel decoder instances for frame
            extraction (torchcodec only). 0 = auto (capped at 16),
            1 = single decoder. Set > 1 to split frame indices across
            multiple decoders in parallel threads.
        """
        self._source = source
        self._num_decode_threads = num_decode_threads
        self._source_bytes = source if isinstance(source, bytes) else None
        self._source_path = source if isinstance(source, str) else None
        self._tmp_path = None
        self._tc_kwargs = {}
        if _BACKEND == "torchcodec":
            kwargs = {"dimension_order": "NHWC"}
            if device == "cuda" and _try_cuda_backend():
                kwargs["device"] = "cuda"
            elif device.startswith("xpu") and _try_xpu_backend():
                kwargs["device"] = device
            self._tc_kwargs = kwargs
            try:
                with self._xpu_slot():
                    self._decoder = VideoDecoder(source, **kwargs)
            except RuntimeError as error:
                if "device" in kwargs:
                    logger.warning(
                        "%s video decoding failed, falling back to CPU: %s",
                        kwargs["device"].upper(),
                        error,
                    )
                    kwargs.pop("device")
                    self._tc_kwargs = kwargs
                    self._decoder = VideoDecoder(source, **kwargs)
                else:
                    raise
            if (
                kwargs.get("device", "cpu").startswith("xpu")
                and num_decode_threads == 0
            ):
                # Auto on XPU means one decoder per codec engine, not one per
                # CPU core, bounded by the process-wide concurrency limit.
                from sglang.srt.utils.video_decoder_xpu import xpu_decoders_per_video

                self._num_decode_threads = xpu_decoders_per_video()
        else:
            from decord import VideoReader, cpu

            if isinstance(source, bytes):
                import tempfile

                fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
                try:
                    os.write(fd, source)
                finally:
                    os.close(fd)
                self._tmp_path = tmp_path
                self._decoder = VideoReader(tmp_path, ctx=cpu(0))
            else:
                self._decoder = VideoReader(source, ctx=cpu(0))

    def __len__(self):
        return len(self._decoder)

    def __getitem__(self, idx):
        """Return one NHWC uint8 frame (numpy on CPU, tensor on an accelerator)."""
        if _BACKEND == "torchcodec":
            with self._xpu_slot():
                frame = self._decoder[idx]
            data = frame.data if hasattr(frame, "data") else frame
            return _as_output(data)
        else:
            frame = self._decoder[idx]
            return frame.asnumpy() if hasattr(frame, "asnumpy") else np.array(frame)

    @property
    def avg_fps(self) -> float:
        if _BACKEND == "torchcodec":
            return self._decoder.metadata.average_fps
        else:
            return self._decoder.get_avg_fps()

    @property
    def frame_shape(self) -> tuple[int, int]:
        if _BACKEND == "torchcodec":
            metadata = self._decoder.metadata
            height = getattr(metadata, "height", None)
            width = getattr(metadata, "width", None)
            if height and width:
                return int(height), int(width)
        shape = self[0].shape
        return int(shape[-3]), int(shape[-2])

    def get_frames_at(self, indices: list):
        """Return NHWC uint8 frames (numpy on CPU, tensor on an accelerator)."""
        if _BACKEND == "torchcodec":
            self._decode_dense_video_on_cpu(num_frames=len(indices))
            with self._xpu_slot():
                batch = self._decoder.get_frames_at(indices)
            self._check_xpu_fallback()
            return _as_output(batch.data)
        else:
            return self._decoder.get_batch(indices).asnumpy()

    def get_frames_as_tensor(self, indices: list):
        """Return frames at given indices as a torch tensor (NHWC, uint8, pinned memory)."""
        import torch

        if _BACKEND == "torchcodec":
            self._decode_dense_video_on_cpu(num_frames=len(indices))
        if (
            _BACKEND == "torchcodec"
            and self._num_decode_threads != 1
            and len(indices) > 1
        ):
            num_threads = self._num_decode_threads
            if num_threads <= 0:
                num_threads = min(os.cpu_count() or 8, 16)
            num_threads = min(num_threads, len(indices))
            if num_threads > 1:
                return self._parallel_decode(indices, num_threads)

        if _BACKEND == "torchcodec":
            with self._xpu_slot():
                batch = self._decoder.get_frames_at(indices)
            self._check_xpu_fallback()
            if _is_cpu_engine() or batch.data.device.type != "cpu":
                return batch.data
            return batch.data.pin_memory()
        else:
            arr = self._decoder.get_batch(indices).asnumpy()
            output = torch.from_numpy(arr)
            return output if _is_cpu_engine() else output.pin_memory()

    def _on_xpu(self) -> bool:
        return self._tc_kwargs.get("device", "cpu").startswith("xpu")

    def _decode_dense_video_on_cpu(self, *, num_frames: int) -> None:
        # WORKAROUND(torchcodec-xpu): intel/torchlib-xpu#162; see note_video_frames.
        if not self._on_xpu():
            return
        from sglang.srt.utils.video_decoder_xpu import note_video_frames

        if not note_video_frames(num_frames):
            return
        self._tc_kwargs = {k: v for k, v in self._tc_kwargs.items() if k != "device"}
        self._decoder = VideoDecoder(self._source, **self._tc_kwargs)
        self._num_decode_threads = 0  # CPU auto: up to 16 parallel decoders

    def _xpu_slot(self):
        """Process-wide concurrency guard for XPU hardware decodes; no-op elsewhere."""
        if not self._on_xpu():
            return nullcontext()
        from sglang.srt.utils.video_decoder_xpu import xpu_decode_slot

        return xpu_decode_slot()

    def _check_xpu_fallback(self):
        if self._on_xpu():
            from sglang.srt.utils.video_decoder_xpu import warn_once_on_cpu_fallback

            warn_once_on_cpu_fallback(self._decoder)

    def _parallel_decode(self, indices, num_threads):
        """Decode frames using multiple VideoDecoder instances in parallel threads."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import torch

        from sglang.srt.platforms import current_platform

        chunks = [list(c) for c in np.array_split(indices, num_threads) if len(c) > 0]
        source = self._source
        kwargs = self._tc_kwargs
        device = None
        if "device" in kwargs:
            device = current_platform.get_device(
                device_id=torch.get_device_module(kwargs["device"]).current_device()
            )

        def _decode_chunk(chunk):
            # The accelerator's current device is thread-local.  Without this,
            # decoder workers created by TP rank > 0 silently default to GPU 0.
            if device is not None:
                current_platform.set_device(device)
            with self._xpu_slot():
                d = VideoDecoder(source, **kwargs)
                return d.get_frames_at(chunk).data

        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            future_to_idx = {
                executor.submit(_decode_chunk, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            results = [None] * len(chunks)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        output = torch.cat(results, dim=0)
        if _is_cpu_engine() or output.device.type != "cpu":
            return output
        return output.pin_memory()

    @property
    def source_bytes(self) -> bytes | None:
        """Return raw video bytes if available (needed for audio extraction)."""
        if self._source_bytes is not None:
            return self._source_bytes
        path = self._tmp_path or self._source_path
        if path is not None:
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    return f.read()
        return None

    def close(self):
        self._decoder = None
        self._source = None
        self._source_bytes = None
        self._source_path = None
        if getattr(self, "_tmp_path", None) is not None:
            if os.path.exists(self._tmp_path):
                os.unlink(self._tmp_path)
            self._tmp_path = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
