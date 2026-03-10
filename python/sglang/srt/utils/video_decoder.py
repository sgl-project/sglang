"""Unified video decoder: torchcodec preferred, decord as fallback."""

import logging

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


class VideoDecoderWrapper:
    """Unified video decoder that uses torchcodec when available, decord as fallback.

    All frames are returned in NHWC uint8 numpy format for consistency.
    """

    def __init__(self, source, device: str = "cpu"):
        """source: file path (str) or video bytes.
        device: "cpu" or "cuda". GPU decoding only supported with torchcodec.
        """
        self._tmp_path = None
        if _BACKEND == "torchcodec":
            kwargs = {"dimension_order": "NHWC"}
            if device == "cuda" and _try_cuda_backend():
                kwargs["device"] = "cuda"
            try:
                self._decoder = VideoDecoder(source, **kwargs)
            except RuntimeError:
                if "device" in kwargs:
                    logger.warning("CUDA video decoding failed, falling back to CPU.")
                    kwargs.pop("device")
                    self._decoder = VideoDecoder(source, **kwargs)
                else:
                    raise
        else:
            from decord import VideoReader, cpu

            if isinstance(source, bytes):
                import os
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
        """Return single frame as numpy NHWC uint8."""
        if _BACKEND == "torchcodec":
            return self._decoder[idx].numpy()
        else:
            frame = self._decoder[idx]
            return frame.asnumpy() if hasattr(frame, "asnumpy") else np.array(frame)

    @property
    def avg_fps(self) -> float:
        if _BACKEND == "torchcodec":
            return self._decoder.metadata.average_fps
        else:
            return self._decoder.get_avg_fps()

    def get_frames_at(self, indices: list) -> np.ndarray:
        """Return frames at given indices as numpy array with shape (N, H, W, C)."""
        if _BACKEND == "torchcodec":
            batch = self._decoder.get_frames_at(indices)
            return batch.data.numpy()
        else:
            return self._decoder.get_batch(indices).asnumpy()

    def get_frames_as_tensor(self, indices: list):
        """Return frames at given indices as a torch tensor (NHWC, uint8, pinned memory)."""
        import torch

        if _BACKEND == "torchcodec":
            batch = self._decoder.get_frames_at(indices)
            return batch.data.pin_memory()
        else:
            arr = self._decoder.get_batch(indices).asnumpy()
            return torch.from_numpy(arr).pin_memory()

    def close(self):
        """Explicitly clean up temporary files."""
        if self._tmp_path is not None:
            import os

            if os.path.exists(self._tmp_path):
                os.unlink(self._tmp_path)
            self._tmp_path = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
