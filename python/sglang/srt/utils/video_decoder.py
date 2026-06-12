"""Unified video decoder: torchcodec preferred, decord as fallback."""

import logging
import os

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


def _decord_batch_to_numpy(batch) -> np.ndarray:
    """Coerce a decord ``get_batch`` return to ``np.ndarray`` (decord 3.3+ returns Tensor)."""
    if hasattr(batch, "asnumpy"):
        return batch.asnumpy()
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dep
        return np.asarray(batch)
    if isinstance(batch, torch.Tensor):
        return batch.cpu().numpy()
    return np.asarray(batch)


class VideoDecoderWrapper:
    """Unified video decoder that uses torchcodec when available, decord as fallback.

    All frames are returned in NHWC uint8 numpy format for consistency.
    """

    def __init__(self, source, device: str = "cpu", num_decode_threads: int = 0):
        """source: file path (str) or video bytes.
        device: "cpu" or "cuda". GPU decoding only supported with torchcodec.
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
        if _BACKEND == "torchcodec":
            kwargs = {"dimension_order": "NHWC"}
            if device == "cuda" and _try_cuda_backend():
                kwargs["device"] = "cuda"
            self._tc_kwargs = kwargs
            try:
                self._decoder = VideoDecoder(source, **kwargs)
            except RuntimeError:
                if "device" in kwargs:
                    logger.warning("CUDA video decoding failed, falling back to CPU.")
                    kwargs.pop("device")
                    self._tc_kwargs = kwargs
                    self._decoder = VideoDecoder(source, **kwargs)
                else:
                    raise
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
            return _decord_batch_to_numpy(self._decoder.get_batch(indices))

    def get_frames_as_tensor(self, indices: list):
        """Return frames at given indices as a torch tensor (NHWC, uint8, pinned memory)."""
        import torch

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
            batch = self._decoder.get_frames_at(indices)
            return batch.data.pin_memory()
        else:
            arr = _decord_batch_to_numpy(self._decoder.get_batch(indices))
            return torch.from_numpy(arr).pin_memory()

    def _parallel_decode(self, indices, num_threads):
        """Decode frames using multiple VideoDecoder instances in parallel threads."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import torch

        chunks = [list(c) for c in np.array_split(indices, num_threads) if len(c) > 0]
        source = self._source
        kwargs = self._tc_kwargs

        def _decode_chunk(chunk):
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

        return torch.cat(results, dim=0).pin_memory()

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
        """Explicitly clean up temporary files."""
        if self._tmp_path is not None:
            if os.path.exists(self._tmp_path):
                os.unlink(self._tmp_path)
            self._tmp_path = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
