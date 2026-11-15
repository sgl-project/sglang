"""High-fidelity nvJPEG decoding for multimodal image processors."""

from __future__ import annotations

import queue
import threading
from functools import lru_cache

import torch

# A decoder retains roughly 15-25 MiB of device-side scratch space. Two
# decoders are enough to overlap JPEG decode without mirroring the much larger
# I/O thread count into HBM usage.
_DECODER_POOL_SIZE = 2
_DECODER_OPTIONS = ":num_cuda_streams=1 :fancy_upsampling=1"


class _NvJpegDecoderPool:
    def __init__(self, device_id: int):
        from nvidia import nvimgcodec

        self._nvimgcodec = nvimgcodec
        self._device_id = device_id
        self._decode_params = nvimgcodec.DecodeParams(
            sample_format=nvimgcodec.SampleFormat.P_RGB,
            apply_exif_orientation=False,
        )
        self._decoders = queue.LifoQueue(maxsize=_DECODER_POOL_SIZE)
        self._created = 0
        self._create_lock = threading.Lock()

    def _acquire(self):
        try:
            return self._decoders.get_nowait()
        except queue.Empty:
            pass

        with self._create_lock:
            if self._created < _DECODER_POOL_SIZE:
                decoder = self._nvimgcodec.Decoder(
                    device_id=self._device_id,
                    max_num_cpu_threads=1,
                    options=_DECODER_OPTIONS,
                )
                self._created += 1
                return decoder

        return self._decoders.get()

    def decode(self, image_bytes: bytes) -> torch.Tensor:
        decoder = self._acquire()
        try:
            stream = torch.cuda.current_stream(self._device_id)
            image = decoder.decode(
                image_bytes,
                params=self._decode_params,
                cuda_stream=stream.cuda_stream,
            )
            if image is None:
                raise RuntimeError("nvImageCodec could not decode the JPEG image")
            return torch.from_dlpack(image.to_dlpack(cuda_stream=stream.cuda_stream))
        finally:
            self._decoders.put(decoder)


@lru_cache(maxsize=None)
def _get_decoder_pool(device_id: int) -> _NvJpegDecoderPool:
    return _NvJpegDecoderPool(device_id)


def decode_jpeg_with_fancy_upsampling(image_bytes: bytes) -> torch.Tensor:
    """Decode a JPEG to contiguous CHW RGB uint8 on the current CUDA device.

    torchvision's CUDA JPEG decoder creates nvJPEG with its default flags,
    which use nearest-neighbor chroma upsampling. nvImageCodec exposes nvJPEG's
    interpolated ("fancy") upsampling and exports the result to PyTorch through
    DLPack without copying it.
    """
    device_id = torch.cuda.current_device()
    image = _get_decoder_pool(device_id).decode(image_bytes)
    if image.ndim != 3 or image.shape[0] != 3 or image.dtype != torch.uint8:
        raise RuntimeError(
            "nvImageCodec returned an invalid JPEG tensor: "
            f"shape={tuple(image.shape)}, dtype={image.dtype}"
        )
    return image
