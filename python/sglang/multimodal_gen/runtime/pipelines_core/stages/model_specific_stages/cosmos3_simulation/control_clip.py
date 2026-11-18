# SPDX-License-Identifier: Apache-2.0
"""Control clip of a Cosmos-Dreams-Transfer request, decoded in one pass.

A thread decodes the clip straight into one host buffer, so the frames are
copied once and a reader can take the early ones while the later ones decode.
"""

import threading
from contextlib import closing

import torch

from sglang.multimodal_gen.runtime.utils.vision import (
    iter_video_frames,
    rgb_hwc_uint8,
)

# Most buffer the decode reserves before it knows how long the clip is, and a
# longer clip doubles it. Arbitrary: host pages are committed only when the
# decode first writes them.
INITIAL_BUFFER_BYTES = 1 << 30


class ControlClipDecode:
    """A control clip decoding on a thread into one ``uint8 [T, H, W, 3]`` buffer.

    Frames become final in order, and ``frames`` may read any frame that ``wait``
    has reported, also after ``close``.
    """

    def __init__(self, path: str, *, max_frames: int) -> None:
        if max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}.")
        self.path = path
        self.max_frames = max_frames
        self._changed = threading.Condition()
        self._buffer: torch.Tensor | None = None
        self._decoded = 0
        self._ended = False
        self._closed = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._decode, name="sglang-control-decode", daemon=True
        )
        self._thread.start()

    def wait(self, num_frames: int | None = None) -> int:
        """Frames decoded once ``num_frames`` are final or the clip has ended
        (None: once it has ended)."""
        with self._changed:
            self._changed.wait_for(
                lambda: (
                    self._ended
                    or (num_frames is not None and self._decoded >= num_frames)
                )
            )
            if self._error is not None:
                raise self._error
            return self._decoded

    @property
    def source_hw(self) -> tuple[int, int]:
        if self.wait(1) == 0:
            raise ValueError(f"No frames decoded from control video {self.path!r}.")
        frame = self.frames(0, 1)
        return int(frame.shape[2]), int(frame.shape[3])

    def frames(self, start: int, stop: int) -> torch.Tensor:
        with self._changed:
            return self._buffer[start:stop].permute(3, 0, 1, 2)

    def close(self) -> None:
        with self._changed:
            self._closed = True
        self._thread.join()

    def _decode(self) -> None:
        try:
            with closing(
                iter_video_frames(self.path, max_frames=self.max_frames)
            ) as decoded_frames:
                for index, frame in enumerate(decoded_frames):
                    pixels = rgb_hwc_uint8(frame)
                    self._buffer_with_room(index, pixels.shape).numpy()[index] = pixels
                    with self._changed:
                        self._decoded = index + 1
                        self._changed.notify_all()
                        if self._closed:
                            break
        except BaseException as error:
            self._error = error
        finally:
            with self._changed:
                self._ended = True
                self._changed.notify_all()

    def _buffer_with_room(
        self, index: int, frame_shape: tuple[int, ...]
    ) -> torch.Tensor:
        # Read unlocked: this runs on the decode thread, the only writer of ``_buffer``.
        buffer = self._buffer
        if buffer is not None and index < buffer.shape[0]:
            return buffer
        if buffer is None:
            frame_bytes = frame_shape[0] * frame_shape[1] * frame_shape[2]
            capacity = max(1, INITIAL_BUFFER_BYTES // frame_bytes)
        else:
            capacity = 2 * buffer.shape[0]
        grown = torch.empty(
            (min(capacity, self.max_frames), *frame_shape), dtype=torch.uint8
        )
        if buffer is not None:
            grown[:index].copy_(buffer[:index])
        with self._changed:
            self._buffer = grown
        return grown
