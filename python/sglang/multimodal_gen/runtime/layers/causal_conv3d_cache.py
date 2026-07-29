# Copyright 2026 The SGLang team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Temporal causal cache for Wan / QwenImage style VAE 3D convolutions.

These VAEs encode and decode a video one temporal chunk at a time, so every
convolution that is causal in time has to carry the tail of the previous chunk
across calls. The cache lives in a :class:`CausalConvCache` installed for the
duration of one encode/decode via :func:`causal_cache_scope`; convolutions look
their entry up by their own module path, so the layout does not depend on the
order in which they run.

The scope also hands each convolution a direct reference to the cache, because
``ContextVar.get`` is not traceable and reading it per convolution breaks the
compiled graph. That reference is scoped: it is restored on exit, including when
the forward pass raises, and a nested scope gives the outer one its own back.
The cache itself is never module state, so the modules stay reusable.
"""

import contextvars
from contextlib import contextmanager
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sglang.multimodal_gen.runtime.platforms import current_platform


class CausalCacheMode(Enum):
    """How a causal conv treats the first chunk of a forward pass.

    ``STREAMING`` is the chunked path: the cache carries state across chunks.
    ``STATELESS`` is the whole-tensor / tiled path: the time dimension is
    zero-padded on every call and nothing is retained.
    ``FIRST_FRAME`` is the whole-tensor path for a single leading frame, where
    the temporal resample convolutions are skipped entirely.
    """

    STREAMING = "streaming"
    STATELESS = "stateless"
    FIRST_FRAME = "first_frame"


class CausalConvCache:
    """Temporal cache for one encode/decode pass, keyed by module path.

    The keys come from :func:`assign_causal_cache_keys`, so the layout does not
    depend on the order in which convolutions happen to run.

    A key maps to one of three states:

    - absent: the conv has not run yet in this pass
    - ``None``: the conv ran once and asked to restart from zeros next time
      (the temporal resample convs use this for their pass-through first chunk)
    - tensor: the retained tail of the previous chunk
    """

    def __init__(
        self,
        mode: CausalCacheMode = CausalCacheMode.STREAMING,
        *,
        retain: bool = True,
    ) -> None:
        self.mode = mode
        # False when the caller knows no later chunk will read the cache, so the
        # convolutions can skip retaining a tail they would only throw away.
        self.retain = retain
        self.chunk_index = 0
        self._store: dict[str, torch.Tensor | None] = {}

    def contains(self, key: str) -> bool:
        return key in self._store

    def get(self, key: str) -> torch.Tensor | None:
        return self._store.get(key)

    def set(self, key: str, value: torch.Tensor | None) -> None:
        self._store[key] = value

    def advance_chunk(self) -> None:
        # Moving past the first chunk with retain=False means the caller lied:
        # the second chunk would read an empty cache, treat itself as the first
        # one, and silently drop the temporal stride.
        assert (
            self.retain or self.chunk_index == 0
        ), "retain=False promised there would be only one chunk"
        self.chunk_index += 1

    def is_first_chunk(self) -> bool:
        return self.chunk_index == 0

    def clear(self) -> None:
        self._store.clear()
        self.chunk_index = 0


_current_cache: contextvars.ContextVar[CausalConvCache | None] = contextvars.ContextVar(
    "causal_conv_cache", default=None
)


def current_causal_cache() -> CausalConvCache | None:
    return _current_cache.get()


@contextmanager
def causal_cache_scope(cache: CausalConvCache | None, consumers=()):
    """Install ``cache`` for the enclosed forward pass.

    ``consumers`` are the modules that will read it. They get a direct reference
    as well as the contextvar: ``ContextVar.get`` is not traceable, so reading it
    per convolution breaks the compiled graph once per call, while an attribute
    read is free. The contextvar remains the source of truth for anything not in
    the list.
    """
    token = _current_cache.set(cache)
    previous = [consumer._active_cache for consumer in consumers]
    for consumer in consumers:
        consumer._active_cache = cache
    try:
        yield cache
    finally:
        # Restore rather than clear, so a nested scope hands the outer one back
        # its own cache.
        for consumer, prior in zip(consumers, previous, strict=True):
            consumer._active_cache = prior
        _current_cache.reset(token)


def should_trim_first_chunk() -> bool:
    """Whether a temporal upsampler must drop the frames the first chunk over-produces.

    Only meaningful while streaming: the whole-tensor paths feed the full clip at
    once and there is nothing to trim.
    """
    cache = current_causal_cache()
    return (
        cache is not None
        and cache.mode is CausalCacheMode.STREAMING
        and cache.is_first_chunk()
    )


def assign_causal_cache_keys(root: nn.Module) -> None:
    """Give every :class:`CausalConv3d` under ``root`` a stable cache key.

    Call once at the end of the VAE's ``__init__``. Using the path from the VAE
    root keeps encoder and decoder keys distinct even if they share a cache.
    """
    for name, module in root.named_modules():
        if isinstance(module, CausalConv3d):
            module.cache_key = name


def _channels_last_3d_supported_by_platform() -> bool:
    """Whether this build and device pair can use channels_last_3d at all.

    Resolved once and cached on each conv at construction: it cannot change
    during the process, and querying the platform from inside a traced region
    breaks the graph (the platform helpers are not traceable Python).
    """
    return hasattr(torch, "channels_last_3d") and (
        current_platform.is_cuda() or current_platform.is_rocm()
    )


def conv3d_weight_is_channels_last_3d(weight: torch.Tensor) -> bool:
    return (
        weight.dim() == 5
        and _channels_last_3d_supported_by_platform()
        and weight.is_contiguous(memory_format=torch.channels_last_3d)
    )


def match_conv3d_input_format(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if x.dim() == 5 and conv3d_weight_is_channels_last_3d(weight):
        return x.contiguous(memory_format=torch.channels_last_3d)
    return x


def is_channels_last_3d(x: torch.Tensor) -> bool:
    """Whether ``x``'s bytes are already ordered NDHWC.

    A pure tensor query, so it stays inside the compiled graph: if a tensor is
    in that layout the platform evidently supports it, and no platform lookup
    is needed.

    Note this is weaker than what ATen's ``suggest_memory_format`` accepts. A
    tensor whose bytes are NDHWC but whose strides are degenerate — which is
    what merging a size-1 dimension produces — passes here and fails there. See
    :func:`flatten_time_into_batch`.
    """
    return (
        x.dim() == 5
        and x.is_contiguous(memory_format=torch.channels_last_3d)
        and not x.is_contiguous()
    )


def flatten_time_into_batch(x: torch.Tensor) -> torch.Tensor:
    """``(b, c, t, h, w) -> (b * t, c, h, w)``, as a view that keeps the layout.

    ``permute(0, 2, 1, 3, 4).reshape(...)`` is already a free view of a
    channels_last_3d tensor, but when ``b * t == 1`` the merged batch dimension
    comes out with stride ``c`` instead of ``h * w * c``. ``is_contiguous``
    accepts that; ``suggest_memory_format`` does not — and interpolate, conv,
    pad and cat all pick their output layout from the latter. So everything
    downstream silently falls back to NCHW and the next causal convolution
    converts the whole activation back.

    Flattening along the NDHWC axis order yields the same view with canonical
    strides.
    """
    b, c, t, h, w = x.shape
    if is_channels_last_3d(x):
        return x.permute(0, 2, 3, 4, 1).reshape(b * t, h, w, c).permute(0, 3, 1, 2)
    return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)


def unflatten_batch_into_time(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Inverse of :func:`flatten_time_into_batch`; same stride fix, here for ``t == 1``."""
    bt, c, h, w = x.shape
    if x.is_contiguous(memory_format=torch.channels_last) and not x.is_contiguous():
        return (
            x.permute(0, 2, 3, 1)
            .reshape(batch_size, bt // batch_size, h, w, c)
            .permute(0, 4, 1, 2, 3)
        )
    return x.view(batch_size, bt // batch_size, c, h, w).permute(0, 2, 1, 3, 4)


def memory_format_of(x: torch.Tensor) -> torch.memory_format:
    """The layout ``x`` is already in, so derived tensors can keep it.

    Concatenating a contiguous tensor onto a channels_last_3d one yields a
    contiguous result, which then has to be converted back before every
    convolution — and the conversion propagates, since a contiguous input
    produces a contiguous output.
    """
    if (
        x.dim() == 5
        and _channels_last_3d_supported_by_platform()
        and x.is_contiguous(memory_format=torch.channels_last_3d)
        and not x.is_contiguous()
    ):
        return torch.channels_last_3d
    return torch.contiguous_format


def causal_cache_frames(
    *,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
) -> int:
    """How many past frames a causal conv must retain.

    A pointwise-in-time conv retains nothing. A symmetrically padded conv moves
    all of its temporal padding to the left, so it retains ``2 * padding[0]``. A
    strided temporal conv retains whatever the stride does not consume.
    """
    if kernel_size[0] == 1:
        return 0
    if padding[0] > 0:
        return 2 * padding[0]
    return kernel_size[0] - stride[0]


def interleave_time(x: torch.Tensor) -> torch.Tensor:
    """``(b, 2c, t, h, w) -> (b, c, 2t, h, w)``, interleaving the halves in time.

    Under channels_last_3d the bytes are ordered ``[b][t][h][w][r][c]`` while the
    result needs ``[b][t][r][h][w][c]``, which no view can express: ``rearrange``
    copies anyway, and hands back a contiguous tensor that the upsample, the 2D
    conv and the next causal conv all then follow into NCDHW. Writing the same
    bytes into a channels_last_3d buffer is the copy it was already paying, and
    it happens before the 2x temporal upsample rather than after it.

    The contiguous path keeps ``rearrange``: there it returns a free view when
    t == 1, which an explicit buffer would turn into a real copy.
    """
    if not is_channels_last_3d(x):
        return rearrange(x, "b (r c) t h w -> b c (t r) h w", r=2)

    b, c2, t, h, w = x.shape
    c = c2 // 2
    out = torch.empty(
        (b, c, 2 * t, h, w),
        dtype=x.dtype,
        device=x.device,
        memory_format=torch.channels_last_3d,
    )
    # r is the fast channel index: the first half of the channels feeds the even
    # output frames.
    out[:, :, 0::2] = x[:, :c]
    out[:, :, 1::2] = x[:, c:]
    return out


class CausalConv3d(nn.Conv3d):
    """Conv3d that is causal in time, with its temporal cache held externally.

    Everything on the instance is fixed at construction; the per-pass state
    lives in the :class:`CausalConvCache` of the enclosing scope.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.padding: tuple[int, int, int]
        self.cache_key: str = ""
        self.cache_frames = causal_cache_frames(
            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )
        # Frames of zeros prepended in time when not streaming. Same as
        # `cache_frames` for symmetrically padded convs; the strided temporal
        # convs override it.
        self.stateless_pad_frames = 2 * self.padding[0]
        self.height_padding = self.padding[1]
        self.width_padding = self.padding[2]
        # Only the time dimension needs explicit padding: it is causal, so the
        # padding is asymmetric. Height and width are symmetric, which is what
        # cuDNN's implicit padding does natively.
        self.padding = (0, 0, 0)
        # Platform capabilities are fixed for the process. Reading them here
        # rather than per call keeps them out of any traced region: the platform
        # helpers are not traceable Python and would break the graph on every
        # convolution.
        # Set by causal_cache_scope; read instead of the contextvar so the
        # compiled graph does not break once per convolution.
        self._active_cache: CausalConvCache | None = None
        self.channels_last_supported = _channels_last_3d_supported_by_platform()
        self.needs_dtype_cast = not current_platform.is_amp_supported()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cache_frames == 0:
            return self._conv(x, time_pad=0)
        cache = self._active_cache
        if cache is None or cache.mode is not CausalCacheMode.STREAMING:
            return self._conv(x, time_pad=self.stateless_pad_frames)
        return self._conv(self._consume_cache(x, cache), time_pad=0)

    def _consume_cache(self, x: torch.Tensor, cache: CausalConvCache) -> torch.Tensor:
        memory_format = self._memory_format_of(x)
        prev = cache.get(self.cache_key)
        if prev is None:
            # Same result as concatenating a zero-filled cache, but in one kernel
            # and without materialising the zeros in the wrong layout first.
            x = F.pad(x, (0, 0, 0, 0, self.cache_frames, 0))
        else:
            x = torch.cat([prev, x], dim=2)
        if cache.retain:
            cache.set(self.cache_key, self._retain_tail(x, memory_format))
        return x

    def _memory_format_of(self, x: torch.Tensor) -> torch.memory_format:
        """Layout of ``x``, without querying the platform inside the trace."""
        if (
            self.channels_last_supported
            and x.dim() == 5
            and x.is_contiguous(memory_format=torch.channels_last_3d)
            and not x.is_contiguous()
        ):
            return torch.channels_last_3d
        return torch.contiguous_format

    def _retain_tail(
        self, x: torch.Tensor, memory_format: torch.memory_format
    ) -> torch.Tensor:
        """Copy out the frames the next chunk needs, in the layout it wants.

        ``clone`` rather than ``contiguous``: the latter hands back a view when
        the slice already has the requested layout, which would keep the whole
        activation alive for as long as the cache does.
        """
        return x[:, :, -self.cache_frames :, :, :].clone(memory_format=memory_format)

    def spatial_padding(self) -> tuple[int, int, int]:
        """Padding handed to the convolution itself, as ``(time, height, width)``."""
        return (0, self.height_padding, self.width_padding)

    def _conv(self, x: torch.Tensor, *, time_pad: int) -> torch.Tensor:
        if time_pad:
            x = F.pad(x, (0, 0, 0, 0, time_pad, 0))
        if self.needs_dtype_cast:
            x = x.to(self.weight.dtype)
        return self._conv_impl(x)

    def _conv_impl(self, x: torch.Tensor) -> torch.Tensor:
        """The convolution itself; ROCm patches this rather than ``forward``.

        The layout match belongs here, not in ``_conv``: a backend that replaces
        this method wants the input in whatever layout suits it. ROCm's unfolded
        Conv2D, for one, immediately gathers into a standard-contiguous buffer,
        so converting to channels_last_3d first is pure loss.
        """
        if (
            self.channels_last_supported
            and x.dim() == 5
            and self.weight.is_contiguous(memory_format=torch.channels_last_3d)
        ):
            x = x.contiguous(memory_format=torch.channels_last_3d)
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.spatial_padding(),
            self.dilation,
            self.groups,
        )


class TimeDownsampleCausalConv3d(CausalConv3d):
    """Temporal downsampling conv of a ``downsample3d`` resample block.

    While streaming, the first chunk is passed through untouched and only its
    tail is retained; the temporal stride starts biting from the second chunk.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # Whole-tensor path: two leading zero frames keep the output at
        # ceil(T / 2), matching the AvgDown3D shortcut of the residual encoder.
        self.stateless_pad_frames = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cache = self._active_cache
        if cache is None:
            return super().forward(x)
        if cache.mode is CausalCacheMode.FIRST_FRAME:
            return x
        if cache.mode is CausalCacheMode.STREAMING and not cache.contains(
            self.cache_key
        ):
            if cache.retain:
                cache.set(
                    self.cache_key, self._retain_tail(x, self._memory_format_of(x))
                )
            return x
        return super().forward(x)


class TimeUpsampleCausalConv3d(CausalConv3d):
    """Temporal upsampling conv of an ``upsample3d`` resample block.

    While streaming, the first chunk is passed through without doubling — the
    decoder's first latent frame maps to a single output frame — and the cache
    is armed so the next chunk starts from zeros.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cache = self._active_cache
        if cache is None:
            return interleave_time(super().forward(x))
        if cache.mode is CausalCacheMode.FIRST_FRAME:
            return x
        if cache.mode is CausalCacheMode.STREAMING and not cache.contains(
            self.cache_key
        ):
            cache.set(self.cache_key, None)
            return x
        return interleave_time(super().forward(x))
