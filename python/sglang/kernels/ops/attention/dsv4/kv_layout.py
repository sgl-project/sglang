"""Paged fp8 / fp4 KV cache layouts of the DeepSeek-V4 family sparse MLA decode kernels.

Block-split layouts store a page as ``page_size`` data rows followed by
``page_size`` scale rows, and the reader requires the page stride to be a
multiple of its TMA row stride, which :meth:`KVLayout.page_bytes` pads to. The
``_NEW`` layouts (FlashMLA's 2026.09.30 kernels) interleave them instead: each
token is its data immediately followed by its scales, with no page padding.
Mirrors ``sgl_kernel/deepseek_v4/kv_layout.cuh``.
"""

from __future__ import annotations

import enum
import functools
from typing import Union


class KVLayout(str, enum.Enum):
    # 448 fp8 nope + 64 bf16 rope, 7 ue8m0 scales (+1 pad) per 64 values.
    V4 = "v4"
    # 512 fp8 (rope quantized too), 16 ue8m0 scales per 32 values.
    V41 = "v41"
    # 512 e2m1 packed two per byte (even index low nibble), 32 e4m3 scales per 16 values.
    V41_FP4 = "v41_fp4"
    # TRT-LLM only: 512 e4m3 values with a fixed external unit scale.
    UNIFORM_FP8 = "uniform_fp8"
    # V41 / V41_FP4 encodings, each token's scales right after its data.
    V41_NEW = "v41_new"
    V41_FP4_NEW = "v41_fp4_new"

    @functools.cached_property
    def is_interleaved(self) -> bool:
        return self in (KVLayout.V41_NEW, KVLayout.V41_FP4_NEW)

    @functools.cached_property
    def encoding(self) -> KVLayout:
        """The block-split layout storing a token with the same bytes."""
        return _INTERLEAVED_ENCODING.get(self, self)

    def with_interleaved(self, interleaved: bool) -> KVLayout:
        """This layout's encoding in the block-split or the interleaved arrangement."""
        if not interleaved:
            return self.encoding
        for new, old in _INTERLEAVED_ENCODING.items():
            if old is self.encoding:
                return new
        raise ValueError(f"{self} has no interleaved arrangement")

    @functools.cached_property
    def data_bytes(self) -> int:
        return {
            KVLayout.V4: 576,
            KVLayout.V41: 512,
            KVLayout.V41_NEW: 512,
            KVLayout.V41_FP4: 256,
            KVLayout.V41_FP4_NEW: 256,
            KVLayout.UNIFORM_FP8: 512,
        }[self]

    @functools.cached_property
    def scale_bytes(self) -> int:
        return {
            KVLayout.V4: 8,
            KVLayout.V41: 16,
            KVLayout.V41_NEW: 16,
            KVLayout.V41_FP4: 32,
            KVLayout.V41_FP4_NEW: 32,
            KVLayout.UNIFORM_FP8: 0,
        }[self]

    @functools.cached_property
    def tile_size(self) -> int:
        """Values sharing one scale."""
        return {
            KVLayout.V4: 64,
            KVLayout.V41: 32,
            KVLayout.V41_NEW: 32,
            KVLayout.V41_FP4: 16,
            KVLayout.V41_FP4_NEW: 16,
            KVLayout.UNIFORM_FP8: 512,
        }[self]

    @functools.cached_property
    def bytes_per_token(self) -> int:
        return self.data_bytes + self.scale_bytes

    @functools.cached_property
    def page_align(self) -> int:
        """Unit the page stride is padded to: the reader's TMA row stride."""
        return {
            KVLayout.V4: 576,
            KVLayout.V41: 512,
            KVLayout.V41_NEW: 528,
            KVLayout.V41_FP4: 256,
            KVLayout.V41_FP4_NEW: 288,
            KVLayout.UNIFORM_FP8: 512,
        }[self]

    @functools.cached_property
    def is_fp4(self) -> bool:
        return self.encoding is KVLayout.V41_FP4

    def page_bytes(self, page_size: int) -> int:
        raw = page_size * self.bytes_per_token
        return -(-raw // self.page_align) * self.page_align

    def row_addressing(self, page_size: int) -> tuple[int, int, int]:
        """(data row stride, scale base, scale row stride) inside a page: slot
        ``i``'s data is at ``i * data_stride``, its scales at ``scale_base + i * scale_stride``."""
        if self.is_interleaved:
            return self.bytes_per_token, self.data_bytes, self.bytes_per_token
        return self.data_bytes, page_size * self.data_bytes, self.scale_bytes

    @functools.cached_property
    def cpp_name(self) -> str:
        """The C++ enumerator, for JIT template arguments. Bare, because it is
        also part of the JIT module name; the headers `using enum` it in."""
        return self.name

    @classmethod
    def parse(cls, value: Union[str, KVLayout]) -> KVLayout:
        if isinstance(value, KVLayout):
            return value
        return cls(str(value).lower())


_INTERLEAVED_ENCODING = {
    KVLayout.V41_NEW: KVLayout.V41,
    KVLayout.V41_FP4_NEW: KVLayout.V41_FP4,
}


def is_valid_kv_layout_pair(kv: KVLayout, extra_kv: KVLayout) -> bool:
    """The (main, extra) cache pairs the decode kernel accepts: identical layouts,
    or the fp4 extra cache next to a V4.1 fp8 main cache of the same arrangement."""
    return extra_kv is kv or (
        kv.encoding is KVLayout.V41
        and extra_kv.encoding is KVLayout.V41_FP4
        and kv.is_interleaved == extra_kv.is_interleaved
    )
