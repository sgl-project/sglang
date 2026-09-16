"""Paged KV cache layout identities used by DeepSeek-V4 sparse attention.

The legacy FlashMLA layouts store ``page_size`` data rows followed by scale
rows. The versioned packed Main-KV layout has its own three-region descriptor
in ``dsv41_main_kv_layout`` and must not use the legacy geometry helpers.
"""

from __future__ import annotations

import enum
from typing import Union


class KVLayout(str, enum.Enum):
    # 448 fp8 nope + 64 bf16 rope, 7 ue8m0 scales (+1 pad) per 64 values.
    V4 = "v4"
    # 512 fp8 (rope quantized too), 16 ue8m0 scales per 32 values.
    V41 = "v41"
    # 512 e2m1 packed two per byte (even index low nibble), 32 e4m3 scales per 16 values.
    V41_FP4 = "v41_fp4"
    # DSV4.1 C1/C2 Main KV only: 448 packed e2m1 noPE values, 28 e4m3
    # scales plus 4 reserved bytes, and 64 bf16 RoPE values.
    DSV41_MAIN_KV_E2M1_BLOCK16_ROPE_BF16_V1 = "dsv41_main_kv_e2m1_block16_rope_bf16_v1"

    def _require_legacy_flashmla_layout(self) -> None:
        if self is KVLayout.DSV41_MAIN_KV_E2M1_BLOCK16_ROPE_BF16_V1:
            raise ValueError(
                f"{self.value} uses MainKVLayoutSpec; it is not a legacy "
                "FlashMLA data/scale layout"
            )

    @property
    def data_bytes(self) -> int:
        self._require_legacy_flashmla_layout()
        return {KVLayout.V4: 576, KVLayout.V41: 512, KVLayout.V41_FP4: 256}[self]

    @property
    def scale_bytes(self) -> int:
        self._require_legacy_flashmla_layout()
        return {KVLayout.V4: 8, KVLayout.V41: 16, KVLayout.V41_FP4: 32}[self]

    @property
    def tile_size(self) -> int:
        """Values sharing one scale."""
        self._require_legacy_flashmla_layout()
        return {KVLayout.V4: 64, KVLayout.V41: 32, KVLayout.V41_FP4: 16}[self]

    @property
    def bytes_per_token(self) -> int:
        return self.data_bytes + self.scale_bytes

    @property
    def page_align(self) -> int:
        """Unit the page stride is padded to: the reader's TMA row stride."""
        self._require_legacy_flashmla_layout()
        return {KVLayout.V4: 576, KVLayout.V41: 512, KVLayout.V41_FP4: 256}[self]

    @property
    def is_fp4(self) -> bool:
        return self in (
            KVLayout.V41_FP4,
            KVLayout.DSV41_MAIN_KV_E2M1_BLOCK16_ROPE_BF16_V1,
        )

    @property
    def is_packed_main_kv(self) -> bool:
        return self is KVLayout.DSV41_MAIN_KV_E2M1_BLOCK16_ROPE_BF16_V1

    def page_bytes(self, page_size: int) -> int:
        raw = page_size * self.bytes_per_token
        return -(-raw // self.page_align) * self.page_align

    def scale_offset(self, page_size: int) -> int:
        """Byte offset of the scale rows inside a page."""
        return page_size * self.data_bytes

    @property
    def cpp_name(self) -> str:
        """The C++ enumerator, for JIT template arguments. Bare, because it is
        also part of the JIT module name; the headers `using enum` it in."""
        return self.name

    @classmethod
    def parse(cls, value: Union[str, KVLayout]) -> KVLayout:
        if isinstance(value, KVLayout):
            return value
        return cls(str(value).lower())


def is_valid_kv_layout_pair(kv: KVLayout, extra_kv: KVLayout) -> bool:
    """The (main, extra) cache pairs the decode kernel accepts: identical layouts,
    or the fp4 extra cache next to a V4.1 fp8 main cache."""
    return extra_kv is kv or (kv is KVLayout.V41 and extra_kv is KVLayout.V41_FP4)
