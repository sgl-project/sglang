"""Row formats of the NPU DeepSeek-V4-family single-KV / indexer-K pools.

One member per physical row shape, one per (generation, pool role). The shared
kernels ``KVLayout`` is the CUDA/TMA kernel contract and carries no arch22/arch35 row
shapes, so the NPU pools resolve their format into this enum at construction
time: ``_window_kv_layout()`` tags the non-compressed pools and
``compressed_kv_layout()`` the compressed ones, so ``_make_kv_pool`` only ever
sees members of this enum. The ``v41_fp4`` value keeps the shared enum's
spelling (the only one an external consumer could match); the other values
are NPU-only spellings — members are consumed by identity, never by value.
"""

import enum

import torch


class NPUDSV4KVLayout(str, enum.Enum):
    V4_BF16 = "v4_bf16"
    V4_FP8 = "v4_fp8"
    V41_SWA_FP8 = "v41_fp8"
    V41_CMP_FP4 = "v41_fp4"

    @property
    def is_compressed(self) -> bool:
        # A property of the pool role, not of the layer's compress ratio -- a
        # ratio-1 aggregate is still read as cmp_kv by the ds41 attention op.
        return self is NPUDSV4KVLayout.V41_CMP_FP4

    @property
    def quant_group(self) -> int:
        """Values sharing one quant scale; None rows are written unquantized."""
        return {
            NPUDSV4KVLayout.V4_BF16: None,
            NPUDSV4KVLayout.V4_FP8: 64,
            NPUDSV4KVLayout.V41_SWA_FP8: 32,
            NPUDSV4KVLayout.V41_CMP_FP4: 16,
        }[self]

    @property
    def quant_mode(self):
        """Quant mode of the epilog write op: int for kv_compress_epilog
        (arch35 window rows), str for kv_compress_epilog_v2 (V4.1 rows)."""
        return {
            NPUDSV4KVLayout.V4_BF16: None,
            # epilog v1 mode; the deployed wheel's encoding, not the one in
            # the current ops-transformer source (which also dropped layout=1).
            NPUDSV4KVLayout.V4_FP8: 2,
            NPUDSV4KVLayout.V41_SWA_FP8: "mxfp8_bf16",
            NPUDSV4KVLayout.V41_CMP_FP4: "mxfp4_bf16",
        }[self]

    @property
    def quant(self) -> tuple:
        """(quant_mode, quant_group_size) for kv_compress_epilog_v2."""
        return self.quant_mode, self.quant_group

    @property
    def row_alignment(self) -> int:
        return {
            NPUDSV4KVLayout.V4_BF16: None,
            NPUDSV4KVLayout.V4_FP8: 128,
            NPUDSV4KVLayout.V41_SWA_FP8: 32,
            NPUDSV4KVLayout.V41_CMP_FP4: 32,
        }[self]

    def kv_info(self, nope_dim: int, rope_dim: int) -> tuple:
        """(kv_dim, kv_dtype) of the single-KV PA_ND buffer: last-dim size in
        store-dtype elements plus the store dtype.

        Arch35 packs nope as fp8 with one scale per quant_group and keeps the rope
        tail in bf16. V4.1 rows are fully quantized by kv_compress_epilog_v2
        (no bf16 tail): dataCol = head_dim (mxfp8) or head_dim // 2 (packed
        mxfp4, two e2m1 per byte), scales are G * bf16.
        """
        if self is NPUDSV4KVLayout.V4_BF16:
            return nope_dim + rope_dim, torch.bfloat16
        if self is NPUDSV4KVLayout.V4_FP8:
            raw = nope_dim + rope_dim * 2 + -(-nope_dim // self.quant_group)
            align = self.row_alignment
            return -(-raw // align) * align, torch.float8_e4m3fn
        head = nope_dim + rope_dim
        assert head > 0, head
        data = head // 2 if self is NPUDSV4KVLayout.V41_CMP_FP4 else head
        raw = data + 2 * -(-head // self.quant_group)
        align = self.row_alignment
        dtype = (
            torch.uint8 if self is NPUDSV4KVLayout.V41_CMP_FP4 else torch.float8_e4m3fn
        )
        return -(-raw // align) * align, dtype

    def indexer_kv_info(self, head_dim: int) -> tuple:
        """(k_dim, scale_dim, k_dtype, scale_dtype) of the indexer K / K-scale
        PA_ND buffers: last-dim sizes in store-dtype elements plus dtypes.

        The indexer rows follow the generation, not the pool side: ratios 1
        and 2 share the same packed-fp4 row shape. The V4.1 scale is one uint8
        e8m0 per 32 values.
        """
        if self in (NPUDSV4KVLayout.V41_SWA_FP8, NPUDSV4KVLayout.V41_CMP_FP4):
            assert head_dim % 32 == 0, head_dim
            return head_dim // 2, head_dim // 32, torch.uint8, torch.uint8
        if self is NPUDSV4KVLayout.V4_FP8:
            return head_dim, 1, torch.float8_e4m3fn, torch.float32
        return head_dim, 1, torch.int8, torch.float16


# Compress ratios whose pool takes the fp4 row: the V4.1 generation. Attached
# outside the class body — a tuple there would be unpacked into str.__new__
# during member creation.
NPUDSV4KVLayout.COMPRESSED_RATIOS = (1, 2)
