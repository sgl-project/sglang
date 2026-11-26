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


def get_a3_c2_state_bank_count(num_req_slots: int) -> int:
    from sglang.srt.layers.dp_attention import get_eager_max_batch_size

    # CompressorV2 needs a bank even for TP-padded rows. Slot 0 is reserved.
    return max(num_req_slots, get_eager_max_batch_size(num_req_slots - 1))


class NPUDSV4KVLayout(str, enum.Enum):
    V4_BF16 = "v4_bf16"
    V4_FP8 = "v4_fp8"
    # A3 uses the same unquantized row for SWA and C1/C2, with BF16 index K.
    V41_BF16 = "v41_bf16"
    V41_SWA_FP8 = "v41_fp8"
    V41_CMP_FP4 = "v41_fp4"

    @property
    def is_compressed(self) -> bool:
        # A property of the pool role, not of the layer's compress ratio -- a
        # ratio-1 aggregate is still read as cmp_kv by the ds41 attention op.
        return self is NPUDSV4KVLayout.V41_CMP_FP4

    @property
    def quant_group(self) -> int:
        """Values sharing one quant scale; None = not used by this row."""
        return {
            NPUDSV4KVLayout.V4_BF16: None,
            NPUDSV4KVLayout.V41_BF16: None,
            NPUDSV4KVLayout.V4_FP8: None,
            NPUDSV4KVLayout.V41_SWA_FP8: 32,
            NPUDSV4KVLayout.V41_CMP_FP4: 16,
        }[self]

    @property
    def quant_mode(self):
        """``kv_compress_epilog_v2`` encoding; None = not written by that op.

        The V4 rows go through ``kv_compress_epilog``, whose encoding comes from
        the runtime quant path and changes with the CANN release.
        """
        return {
            NPUDSV4KVLayout.V4_BF16: None,
            NPUDSV4KVLayout.V41_BF16: None,
            NPUDSV4KVLayout.V4_FP8: None,
            NPUDSV4KVLayout.V41_SWA_FP8: "mxfp8_bf16",
            NPUDSV4KVLayout.V41_CMP_FP4: "mxfp4_bf16",
        }[self]

    @property
    def quant(self) -> tuple:
        """(quant_mode, quant_group_size) for kv_compress_epilog_v2."""
        return self.quant_mode, self.quant_group

    @property
    def row_alignment(self) -> int:
        """Row-width alignment; None = not used by this row."""
        return {
            NPUDSV4KVLayout.V4_BF16: None,
            NPUDSV4KVLayout.V41_BF16: None,
            NPUDSV4KVLayout.V4_FP8: None,
            NPUDSV4KVLayout.V41_SWA_FP8: 32,
            NPUDSV4KVLayout.V41_CMP_FP4: 32,
        }[self]

    def kv_info(self, nope_dim: int, rope_dim: int) -> tuple:
        """(kv_dim, kv_dtype) of the single-KV PA_ND buffer: last-dim size in
        store-dtype elements plus the store dtype.

        The arch35 V4 encoding belongs to the runtime quant path (608B on CANN
        9.2, 640B on 9.1), so its width is not derived here. V4.1 rows are fully
        quantized by kv_compress_epilog_v2 (no bf16 tail): dataCol = head_dim
        (mxfp8) or head_dim // 2 (packed mxfp4, two e2m1 per byte), scales are
        G * bf16.
        """
        if self in (NPUDSV4KVLayout.V4_BF16, NPUDSV4KVLayout.V41_BF16):
            return nope_dim + rope_dim, torch.bfloat16
        if self is NPUDSV4KVLayout.V4_FP8:
            from sgl_kernel_npu.operators.mem_cache import get_npu_kv_quant_path

            return (
                get_npu_kv_quant_path().packed_dim(nope_dim, rope_dim),
                torch.float8_e4m3fn,
            )
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
        if self is NPUDSV4KVLayout.V41_BF16:
            return head_dim, 0, torch.bfloat16, torch.float32
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
