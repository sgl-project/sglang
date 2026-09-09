"""Built-in DeepSeek V4 packed KV-cache primitives.

This module intentionally has no dependency on the external ``kvbit`` package.
It owns the fixed DSV4 layout, a CPU reference codec, and the capability gate
used before the target worker may allocate packed SWA storage.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAS_TRITON = False

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.mem_cache import kvbit_dsv4_layout_constants as abi
from sglang.srt.mem_cache.kvbit_dsv4_codec import (
    DSV4_INT4_LAYOUT,
    DSV4KVBitLayout,
    decode_dsv4_int4_reference,
    encode_dsv4_int4_reference,
    layout_for_row_bytes,
    validate_dsv4_int4_geometry,
)
from sglang.srt.mem_cache.kvbit_dsv4_runtime import (
    DSV4KVBitRuntimeCapability,
    require_dsv4_kvbit_runtime_capability,
)
from sglang.srt.mem_cache.memory_pool import KVCache

DSV4_NATIVE_SWA_ROW_BYTES = abi.NATIVE_ROW_BYTES

# Preserve the public reference-codec and capability imports.
__all__ = [
    "DSV4_INT4_LAYOUT",
    "DSV4KVBitLayout",
    "DSV4KVBitRuntimeCapability",
    "decode_dsv4_int4_reference",
    "encode_dsv4_int4_reference",
    "require_dsv4_kvbit_runtime_capability",
]


def get_dsv4_int4_layout() -> DSV4KVBitLayout:
    return DSV4KVBitLayout(envs.SGLANG_DSV4_INT4_LAYOUT.get())


def dsv4_kvbit_enabled_for_worker(
    *, kv_cache_dtype: str | None, is_draft_worker: bool
) -> bool:
    return kv_cache_dtype == "int4" and not is_draft_worker


def dsv4_kvbit_target_persistent_savings(
    *,
    swa_ratio: float,
    num_target_layers: int,
    num_c4_layers: int,
    num_c128_layers: int,
    c4_shrink_factor: float = 1.0,
    layout: DSV4KVBitLayout = DSV4_INT4_LAYOUT,
) -> float:
    row_saving = DSV4_NATIVE_SWA_ROW_BYTES - layout.row_bytes
    return row_saving * (
        swa_ratio * num_target_layers
        + num_c4_layers / (4 * c4_shrink_factor)
        + num_c128_layers / 128
    )


def merge_attention_states_natural_log(
    left_output: torch.Tensor,
    left_lse: torch.Tensor,
    right_output: torch.Tensor,
    right_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge independently normalized attention states using natural-log LSE."""
    if left_output.shape != right_output.shape:
        raise ValueError(
            f"attention output shapes must match, got "
            f"{left_output.shape} and {right_output.shape}"
        )
    expected_lse_shape = (
        left_output.shape[0],
        left_output.shape[2],
        left_output.shape[1],
    )
    if left_lse.shape != expected_lse_shape or right_lse.shape != expected_lse_shape:
        raise ValueError(
            f"LSE shape must be {expected_lse_shape} for output "
            f"{left_output.shape}, got {left_lse.shape} and {right_lse.shape}"
        )

    left_lse = torch.where(torch.isposinf(left_lse), -torch.inf, left_lse.float())
    right_lse = torch.where(torch.isposinf(right_lse), -torch.inf, right_lse.float())
    merged_lse = torch.logaddexp(left_lse, right_lse)
    left_weight = torch.exp(left_lse - merged_lse)
    right_weight = torch.exp(right_lse - merged_lse)
    left_weight = torch.where(
        torch.isfinite(left_lse), left_weight, torch.zeros_like(left_weight)
    )
    right_weight = torch.where(
        torch.isfinite(right_lse), right_weight, torch.zeros_like(right_weight)
    )
    left_weight = left_weight.transpose(1, 2).unsqueeze(-1)
    right_weight = right_weight.transpose(1, 2).unsqueeze(-1)
    output = left_output.float() * left_weight + right_output.float() * right_weight
    return output.to(left_output.dtype), merged_lse


if _HAS_TRITON:

    @triton.jit
    def _dsv4_int4_pack_scatter_kernel(
        kv_ptr,
        loc_ptr,
        packed_ptr,
        stride_kv_row,
        stride_loc,
        stride_page,
        num_pages,
        PAGE_SIZE: tl.constexpr,
        ROW_BYTES: tl.constexpr,
        CODE_BYTES: tl.constexpr,
        HEADER_OFFSET: tl.constexpr,
        ROPE_OFFSET: tl.constexpr,
        PAYLOAD_BYTES: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        loc = tl.load(loc_ptr + row * stride_loc).to(tl.int64)
        valid_loc = (loc >= 0) & (loc < num_pages * PAGE_SIZE)
        page = loc // PAGE_SIZE
        page_offset = loc % PAGE_SIZE
        dst = packed_ptr + page * stride_page + page_offset * ROW_BYTES

        offs = tl.arange(0, 512)
        values = tl.load(kv_ptr + row * stride_kv_row + offs).to(tl.float32)
        grouped = tl.reshape(values, [16, 32])
        max_abs = tl.max(tl.abs(grouped), axis=1)
        stored_step = tl.minimum(tl.div_rn(max_abs, 7.0), 448.0).to(tl.float8e4nv)
        quant_step = stored_step.to(tl.float32)
        has_step = quant_step > 0
        safe_step = tl.where(has_step, quant_step, 1.0)
        normalized = tl.minimum(
            tl.maximum(tl.div_rn(grouped, safe_step[:, None]), -7.0), 7.0
        )
        rounded = tl.inline_asm_elementwise(
            "cvt.rni.s32.f32 $0, $1;",
            constraints="=r,f",
            args=[normalized],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        rounded = tl.where(has_step[:, None], rounded, 0)
        codes = tl.minimum(tl.maximum(rounded, -7), 7).to(tl.int8)
        paired = tl.reshape(codes.to(tl.uint8) & 0x0F, [16, 16, 2])
        low, high = tl.split(paired)
        packed_codes = tl.reshape((low | (high << 4)), [256])
        code_offsets = tl.arange(0, 256)
        tl.store(
            dst + code_offsets,
            packed_codes,
            mask=valid_loc & (code_offsets < CODE_BYTES),
        )

        group_offsets = tl.arange(0, 16)
        stored_step_bits = stored_step.to(tl.uint8, bitcast=True)
        tl.store(
            dst + HEADER_OFFSET + group_offsets,
            tl.where(group_offsets < NUM_GROUPS, stored_step_bits, 0),
            mask=valid_loc,
        )

        rope_offsets = tl.arange(0, 64)
        rope = tl.load(kv_ptr + row * stride_kv_row + 448 + rope_offsets).to(
            tl.bfloat16
        )
        tl.store(
            (dst + ROPE_OFFSET).to(tl.pointer_type(tl.bfloat16)) + rope_offsets,
            rope,
            mask=valid_loc,
        )
        if ROW_BYTES > PAYLOAD_BYTES:
            pad_offsets = tl.arange(0, 16)
            tl.store(dst + PAYLOAD_BYTES + pad_offsets, 0, mask=valid_loc)

    @triton.jit
    def _dsv4_int4_sparse_decode_kernel(
        q_ptr,
        packed_ptr,
        indices_ptr,
        lengths_ptr,
        sink_ptr,
        output_ptr,
        lse_ptr,
        stride_q_row,
        stride_q_head,
        stride_page,
        stride_indices_row,
        stride_output_row,
        stride_output_head,
        stride_lse_row,
        num_pages,
        softmax_scale,
        num_heads: tl.constexpr,
        index_width: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        ROW_BYTES: tl.constexpr,
        HEADER_OFFSET: tl.constexpr,
        ROPE_OFFSET: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_SINK: tl.constexpr,
    ):
        query_row = tl.program_id(0)
        head = tl.program_id(1)
        length = tl.minimum(
            tl.maximum(tl.load(lengths_ptr + query_row), 0), index_width
        )

        q_offsets = tl.arange(0, 512)
        q = tl.load(
            q_ptr + query_row * stride_q_row + head * stride_q_head + q_offsets
        ).to(tl.float32)
        max_score = (
            tl.load(sink_ptr + head).to(tl.float32) if HAS_SINK else -float("inf")
        )
        normalizer = 1.0 if HAS_SINK else 0.0
        accumulator = tl.zeros([512], dtype=tl.float32)

        start = 0
        while start < length:
            token_offsets = start + tl.arange(0, BLOCK_N)
            valid_token = token_offsets < length
            loc = tl.load(
                indices_ptr + query_row * stride_indices_row + token_offsets,
                mask=valid_token & (token_offsets < index_width),
                other=-1,
            ).to(tl.int64)
            valid_token = valid_token & (loc >= 0) & (loc < num_pages * PAGE_SIZE)
            page = loc // PAGE_SIZE
            page_offset = loc % PAGE_SIZE
            row_ptr = (
                packed_ptr
                + page[:, None] * stride_page
                + page_offset[:, None] * ROW_BYTES
            )

            score = tl.zeros([BLOCK_N], dtype=tl.float32)
            for group in range(NUM_GROUPS):
                byte_offsets = tl.arange(0, 16)
                packed_codes = tl.load(
                    row_ptr + group * 16 + byte_offsets[None, :],
                    mask=valid_token[:, None],
                    other=0,
                )
                unsigned_codes = tl.interleave(
                    (packed_codes & 0x0F).to(tl.float32),
                    ((packed_codes >> 4) & 0x0F).to(tl.float32),
                )
                signed_codes = tl.where(
                    unsigned_codes >= 8.0,
                    unsigned_codes - 16.0,
                    unsigned_codes,
                )
                scale_bits = tl.load(
                    row_ptr + HEADER_OFFSET + group,
                    mask=valid_token[:, None],
                    other=0,
                )
                scale = (
                    tl.reshape(scale_bits, [BLOCK_N])
                    .to(tl.float8e4nv, bitcast=True)
                    .to(tl.float32)
                )
                values = (signed_codes * scale[:, None]).to(tl.bfloat16).to(tl.float32)
                dim_offsets = group * 32 + tl.arange(0, 32)
                q_group = tl.gather(q, dim_offsets, axis=0)
                score += tl.sum(q_group[None, :] * values, axis=1)

            rope_offsets = tl.arange(0, 64)
            rope = tl.load(
                (row_ptr + ROPE_OFFSET).to(tl.pointer_type(tl.bfloat16))
                + rope_offsets[None, :],
                mask=valid_token[:, None],
                other=0.0,
            ).to(tl.float32)
            q_rope = tl.gather(q, 448 + rope_offsets, axis=0)
            score += tl.sum(q_rope[None, :] * rope, axis=1)
            score = tl.where(valid_token, score * softmax_scale, -float("inf"))

            has_valid_token = tl.sum(valid_token.to(tl.int32), axis=0) > 0
            next_max = tl.where(
                has_valid_token,
                tl.maximum(max_score, tl.max(score, axis=0)),
                max_score,
            )
            rescale = tl.where(has_valid_token, tl.exp(max_score - next_max), 1.0)
            probabilities = tl.exp(score - next_max)
            probabilities = tl.where(valid_token, probabilities, 0.0)
            accumulator *= rescale

            for group in range(NUM_GROUPS):
                byte_offsets = tl.arange(0, 16)
                packed_codes = tl.load(
                    row_ptr + group * 16 + byte_offsets[None, :],
                    mask=valid_token[:, None],
                    other=0,
                )
                unsigned_codes = tl.interleave(
                    (packed_codes & 0x0F).to(tl.float32),
                    ((packed_codes >> 4) & 0x0F).to(tl.float32),
                )
                signed_codes = tl.where(
                    unsigned_codes >= 8.0,
                    unsigned_codes - 16.0,
                    unsigned_codes,
                )
                scale_bits = tl.load(
                    row_ptr + HEADER_OFFSET + group,
                    mask=valid_token[:, None],
                    other=0,
                )
                scale = (
                    tl.reshape(scale_bits, [BLOCK_N])
                    .to(tl.float8e4nv, bitcast=True)
                    .to(tl.float32)
                )
                values = (signed_codes * scale[:, None]).to(tl.bfloat16).to(tl.float32)
                partial = tl.sum(probabilities[:, None] * values, axis=0)
                relative = tl.maximum(tl.minimum(q_offsets - group * 32, 31), 0)
                accumulator += tl.where(
                    (q_offsets >= group * 32) & (q_offsets < (group + 1) * 32),
                    tl.gather(partial, relative, axis=0),
                    0.0,
                )

            rope_partial = tl.sum(probabilities[:, None] * rope, axis=0)
            rope_relative = tl.maximum(tl.minimum(q_offsets - 448, 63), 0)
            accumulator += tl.where(
                q_offsets >= 448,
                tl.gather(rope_partial, rope_relative, axis=0),
                0.0,
            )
            normalizer = normalizer * rescale + tl.sum(probabilities, axis=0)
            max_score = next_max
            start += BLOCK_N

        output_offsets = tl.arange(0, 512)
        has_mass = normalizer > 0
        tl.store(
            output_ptr
            + query_row * stride_output_row
            + head * stride_output_head
            + output_offsets,
            tl.where(has_mass, accumulator / normalizer, 0.0),
        )
        tl.store(
            lse_ptr + query_row * stride_lse_row + head,
            tl.where(has_mass, max_score + tl.log(normalizer), -float("inf")),
        )


def _reshape_packed_rows(
    packed: torch.Tensor,
    *,
    page_size: int,
) -> torch.Tensor:
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("packed cache must be a rank-2 torch.uint8 tensor")
    if page_size not in (2, 64, 256):
        raise ValueError(f"DSV4 INT4 page_size must be 2, 64 or 256, got {page_size}")
    if not packed.is_contiguous() or packed.shape[1] % page_size:
        raise ValueError("packed cache must contain contiguous, whole fixed-size rows")
    layout = layout_for_row_bytes(packed.shape[1] // page_size)
    return packed.view(-1, layout.row_bytes)


def write_dsv4_int4_packed(
    kv: torch.Tensor,
    loc: torch.Tensor,
    packed: torch.Tensor,
    *,
    page_size: int,
) -> None:
    """G32 signed-INT4/E4M3 encode and scatter DSV4 rows."""
    if kv.device.type != "cuda" or not _HAS_TRITON:
        raise RuntimeError("DSV4 KVBit packed writes require CUDA and Triton")
    if kv.ndim != 2 or kv.shape[-1] != 512:
        raise ValueError(f"kv must have shape (tokens, 512), got {kv.shape}")
    if not kv.is_floating_point() or kv.stride(-1) != 1:
        raise ValueError(
            "kv must be floating point and contiguous along its last dimension"
        )
    if loc.ndim != 1 or loc.shape[0] != kv.shape[0]:
        raise ValueError(f"loc must have shape ({kv.shape[0]},), got {loc.shape}")
    if loc.dtype not in (torch.int32, torch.int64):
        raise ValueError("loc must be int32 or int64")
    if kv.device != loc.device or kv.device != packed.device:
        raise ValueError("kv, loc, and packed cache must be on the same device")
    rows = _reshape_packed_rows(packed, page_size=page_size)
    if kv.shape[0] == 0:
        return
    _dsv4_int4_pack_scatter_kernel[(kv.shape[0],)](
        kv,
        loc,
        packed,
        kv.stride(0),
        loc.stride(0),
        packed.stride(0),
        packed.shape[0],
        PAGE_SIZE=page_size,
        ROW_BYTES=rows.shape[-1],
        CODE_BYTES=abi.CODE_BYTES,
        HEADER_OFFSET=abi.HEADER_OFFSET,
        ROPE_OFFSET=abi.ROPE_OFFSET,
        PAYLOAD_BYTES=abi.PAYLOAD_BYTES,
        NUM_GROUPS=abi.NUM_GROUPS,
        num_warps=4,
        num_stages=1,
    )


def _dsv4_sparse_decode_reference(
    q: torch.Tensor,
    packed: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    attn_sink: torch.Tensor | None,
    *,
    page_size: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = _reshape_packed_rows(packed, page_size=page_size)
    output = torch.empty_like(q)
    lse = torch.empty(
        (q.shape[0], q.shape[2], q.shape[1]),
        dtype=torch.float32,
        device=q.device,
    )
    for row in range(q.shape[0]):
        count = max(0, min(int(lengths[row]), indices.shape[-1]))
        selected = indices[row, 0, :count].to(torch.long)
        selected = selected[(selected >= 0) & (selected < rows.shape[0])]
        stored = decode_dsv4_int4_reference(rows[selected].cpu()).to(q.device)
        scores = torch.einsum("qhd,kd->qhk", q[row].float(), stored.float())
        scores.mul_(softmax_scale)
        if attn_sink is not None:
            sink = attn_sink.float().view(1, -1, 1)
            scores = torch.cat((scores, sink), dim=-1)
        if scores.shape[-1] == 0:
            output[row].zero_()
            lse[row].fill_(-torch.inf)
        else:
            probabilities = torch.softmax(scores, dim=-1)
            value_probabilities = (
                probabilities[..., :-1] if attn_sink is not None else probabilities
            )
            output[row] = torch.einsum(
                "qhk,kd->qhd", value_probabilities, stored.float()
            ).to(q.dtype)
            lse[row] = torch.logsumexp(scores, dim=-1).transpose(0, 1)
    return output, lse


def dsv4_kvbit_sparse_decode(
    q: torch.Tensor,
    packed: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    attn_sink: torch.Tensor | None,
    *,
    page_size: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct sparse attention over packed DSV4 Q/K/V=512 rows."""
    if page_size not in (2, 64, 256):
        raise ValueError(
            f"DSV4 KVBit requires page_size in (2, 64, 256), got {page_size}"
        )
    if q.ndim != 4 or q.shape[1] != 1 or q.shape[-1] != 512:
        raise ValueError(f"q must have shape (tokens, 1, heads, 512), got {q.shape}")
    if indices.ndim != 3 or indices.shape[:2] != q.shape[:2]:
        raise ValueError(
            f"indices must have shape ({q.shape[0]}, 1, width), got {indices.shape}"
        )
    rows = _reshape_packed_rows(packed, page_size=page_size)
    if lengths.shape != (q.shape[0],) or lengths.dtype != torch.int32:
        raise ValueError("lengths must be an int32 vector with one entry per query")
    if indices.dtype != torch.int32 or indices.stride(-1) != 1:
        raise ValueError("indices must be int32 and contiguous along the sparse width")
    if not lengths.is_contiguous():
        raise ValueError("lengths must be contiguous")
    if attn_sink is not None and (
        attn_sink.shape != (q.shape[2],)
        or attn_sink.dtype != torch.float32
        or not attn_sink.is_contiguous()
    ):
        raise ValueError(
            "attn_sink must be a contiguous float32 vector with one entry per head"
        )
    for tensor in (packed, indices, lengths, attn_sink):
        if tensor is not None and tensor.device != q.device:
            raise ValueError("all sparse decode tensors must be on the same device")
    if q.device.type != "cuda" or not _HAS_TRITON:
        return _dsv4_sparse_decode_reference(
            q,
            packed,
            indices,
            lengths,
            attn_sink,
            page_size=page_size,
            softmax_scale=softmax_scale,
        )

    q_contiguous = q.contiguous()
    output = torch.empty_like(q_contiguous)
    lse = torch.empty(
        (q.shape[0], q.shape[2], 1),
        dtype=torch.float32,
        device=q.device,
    )
    _dsv4_int4_sparse_decode_kernel[(q.shape[0], q.shape[2])](
        q_contiguous,
        packed,
        indices,
        lengths,
        attn_sink if attn_sink is not None else q,
        output,
        lse,
        q_contiguous.stride(0),
        q_contiguous.stride(2),
        packed.stride(0),
        indices.stride(0),
        output.stride(0),
        output.stride(2),
        lse.stride(0),
        packed.shape[0],
        softmax_scale,
        num_heads=q.shape[2],
        index_width=indices.shape[-1],
        PAGE_SIZE=page_size,
        ROW_BYTES=rows.shape[-1],
        HEADER_OFFSET=abi.HEADER_OFFSET,
        ROPE_OFFSET=abi.ROPE_OFFSET,
        NUM_GROUPS=abi.NUM_GROUPS,
        BLOCK_N=16,
        HAS_SINK=attn_sink is not None,
        num_warps=4,
        num_stages=1,
    )
    return output, lse


class DSV4KVBitPackedSWAPool(KVCache):
    """Persistent packed target DSV4 rows with no native shadow or scratch."""

    is_dsv4_kvbit_packed_swa = True
    is_dsv4_kvbit_packed = True

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: int | None = None,
        end_layer: int | None = None,
        layout: DSV4KVBitLayout | None = None,
    ):
        validate_dsv4_int4_geometry(qk_nope_head_dim, qk_rope_head_dim)
        if page_size not in (2, 64, 256):
            raise ValueError(
                f"DSV4 KVBit requires page_size in (2, 64, 256), got {page_size}"
            )
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.layout = layout if layout is not None else get_dsv4_int4_layout()
        self.store_dtype = torch.uint8
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_cache_total_dim = self.layout.row_bytes
        self.bytes_per_page_padded = self.page_size * self.layout.row_bytes
        self.num_pages = (self.size + self.page_size + 1) // self.page_size
        with (
            self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
            (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ),
        ):
            self.kv_buffer = [
                torch.zeros(
                    self.num_pages,
                    self.bytes_per_page_padded,
                    dtype=torch.uint8,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

    def get_bytes_per_token(self) -> int:
        return self.layout.row_bytes

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.kv_buffer[layer_id - self.start_layer]

    def set_key_buffer(self, *args, **kwargs) -> None:
        raise RuntimeError("DSV4 KVBit accepts only fused BF16 direct writes")

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        write_dsv4_int4_packed(
            cache_k,
            loc,
            self.kv_buffer[layer_id - self.start_layer],
            page_size=self.page_size,
        )

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError("DSV4 KVBit uses a single packed K/V buffer")

    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("DSV4 KVBit uses a single packed K/V buffer")

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError("DSV4 KVBit uses direct packed writes")
