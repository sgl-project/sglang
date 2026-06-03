from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional, TypeAlias, Union, cast

import torch

from sglang.jit_kernel.dsv4 import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_forward,
    compress_norm_rope_store,
)
from sglang.jit_kernel.utils import is_hip_runtime
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import DSV4Metadata
    from sglang.srt.layers.attention.dsv4.compressor import Compressor
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


CompressMetadata: TypeAlias = Union[CompressorDecodePlan, CompressorPrefillPlan]
# NOTE: alias for backward compatibility
FusedCompressMetadata: TypeAlias = CompressMetadata

_is_hip = is_hip_runtime()

if _is_hip:
    import triton
    import triton.language as tl

    @triton.jit
    def _c128_compress_decode_kernel(
        buf_ptr,
        input_ptr,
        ape_ptr,
        out_ptr,
        plan_ptr,
        buf_stride_slot,
        input_stride_b,
        ape_stride_r,
        out_stride_b,
        bs,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        COMPRESS_RATIO: tl.constexpr,
    ):
        """Fused C128 decode: write to state buffer + online softmax-pool.

        plan_ptr points to int32 view: [bs, 4] where each row is
        {seq_len, write_loc, read_page_0, read_page_1}.
        """
        bid = tl.program_id(0)
        if bid >= bs:
            return

        # Parse plan
        plan_base = plan_ptr + bid * 4
        seq_len = tl.load(plan_base).to(tl.int32)
        write_loc = tl.load(plan_base + 1).to(tl.int32)
        read_page_0 = tl.load(plan_base + 2).to(tl.int32)

        d = tl.arange(0, BLOCK_D)
        last_dim: tl.constexpr = HEAD_DIM * 2

        # Step 1: Write kv_score_input to state buffer at write_loc
        d_mask_full = d < last_dim
        input_val = tl.load(
            input_ptr + bid * input_stride_b + d, mask=d_mask_full, other=0.0
        )
        tl.store(buf_ptr + write_loc * buf_stride_slot + d, input_val, mask=d_mask_full)

        # Step 2: Check boundary condition
        d_mask_hd = d < HEAD_DIM
        if seq_len % COMPRESS_RATIO != 0:
            tl.store(
                out_ptr + bid * out_stride_b + d,
                tl.zeros([BLOCK_D], tl.float32),
                mask=d_mask_hd,
            )
            return

        # Step 3: Online softmax-pool over 128 slots in the page
        page_base = read_page_0 * COMPRESS_RATIO * buf_stride_slot
        m_prev = tl.full([BLOCK_D], float("-inf"), tl.float32)
        kv_acc = tl.zeros([BLOCK_D], tl.float32)
        w_acc = tl.zeros([BLOCK_D], tl.float32)

        for k in tl.static_range(COMPRESS_RATIO):
            slot_addr = page_base + k * buf_stride_slot
            kv_val = tl.load(buf_ptr + slot_addr + d, mask=d_mask_hd, other=0.0).to(
                tl.float32
            )
            sc_val = tl.load(
                buf_ptr + slot_addr + HEAD_DIM + d, mask=d_mask_hd, other=0.0
            ).to(tl.float32)
            ape_val = tl.load(
                ape_ptr + k * ape_stride_r + d, mask=d_mask_hd, other=0.0
            ).to(tl.float32)
            score_k = sc_val + ape_val

            m_new = tl.maximum(m_prev, score_k)
            exp_old = tl.where(m_prev == float("-inf"), 0.0, tl.exp(m_prev - m_new))
            exp_cur = tl.where(score_k == float("-inf"), 0.0, tl.exp(score_k - m_new))
            kv_acc = kv_acc * exp_old + exp_cur * kv_val
            w_acc = w_acc * exp_old + exp_cur
            m_prev = m_new

        compressed = kv_acc / w_acc
        tl.store(out_ptr + bid * out_stride_b + d, compressed, mask=d_mask_hd)

    @triton.jit
    def _c128_compress_prefill_write_kernel(
        buf_ptr,
        input_ptr,
        plan_w_ptr,
        buf_stride_slot,
        input_stride_b,
        num_w,
        BLOCK_D: tl.constexpr,
        LAST_DIM: tl.constexpr,
    ):
        """Prefill write phase: scatter kv_score_input tokens into state buffer."""
        wid = tl.program_id(0)
        if wid >= num_w:
            return

        # WritePlan: {ragged_id(u32), write_loc(i32)} = 8 bytes = 2 int32s
        plan_base = plan_w_ptr + wid * 2
        ragged_id = (tl.load(plan_base).to(tl.int32)) & 0xFFFF
        write_loc = tl.load(plan_base + 1).to(tl.int32)

        d = tl.arange(0, BLOCK_D)
        d_mask = d < LAST_DIM

        if write_loc >= 0:
            input_val = tl.load(
                input_ptr + ragged_id * input_stride_b + d, mask=d_mask, other=0.0
            )
            tl.store(buf_ptr + write_loc * buf_stride_slot + d, input_val, mask=d_mask)

    @triton.jit
    def _c128_compress_prefill_compress_kernel(
        buf_ptr,
        ape_ptr,
        out_ptr,
        plan_c_ptr,
        buf_stride_slot,
        ape_stride_r,
        out_stride_b,
        num_c,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        COMPRESS_RATIO: tl.constexpr,
    ):
        """Prefill compress phase: online softmax-pool for each compress plan entry."""
        cid = tl.program_id(0)
        if cid >= num_c:
            return

        # CompressPlan: {seq_len(u32), ragged_id(u16)|buffer_len(u16), read_page_0(i32), read_page_1(i32)}
        plan_base = plan_c_ptr + cid * 4
        read_page_0 = tl.load(plan_base + 2).to(tl.int32)

        d = tl.arange(0, BLOCK_D)
        d_mask_hd = d < HEAD_DIM

        if read_page_0 < 0:
            tl.store(
                out_ptr + cid * out_stride_b + d,
                tl.zeros([BLOCK_D], tl.float32),
                mask=d_mask_hd,
            )
            return

        page_base = read_page_0 * COMPRESS_RATIO * buf_stride_slot
        m_prev = tl.full([BLOCK_D], float("-inf"), tl.float32)
        kv_acc = tl.zeros([BLOCK_D], tl.float32)
        w_acc = tl.zeros([BLOCK_D], tl.float32)

        for k in tl.static_range(COMPRESS_RATIO):
            slot_addr = page_base + k * buf_stride_slot
            kv_val = tl.load(buf_ptr + slot_addr + d, mask=d_mask_hd, other=0.0).to(
                tl.float32
            )
            sc_val = tl.load(
                buf_ptr + slot_addr + HEAD_DIM + d, mask=d_mask_hd, other=0.0
            ).to(tl.float32)
            ape_val = tl.load(
                ape_ptr + k * ape_stride_r + d, mask=d_mask_hd, other=0.0
            ).to(tl.float32)
            score_k = sc_val + ape_val

            m_new = tl.maximum(m_prev, score_k)
            exp_old = tl.where(m_prev == float("-inf"), 0.0, tl.exp(m_prev - m_new))
            exp_cur = tl.where(score_k == float("-inf"), 0.0, tl.exp(score_k - m_new))
            kv_acc = kv_acc * exp_old + exp_cur * kv_val
            w_acc = w_acc * exp_old + exp_cur
            m_prev = m_new

        compressed = kv_acc / w_acc
        tl.store(out_ptr + cid * out_stride_b + d, compressed, mask=d_mask_hd)


def _compress_forward_c128_triton(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
    head_dim: int,
) -> torch.Tensor:
    """Triton C128 compress_forward for HIP (wave64).

    Fuses write + online-softmax-pool into Triton kernels.
    CUDA graph compatible.
    """
    num_total_slots = kv_score_buffer.shape[0] * kv_score_buffer.shape[1]
    num_pages = kv_score_buffer.shape[0]
    last_dim = kv_score_buffer.shape[-1]
    compress_ratio = 128

    buf_flat = kv_score_buffer.view(-1, last_dim)
    buf_stride_slot = last_dim  # elements per slot

    BLOCK_D = triton.next_power_of_2(last_dim)

    if plan.is_decode:
        # Decode path: single kernel does write + compress
        plan_raw = plan[1].view(torch.int32)  # [bs, 4]
        bs = plan_raw.shape[0]
        out = torch.empty(
            bs, head_dim, dtype=torch.float32, device=kv_score_input.device
        )

        if bs > 0 and num_total_slots > 0:
            grid = (bs,)
            _c128_compress_decode_kernel[grid](
                buf_flat,
                kv_score_input,
                ape,
                out,
                plan_raw,
                buf_stride_slot,
                kv_score_input.stride(0),
                ape.stride(0),
                out.stride(0),
                bs,
                HEAD_DIM=head_dim,
                BLOCK_D=triton.next_power_of_2(head_dim),
                COMPRESS_RATIO=compress_ratio,
                num_warps=8,
            )
        return out
    else:
        # Prefill path: separate write kernel + compress kernel
        plan_c_raw = plan[1].view(torch.int32)  # [num_c, 4]
        plan_w = plan[2]  # [num_w, 8] uint8
        plan_w_raw = plan_w.view(torch.int32)  # [num_w, 2]
        num_c = plan_c_raw.shape[0]
        num_w = plan_w_raw.shape[0]

        out = torch.empty(
            num_c, head_dim, dtype=torch.float32, device=kv_score_input.device
        )

        # Phase 1: Write
        if num_w > 0 and num_total_slots > 0:
            grid_w = (num_w,)
            _c128_compress_prefill_write_kernel[grid_w](
                buf_flat,
                kv_score_input,
                plan_w_raw,
                buf_stride_slot,
                kv_score_input.stride(0),
                num_w,
                BLOCK_D=BLOCK_D,
                LAST_DIM=last_dim,
                num_warps=4,
            )

        # Phase 2: Compress
        if num_c > 0 and num_pages > 0:
            grid_c = (num_c,)
            _c128_compress_prefill_compress_kernel[grid_c](
                buf_flat,
                ape,
                out,
                plan_c_raw,
                buf_stride_slot,
                ape.stride(0),
                out.stride(0),
                num_c,
                HEAD_DIM=head_dim,
                BLOCK_D=triton.next_power_of_2(head_dim),
                COMPRESS_RATIO=compress_ratio,
                num_warps=8,
            )

        return out


def _use_online_compress(compress_ratio: int) -> bool:
    """Online state-pool path is c128-only."""
    return compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()


def _extract_positions_from_plan(
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
    compress_ratio: int,
) -> torch.Tensor:
    """Extract RoPE positions from plan tensors (decode or prefill).

    DecodePlan layout: [bs, 16] uint8, first 4 bytes = uint32 seq_len.
    CompressPlan layout: [num_c, 16] uint8, first 4 bytes = uint32 seq_len.
    Position for RoPE = seq_len - compress_ratio.
    """
    plan_tensor = plan[1]  # plan_d or plan_c
    seq_lens = plan_tensor[:, :4].contiguous().view(torch.int32).squeeze(-1)
    positions = seq_lens.to(torch.int32) - compress_ratio
    return positions


def _compress_forward_c128_fallback(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
    head_dim: int,
) -> torch.Tensor:
    """PyTorch fallback for C128 compress_forward on HIP (wave64).

    Fully vectorized, compatible with CUDA graph capture.
    kv_score_buffer: [num_pages, 128, head_dim * 2]
    ape: [128, head_dim]

    IMPORTANT: This also performs the write to state buffer (like the JIT kernel).
    The JIT kernel does: (1) write kv_score_input to buffer, (2) compress from buffer.
    """
    num_total_slots = kv_score_buffer.shape[0] * kv_score_buffer.shape[1]
    num_pages = kv_score_buffer.shape[0]
    last_dim = kv_score_buffer.shape[-1]

    # Step 1: WRITE kv_score_input to state buffer
    if num_total_slots > 0:
        buf_flat = kv_score_buffer.view(-1, last_dim)
        if plan.is_decode:
            # Decode: plan_d has write_loc per batch item
            plan_raw = plan[1].view(torch.int32)  # [bs, 4]
            write_locs = plan_raw[:, 1].long()
            # Only write valid locations (>= 0 and < buffer size)
            valid_write = (write_locs >= 0) & (write_locs < num_total_slots)
            if valid_write.any():
                buf_flat[write_locs[valid_write]] = kv_score_input[valid_write]
        else:
            # Prefill: plan_w has {ragged_id, write_loc} per write entry
            plan_w = plan[2]  # [num_w, 8] uint8 = WritePlan
            if plan_w.shape[0] > 0:
                plan_w_raw = plan_w.view(torch.int32)  # [num_w, 2]
                ragged_ids = plan_w_raw[:, 0].long() & 0xFFFF
                write_locs = plan_w_raw[:, 1].long()
                valid_write = (write_locs >= 0) & (write_locs < num_total_slots)
                ragged_ids_safe = ragged_ids.clamp(
                    min=0, max=kv_score_input.shape[0] - 1
                )
                if valid_write.any():
                    buf_flat[write_locs[valid_write]] = kv_score_input[
                        ragged_ids_safe[valid_write]
                    ]

    # Step 2: COMPRESS (read from buffer page and do softmax-pool)
    plan_c = plan[1]  # plan_d for decode, plan_c for prefill
    num_tokens = plan_c.shape[0]
    if num_pages == 0 or num_tokens == 0:
        return kv_score_input.new_zeros(num_tokens, head_dim)

    plan_c_raw = plan_c.view(torch.int32)  # [N, 4]
    read_page_0 = plan_c_raw[:, 2].long()
    # Use torch.where instead of clamp to handle -1 (invalid) gracefully
    valid_read = (read_page_0 >= 0) & (read_page_0 < num_pages)
    read_page_0_safe = torch.where(
        valid_read, read_page_0, torch.zeros_like(read_page_0)
    )

    gathered = kv_score_buffer[read_page_0_safe]  # [N, 128, head_dim*2]
    kv = gathered[:, :, :head_dim].float()
    score = gathered[:, :, head_dim:].float() + ape.float().unsqueeze(0)
    weights = score.softmax(dim=1)
    out = (weights * kv).sum(dim=1)

    # For decode: zero out non-boundary tokens (seq_len % 128 != 0)
    # so they don't corrupt kvcache location 0 when stored.
    if plan.is_decode:
        seq_lens = plan_c_raw[:, 0].to(torch.int32)
        is_boundary = (seq_lens % 128 == 0).unsqueeze(-1)  # [N, 1]
        out = torch.where(is_boundary, out, torch.zeros_like(out))

    return out.to(kv_score_input.dtype)


class CompressorBackendMixin:
    def __init__(self):
        super().__init__()
        self.forward_metadata: DSV4Metadata

    def _get_paged_compress_metadata(self, compress_ratio: int) -> CompressMetadata:
        attr_name = f"c{compress_ratio}_compress_metadata"
        return getattr(self.forward_metadata, attr_name)

    def _get_out_loc(self, compress_ratio: int) -> torch.Tensor:
        attr_name = f"c{compress_ratio}_out_loc"
        return getattr(self.forward_metadata.core_metadata, attr_name)

    def _forward_compress_all_in_one(
        self,
        *,
        kv_score_buffer: torch.Tensor,
        kv_score_input: torch.Tensor,
        ape: torch.Tensor,
        head_dim: int,
        norm: RMSNorm,
        freqs_cis_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        is_indexer: bool,
        rotate: bool,
        compress_ratio: int,
        page_size: int,
        out_loc: torch.Tensor,
        use_fp4_indexer: bool = False,
    ) -> None:
        assert compress_ratio == 4 or compress_ratio == 128
        assert rotate == is_indexer == (head_dim == 128)
        if use_fp4_indexer:
            assert is_indexer
            assert compress_ratio == 4
            assert head_dim == 128

        plan = self._get_paged_compress_metadata(compress_ratio)
        is_online = _use_online_compress(compress_ratio)
        if is_online:
            kv_score_buffer = kv_score_buffer.view(-1, 1, head_dim * 3)
        else:
            coff = 2 if is_overlap_compress(compress_ratio) else 1
            last_dim = 2 * head_dim * coff
            assert kv_score_buffer.shape[-1] == last_dim
            kv_score_buffer = kv_score_buffer.view(-1, compress_ratio, last_dim)

        # Step 1: compress_forward
        kv_compressed = compress_forward(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score_input,
            ape=ape.view(-1, head_dim),
            plan=plan,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            is_online=is_online,
        )

        # Step 2: norm + rope + store
        compress_norm_rope_store(
            kv_compressed,
            plan,
            norm_weight=norm.weight,
            norm_eps=norm.variance_epsilon,
            freq_cis=freqs_cis_cache,
            out_loc=out_loc,
            kvcache=kv_cache,
            page_size=page_size,
            use_fp4=use_fp4_indexer,
        )

    def forward_unified(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return

        token_to_kv_pool = self.token_to_kv_pool
        token_to_kv_pool = cast("DeepSeekV4TokenToKVPool", token_to_kv_pool)
        kv_score_input = compressor.compute_kv_score(x, forward_batch)

        state_pool = compressor.get_state_pool(self)
        if _is_hip and not envs.SGLANG_OPT_USE_JIT_NORM.get():
            self._forward_unified_hip(
                token_to_kv_pool=token_to_kv_pool,
                kv_score_input=kv_score_input,
                state_pool=state_pool,
                compressor=compressor,
                layer_id=layer_id,
            )
        else:
            out_loc = self._get_out_loc(compressor.ratio)
            use_fp4_indexer = (
                compressor.is_in_indexer and self.enable_deepseek_v4_fp4_indexer
            )
            if compressor.is_in_indexer:
                kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
                page_size = token_to_kv_pool.get_index_k_page_size()
            else:
                _, _, compress_kv_pool = token_to_kv_pool.layer_mapping[layer_id]
                assert compress_kv_pool is not None
                kv_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)
                if hasattr(compress_kv_pool, "translate_loc_to_hisparse_device"):
                    # The v2 compressor writes directly into the raw C4 KV tensor.
                    # HiSparse C4 therefore needs the physical C4 location here.
                    out_loc = compress_kv_pool.translate_loc_to_hisparse_device(out_loc)
            self._forward_compress_all_in_one(
                kv_score_buffer=state_pool.kv_score_buffer.kv_score,
                kv_score_input=kv_score_input,
                ape=compressor.ape,
                head_dim=compressor.head_dim,
                norm=compressor.norm,
                freqs_cis_cache=compressor.freqs_cis,
                kv_cache=kv_cache.view(dtype=torch.uint8),
                is_indexer=compressor.is_in_indexer,
                rotate=compressor.rotate,
                compress_ratio=compressor.ratio,
                page_size=page_size,
                out_loc=out_loc,
                use_fp4_indexer=use_fp4_indexer,
            )

    def _forward_unified_hip(
        self,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        kv_score_input: torch.Tensor,
        state_pool,
        compressor: Compressor,
        layer_id: int,
    ) -> None:
        """HIP-specific forward path using PyTorch/Triton fallbacks."""
        from sglang.srt.layers.attention.dsv4.quant_k_cache import (
            quant_to_nope_fp8_rope_bf16_pack_triton,
        )
        from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation
        from sglang.srt.layers.attention.nsa.triton_kernel import act_quant
        from sglang.srt.layers.deepseek_v4_rope import fused_norm_rope_inplace_triton

        compress_ratio = compressor.ratio
        head_dim = compressor.head_dim
        is_indexer = compressor.is_in_indexer

        plan = self._get_paged_compress_metadata(compress_ratio)
        out_loc = self._get_out_loc(compress_ratio)

        # Step 1: compress_forward (always use JIT for both C4 and C128)
        coff = 2 if is_overlap_compress(compress_ratio) else 1
        last_dim = 2 * head_dim * coff
        kv_score_buffer = state_pool.kv_score_buffer.kv_score
        kv_score_buffer = kv_score_buffer.view(-1, compress_ratio, last_dim)

        kv_compressed = compress_forward(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score_input,
            ape=compressor.ape.view(-1, head_dim),
            plan=plan,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            is_online=False,
        )

        if kv_compressed.shape[0] == 0:
            return

        # For decode: zero out non-boundary tokens to prevent corrupting kvcache loc 0.
        if plan.is_decode:
            plan_raw = plan[1].view(torch.int32)
            seq_lens_plan = plan_raw[:, 0].to(torch.int32)
            is_boundary = (seq_lens_plan % compress_ratio == 0).unsqueeze(-1)
            kv_compressed = torch.where(
                is_boundary, kv_compressed, torch.zeros_like(kv_compressed)
            )

        # Step 2: norm + rope (Triton fallback for precision parity with V1)
        positions = _extract_positions_from_plan(plan, compress_ratio)
        positions_safe = positions.clamp(min=0)

        fused_norm_rope_inplace_triton(
            kv_compressed,
            compressor.norm.weight,
            compressor.norm.variance_epsilon,
            compressor.freqs_cis,
            positions=positions_safe,
        )

        # Step 3: optional Hadamard rotation for indexer
        if compressor.rotate:
            kv_compressed = rotate_activation(kv_compressed)

        # Step 4: store to kvcache
        # For decode: store ALL tokens. Non-boundary tokens have out_loc=0 (safe).
        # For prefill: plan_c already only contains valid entries.
        if plan.is_decode:
            kv_to_store = kv_compressed
            out_loc_to_store = out_loc
        else:
            kv_to_store = kv_compressed
            plan_raw = plan[1].view(torch.int32)
            ragged_ids = plan_raw[:, 1].to(torch.int32) & 0xFFFF
            out_loc_to_store = out_loc[ragged_ids.long()]

        if kv_to_store.shape[0] == 0:
            return

        if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            # fused kernel: BF16 in -> FP8 quant + paged scatter in one launch
            if is_indexer:
                token_to_kv_pool.set_index_k_fused(
                    layer_id=layer_id,
                    loc=out_loc_to_store,
                    cache_k=kv_to_store,
                )
            else:
                token_to_kv_pool.set_extra_key_buffer_fused(
                    layer_id=layer_id,
                    loc=out_loc_to_store,
                    cache_k=kv_to_store,
                )
        else:
            if is_indexer:
                kv_fp8, kv_scale = act_quant(kv_to_store)
                token_to_kv_pool.set_index_k_scale_buffer(
                    layer_id=layer_id,
                    loc=out_loc_to_store,
                    index_k=kv_fp8,
                    index_k_scale=kv_scale,
                )
            else:
                pack = quant_to_nope_fp8_rope_bf16_pack_triton(kv_to_store.bfloat16())
                token_to_kv_pool.set_extra_key_buffer(layer_id, out_loc_to_store, pack)

    # NOTE: alias for backward compatibility
    forward_indexer_compressor = forward_unified
    forward_core_compressor = forward_unified


def is_overlap_compress(compress_ratio: int) -> bool:
    return compress_ratio == 4


def create_paged_compressor_data(
    compress_ratio: Literal[4, 128],
    *,
    is_prefill: bool,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: Optional[torch.Tensor] = None,
    seq_lens_cpu: Optional[List[int]] = None,
    extend_lens_cpu: Optional[List[int]] = None,
    use_prefill_cuda_graph: bool = False,
    num_q_tokens: Optional[int] = None,
) -> CompressMetadata:
    """Build the paged compress metadata (= the plan).

    State-pool slot translation is done inside the C++ planner; the
    Python side just hands the relevant tensors over.
    """
    if _use_online_compress(compress_ratio):
        return _create_online_paged_compressor_data(
            is_prefill=is_prefill,
            token_to_kv_pool=token_to_kv_pool,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_lens=extend_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_lens_cpu=extend_lens_cpu,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
            num_q_tokens=num_q_tokens,
        )

    swa_page_size = token_to_kv_pool.swa_page_size
    ring_size = token_to_kv_pool.get_ring_size(compress_ratio=compress_ratio)
    # NOTE: This is actually a proxy, which encounter some bug with tvm-ffi.
    # As a workaround, we use `.detach()` to get the real tensor.
    full_to_swa = token_to_kv_pool.full_to_swa_index_mapping.detach()
    req_pool_indices_i64 = req_pool_indices.to(torch.int64)

    if is_prefill:
        assert extend_lens is not None
        if seq_lens_cpu is not None:
            assert extend_lens_cpu is not None
            seq_lens_planner = torch.tensor(seq_lens_cpu, dtype=torch.int64)
            extend_lens_planner = torch.tensor(extend_lens_cpu, dtype=torch.int64)
            num_q_tokens = sum(extend_lens_cpu)
        else:
            assert num_q_tokens is not None
            seq_lens_planner = seq_lens.to(torch.int64)
            extend_lens_planner = extend_lens.to(torch.int64)

        return CompressorPrefillPlan.generate(
            compress_ratio=compress_ratio,
            req_pool_indices=req_pool_indices_i64,
            seq_lens=seq_lens_planner,
            extend_lens=extend_lens_planner,
            req_to_token=req_to_token,
            full_to_swa=full_to_swa,
            swa_page_size=swa_page_size,
            ring_size=ring_size,
            num_q_tokens=num_q_tokens,
            use_cuda_graph=use_prefill_cuda_graph,
        )
    else:
        return CompressorDecodePlan.generate(
            compress_ratio=compress_ratio,
            req_pool_indices=req_pool_indices_i64,
            req_to_token=req_to_token,
            full_to_swa=full_to_swa,
            seq_lens=seq_lens.to(torch.int64),
            swa_page_size=swa_page_size,
            ring_size=ring_size,
        )


def _create_online_paged_compressor_data(
    *,
    is_prefill: bool,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: Optional[torch.Tensor],
    seq_lens_cpu: Optional[List[int]],
    extend_lens_cpu: Optional[List[int]],
    use_prefill_cuda_graph: bool,
    num_q_tokens: Optional[int],
) -> CompressMetadata:
    assert not use_prefill_cuda_graph, "online c128 doesn't support cuda graph"

    swa_page_size = int(token_to_kv_pool.swa_page_size)
    full_to_swa = token_to_kv_pool.full_to_swa_index_mapping.detach()
    req_pool_indices = req_pool_indices.to(torch.int64)

    if is_prefill:
        # Sync-on-entry: catch IMA from a prior layer / kernel BEFORE we touch
        # anything in this builder, so blame doesn't land on us spuriously.
        assert extend_lens is not None
        if seq_lens_cpu is not None:
            assert extend_lens_cpu is not None
            seq_lens_planner = torch.tensor(seq_lens_cpu, dtype=torch.int64)
            extend_lens_planner = torch.tensor(extend_lens_cpu, dtype=torch.int64)
            num_q_tokens_planner = sum(extend_lens_cpu)
        else:
            assert num_q_tokens is not None
            seq_lens_planner = seq_lens.to(torch.int64)
            extend_lens_planner = extend_lens.to(torch.int64)
            num_q_tokens_planner = num_q_tokens

        return CompressorPrefillPlan.generate_online(
            seq_lens=seq_lens_planner,
            extend_lens=extend_lens_planner,
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            full_to_swa=full_to_swa,
            num_q_tokens=int(num_q_tokens_planner),
            swa_page_size=swa_page_size,
        )
    else:
        return CompressorDecodePlan.generate_online(
            seq_lens=seq_lens.to(torch.int64),
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            full_to_swa=full_to_swa,
            swa_page_size=swa_page_size,
        )
