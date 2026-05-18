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
            write_locs_safe = torch.where(
                valid_write, write_locs, torch.zeros_like(write_locs)
            )
            buf_flat[write_locs_safe] = kv_score_input
        else:
            # Prefill: plan_w has {ragged_id, write_loc} per write entry
            plan_w = plan[2]  # [num_w, 8] uint8 = WritePlan
            if plan_w.shape[0] > 0:
                plan_w_raw = plan_w.view(torch.int32)  # [num_w, 2]
                ragged_ids = plan_w_raw[:, 0].long() & 0xFFFF
                write_locs = plan_w_raw[:, 1].long()
                valid_write = (write_locs >= 0) & (write_locs < num_total_slots)
                write_locs_safe = torch.where(
                    valid_write, write_locs, torch.zeros_like(write_locs)
                )
                ragged_ids_safe = ragged_ids.clamp(
                    min=0, max=kv_score_input.shape[0] - 1
                )
                buf_flat[write_locs_safe] = kv_score_input[ragged_ids_safe]

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

    # NOTE: Will be overridden
    def _maybe_upgrade_forward_metadata(self): ...

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
    ) -> None:
        assert compress_ratio == 4 or compress_ratio == 128
        assert rotate == is_indexer == (head_dim == 128)

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
            out_loc=self._get_out_loc(compress_ratio),
            kvcache=kv_cache,
            page_size=page_size,
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

        self._maybe_upgrade_forward_metadata()
        token_to_kv_pool = forward_batch.token_to_kv_pool
        token_to_kv_pool = cast("DeepSeekV4TokenToKVPool", token_to_kv_pool)
        kv_score_input = compressor.compute_kv_score(x, forward_batch)
        state_pool = compressor.get_state_pool(forward_batch)

        if _is_hip:
            self._forward_unified_hip(
                token_to_kv_pool=token_to_kv_pool,
                kv_score_input=kv_score_input,
                state_pool=state_pool,
                compressor=compressor,
                layer_id=layer_id,
            )
        else:
            if compressor.is_in_indexer:
                kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
                page_size = token_to_kv_pool.get_index_k_page_size()
            else:
                kv_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)
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

        # Step 1: compress_forward
        coff = 2 if is_overlap_compress(compress_ratio) else 1
        last_dim = 2 * head_dim * coff
        kv_score_buffer = state_pool.kv_score_buffer.kv_score
        kv_score_buffer = kv_score_buffer.view(-1, compress_ratio, last_dim)

        if compress_ratio == 128:
            kv_compressed = _compress_forward_c128_fallback(
                kv_score_buffer=kv_score_buffer,
                kv_score_input=kv_score_input,
                ape=compressor.ape.view(-1, head_dim),
                plan=plan,
                head_dim=head_dim,
            )
        else:
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
        # The JIT kernels only output valid data at compression boundaries.
        if plan.is_decode:
            plan_raw = plan[1].view(torch.int32)
            seq_lens_plan = plan_raw[:, 0].to(torch.int32)
            is_boundary = (seq_lens_plan % compress_ratio == 0).unsqueeze(-1)
            kv_compressed = torch.where(
                is_boundary, kv_compressed, torch.zeros_like(kv_compressed)
            )

        # Step 2: norm + rope (Triton, works on wave64)
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
