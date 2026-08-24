from __future__ import annotations

from functools import lru_cache

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor
from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.utils import is_npu, is_npu_atlas_a5

if is_npu():
    import torch_npu
    import triton
    import triton.language as tl

    from sglang.srt.hardware_backend.npu.utils import get_indexer_weight_stream

_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()


@lru_cache(maxsize=1)
def _create_hadamard_128_cpu() -> torch.Tensor:
    """Build an unnormalized BF16 Sylvester H128 once on the CPU."""
    matrix = [[1.0]]
    while len(matrix) < 128:
        top = [row + row for row in matrix]
        bottom = [row + [-value for value in row] for row in matrix]
        matrix = top + bottom
    return torch.tensor(matrix, dtype=torch.bfloat16).contiguous()


def create_npu_hadamard_128(head_dim: int, device) -> torch.Tensor | None:
    """Create the A5-only H128 buffer after the NPU runtime is initialized."""
    if head_dim != 128 or not is_npu_atlas_a5():
        return None
    return _create_hadamard_128_cpu().to(device=device)


if is_npu():

    @lru_cache(maxsize=None)
    def _get_npu_cube_core_count(device_index: int) -> int:
        # Importing the backend registers the Ascend Triton runtime on releases
        # where it is not selected until the first kernel launch.
        try:
            import triton.backends.ascend.runtime  # noqa: F401
        except ImportError:
            pass

        properties = triton.runtime.driver.active.utils.get_device_properties(
            device_index
        )
        cube_core_count = int(properties.get("num_aicore", 0))
        if cube_core_count <= 0:
            raise RuntimeError(
                "Failed to query num_aicore from Ascend device properties: "
                f"{properties}"
            )
        return cube_core_count

    @triton.jit(do_not_specialize=["num_rows"])
    def _npu_hadamard_128_gemm_quant_fp8_kernel(
        input_ptr,
        hadamard_ptr,
        quantized_ptr,
        scale_ptr,
        num_rows,
        BLOCK_ROWS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HADAMARD_SCALE: tl.constexpr,
        FP8_MAX: tl.constexpr,
    ):
        """Compute BF16 X @ H128 and per-row FP8 dynamic quantization."""
        pid = tl.program_id(0)
        num_programs = tl.num_programs(0)
        num_row_tiles = tl.cdiv(num_rows, BLOCK_ROWS)
        tiles_per_program = num_row_tiles // num_programs
        extra_tiles = num_row_tiles - tiles_per_program * num_programs
        tiles_before = tl.minimum(pid, extra_tiles)
        first_row_tile = pid * tiles_per_program + tiles_before
        next_tiles_before = tl.minimum(pid + 1, extra_tiles)
        row_tile_count = tiles_per_program + next_tiles_before - tiles_before
        last_row_tile = first_row_tile + row_tile_count

        row_lane = tl.arange(0, BLOCK_ROWS).to(tl.int32)
        dim_lane = tl.arange(0, HEAD_DIM).to(tl.int32)
        hadamard_offsets = dim_lane[:, None] * HEAD_DIM + dim_lane[None, :]
        hadamard = tl.load(hadamard_ptr + hadamard_offsets)

        for row_tile in tl.range(first_row_tile, last_row_tile):
            rows = row_tile * BLOCK_ROWS + row_lane
            row_mask = rows < num_rows
            input_offsets = rows[:, None] * HEAD_DIM + dim_lane[None, :]
            values = tl.load(
                input_ptr + input_offsets,
                mask=row_mask[:, None],
                other=0.0,
            )
            rotated = tl.dot(values, hadamard) * HADAMARD_SCALE
            rotated = rotated.to(tl.bfloat16, fp_downcast_rounding="rtne").to(
                tl.float32
            )

            abs_max = tl.max(tl.abs(rotated), axis=1)
            scales = abs_max * (1.0 / FP8_MAX)
            nonzero = abs_max > 0.0
            safe_scales = tl.where(nonzero, scales, 1.0)
            scaled = rotated / safe_scales[:, None]
            scaled = tl.where(nonzero[:, None], scaled, 0.0)
            quantized = tl.clamp(scaled, -FP8_MAX, FP8_MAX).to(
                quantized_ptr.dtype.element_ty,
                fp_downcast_rounding="rtne",
            )

            tl.store(
                quantized_ptr + input_offsets,
                quantized,
                mask=row_mask[:, None],
            )
            tl.store(scale_ptr + rows, scales, mask=row_mask)


def _quantize_npu_indexer_activation(
    x: torch.Tensor,
    hadamard: torch.Tensor,
    dst_type: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse normalized H128 GEMM and row-wise E4M3 quantization on A5."""
    if x.device.type != "npu":
        raise RuntimeError("NPU fused Hadamard quantization requires an NPU tensor.")
    if x.dtype != torch.bfloat16:
        raise TypeError(
            "NPU fused Hadamard quantization requires BF16 input, " f"got {x.dtype}."
        )
    if x.size(-1) != 128:
        raise ValueError(
            "NPU fused Hadamard quantization requires last dimension 128, "
            f"got {x.size(-1)}."
        )
    if not x.is_contiguous():
        raise ValueError(
            "NPU fused Hadamard quantization requires a contiguous tensor."
        )
    if dst_type != torch.float8_e4m3fn:
        raise ValueError(
            "NPU fused Hadamard quantization only supports "
            f"torch.float8_e4m3fn, got {dst_type}."
        )
    if (
        hadamard.device != x.device
        or hadamard.dtype != torch.bfloat16
        or hadamard.shape != (128, 128)
        or not hadamard.is_contiguous()
    ):
        raise ValueError(
            "Hadamard matrix must be a contiguous BF16 [128, 128] tensor "
            "on the input NPU."
        )

    quantized = torch.empty_like(x, dtype=dst_type)
    scales = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    num_rows = x.numel() // 128
    if num_rows == 0:
        return quantized, scales

    if num_rows <= 16:
        block_rows = 16
    elif num_rows <= 32:
        block_rows = 32
    else:
        block_rows = 64
    device_index = x.device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    cube_core_count = _get_npu_cube_core_count(device_index)
    grid = (min(triton.cdiv(num_rows, block_rows), cube_core_count),)
    _npu_hadamard_128_gemm_quant_fp8_kernel[grid](
        x,
        hadamard,
        quantized,
        scales,
        num_rows,
        BLOCK_ROWS=block_rows,
        HEAD_DIM=128,
        HADAMARD_SCALE=0.08838834764831845,
        FP8_MAX=448.0,
    )
    return quantized, scales


def _normalize_npu_topk_result(topk_result, index_topk: int) -> torch.Tensor:
    if isinstance(topk_result, (tuple, list)):
        topk_result = topk_result[0]
    return topk_result.reshape(-1, index_topk)


def _resolve_eager_indexer_batch_size(
    padded_batch_size: int,
    num_kv_sequences: int,
    query_tokens_per_request: int,
    *,
    is_prefill: bool,
    graph_mode: bool,
) -> int:
    if is_prefill or graph_mode:
        return padded_batch_size
    return min(padded_batch_size, num_kv_sequences * query_tokens_per_request)


class DSANPUIndexerMixin:
    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        layer_scatter_modes=None,
        dynamic_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
        )
        bs = q_lora.shape[0]
        if (
            not is_prefill
            and not get_attn_backend().graph_mode
            and forward_batch.forward_mode.is_idle()
        ):
            # Idle early-return must precede the forward_metadata access:
            # eager _execute_idle drops forward_metadata (None) for
            # unpadded idle batches before running the model forward.
            return torch.empty((0, self.index_topk), dtype=torch.int32, device=x.device)
        if get_attn_backend().forward_metadata.seq_lens_cpu_int is None:
            actual_seq_lengths_kv = get_attn_backend().forward_metadata.seq_lens
        else:
            actual_seq_lengths_kv = get_attn_backend().forward_metadata.seq_lens_cpu_int

        kv_pool = get_token_to_kv_pool()
        use_quant_lightning_indexer = kv_pool.enable_npu_quant_lightning_indexer
        if use_quant_lightning_indexer and self._npu_hadamard_128 is None:
            raise RuntimeError(
                "The quantized NPU DSA Indexer requires an Atlas A5 H128 buffer."
            )
        # if (
        #     use_quant_lightning_indexer
        #     and is_prefill
        #     and self.dsa_enable_prefill_cp
        #     and forward_batch.attn_cp_metadata is not None
        # ):
        #     raise NotImplementedError(
        #         "The quantized NPU DSA Indexer does not support prefill context "
        #         "parallelism yet."
        #     )

        if self.rotary_emb.is_neox_style:
            if not hasattr(forward_batch, "npu_indexer_sin_cos_cache"):
                cos_sin = self.rotary_emb.cos_sin_cache[positions]
                cos, sin = cos_sin.chunk(2, dim=-1)
                cos = cos.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                sin = sin.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                forward_batch.npu_indexer_sin_cos_cache = (sin, cos)
            else:
                sin, cos = forward_batch.npu_indexer_sin_cos_cache

            if self.alt_stream is not None:
                self.alt_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(self.alt_stream):
                    q_lora = (
                        (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                    )
                    q = self.wq_b(q_lora)[
                        0
                    ]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
                    q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
                    q_pe, q_nope = torch.split(
                        q,
                        [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                        dim=-1,
                    )  # [bs, 64, 64 + 64]
                    q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
                    q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin).view(
                        bs, self.n_heads, self.rope_head_dim
                    )  # [bs, n, d]
                    q = torch.cat([q_pe, q_nope], dim=-1)
                    q.record_stream(self.alt_stream)
                    q_rope_event = self.alt_stream.record_event()
            else:
                q_lora = (
                    (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                )
                q = self.wq_b(q_lora)[
                    0
                ]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
                q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
                q_pe, q_nope = torch.split(
                    q,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )  # [bs, 64, 64 + 64]
                q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
                q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin).view(
                    bs, self.n_heads, self.rope_head_dim
                )  # [bs, n, d]
                q = torch.cat([q_pe, q_nope], dim=-1)

            if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    x = x.view(-1, self.hidden_size)
                    weights = self.weights_proj(x.float())[0].to(torch.bfloat16)
                    weights.record_stream(indexer_weight_stream)
                    weights_event = indexer_weight_stream.record_event()
            else:
                x = x.view(-1, self.hidden_size)
                weights = self.weights_proj(x.float())[0].to(torch.bfloat16)

            k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
            k = self.k_norm(k_proj)
            if (
                _use_ag_after_qlora
                and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
                and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
            ):
                k = scattered_to_tp_attn_full(k, forward_batch)
            k_pe, k_nope = torch.split(
                k,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64 + 64]

            k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
            k_pe = torch.ops.npu.npu_rotary_mul(k_pe, cos, sin).view(
                bs, 1, self.rope_head_dim
            )  # [bs, 1, d]
            k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)  # [bs, 1, 128]

        else:
            if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    x = x.view(-1, self.hidden_size)
                    weights = self.weights_proj(x.float())[0].to(torch.bfloat16)
                    weights.record_stream(indexer_weight_stream)
                    weights_event = indexer_weight_stream.record_event()
            else:
                x = x.view(-1, self.hidden_size)
                weights = self.weights_proj(x.float())[0].to(torch.bfloat16)

            q_lora = (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
            q = self.wq_b(q_lora)[0]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
            q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
            q_pe, q_nope = torch.split(
                q,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64, 64 + 64]

            k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
            k = self.k_norm(k_proj)
            k_pe, k_nope = torch.split(
                k,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64 + 64]

            k_pe = k_pe.unsqueeze(1)

            if layer_id == get_token_to_kv_pool().start_layer:
                self.rotary_emb.sin_cos_cache = (
                    self.rotary_emb.cos_sin_cache.index_select(0, positions)
                )

            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            k_pe = k_pe.squeeze(1)
            q = torch.cat([q_pe, q_nope], dim=-1)
            k = torch.cat([k_pe, k_nope], dim=-1)

        if (
            is_prefill
            and self.dsa_enable_prefill_cp
            and forward_batch.attn_cp_metadata is not None
        ):
            k = cp_all_gather_rerange_output(
                k.contiguous().view(-1, self.head_dim),
                self.cp_size,
                forward_batch,
                torch.npu.current_stream(),
            )

        if use_quant_lightning_indexer:
            k, k_scale = _quantize_npu_indexer_activation(
                k,
                self._npu_hadamard_128,
                kv_pool.dtype,
            )
            kv_pool.set_index_k_scale_buffer(
                layer_id, forward_batch.out_cache_loc, k_scale
            )
        kv_pool.set_index_k_buffer(layer_id, forward_batch.out_cache_loc, k)
        query_tokens_per_request = (
            get_attn_backend().speculative_num_draft_tokens
            if (
                forward_batch.forward_mode.is_draft_extend_v2()
                or forward_batch.forward_mode.is_target_verify()
            )
            else 1
        )
        indexer_bs = bs
        if not is_prefill and not get_attn_backend().graph_mode:
            indexer_bs = _resolve_eager_indexer_batch_size(
                bs,
                actual_seq_lengths_kv.numel(),
                query_tokens_per_request,
                is_prefill=False,
                graph_mode=False,
            )
        if is_prefill:
            if (
                self.dsa_enable_prefill_cp
                and forward_batch.attn_cp_metadata is not None
            ):
                get_attn_backend().forward_metadata.actual_seq_lengths_q = (
                    forward_batch.attn_cp_metadata.actual_seq_q_prev_tensor,
                    forward_batch.attn_cp_metadata.actual_seq_q_next_tensor,
                )
                if sum(forward_batch.extend_prefix_lens_cpu) > 0:
                    total_kv_len_prev_tensor = (
                        forward_batch.attn_cp_metadata.kv_len_prev_tensor
                        + forward_batch.extend_prefix_lens.squeeze()
                    )
                    total_kv_len_next_tensor = (
                        forward_batch.attn_cp_metadata.kv_len_next_tensor
                        + forward_batch.extend_prefix_lens.squeeze()
                    )
                    get_attn_backend().forward_metadata.actual_seq_lengths_kv = (
                        total_kv_len_prev_tensor,
                        total_kv_len_next_tensor,
                    )
                else:
                    get_attn_backend().forward_metadata.actual_seq_lengths_kv = (
                        forward_batch.attn_cp_metadata.kv_len_prev_tensor,
                        forward_batch.attn_cp_metadata.kv_len_next_tensor,
                    )
                actual_seq_lengths_q = (
                    get_attn_backend().forward_metadata.actual_seq_lengths_q
                )
                actual_seq_lengths_kv = (
                    get_attn_backend().forward_metadata.actual_seq_lengths_kv
                )
            else:
                actual_seq_lengths_kv = forward_batch.seq_lens
                actual_seq_lengths_q = forward_batch.extend_seq_lens.cumsum(dim=0)
        else:
            if get_attn_backend().forward_metadata.actual_seq_lengths_q is None:
                if (
                    forward_batch.forward_mode.is_draft_extend_v2()
                    or forward_batch.forward_mode.is_target_verify()
                ):
                    num_draft_tokens = get_attn_backend().speculative_num_draft_tokens
                    actual_seq_lengths_q = torch.arange(
                        num_draft_tokens,
                        num_draft_tokens + indexer_bs,
                        num_draft_tokens,
                        dtype=torch.int32,
                        device=k.device,
                    )
                else:
                    actual_seq_lengths_q = torch.tensor(
                        [1 + i for i in range(indexer_bs)],
                        dtype=torch.int32,
                        device=k.device,
                    )
            else:
                actual_seq_lengths_q = (
                    get_attn_backend().forward_metadata.actual_seq_lengths_q
                )

        past_key_states = kv_pool.get_index_k_buffer(layer_id)
        if use_quant_lightning_indexer:
            past_key_states_scale = kv_pool.get_index_k_scale_buffer(layer_id)

        if self.rotary_emb.is_neox_style and self.alt_stream is not None:
            torch.npu.current_stream().wait_event(q_rope_event)
        if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
            torch.npu.current_stream().wait_event(weights_event)
        if (
            _use_ag_after_qlora
            and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
            and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
        ):
            weights = scattered_to_tp_attn_full(weights, forward_batch)
        block_table = get_attn_backend().forward_metadata.block_tables
        if (
            is_prefill
            and self.dsa_enable_prefill_cp
            and forward_batch.attn_cp_metadata is not None
        ):
            block_table = block_table[: actual_seq_lengths_q[0].numel()]
            topk_indices = self.do_npu_cp_balance_indexer(
                q.view(-1, self.n_heads, self.head_dim),
                past_key_states,
                weights,
                actual_seq_lengths_q,
                actual_seq_lengths_kv,
                block_table,
                use_quant_lightning_indexer,
                layer_id,
            )
            return topk_indices
        else:
            block_table = (
                block_table[: actual_seq_lengths_q.size()[0]]
                if is_prefill
                else block_table
            )

            indexer_query = q.view(-1, self.n_heads, self.head_dim)
            indexer_weights = weights
            if indexer_bs != bs:
                indexer_query = indexer_query[:indexer_bs]
                indexer_weights = indexer_weights[:indexer_bs]
            if use_quant_lightning_indexer:
                indexer_query, indexer_query_scale = _quantize_npu_indexer_activation(
                    indexer_query,
                    self._npu_hadamard_128,
                    kv_pool.dtype,
                )
                topk_indices = torch_npu.npu_quant_lightning_indexer(
                    query=indexer_query,
                    key=past_key_states,
                    weights=indexer_weights,
                    query_dequant_scale=indexer_query_scale,
                    key_dequant_scale=past_key_states_scale,
                    actual_seq_lengths_query=actual_seq_lengths_q.to(torch.int32),
                    actual_seq_lengths_key=actual_seq_lengths_kv.to(k.device).to(
                        torch.int32
                    ),
                    block_table=block_table,
                    layout_query="TND",
                    layout_key="PA_BSND",
                    sparse_count=self.index_topk,
                    sparse_mode=3,
                    query_quant_mode=0,
                    key_quant_mode=0,
                )
            else:
                topk_indices = torch_npu.npu_lightning_indexer(
                    query=indexer_query,
                    key=past_key_states,
                    weights=indexer_weights,
                    actual_seq_lengths_query=actual_seq_lengths_q.to(torch.int32),
                    actual_seq_lengths_key=actual_seq_lengths_kv.to(k.device).to(
                        torch.int32
                    ),
                    block_table=block_table,
                    layout_query="TND",
                    layout_key="PA_BSND",
                    sparse_count=self.index_topk,
                    sparse_mode=3,
                )
            # IndexShare and NPU attention consume a stable [T, K] contract.
            return _normalize_npu_topk_result(topk_indices, self.index_topk)

    def do_npu_cp_balance_indexer(
        self,
        q,
        past_key_states,
        indexer_weights,
        actual_seq_lengths_q,
        actual_seq_lengths_kv,
        block_table,
        use_quant_lightning_indexer,
        layer_id,
    ):
        q_prev, q_next = torch.split(q, (q.size(0) + 1) // 2, dim=0)
        weights_prev, weights_next = None, None
        if indexer_weights is not None:
            weights_prev, weights_next = torch.split(
                indexer_weights, (indexer_weights.size(0) + 1) // 2, dim=0
            )
            weights_prev = weights_prev.contiguous().view(-1, weights_prev.shape[-1])
            weights_next = weights_next.contiguous().view(-1, weights_next.shape[-1])

        actual_seq_lengths_q_prev, actual_seq_lengths_q_next = actual_seq_lengths_q
        actual_seq_lengths_kv_prev, actual_seq_lengths_kv_next = actual_seq_lengths_kv

        if use_quant_lightning_indexer:
            q_prev, q_prev_scale = _quantize_npu_indexer_activation(
                q_prev,
                self._npu_hadamard_128,
                get_token_to_kv_pool().dtype,
            )
            past_key_states_scale = get_token_to_kv_pool().get_index_k_scale_buffer(
                layer_id
            )
            topk_indices_prev = torch_npu.npu_quant_lightning_indexer(
                query=q_prev,
                key=past_key_states,
                weights=weights_prev,
                query_dequant_scale=q_prev_scale,
                key_dequant_scale=past_key_states_scale,
                actual_seq_lengths_query=actual_seq_lengths_q_prev.to(
                    device=q.device, dtype=torch.int32
                ),
                actual_seq_lengths_key=actual_seq_lengths_kv_prev.to(
                    device=q.device, dtype=torch.int32
                ),
                block_table=block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
                query_quant_mode=0,
                key_quant_mode=0,
            )
            q_next, q_next_scale = _quantize_npu_indexer_activation(
                q_next,
                self._npu_hadamard_128,
                get_token_to_kv_pool().dtype,
            )
            topk_indices_next = torch_npu.npu_quant_lightning_indexer(
                query=q_next,
                key=past_key_states,
                weights=weights_next,
                query_dequant_scale=q_next_scale,
                key_dequant_scale=past_key_states_scale,
                actual_seq_lengths_query=actual_seq_lengths_q_next.to(
                    device=q.device, dtype=torch.int32
                ),
                actual_seq_lengths_key=actual_seq_lengths_kv_next.to(
                    device=q.device, dtype=torch.int32
                ),
                block_table=block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
                query_quant_mode=0,
                key_quant_mode=0,
            )
            return torch.cat([topk_indices_prev, topk_indices_next], dim=0).squeeze(1)

        else:
            topk_indices_prev = torch_npu.npu_lightning_indexer(
                query=q_prev,
                key=past_key_states,
                weights=weights_prev,
                actual_seq_lengths_query=actual_seq_lengths_q_prev.to(
                    device=q.device, dtype=torch.int32
                ),
                actual_seq_lengths_key=actual_seq_lengths_kv_prev.to(
                    device=q.device, dtype=torch.int32
                ),
                block_table=block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
            )
            topk_indices_next = torch_npu.npu_lightning_indexer(
                query=q_next,
                key=past_key_states,
                weights=weights_next,
                actual_seq_lengths_query=actual_seq_lengths_q_next.to(
                    device=q.device, dtype=torch.int32
                ),
                actual_seq_lengths_key=actual_seq_lengths_kv_next.to(
                    device=q.device, dtype=torch.int32
                ),
                block_table=block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
            )
            return torch.cat([topk_indices_prev[0], topk_indices_next[0]], dim=0).squeeze(1)


def scattered_to_tp_attn_full(
    hidden_states: torch.Tensor,
    forward_batch,
) -> torch.Tensor:
    hidden_states, local_hidden_states = (
        torch.empty(
            (forward_batch.input_ids.shape[0], hidden_states.shape[1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ),
        hidden_states,
    )
    attn_tp_all_gather_into_tensor(hidden_states, local_hidden_states.contiguous())
    return hidden_states
