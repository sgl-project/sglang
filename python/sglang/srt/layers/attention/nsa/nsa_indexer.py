from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
from einops import rearrange

from sglang.srt.custom_op import CustomOp
from sglang.srt.layers.layernorm import LayerNorm
from sglang.srt.utils import add_prefix, ceil_align, is_cuda, is_hip, is_npu

if is_cuda():
    try:
        import deep_gemm
    except ImportError as e:
        deep_gemm = e


from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.nsa.utils import (
    NSA_DUAL_STREAM,
    cp_all_gather_rerange_output,
    is_nsa_enable_prefill_cp,
)
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

DUAL_STREAM_TOKEN_THRESHOLD = 1024 if is_cuda() else 0


class BaseIndexerMetadata(ABC):
    @abstractmethod
    def get_seqlens_int32(self) -> torch.Tensor:
        """
        Return: (batch_size,) int32 tensor
        """

    @abstractmethod
    def get_page_table_64(self) -> torch.Tensor:
        """
        Return: (batch_size, num_blocks) int32, page table.
                The page size of the table is 64.
        """

    @abstractmethod
    def get_seqlens_expanded(self) -> torch.Tensor:
        """
        Return: (sum_extend_seq_len,) int32 tensor
        """

    @abstractmethod
    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """
        Perform topk selection on the logits and possibly transform the result.

        NOTE that attention backend may override this function to do some
        transformation, which means the result of this topk_transform may not
        be the topk indices of the input logits.

        Return: Anything, since it will be passed to the attention backend
                for further processing on sparse attention computation.
                Don't assume it is the topk indices of the input logits.
        """


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    from sgl_kernel import hadamard_transform

    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
    return hadamard_transform(x, scale=hidden_size**-0.5)


class Indexer(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_id: int,
        scale_fmt: Optional[str],
        block_size: int = 128,
        rope_scaling: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_attention_tp_size()
            self.cp_rank = get_attention_tp_rank()
        else:
            self.cp_size = None
            self.cp_rank = None
        if is_cuda():
            self.sm_count = deep_gemm.get_num_sms()
            self.half_device_sm_count = ceil_align(self.sm_count // 2, 8)

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )

        self.wk = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wk", prefix),
        )
        # NOTE: weights_proj in the checkpoint is stored in bf16, while the parameters here are stored in fp32 for convenience
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            params_dtype=torch.float32,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.k_norm = LayerNorm(self.head_dim, dtype=torch.float32)
        self.rotary_emb = get_rope_wrapper(
            rope_head_dim,
            rotary_dim=rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,  # type: ignore
            rope_scaling=rope_scaling,
            is_neox_style=True,
            device=get_global_server_args().device,
        )
        self.block_size = block_size
        self.scale_fmt = scale_fmt
        self.softmax_scale = self.head_dim**-0.5

    @torch.compile(dynamic=True)
    def _get_logits_head_gate(self, x: torch.Tensor, q_scale: torch.Tensor):
        weights, _ = self.weights_proj(x.float())
        weights = weights * self.n_heads**-0.5
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        return weights

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        positions: torch.Tensor,
        enable_dual_stream: bool,
        forward_batch: ForwardBatch,
    ):
        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                self.half_device_sm_count
            ):
                query, _ = self.wq_b(q_lora)
                query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
                q_rope, _ = torch.split(
                    query,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )
            with torch.cuda.stream(self.alt_stream):
                # TODO we should also put DeepGEMM half SM here?
                key, _ = self.wk(x)
                key = self.k_norm(key)

                k_rope, _ = torch.split(
                    key,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )

            current_stream.wait_stream(self.alt_stream)
        else:
            query, _ = self.wq_b(q_lora)
            query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
            q_rope, _ = torch.split(
                query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )
            key, _ = self.wk(x)
            key = self.k_norm(key)
            k_rope, _ = torch.split(
                key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )

        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

        query[..., : self.rope_head_dim] = q_rope
        key[..., : self.rope_head_dim] = k_rope

        # allgather+rerrange
        if forward_batch.nsa_cp_metadata is not None and self.nsa_enable_prefill_cp:
            key = cp_all_gather_rerange_output(
                key.contiguous(),
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )

        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            query = rotate_activation(query)

            with torch.cuda.stream(self.alt_stream):
                key = rotate_activation(key)
            current_stream.wait_stream(self.alt_stream)
        else:
            query = rotate_activation(query)
            key = rotate_activation(key)

        return query, key

    def _get_k_bf16(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        enable_dual_stream: bool,
    ):
        # Compute only key, skip query
        key, _ = self.wk(x)
        key = self.k_norm(key)
        k_rope, _ = torch.split(
            key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

        _, k_rope = self.rotary_emb(positions, k_rope, k_rope)
        key[..., : self.rope_head_dim] = k_rope
        key = rotate_activation(key)

        return key

    def _get_topk_paged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        # NOTE(dark): blocksize = 64 is hardcoded in deep_gemm
        assert page_size == 64, "only support page size 64"

        # NOTE(dark): this support extend/decode/decode+graph
        block_tables = metadata.get_page_table_64()

        max_seq_len = block_tables.shape[1] * page_size
        kv_cache_fp8 = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=layer_id
        )

        blocksize = page_size
        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
        ):
            seqlens_32 = metadata.get_seqlens_expanded()
        else:
            seqlens_32 = metadata.get_seqlens_int32()
        # NOTE(dark): 132 is SM count on H200/B200, not magic number
        schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
            seqlens_32, blocksize, self.sm_count
        )

        assert len(q_fp8.shape) == 3
        q_fp8 = q_fp8.unsqueeze(1)  # the next_n dim is 1 now
        assert len(kv_cache_fp8.shape) == 2
        block_kv = 64
        num_heads_kv = 1
        head_dim_with_sf = 132
        kv_cache_fp8 = kv_cache_fp8.view(
            kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)

        logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            seqlens_32,
            block_tables,
            schedule_metadata,
            max_seq_len,
            clean_logits=False,
        )

        # NOTE(dark): logits should be cleaned in topk_transform
        topk_result = metadata.topk_transform(logits, self.index_topk)
        return topk_result

    def _should_chunk_mqa_logits(
        self, num_q: int, num_k: int, device: torch.device
    ) -> Tuple[bool, int]:
        """
        Detect whether we need to chunk the MQA logits computation to avoid OOM
        Return: (need_chunk, free_mem)
        """
        # Quick static check for normal batches
        if num_q * num_k < 8_000_000:  # 8M elements ≈ 32MB logits
            return False, 0

        free_mem, total_mem = torch.cuda.mem_get_info(device)
        bytes_per_elem = 4  # float32
        logits_bytes = num_q * num_k * bytes_per_elem

        # Logits should not exceed 50% of free memory or 30% of total memory
        need_chunk = (logits_bytes * 2 > free_mem) or (logits_bytes > total_mem * 0.3)
        return need_chunk, free_mem

    def _get_topk_ragged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        assert forward_batch.forward_mode.is_extend_without_speculative()

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)
        k_fp8_list = []
        k_scale_list = []
        ks_list = []
        ke_list = []
        # Token-to-batch mapping for PAGED chunk alignment
        token_to_batch_idx: List[int] = []

        q_offset = 0
        k_offset = 0

        seq_lens_expanded = metadata.get_seqlens_expanded()
        block_tables = metadata.get_page_table_64()

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )

        for i in range(forward_batch.batch_size):
            seq_len = forward_batch.seq_lens_cpu[i].item()
            assert isinstance(seq_len, int)
            # Use fused Triton kernel to get both K and scale in a single call
            k_fp8, k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_buffer(
                layer_id,
                seq_len,
                block_tables[i],
            )
            extend_seq_len = forward_batch.extend_seq_lens_cpu[i]
            ks = torch.full(
                (extend_seq_len,), k_offset, dtype=torch.int32, device="cuda"
            )
            ke = ks + seq_lens_expanded[q_offset : q_offset + extend_seq_len]
            k_fp8_list.append(k_fp8)
            k_scale_list.append(k_scale)
            ks_list.append(ks)
            ke_list.append(ke)

            token_to_batch_idx.extend([i] * extend_seq_len)
            q_offset += extend_seq_len
            k_offset += seq_len

        k_fp8 = torch.cat(k_fp8_list, dim=0).view(torch.float8_e4m3fn)
        k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
        kv_fp8 = (k_fp8, k_scale)
        ks = torch.cat(ks_list, dim=0)
        ke = torch.cat(ke_list, dim=0)

        # Suppose there are two requests, with extend_seq_len = [3, 2]
        # and seq_lens = [10, 4]
        # The logits matrix looks like this, with * representing the valid logits
        # and - representing the invalid logits:
        #
        #  ********--|----
        #  *********-|----
        #  **********|----
        #  ----------|***-
        #  ----------|****
        #
        # ks = [0, 0, 0, 10, 10]
        # ke = [8, 9, 10, 13, 14]

        token_nums, _, _ = q_fp8.shape
        device = q_fp8.device

        # Check if we need to chunk to avoid OOM
        need_chunk, free_mem = self._should_chunk_mqa_logits(q_offset, k_offset, device)

        if not need_chunk:
            logits = deep_gemm.fp8_mqa_logits(
                q_fp8[:q_offset],
                kv_fp8,
                weights[:q_offset],
                ks,
                ke,
                clean_logits=False,
            )
            assert logits.shape[0] == len(seq_lens_expanded)
            assert logits.shape[1] == k_offset

            raw_topk_result = metadata.topk_transform(logits, self.index_topk, ks=ks)
            topk_result = torch.full(
                (token_nums, self.index_topk),
                -1,
                device=device,
                dtype=torch.int32,
            )
            topk_result[:q_offset] = raw_topk_result
            return topk_result

        # Chunk path
        bytes_per_elem = 4  # float32
        bytes_per_row = k_offset * bytes_per_elem
        # Reserve 50% of free memory for logits
        max_rows = max(1, int((free_mem * 0.5) // max(bytes_per_row, 1)))
        max_rows = min(max_rows, q_offset)

        global_topk_offset = metadata.attn_metadata.topk_indices_offset

        assert (
            seq_lens_expanded.shape[0] == q_offset
        ), f"seq_lens_expanded length mismatch: {seq_lens_expanded.shape[0]} != {q_offset}"
        if global_topk_offset is not None:
            assert (
                global_topk_offset.shape[0] >= q_offset
            ), f"topk_indices_offset too short: {global_topk_offset.shape[0]} < {q_offset}"

        topk_result = torch.full(
            (token_nums, self.index_topk), -1, device=device, dtype=torch.int32
        )

        # Only materialize batch index tensor when PAGED path needs it
        token_to_batch_idx_tensor = None
        if global_topk_offset is None:
            token_to_batch_idx_tensor = torch.tensor(
                token_to_batch_idx, dtype=torch.long, device=device
            )

        start = 0
        while start < q_offset:
            end = min(start + max_rows, q_offset)

            logits_chunk = deep_gemm.fp8_mqa_logits(
                q_fp8[start:end],
                kv_fp8,
                weights[start:end],
                ks[start:end],
                ke[start:end],
                clean_logits=False,
            )

            lengths_chunk = seq_lens_expanded[start:end]

            # RAGGED: use global offset; PAGED: construct local cu_seqlens_q per chunk
            if global_topk_offset is not None:
                # RAGGED path
                topk_offset_chunk = global_topk_offset[start:end]
                cu_seqlens_q_chunk = None
                batch_idx_chunk = None
            else:
                # PAGED path: treat each token as a length-1 sequence
                topk_offset_chunk = None
                B_chunk = logits_chunk.shape[0]
                cu_seqlens_q_chunk = torch.ones(
                    B_chunk, dtype=torch.int32, device=device
                )
                batch_idx_chunk = token_to_batch_idx_tensor[start:end]

            raw_topk_chunk = metadata.topk_transform(
                logits_chunk,
                self.index_topk,
                ks=ks[start:end],
                cu_seqlens_q=cu_seqlens_q_chunk,
                ke_offset=lengths_chunk,
                batch_idx_list=batch_idx_chunk,
                topk_indices_offset_override=topk_offset_chunk,
            )
            topk_result[start:end] = raw_topk_chunk
            start = end

        return topk_result

    def _forward_cuda_k_only(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        act_quant,
        enable_dual_stream: bool,
        metadata: BaseIndexerMetadata,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        assert forward_batch.forward_mode.is_extend_without_speculative()

        # Fast path: only compute and store k cache, skip all q and weights ops
        key = self._get_k_bf16(x, positions, enable_dual_stream)
        k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)

        if not forward_batch.out_cache_loc.is_contiguous():
            forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()
        forward_batch.token_to_kv_pool.set_index_k_scale_buffer(
            layer_id=layer_id,
            loc=forward_batch.out_cache_loc,
            index_k=k_fp8,
            index_k_scale=k_scale,
        )

        # MHA doesn't need topk_indices
        if not return_indices:
            return None

        # MLA: use dummy logits with topk kernel's fast path to generate indices
        # When length <= 2048, naive_topk_cuda directly generates [0,1,...,length-1,-1,...]
        seq_lens_expanded = metadata.get_seqlens_expanded()
        dummy_logits = torch.zeros(
            seq_lens_expanded.shape[0],
            self.index_topk,
            dtype=torch.float32,
            device=x.device,
        )
        return metadata.topk_transform(dummy_logits, self.index_topk)

    def _get_topk_ragged_with_cp(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
        kv_len: int,
        actual_seq_q: int,
        cp_index: List[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)
        k_fp8_list = []
        k_scale_list = []
        ks_list = []
        ke_offset_list = []
        offset = 0
        actual_seq_q_list = []
        batch_idx_list = []

        block_tables = metadata.get_page_table_64()

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        if cp_index is not None:
            # TODO Multi-batch support has accuracy issues
            for batch_idx, start_seq_position, end_seq_position in cp_index:
                pre_chunk_offset = (
                    forward_batch.seq_lens_cpu[batch_idx].item()
                    - forward_batch.extend_seq_lens_cpu[batch_idx]
                )
                start_seq_position += pre_chunk_offset
                end_seq_position += pre_chunk_offset
                if offset == 0 and batch_idx != 0:
                    offset += forward_batch.extend_seq_lens_cpu[batch_idx - 1]
                k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
                    layer_id,
                    end_seq_position,
                    block_tables[batch_idx],
                )
                k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
                    layer_id,
                    end_seq_position,
                    block_tables[batch_idx],
                )

                extend_seq_len = end_seq_position - start_seq_position
                ks = torch.full(
                    (extend_seq_len,), offset, dtype=torch.int32, device="cuda"
                )
                k_fp8_list.append(k_fp8)
                k_scale_list.append(k_scale)
                ks_list.append(ks)
                ke_offset = torch.arange(
                    start_seq_position + 1,
                    end_seq_position + 1,
                    dtype=torch.int32,
                    device="cuda",
                )
                ke_offset_list.append(ke_offset)
                actual_seq_q = torch.tensor(
                    [extend_seq_len], dtype=torch.int32, device="cuda"
                )
                actual_seq_q_list.append(actual_seq_q)
                batch_idx_list.append(batch_idx)

            k_fp8 = torch.cat(k_fp8_list, dim=0).view(torch.float8_e4m3fn)
            k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
            kv_fp8 = (k_fp8, k_scale)
            ks = torch.cat(ks_list, dim=0)
            ke_offset = torch.cat(ke_offset_list, dim=0)
            ke = ks + ke_offset
            actual_seq_q = torch.cat(actual_seq_q_list, dim=0)
            logits = deep_gemm.fp8_mqa_logits(
                q_fp8,
                kv_fp8,
                weights,
                ks,
                ke,
                clean_logits=False,
            )
            topk_result = metadata.topk_transform(
                logits,
                self.index_topk,
                ks=ks,
                cu_seqlens_q=actual_seq_q,
                ke_offset=ke_offset,
                batch_idx_list=batch_idx_list,
            )
        else:
            kv_len = (
                forward_batch.seq_lens_cpu[0].item()
                - forward_batch.extend_seq_lens_cpu[0]
                + kv_len
            )
            k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
                layer_id,
                kv_len,
                block_tables[0],
            )
            k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
                layer_id,
                kv_len,
                block_tables[0],
            )

            k_fp8 = k_fp8.view(torch.float8_e4m3fn)
            k_scale = k_scale.view(torch.float32).squeeze(-1)
            kv_fp8 = (k_fp8, k_scale)
            ks = torch.full((actual_seq_q,), offset, dtype=torch.int32, device="cuda")
            ke_offset = torch.arange(
                (kv_len - actual_seq_q) + 1,
                kv_len + 1,
                dtype=torch.int32,
                device="cuda",
            )
            ke = ks + ke_offset

            logits = deep_gemm.fp8_mqa_logits(
                q_fp8,
                kv_fp8,
                weights,
                ks,
                ke,
                clean_logits=False,
            )
            actual_seq_q = torch.tensor([actual_seq_q], dtype=torch.int32).to(
                device="cuda", non_blocking=True
            )
            topk_result = metadata.topk_transform(
                logits,
                self.index_topk,
                ks=ks,
                cu_seqlens_q=actual_seq_q,
                ke_offset=ke_offset,
            )

        return topk_result

    def forward_indexer(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        forward_batch: ForwardBatch,
        topk: int,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        if not is_npu():
            from sglang.srt.layers.attention.nsa.tilelang_kernel import fp8_index

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"

        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)

        # logits = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)
        k_fp8_list = []
        k_scale_list = []

        topk_indices_list = []

        block_tables = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :
        ]
        strided_indices = torch.arange(
            0, block_tables.shape[-1], page_size, device="cuda"
        )
        block_tables = block_tables[:, strided_indices] // page_size

        q_len_start = 0

        for i in range(forward_batch.batch_size):
            seq_len = forward_batch.seq_lens[i].item()
            q_len = (
                forward_batch.extend_seq_lens_cpu[i]
                if forward_batch.forward_mode.is_extend()
                else 1
            )
            q_len_end = q_len_start + q_len

            q_fp8_partial = q_fp8[q_len_start:q_len_end]
            q_fp8_partial = q_fp8_partial.unsqueeze(0).contiguous()

            weights_partial = weights[q_len_start:q_len_end]
            weights_partial = weights_partial.squeeze(-1).unsqueeze(0).contiguous()

            k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
                layer_id,
                seq_len,
                block_tables[i],
            )
            k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
                layer_id,
                seq_len,
                block_tables[i],
            )

            k_fp8 = k_fp8.view(torch.float8_e4m3fn).unsqueeze(0).contiguous()
            k_scale = k_scale.view(torch.float32).squeeze(-1).unsqueeze(0).contiguous()

            index_score = fp8_index(
                q_fp8_partial,
                weights_partial,
                k_fp8,
                k_scale,
            )
            end_pos = seq_len
            topk_indices = index_score.topk(min(topk, end_pos), dim=-1)[1].squeeze(0)

            pad_len = ceil_align(topk_indices.shape[-1], 2048) - topk_indices.shape[-1]
            topk_indices = torch.nn.functional.pad(
                topk_indices, (0, pad_len), "constant", -1
            )

            topk_indices_list.append(topk_indices)

            q_len_start = q_len_end

        topk_indices = torch.cat(topk_indices_list, dim=0)
        return topk_indices

    def forward_cuda(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        if is_hip():
            from sglang.srt.layers.attention.nsa.tilelang_kernel import act_quant
        elif not is_npu():
            from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        metadata = forward_batch.attn_backend.get_indexer_metadata(
            layer_id, forward_batch
        )

        enable_dual_stream = (
            NSA_DUAL_STREAM
            and self.alt_stream is not None
            and get_is_capture_mode()
            and q_lora.shape[0] > 0
            and q_lora.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
        )

        # skip NSA if attention backend choose to skip this batch
        if metadata is None:
            return None

        # Determine if should skip topk based on sequence length
        # We can only skip the logits computation if cuda graph is not involved
        skip_logits_computation = False
        if forward_batch.forward_mode.is_extend_without_speculative():
            if forward_batch.seq_lens_cpu is not None:
                max_kv_len = forward_batch.seq_lens_cpu.max().item()
                skip_logits_computation = max_kv_len <= self.index_topk

        # Optimization: fast path when skipping topk computation
        if skip_logits_computation and (not self.nsa_enable_prefill_cp):
            return self._forward_cuda_k_only(
                x,
                positions,
                forward_batch,
                layer_id,
                act_quant,
                enable_dual_stream,
                metadata,
                return_indices,
            )

        query, key = self._get_q_k_bf16(
            q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
        )

        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
            with torch.cuda.stream(self.alt_stream):
                k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)
            current_stream.wait_stream(self.alt_stream)
        else:
            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
            k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)

        # k_fp8: (seq_len, head_dim) fp8_e4m3fn
        # k_buffer: (num_total_tokens + page_size, head_dim) fp8_e4m3fn
        # k_scale: (seq_len, head_dim // block_size = 1) fp8_e4m3fn
        # k_scale_cache: (num_total_tokens + page_size, head_dim // block_size = 1) fp8_e4m3fn
        if not forward_batch.out_cache_loc.is_contiguous():
            forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()
        forward_batch.token_to_kv_pool.set_index_k_scale_buffer(
            layer_id=layer_id,
            loc=forward_batch.out_cache_loc,
            index_k=k_fp8,
            index_k_scale=k_scale,
        )

        weights = self._get_logits_head_gate(x, q_scale)

        if is_cuda():
            assert forward_batch.seq_lens_cpu is not None
            if len(forward_batch.seq_lens_cpu) == 0:
                # this seems b/c max-pad, no worries?
                # if x.shape[0] != 0:
                #     print(
                #         "HACK: seq_lens empty but x not empty, hackily return all-invalid topk_result"
                #     )
                return torch.full(
                    (x.shape[0], self.index_topk), -1, dtype=torch.int, device="cuda"
                )

            if (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend()
            ):
                topk_result = self._get_topk_paged(
                    forward_batch, layer_id, q_fp8, weights, metadata
                )
            else:
                if (
                    forward_batch.nsa_cp_metadata is not None
                    and self.nsa_enable_prefill_cp
                ):
                    kv_len_prev = forward_batch.nsa_cp_metadata.kv_len_prev
                    kv_len_next = forward_batch.nsa_cp_metadata.kv_len_next
                    actual_seq_q_prev = forward_batch.nsa_cp_metadata.actual_seq_q_prev
                    actual_seq_q_next = forward_batch.nsa_cp_metadata.actual_seq_q_next

                    # TODO support mutil-batch
                    # cp_batch_seq_index_prev = forward_batch.nsa_cp_metadata["cp_batch_seq_index_prev"]
                    # cp_batch_seq_index_next = forward_batch.nsa_cp_metadata["cp_batch_seq_index_next"]
                    # TODO prev, next, combined into a single call
                    q_fp8_prev, q_fp8_next = torch.split(
                        q_fp8, (q_fp8.shape[0] + 1) // 2, dim=0
                    )
                    weights_prev, weights_next = torch.split(
                        weights, (weights.shape[0] + 1) // 2, dim=0
                    )
                    topk_result_prev = self._get_topk_ragged_with_cp(
                        forward_batch,
                        layer_id,
                        q_fp8_prev,
                        weights_prev,
                        metadata,
                        kv_len_prev,
                        actual_seq_q_prev,
                    )

                    topk_result_next = self._get_topk_ragged_with_cp(
                        forward_batch,
                        layer_id,
                        q_fp8_next,
                        weights_next,
                        metadata,
                        kv_len_next,
                        actual_seq_q_next,
                    )
                    return torch.cat([topk_result_prev, topk_result_next], dim=0)
                else:
                    topk_result = self._get_topk_ragged(
                        forward_batch, layer_id, q_fp8, weights, metadata
                    )
        else:
            topk_result = self.forward_indexer(
                q_fp8.contiguous(),
                weights,
                forward_batch,
                topk=self.index_topk,
                layer_id=layer_id,
            )
        return topk_result

    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> torch.Tensor:
        if forward_batch.attn_backend.forward_metadata.seq_lens_cpu_int is None:
            actual_seq_lengths_kv = forward_batch.attn_backend.forward_metadata.seq_lens
        else:
            actual_seq_lengths_kv = (
                forward_batch.attn_backend.forward_metadata.seq_lens_cpu_int
            )
        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend()
        )

        cos_sin = self.rotary_emb.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
        sin = sin.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)

        bs = x.shape[0]
        q = self.wq_b(q_lora)[0]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
        q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
        q_pe, q_nope = torch.split(
            q,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )  # [bs, 64, 64 + 64]

        q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
        q_pe = torch.ops.npu.npu_rotary_mul(q_pe, cos, sin).view(
            bs, self.n_heads, self.rope_head_dim
        )  # [bs, n, d]
        q = torch.cat([q_pe, q_nope], dim=-1)

        k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
        k = self.k_norm(k_proj)
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

        if (
            is_prefill
            and self.nsa_enable_prefill_cp
            and forward_batch.nsa_cp_metadata is not None
        ):
            k = cp_all_gather_rerange_output(
                k.contiguous().view(-1, self.head_dim),
                self.cp_size,
                forward_batch,
                torch.npu.current_stream(),
            )

        forward_batch.token_to_kv_pool.set_index_k_buffer(
            layer_id, forward_batch.out_cache_loc, k
        )
        if is_prefill:
            if self.nsa_enable_prefill_cp and forward_batch.nsa_cp_metadata is not None:
                forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q = (
                    forward_batch.nsa_cp_metadata.actual_seq_q_prev_tensor,
                    forward_batch.nsa_cp_metadata.actual_seq_q_next_tensor,
                )
                forward_batch.attn_backend.forward_metadata.actual_seq_lengths_kv = (
                    forward_batch.nsa_cp_metadata.kv_len_prev_tensor,
                    forward_batch.nsa_cp_metadata.kv_len_next_tensor,
                )
                actual_seq_lengths_q = (
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q
                )
                actual_seq_lengths_kv = (
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_kv
                )
            else:
                actual_seq_lengths_kv = forward_batch.seq_lens
                actual_seq_lengths_q = forward_batch.seq_lens.cumsum(dim=0)
        else:
            if forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q is None:
                if (
                    forward_batch.forward_mode.is_draft_extend_v2()
                    or forward_batch.forward_mode.is_target_verify()
                    or forward_batch.forward_mode.is_draft_extend()
                ):
                    num_draft_tokens = (
                        forward_batch.attn_backend.speculative_num_draft_tokens
                    )
                    actual_seq_lengths_q = torch.arange(
                        num_draft_tokens,
                        num_draft_tokens + bs,
                        num_draft_tokens,
                        dtype=torch.int32,
                        device=k.device,
                    )
                else:
                    actual_seq_lengths_q = torch.tensor(
                        [1 + i * 1 for i in range(bs)],
                        dtype=torch.int32,
                        device=k.device,
                    )
            else:
                actual_seq_lengths_q = (
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q
                )

        past_key_states = forward_batch.token_to_kv_pool.get_index_k_buffer(layer_id)

        x = x.view(-1, self.hidden_size)
        weights = self.weights_proj(x.float())[0].to(torch.bfloat16)
        block_table = forward_batch.attn_backend.forward_metadata.block_tables
        if (
            is_prefill
            and self.nsa_enable_prefill_cp
            and forward_batch.nsa_cp_metadata is not None
        ):
            block_table = block_table[: actual_seq_lengths_q[0].numel()]
            topk_indices = self.do_npu_cp_balance_indexer(
                q.view(-1, self.n_heads, self.head_dim),
                past_key_states,
                weights,
                actual_seq_lengths_q,
                actual_seq_lengths_kv,
                block_table,
            )
        else:
            block_table = (
                block_table[: actual_seq_lengths_q.size()[0]]
                if is_prefill
                else block_table
            )

            topk_indices = torch.ops.custom.npu_lightning_indexer(
                query=q.view(-1, self.n_heads, self.head_dim),
                key=past_key_states,
                weights=weights,
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

        return topk_indices

    def do_npu_cp_balance_indexer(
        self,
        q,
        past_key_states,
        indexer_weights,
        actual_seq_lengths_q,
        actual_seq_lengths_kv,
        block_table,
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

        topk_indices_prev = torch.ops.custom.npu_lightning_indexer(
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
        topk_indices_next = torch.ops.custom.npu_lightning_indexer(
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
        return topk_indices_prev, topk_indices_next
