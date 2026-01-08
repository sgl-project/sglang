from __future__ import annotations

"""
end to end attention solution with aiter kernels
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, List

import torch
import triton

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import (
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

try:
    from aiter import (
        flash_attn_varlen_func,
        flash_attn_varlen_fp8_pertensor_func,
        dtypes,
        get_pa_metadata_info_v1,
        get_pa_metadata_v1,
        pa_persistent_fwd,
    )
    from aiter.mla import mla_decode_fwd, mla_prefill_fwd
except ImportError:
    print(
        "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
    )

from sglang.srt.configs.model_config import AttentionArch


@dataclass
class ForwardMetadata:
    # kv_indptr and kv_indices are only used in MLA mode, optional for non-MLA mode
    kv_indptr: Optional[torch.Tensor]
    kv_indices: Optional[torch.Tensor]
    qo_indptr: Optional[torch.Tensor]
    kv_last_page_len: Optional[torch.Tensor]
    max_q_len: Optional[int]
    max_kv_len: Optional[int]
    page_table: Optional[torch.Tensor]
    kv_lens: Optional[torch.Tensor]
    # PA metadata for pa_persistent_fwd (only used in decode mode, non-MLA)
    pa_metadata_qo_indptr: Optional[torch.Tensor] = None
    pa_metadata_pages_kv_indptr: Optional[torch.Tensor] = None
    pa_metadata_kv_indices: Optional[torch.Tensor] = None
    pa_metadata_context_lens: Optional[torch.Tensor] = None
    pa_metadata_max_qlen: Optional[int] = None
    pa_metadata_tp_q_head_num: Optional[int] = None


_AITER_PARTITION_SIZE_ROCM = 256


class AiterAttnBackend(AttentionBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        # Lazy import to avoid the initialization of cuda context
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd,
        )

        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)

        self.device = model_runner.device
        self.is_multimodal = model_runner.model_config.is_multimodal
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.page_size = model_runner.page_size
        self.head_dim = model_runner.model_config.head_dim
        mapping = getattr(
            model_runner.token_to_kv_pool, "full_attention_layer_id_mapping", None
        )

        if isinstance(mapping, dict) and mapping:
            first_full_attn_id = next(iter(mapping.keys()))
        else:
            first_full_attn_id = 0
        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(
            first_full_attn_id
        ).shape[-1]
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.q_dtype = model_runner.dtype  # Save q dtype for pa_metadata building

        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA

        # Parse constants
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill

        max_bs = model_runner.req_to_token_pool.size

        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        self.qo_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )
        # Pre-initialized qo_indptr for pa_persistent_fwd decode mode: [0, 1, 2, ..., max_bs]
        # In decode mode, each sequence has 1 token, so this is always [0, 1, 2, ..., batch_size]
        self.pa_decode_qo_indptr = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=model_runner.device
        )
        self.seq_lens = torch.zeros(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        self.page_table = torch.zeros(
            (max_bs, self.max_context_len // self.page_size), dtype=torch.int32, device=model_runner.device
        )

        # Create prefill indices updater
        if not skip_prefill:
            self.indices_updater_prefill = AiterIndicesUpdaterPrefill(
                model_runner, self
            )
            if self.use_mla:
                self.mla_indices_updater_prefill = AiterMlaIndicesUpdaterPrefill(
                    model_runner, self
                )

        # aiter kernel related initialization
        self.max_num_partitions = (
            self.max_context_len + _AITER_PARTITION_SIZE_ROCM - 1
        ) // _AITER_PARTITION_SIZE_ROCM

        nbyes_per_qo_elem = torch.finfo(torch.float32).bits // 8

        if not self.use_mla:
            self.workspace_buffer = torch.empty(
                (max_bs * self.num_head * self.max_num_partitions * self.head_dim)
                * nbyes_per_qo_elem
                + 2 * (max_bs * self.num_head * self.max_num_partitions) * 4,
                dtype=torch.uint8,
                device=self.device,
            )
            # Pre-allocate buffers for pa_persistent_fwd (used in both CUDA graph and non-CUDA graph modes)
            max_num_blocks_per_seq = (self.max_context_len + self.page_size - 1) // self.page_size
            max_total_blocks = max_bs * max_num_blocks_per_seq
            self.pa_kv_indices = torch.zeros(
                max_total_blocks, dtype=torch.int32, device=self.device
            )
            # Pre-allocate pa_kv_indptr buffer (similar to self.kv_indptr, but dedicated for pa_persistent_fwd)
            self.pa_kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=self.device
            )
            # Pre-initialized batch indices [0, 1, 2, ..., max_bs-1] for Triton kernel
            self.pa_batch_indices = torch.arange(
                0, max_bs, dtype=torch.int32, device=self.device
            )

        self.scale = float(1.0 / (self.head_dim**0.5))
        self.k_scale = self.v_scale = torch.tensor([1.0], dtype=torch.float32).to(
            self.device
        )
        self.fp8_scale = 1.0  # Store as scalar to avoid sync in kernel calls

        self.logits_soft_cap = 0.0

        self.forward_metadata: ForwardMetadata = None

        if self.use_mla:
            self.qo_indptr_ = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

            self.enable_dp_attention = is_dp_attention_enabled()
        
        self.pa_metadata_buffers = None
        
        k_buffer, _ = model_runner.token_to_kv_pool.get_kv_buffer(first_full_attn_id)
        num_slots, num_kv_heads, _ = k_buffer.shape
        block_size = self.page_size
        num_blocks = num_slots // block_size
        max_total_tokens = num_blocks * block_size
        self.k_qscale = torch.ones(
            num_kv_heads, max_total_tokens, dtype=torch.float32, device=self.device
        )
        self.v_qscale = torch.ones(
            num_kv_heads, max_total_tokens, dtype=torch.float32, device=self.device
        )
           

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for triton attention backend."""

        bs = forward_batch.batch_size
        kv_indptr = self.kv_indptr
        spec_info = forward_batch.spec_info
        qo_indptr = None
        kv_last_page_len = None
        max_q_len = None
        page_table = None

        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None:
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = torch.empty(
                    forward_batch.seq_lens_sum, dtype=torch.int32, device=self.device
                )
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                bs = kv_indptr.shape[0] - 1

            if self.use_mla:
                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(self.kv_last_page_len[:bs], dim=0)
                kv_last_page_len = self.kv_last_page_len[:bs]
                max_q_len = 1
            page_table = forward_batch.req_to_token_pool.req_to_token[forward_batch.req_pool_indices, :]

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                None,
                page_table,
                forward_batch.seq_lens,
            )
            
            # Build pa_metadata for pa_persistent_fwd (only for non-MLA decode mode)
            if not self.use_mla:
                self._build_pa_metadata_for_decode(bs, tp_q_head_num=self.num_head)

        elif forward_batch.forward_mode.is_draft_extend():
            if self.use_mla:
                kv_indices, kv_indptr, qo_indptr, _ = (
                    spec_info.generate_attn_arg_prefill(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.seq_lens_sum,
                        self.req_to_token,
                    )
                )
                self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                self.kv_last_page_len[:bs],
                max(forward_batch.extend_seq_lens_cpu),
                forward_batch.seq_lens_cpu.max().item(),
                None,
                None,
            )
            else:
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    prefix_lens=None,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=forward_batch.spec_info,
                )
                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    self.indices_updater_prefill.max_q_len,
                    self.indices_updater_prefill.max_kv_len,
                    None,
                    None,
                )
        elif forward_batch.forward_mode.is_target_verify():
            if self.use_mla:
                draft_num = spec_info.draft_token_num
                kv_lens = forward_batch.seq_lens + draft_num
                kv_lens_sum = forward_batch.seq_lens_sum + draft_num * bs
                device = forward_batch.seq_lens.device

                qo_indptr = torch.arange(
                    0,
                    (1 + bs) * draft_num,
                    step=draft_num,
                    dtype=torch.int32,
                    device=device,
                )
                kv_indptr = self.kv_indptr
                kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = torch.empty(
                    kv_lens_sum,
                    dtype=torch.int32,
                    device=device,
                )
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    kv_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    self.kv_last_page_len[:bs],
                    draft_num,
                    None,
                    None,
                    None,
                )
            else:
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    prefix_lens=None,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=forward_batch.spec_info,
                )
                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    self.indices_updater_prefill.max_q_len,
                    self.indices_updater_prefill.max_kv_len,
                    None,
                    None,
                )
        else:
            prefix_lens = forward_batch.extend_prefix_lens
            prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu

            if self.use_mla:
                self.mla_indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    forward_batch.extend_seq_lens,
                    forward_batch.extend_seq_lens.max().item(),
                    forward_batch.seq_lens.max().item(),
                    spec_info=None,
                )

                kv_indices = self.mla_indices_updater_prefill.kv_indices

                self.forward_metadata = ForwardMetadata(
                    self.mla_indices_updater_prefill.kv_indptr,
                    kv_indices,
                    self.mla_indices_updater_prefill.qo_indptr,
                    self.kv_last_page_len[:bs],
                    self.mla_indices_updater_prefill.max_q_len,
                    self.mla_indices_updater_prefill.max_kv_len,
                    None,
                    None,
                )
            else:
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    prefix_lens,
                    prefix_lens_cpu=prefix_lens_cpu,
                    seq_lens_cpu=forward_batch.seq_lens_cpu,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=None,
                )
                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    self.indices_updater_prefill.max_q_len,
                    self.indices_updater_prefill.max_kv_len,
                    None,
                    forward_batch.seq_lens,
                )  
        if self.page_size > 1 and self.forward_metadata.page_table is not None:
            strided_indices = torch.arange(
                0, self.forward_metadata.page_table.shape[1], self.page_size, device=self.device
            )
            self.forward_metadata.page_table = (
                self.forward_metadata.page_table[:, strided_indices] // self.page_size
            )

    def _allocate_pa_metadata_buffers(
        self,
        work_metadata_ptrs_size,
        work_metadata_ptrs_type,
        work_indptr_size,
        work_indptr_type,
        work_info_size,
        work_info_type,
        reduce_indptr_size,
        reduce_indptr_type,
        reduce_final_map_size,
        reduce_final_map_type,
        reduce_partial_map_size,
        reduce_partial_map_type,
    ):
        """Allocate or reuse pa_metadata buffers."""
        if self.pa_metadata_buffers is None:
            self.pa_metadata_buffers = {}
        
        def _get_size_val(size):
            return size[0] if isinstance(size, tuple) else size
        
        # Allocate work_metadata_ptrs
        size_val = _get_size_val(work_metadata_ptrs_size)
        if ("work_metadata_ptrs" not in self.pa_metadata_buffers or 
            self.pa_metadata_buffers["work_metadata_ptrs"].shape[0] < size_val):
            self.pa_metadata_buffers["work_metadata_ptrs"] = torch.empty(
                work_metadata_ptrs_size, dtype=work_metadata_ptrs_type, device=self.device
            )
        
        # Allocate work_indptr
        size_val = _get_size_val(work_indptr_size)
        if ("work_indptr" not in self.pa_metadata_buffers or 
            self.pa_metadata_buffers["work_indptr"].shape[0] < size_val):
            self.pa_metadata_buffers["work_indptr"] = torch.zeros(
                work_indptr_size, dtype=work_indptr_type, device=self.device
            )
        else:
            self.pa_metadata_buffers["work_indptr"].zero_()
        
        # Allocate work_info
        size_val = _get_size_val(work_info_size)
        if ("work_info" not in self.pa_metadata_buffers or 
            len(self.pa_metadata_buffers["work_info"].shape) < len(work_info_size) or
            self.pa_metadata_buffers["work_info"].shape[0] < size_val):
            self.pa_metadata_buffers["work_info"] = torch.zeros(
                work_info_size, dtype=work_info_type, device=self.device
            )
        else:
            self.pa_metadata_buffers["work_info"].zero_()
        
        # Allocate reduce_indptr
        size_val = _get_size_val(reduce_indptr_size)
        if ("reduce_indptr" not in self.pa_metadata_buffers or 
            self.pa_metadata_buffers["reduce_indptr"].shape[0] < size_val):
            self.pa_metadata_buffers["reduce_indptr"] = torch.zeros(
                reduce_indptr_size, dtype=reduce_indptr_type, device=self.device
            )
        else:
            self.pa_metadata_buffers["reduce_indptr"].zero_()
        
        # Allocate reduce_final_map
        size_val = _get_size_val(reduce_final_map_size)
        if ("reduce_final_map" not in self.pa_metadata_buffers or 
            len(self.pa_metadata_buffers["reduce_final_map"].shape) < len(reduce_final_map_size) or
            self.pa_metadata_buffers["reduce_final_map"].shape[0] < size_val):
            self.pa_metadata_buffers["reduce_final_map"] = torch.zeros(
                reduce_final_map_size, dtype=reduce_final_map_type, device=self.device
            )
        else:
            self.pa_metadata_buffers["reduce_final_map"].zero_()
        
        # Allocate reduce_partial_map
        reduce_partial_map_size_val = reduce_partial_map_size if isinstance(reduce_partial_map_size, int) else reduce_partial_map_size[0]
        if ("reduce_partial_map" not in self.pa_metadata_buffers or 
            self.pa_metadata_buffers["reduce_partial_map"].shape[0] < reduce_partial_map_size_val):
            self.pa_metadata_buffers["reduce_partial_map"] = torch.zeros(
                reduce_partial_map_size, dtype=reduce_partial_map_type, device=self.device
            )
        else:
            self.pa_metadata_buffers["reduce_partial_map"].zero_()

    def _build_pa_metadata_for_decode(
        self, 
        batch_size: int, 
        tp_q_head_num: Optional[int] = None,
    ):
        """Build pa_metadata buffers for pa_persistent_fwd in decode mode.
        
        This method prepares all metadata buffers needed for pa_persistent_fwd kernel.
        The metadata can be reused across multiple layers in the same forward pass.
        
        Args:
            batch_size: Batch size for the current forward pass
            tp_q_head_num: Number of Q heads per TP rank. If None, uses self.num_head.
        """
        
        
        max_qlen = 1
        
        # Use provided tp_q_head_num or default to self.num_head
        if tp_q_head_num is None:
            tp_q_head_num = self.num_head
        
        kv_dtype_for_metadata = dtypes.fp8
        (
            (work_metadata_ptrs_size, work_metadata_ptrs_type),
            (work_indptr_size, work_indptr_type),
            (work_info_size, work_info_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_pa_metadata_info_v1(
            batch_size,
            max_qlen,
            tp_q_head_num,
            self.q_dtype,
            kv_dtype_for_metadata,
            is_sparse=0,  # 0 for non-sparse attention
            fast_mode=True,
        )

        
        # Allocate metadata buffers with reuse optimization for multi-layer forward passes
        self._allocate_pa_metadata_buffers(
            work_metadata_ptrs_size,
            work_metadata_ptrs_type,
            work_indptr_size,
            work_indptr_type,
            work_info_size,
            work_info_type,
            reduce_indptr_size,
            reduce_indptr_type,
            reduce_final_map_size,
            reduce_final_map_type,
            reduce_partial_map_size,
            reduce_partial_map_type,
        )
        
        # Get qo_indptr for decode mode (each sequence has 1 token)
        # pa_decode_qo_indptr is pre-initialized to [0, 1, 2, ..., max_bs], just take the slice
        qo_indptr = self.pa_decode_qo_indptr[: batch_size + 1]
        
        # Get context_lens (kv_lens is always set before calling _build_pa_metadata_for_decode)
        # Note: kv_lens comes from self.seq_lens which is already int32
        context_lens = self.forward_metadata.kv_lens
        
        kernel_block_size = self.page_size
        num_blocks_per_seq = (context_lens + kernel_block_size - 1) // kernel_block_size
        # Use dedicated pa_kv_indptr buffer (similar to self.kv_indptr, but for pa_persistent_fwd)
        pages_kv_indptr = self.pa_kv_indptr[: batch_size + 1]
        pages_kv_indptr[1 : batch_size + 1] = torch.cumsum(num_blocks_per_seq, dim=0)
        
        # Convert page_table to kv_indices (block indices) using Triton kernel to avoid sync
        # page_table shape: [batch_size, max_num_blocks_per_seq]
        # Note: page_table comes from self.page_table which is already int32 and always set before this call
        page_table = self.forward_metadata.page_table
        
        # Use Triton kernel to gather kv_indices from page_table (avoids high-level indexing sync)
        create_flashinfer_kv_indices_triton[(batch_size,)](
            page_table,
            self.pa_batch_indices[:batch_size],  # [0, 1, 2, ..., batch_size-1]
            num_blocks_per_seq,
            pages_kv_indptr,
            None,  # kv_start_idx
            self.pa_kv_indices,
            page_table.stride(0),
        )
        # Use the full buffer - pa_persistent_fwd reads only valid elements based on pages_kv_indptr
        kv_indices = self.pa_kv_indices
        
        get_pa_metadata_v1(
            seqlens_qo_indptr=qo_indptr,
            pages_kv_indptr=pages_kv_indptr,
            context_lens=context_lens.int(),
            num_heads_per_head_k=tp_q_head_num // self.num_kv_head,
            num_heads_k=self.num_kv_head,
            is_causal=True,
            work_metadata_ptrs=self.pa_metadata_buffers["work_metadata_ptrs"],
            work_indptr=self.pa_metadata_buffers["work_indptr"],
            work_info=self.pa_metadata_buffers["work_info"],
            reduce_indptr=self.pa_metadata_buffers["reduce_indptr"],
            reduce_final_map=self.pa_metadata_buffers["reduce_final_map"],
            reduce_partial_map=self.pa_metadata_buffers["reduce_partial_map"],
            kv_granularity=max(kernel_block_size, 16),
            block_size=kernel_block_size,
            max_seqlen_qo=max_qlen,
            uni_seqlen_qo=max_qlen,
            fast_mode=True,
            topk=-1,
            max_split_per_batch=-1,
        )
        
        # Store computed values in ForwardMetadata for reuse in forward_decode
        self.forward_metadata.pa_metadata_qo_indptr = qo_indptr
        self.forward_metadata.pa_metadata_pages_kv_indptr = pages_kv_indptr
        self.forward_metadata.pa_metadata_kv_indices = kv_indices
        self.forward_metadata.pa_metadata_context_lens = context_lens
        self.forward_metadata.pa_metadata_max_qlen = max_qlen
        self.forward_metadata.pa_metadata_tp_q_head_num = tp_q_head_num

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        self.cuda_graph_kv_last_page_len = torch.ones(max_bs, dtype=torch.int)
        if kv_indices_buf is None:
            self.cuda_graph_kv_indices = torch.zeros(
                (max_bs * self.max_context_len),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        # Always use preshuffle layout for pa_fwd_asm
        self.page_table = torch.zeros(
            (max_bs, self.max_context_len // self.page_size), dtype=torch.int32, device=self.device
        )
        self.seq_lens = torch.zeros(
            (max_bs,), dtype=torch.int32, device=self.device
        )
        self.strided_indices = torch.arange(
            0, self.max_context_len, self.page_size, device=self.device
        )
        
        # Pre-allocate buffers for pa_metadata in CUDA graph mode (non-MLA decode)
        if not self.use_mla:
            # Pre-allocate pa_metadata buffers for CUDA graph compatibility
            # These buffers will be reused in capture and replay phases
            # Use max_bs and max_qlen=1 (decode mode) to calculate buffer sizes
            max_qlen = 1  # decode mode
            kv_dtype_for_metadata = dtypes.fp8
            (
                (work_metadata_ptrs_size, work_metadata_ptrs_type),
                (work_indptr_size, work_indptr_type),
                (work_info_size, work_info_type),
                (reduce_indptr_size, reduce_indptr_type),
                (reduce_final_map_size, reduce_final_map_type),
                (reduce_partial_map_size, reduce_partial_map_type),
            ) = get_pa_metadata_info_v1(
                max_bs,
                max_qlen,
                self.num_head,  # Use self.num_head as default tp_q_head_num
                self.q_dtype,
                kv_dtype_for_metadata,
                is_sparse=0,
                fast_mode=True,
            )
            
            # Pre-allocate buffers with maximum size for CUDA graph compatibility
            self._allocate_pa_metadata_buffers(
                work_metadata_ptrs_size,
                work_metadata_ptrs_type,
                work_indptr_size,
                work_indptr_type,
                work_info_size,
                work_info_type,
                reduce_indptr_size,
                reduce_indptr_type,
                reduce_final_map_size,
                reduce_final_map_type,
                reduce_partial_map_size,
                reduce_partial_map_type,
            )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        if forward_mode.is_decode_or_idle():
            if self.use_mla:
                # MLA mode: kv_indptr and kv_indices are used in forward_decode
                kv_indptr = self.kv_indptr
                kv_indices = self.cuda_graph_kv_indices
                if spec_info is None:
                    kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                    kv_indptr = kv_indptr[: bs + 1]
                    create_flashinfer_kv_indices_triton[(bs,)](
                        self.req_to_token,
                        req_pool_indices,
                        seq_lens,
                        kv_indptr,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                    )
                else:
                    kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(
                    self.cuda_graph_kv_last_page_len[:bs], dim=0
                )
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = 1
                
                page_table = self.page_table[:bs, :]
                self.seq_lens[:bs].copy_(seq_lens, non_blocking=True)
                seq_lens = self.seq_lens[:bs]
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    None,
                    page_table,
                    seq_lens,
                )
            else:
                # Non-MLA decode mode: kv_indptr and kv_indices are NOT used in forward_decode
                # (forward_decode uses pa_metadata_pages_kv_indptr and pa_metadata_kv_indices instead)
                page_table = self.page_table[:bs, :]
                self.seq_lens[:bs].copy_(seq_lens, non_blocking=True)
                seq_lens_persistent = self.seq_lens[:bs]
                self.forward_metadata = ForwardMetadata(
                    None,  # kv_indptr not used in non-MLA decode mode
                    None,  # kv_indices not used in non-MLA decode mode
                    None,  # qo_indptr will be set by _build_pa_metadata_for_decode
                    None,  # kv_last_page_len not used in non-MLA mode
                    1,  # max_q_len = 1 for decode mode
                    None,  # max_kv_len
                    page_table,
                    seq_lens_persistent,
                )
                
                # Build pa_metadata using CUDA graph buffers
                self._build_pa_metadata_for_decode(bs, tp_q_head_num=self.num_head)
                return  # Early return for non-MLA decode mode

        elif forward_mode.is_target_verify():
            if self.use_mla:
                qo_indptr = self.qo_indptr[: bs + 1]
                qo_indptr[: bs + 1] = torch.arange(
                    0,
                    (1 + bs) * self.num_draft_tokens,
                    step=self.num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                )
                kv_indptr = self.kv_indptr[: bs + 1]
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                kv_indices = self.cuda_graph_kv_indices
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = self.num_draft_tokens

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    None,
                )
            else:
                seq_lens_sum = seq_lens.sum().item()
                self.indices_updater_prefill.update(
                    req_pool_indices,
                    seq_lens,
                    seq_lens_sum,
                    prefix_lens=None,
                    encoder_lens=encoder_lens,
                    spec_info=spec_info,
                )
                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    self.indices_updater_prefill.max_q_len,
                    self.indices_updater_prefill.max_kv_len,
                )
        elif forward_mode.is_draft_extend():
            num_tokens_per_bs = self.speculative_num_steps + 1
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                step=num_tokens_per_bs,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = num_tokens_per_bs
            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                None,
            )
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: Optional[torch.Tensor] = None,
    ):
        if forward_mode.is_decode_or_idle():
            # Common setup for both MLA and non-MLA modes
            page_table_persistent = self.page_table
            seq_lens_persistent = self.seq_lens
            seq_lens_persistent.fill_(0)
            page_table_persistent.fill_(0)
            seq_lens_persistent[:bs].copy_(seq_lens, non_blocking=True)
            max_seq_pages = (seq_lens_cpu.max().item() + self.page_size - 1) // self.page_size + 1
            page_table = self.req_to_token[req_pool_indices[:, None], self.strided_indices[:max_seq_pages][None, :],]
            page_table_persistent[:bs, :max_seq_pages].copy_(page_table // self.page_size, non_blocking=True)
            
            if self.use_mla:
                # MLA mode: kv_indptr and kv_indices are used in forward_decode
                kv_indptr = self.kv_indptr
                kv_indices = self.cuda_graph_kv_indices
                if spec_info is None:
                    kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens[:bs], dim=0)
                    kv_indptr = kv_indptr[: bs + 1]
                    create_flashinfer_kv_indices_triton[(bs,)](
                        self.req_to_token,
                        req_pool_indices[:bs],
                        seq_lens[:bs],
                        kv_indptr,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                    )
                else:
                    kv_indptr[: spec_info.kv_indptr.shape[0]] = spec_info.kv_indptr
                    kv_indices[: spec_info.kv_indices.shape[0]] = spec_info.kv_indices
                
                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(
                    self.cuda_graph_kv_last_page_len[:bs], dim=0
                )
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = 1
                
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    None,
                    page_table_persistent[:bs, :max_seq_pages],
                    seq_lens_persistent[:bs],
                )
            else:
                # Non-MLA decode mode: kv_indptr and kv_indices are NOT used in forward_decode
                # (forward_decode uses pa_metadata_pages_kv_indptr and pa_metadata_kv_indices instead)
                self.forward_metadata = ForwardMetadata(
                    None,  # kv_indptr not used in non-MLA decode mode
                    None,  # kv_indices not used in non-MLA decode mode
                    None,  
                    None,  # kv_last_page_len not used in non-MLA mode
                    1,  # max_q_len = 1 for decode mode
                    None,  # max_kv_len
                    page_table_persistent[:bs, :max_seq_pages],
                    seq_lens_persistent[:bs],
                )
                
                # Rebuild pa_metadata using CUDA graph buffers (updates content, keeps same addresses)
                self._build_pa_metadata_for_decode(bs, tp_q_head_num=self.num_head)

        elif forward_mode.is_target_verify():
            bs = len(req_pool_indices)
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            kv_lens = seq_lens + self.num_draft_tokens
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                kv_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
        elif forward_mode.is_draft_extend():
            seq_lens = seq_lens[:bs]
            accept_lens = spec_info.accept_length[:bs]
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[1 : bs + 1] = torch.cumsum(accept_lens, dim=0)
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
        else:
            raise ValueError("Invalid forward mode")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        self.logits_soft_cap = layer.logit_cap

        if save_kv_cache:
            assert k is not None
            assert v is not None
            if self.use_mla:
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
            else:
                # Shuffle operation is already fused in rotary_emb, so just save directly
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        if self.use_mla:
            max_q_len = self.forward_metadata.max_q_len
            max_kv_len = self.forward_metadata.max_kv_len
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            qo_indptr = self.forward_metadata.qo_indptr
            K_Buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            V_Buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
            kv_lora_rank = V_Buffer.shape[-1]
            qk_rope_head_dim = K_Buffer.shape[-1] - kv_lora_rank
            qk_nope_head_dim = k.shape[-1] - qk_rope_head_dim
            assert len(q.shape) == 3
            assert len(k.shape) == 3
            assert len(v.shape) == 3

            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
            ):
                if kv_indices.shape[0] == 0:
                    o = flash_attn_varlen_func(
                        q,
                        k,
                        v,
                        qo_indptr,
                        qo_indptr,
                        max_q_len,
                        max_q_len,
                        softmax_scale=layer.scaling,
                        causal=True,
                    )
                    return o
                elif layer.qk_head_dim != (kv_lora_rank + qk_rope_head_dim):
                    K_Buffer = torch.index_select(K_Buffer, 0, kv_indices)
                    kvc, k_pe = torch.split(
                        K_Buffer, [kv_lora_rank, qk_rope_head_dim], dim=-1
                    )
                    kvprefix = layer.kv_b_proj(kvc.contiguous())[0]

                    kvprefix = kvprefix.view(
                        -1, layer.tp_k_head_num, qk_nope_head_dim + layer.v_head_dim
                    )
                    k_prefix, v_prefix = torch.split(
                        kvprefix, [qk_nope_head_dim, layer.v_head_dim], dim=-1
                    )
                    k_prefix = torch.cat(
                        [
                            k_prefix,
                            torch.broadcast_to(
                                k_pe,
                                (k_pe.shape[0], layer.tp_k_head_num, k_pe.shape[2]),
                            ),
                        ],
                        dim=-1,
                    )
                    assert (
                        forward_batch.extend_prefix_lens.shape
                        == forward_batch.extend_seq_lens.shape
                    )

                    k = k_prefix
                    v = v_prefix

                    o = flash_attn_varlen_func(
                        q,
                        k,
                        v,
                        qo_indptr,
                        kv_indptr,
                        max_q_len,
                        max_kv_len,
                        softmax_scale=layer.scaling,
                        causal=True,
                    )
                    return o

                else:
                    if layer.qk_head_dim != layer.v_head_dim:
                        o = q.new_empty(
                            (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                        )
                    else:
                        o = torch.empty_like(q)

                    mla_prefill_fwd(
                        q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                        qo_indptr,
                        kv_indptr,
                        kv_indices,
                        self.forward_metadata.kv_last_page_len,
                        self.forward_metadata.max_q_len,
                        layer.scaling,
                        layer.logit_cap,
                    )
                    K_Buffer = K_Buffer.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
                    return o
            elif forward_batch.forward_mode.is_target_verify():
                o = q.new_empty((q.shape[0], layer.tp_q_head_num, layer.v_head_dim))
                mla_decode_fwd(
                    q,
                    K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                    o,
                    self.forward_metadata.qo_indptr,
                    self.forward_metadata.kv_indptr,
                    self.forward_metadata.kv_indices,
                    self.forward_metadata.kv_last_page_len,
                    self.forward_metadata.max_q_len,
                    layer.scaling,
                    layer.logit_cap,
                )
                K_Buffer = K_Buffer.view(-1, 1, layer.qk_head_dim)
                return o
            elif forward_batch.forward_mode.is_draft_extend():
                o = q.new_empty((q.shape[0], layer.tp_q_head_num, layer.v_head_dim))
                kv_indptr = self.forward_metadata.kv_indptr
                kv_indices = self.forward_metadata.kv_indices
                mla_prefill_fwd(
                    q,
                    K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                    o,
                    self.forward_metadata.qo_indptr,
                    self.forward_metadata.kv_indptr,
                    self.forward_metadata.kv_indices,
                    self.forward_metadata.kv_last_page_len,
                    self.forward_metadata.max_q_len,
                    layer.scaling,
                    layer.logit_cap,
                )
                K_Buffer = K_Buffer.view(-1, 1, layer.qk_head_dim)
                return o
            else:
                raise ValueError(
                    f"Invalid forward mode for MLA prefill: {forward_batch.forward_mode=}"
                )
        else:
            bs0 = forward_batch.batch_size + 1

            q_fp8 = q.to(dtypes.fp8)
            
            o = flash_attn_varlen_fp8_pertensor_func(
                q=q_fp8,
                k=k,
                v=v,
                cu_seqlens_q=self.qo_indptr[:bs0],
                cu_seqlens_k=self.forward_metadata.kv_indptr[:bs0],
                max_seqlen_q=self.forward_metadata.max_q_len,
                max_seqlen_k=self.forward_metadata.max_kv_len,
                softmax_scale=layer.scaling,
                causal=True,
                q_descale=None,
                k_descale=None,
                v_descale=None,
            )

            return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # Create o as 3D tensor [batch_size, num_heads, head_dim] for both MLA and pa_fwd_asm
        # In decode mode, q.shape[0] equals batch_size (each sequence has 1 token)
        # Use q.shape[0] instead of forward_batch.batch_size to be safe
        batch_size = q.shape[0]
        head_dim_out = layer.v_head_dim if layer.qk_head_dim != layer.v_head_dim else layer.head_dim
        o = q.new_empty((batch_size, layer.tp_q_head_num, head_dim_out))

        if save_kv_cache:
            if self.use_mla:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )
            else:
                # Shuffle operation is already fused in rotary_emb, so just save directly
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        if self.use_mla:
            k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            mla_decode_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k_buffer.view(-1, 1, 1, layer.qk_head_dim),
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                self.forward_metadata.qo_indptr,
                self.forward_metadata.kv_indptr,
                self.forward_metadata.kv_indices,
                self.forward_metadata.kv_last_page_len,
                self.forward_metadata.max_q_len,
                layer.scaling,
                layer.logit_cap,
            )
            k_buffer = k_buffer.view(-1, 1, layer.qk_head_dim)
        else:
            k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            num_slots, num_kv_heads, head_size = k_buffer.shape
            block_size = self.page_size
            num_blocks = num_slots // block_size
            k_buffer = k_buffer[:num_blocks * block_size].view(num_blocks, block_size, num_kv_heads, head_size)
            v_buffer = v_buffer[:num_blocks * block_size].view(num_blocks, block_size, num_kv_heads, head_size)


            quant_dtype = dtypes.fp8
            x = 16 // quant_dtype.itemsize
            k_cache_template = torch.empty(
                [num_blocks, num_kv_heads, head_size // x, block_size, x],
                dtype=k_buffer.dtype,
                device="meta",
            )
            # V: [num_blocks, block_size, num_kv_heads, head_size] -> [num_blocks, num_kv_heads, block_size // x, head_size, x]
            v_cache_template = torch.empty(
                [num_blocks, num_kv_heads, block_size // x, head_size, x],
                dtype=v_buffer.dtype,
                device="meta",
            )
            new_key_cache = k_buffer.view_as(k_cache_template)
            new_value_cache = v_buffer.view_as(v_cache_template)
            
            total_tokens = num_blocks * block_size
            k_qscale = self.k_qscale[:, :total_tokens]
            v_qscale = self.v_qscale[:, :total_tokens]
            
            q = q.view(batch_size, layer.tp_q_head_num, layer.head_dim)
    

            assert self.forward_metadata.pa_metadata_qo_indptr is not None, "pa_metadata_qo_indptr should be set by _build_pa_metadata_for_decode"
            assert self.forward_metadata.pa_metadata_pages_kv_indptr is not None, "pa_metadata_pages_kv_indptr should be set by _build_pa_metadata_for_decode"
            assert self.forward_metadata.pa_metadata_kv_indices is not None, "pa_metadata_kv_indices should be set by _build_pa_metadata_for_decode"
            assert self.forward_metadata.pa_metadata_context_lens is not None, "pa_metadata_context_lens should be set by _build_pa_metadata_for_decode"
            assert self.forward_metadata.pa_metadata_max_qlen is not None, "pa_metadata_max_qlen should be set by _build_pa_metadata_for_decode"
            
            qo_indptr = self.forward_metadata.pa_metadata_qo_indptr
            kv_indptr = self.forward_metadata.pa_metadata_pages_kv_indptr
            kv_indices = self.forward_metadata.pa_metadata_kv_indices
            context_lens = self.forward_metadata.pa_metadata_context_lens
            max_qlen = self.forward_metadata.pa_metadata_max_qlen
            
            
            _, _ = pa_persistent_fwd(
                Q=q,
                K=new_key_cache,
                V=new_value_cache,
                output=o,
                max_qlen=max_qlen,
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                context_lens=context_lens,
                work_indptr=self.pa_metadata_buffers["work_indptr"],
                work_info=self.pa_metadata_buffers["work_info"],
                reduce_indptr=self.pa_metadata_buffers["reduce_indptr"],
                reduce_final_map=self.pa_metadata_buffers["reduce_final_map"],
                reduce_partial_map=self.pa_metadata_buffers["reduce_partial_map"],
                K_QScale=k_qscale,
                V_QScale=v_qscale,
                softmax_scale=layer.scaling,
                mask=1,  
            )
        return o.view(-1, layer.tp_q_head_num * head_dim_out)


class AiterIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.update = self.update_single_wrapper

        self.kv_indices = None
        self.max_q_len = 0
        self.max_kv_len = 0

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: Optional[List[int]],
        seq_lens_cpu: Optional[torch.Tensor],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: Optional[List[int]],
        seq_lens_cpu: Optional[torch.Tensor],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
    ):

        kv_start_idx = None
        kv_indptr = self.kv_indptr
        qo_indptr = self.qo_indptr
        paged_kernel_lens = seq_lens
        paged_kernel_lens_sum = seq_lens_sum

        bs = len(req_pool_indices)
        if spec_info is None:
            # Normal extend
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            # (TODO: Kk) WA - CI test_moe_eval_accuracy_large.py
            # mha_batch_prefill reads 128 data to do computatoin
            # if real data is not long enough then original padding value 0 is used
            # but the 0 location will be made nan (noqa) in cuda graph capture mode
            # this will cause the output tensor value becomes nan
            # WA is to assure that last index of pool not changed
            kv_indices = torch.empty(
                paged_kernel_lens_sum + 256,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
            if seq_lens_cpu is None:
                token_num = kv_indptr[-1]
                kv_indices[token_num:] = kv_indices[0]
            else:
                token_num = torch.cumsum(seq_lens_cpu, dim=0)[-1]
                kv_indices[token_num:] = kv_indices[0]

            if seq_lens_cpu is None:
                self.max_kv_len = torch.max(paged_kernel_lens).item()
            else:
                self.max_kv_len = torch.max(seq_lens_cpu).item()

            extend_lens = seq_lens - prefix_lens
            if seq_lens_cpu is not None and prefix_lens_cpu is not None:
                prefix_lens_cpu_torch = torch.tensor(prefix_lens_cpu)
                extend_lens_cpu = seq_lens_cpu - prefix_lens_cpu_torch
                self.max_q_len = torch.max(extend_lens_cpu).item()
            else:
                self.max_q_len = torch.max(extend_lens).item()

            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
        else:
            kv_indices, kv_indptr, qo_indptr, _ = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    paged_kernel_lens,
                    paged_kernel_lens_sum,
                    self.req_to_token,
                )
            )

        self.kv_indices = kv_indices


class AiterMlaIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.update = self.update_single_wrapper

        self.kv_indptr = None
        self.kv_indices = None
        self.qo_indptr = None
        self.kv_last_page_len = None
        self.max_q_len = 0
        self.max_kv_len = 0

    def update(
        self,
        req_pool_indices: torch.Tensor,
        kv_lens: torch.Tensor,
        kv_lens_sum: int,
        extend_lens: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        spec_info: Optional[SpecInput],
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        kv_lens: torch.Tensor,
        kv_lens_sum: int,
        extend_lens: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        spec_info: Optional[SpecInput],
    ):
        bs = len(req_pool_indices)

        kv_indptr = self.attn_backend.kv_indptr

        if spec_info is None:
            # Normal extend
            kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                kv_lens_sum,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                kv_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            qo_indptr = self.attn_backend.qo_indptr
            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
        else:
            kv_indices, kv_indptr, qo_indptr, _ = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    kv_lens,
                    kv_lens_sum,
                    self.req_to_token,
                )
            )

        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices
        self.qo_indptr = qo_indptr
        self.max_q_len = max_q_len
        self.max_kv_len = max_kv_len


class AiterMultiStepDraftBackend:
    """
    Wrap multiple triton attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        from sglang.srt.speculative.spec_utils import generate_draft_decode_kv_indices

        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                AiterAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                )
            )
        self.max_context_len = self.attn_backends[0].max_context_len
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.device = model_runner.device
        # Cached variables for generate_draft_decode_kv_indices
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.page_size = model_runner.server_args.page_size
        assert self.page_size == 1, "Page size must be 1"

    def common_template(
        self, forward_batch: ForwardBatch, kv_indices_buffer: torch.Tensor, call_fn: int
    ):
        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        self.generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            triton.next_power_of_2(num_seqs),
            triton.next_power_of_2(self.speculative_num_steps),
            triton.next_power_of_2(bs),
            self.page_size,
        )

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices = torch.empty(
            (
                self.speculative_num_steps,
                forward_batch.batch_size * self.topk * self.max_context_len,
            ),
            dtype=torch.int32,
            device=self.device,
        )

        def call_fn(i, forward_batch):
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, max_num_tokens * self.max_context_len),
            dtype=torch.int32,
            device=self.device,
        )
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, kv_indices_buf=self.cuda_graph_kv_indices[i]
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=None,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)
