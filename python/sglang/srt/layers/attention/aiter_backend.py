from __future__ import annotations

"""
end to end attention solution with aiter kernels
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

import torch
import triton

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.triton_ops.aiter_unified_attention import (
    scatter_ragged_to_page_table_kernel,
    scatter_req_to_token_to_page_table_kernel,
)
from sglang.srt.layers.attention.utils import (
    create_flashinfer_kv_indices_triton,
    create_flashmla_kv_indices_triton,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_gfx95_supported

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

try:
    from aiter import (
        flash_attn_varlen_func,
        get_mla_metadata_info_v1,
        get_mla_metadata_v1,
        get_ps_metadata_info_v1,
        get_ps_metadata_v1,
        mha_batch_prefill_func,
        mla_prefill_ps_asm_fwd,
        mla_reduce_v1,
        paged_attention_ragged,
    )
    from aiter.mla import mla_decode_fwd, mla_prefill_fwd
    from aiter.ops.triton.attention.unified_attention import unified_attention
except ImportError:
    print(
        "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
    )

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.utils import (
    launch_reshape_and_cache_flash,
    pad_sequence_with_mask,
)
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

# Use aiter mla persist design for fp8-kv cache
_use_mla_ps_kernel = get_bool_env_var("SGLANG_AITER_MLA_PERSIST", "True")

# Use fp8 prefill only on gfx95
_use_fp8_prefill_attn = (
    get_bool_env_var("SGLANG_AITER_FP8_PREFILL_ATTN", "True") and is_gfx95_supported()
)

# Persist
# fast_mode=True if _use_mla_ps_kernel else False
# intra_batch_mode=False if _use_mla_ps_kernel else True

# fake non-ps, intra_batch_mode needs to be True for non-ps-mode
fast_mode = False
intra_batch_mode = True if _use_mla_ps_kernel else False


class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


@dataclass
class ForwardMetadata:
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor
    kv_last_page_len: torch.Tensor
    max_q_len: int
    max_kv_len: Optional[int]
    work_metadata: Optional[torch.Tensor] = None
    work_info_set: Optional[torch.Tensor] = None
    work_indptr: Optional[torch.Tensor] = None
    reduce_indptr: Optional[torch.Tensor] = None
    reduce_final_map: Optional[torch.Tensor] = None
    reduce_partial_map: Optional[torch.Tensor] = None
    num_kv_splits: Optional[int] = None
    run_graph: Optional[bool] = True
    custom_mask: Optional[torch.Tensor] = None
    mask_indptr: Optional[torch.Tensor] = None
    max_extend_len: Optional[int] = None
    fp8_prefill_kv_indices: Optional[torch.Tensor] = None
    swa_page_table: Optional[torch.Tensor] = None


global_workspace_buffer = None


_AITER_PARTITION_SIZE_ROCM = 256


class AiterAttnBackend(AttentionBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        topk: int = 1,
    ):
        super().__init__()
        # Lazy import to avoid the initialization of cuda context
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd,
        )

        self.input_dtype = model_runner.model_config.dtype

        self.page_size = model_runner.server_args.page_size

        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)

        self.device = model_runner.device
        self.is_multimodal = model_runner.model_config.is_multimodal
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.topk = topk
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.kv_cache_dtype = model_runner.kv_cache_dtype

        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA

        # Get v_head_dim based on model type
        if self.use_mla:
            # For MLA models, get v_head_dim from model config
            self.v_head_dim = model_runner.model_config.v_head_dim
        elif hasattr(model_runner.token_to_kv_pool, "get_v_head_dim"):
            # For hybrid models (Mamba+attention, GDN, Kimi linear),
            # layer_id=0 may not be a full attention layer
            self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
        else:
            self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[
                -1
            ]

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
        # qo_indptr for the unified-attn decode path (q_len == 1 per request)
        # is always arange(0, bs+1); precompute once to avoid a per-step cumsum.
        self.qo_indptr_unified_decode = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=model_runner.device
        )
        self.mask_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int64, device=model_runner.device
        )
        self._kv_indices_scratch: Optional[torch.Tensor] = None

        # Create prefill indices updater
        if not skip_prefill:
            self.indices_updater_prefill = AiterIndicesUpdaterPrefill(
                model_runner, self
            )
            if self.use_mla:
                self.mla_indices_updater_prefill = AiterMlaIndicesUpdaterPrefill(
                    model_runner, self
                )

        # sliding window attention
        self.use_sliding_window_kv_pool = (
            isinstance(model_runner.token_to_kv_pool, SWAKVPool)
            and model_runner.token_to_kv_pool.swa_layer_nums > 0
        )

        if self.use_sliding_window_kv_pool:
            self.token_to_kv_pool = model_runner.token_to_kv_pool
            self.use_triton_unified_attention = True
        else:
            self.use_triton_unified_attention = get_bool_env_var(
                "SGLANG_USE_AITER_UNIFIED_ATTN"
            )

        # When topk == 1 the EAGLE draft chain is linear, so target_verify's
        # mask reduces to pure causal and can go through unified_attention
        # instead of the legacy triton extend_attention_fwd. Gated on non-MLA
        # (MLA has its own verify path) and env var for opt-out.
        self._use_unified_verify = (
            self.use_triton_unified_attention
            and not self.use_mla
            and self.topk == 1
            and get_bool_env_var("SGLANG_AITER_UNIFIED_VERIFY", "1")
        )

        # aiter kernel related initialization
        self.max_num_partitions = (
            self.max_context_len + _AITER_PARTITION_SIZE_ROCM - 1
        ) // _AITER_PARTITION_SIZE_ROCM

        nbyes_per_qo_elem = torch.finfo(torch.float32).bits // 8

        if not (self.use_mla or self.use_triton_unified_attention):
            self.workspace_buffer = torch.empty(
                (max_bs * self.num_head * self.max_num_partitions * self.head_dim)
                * nbyes_per_qo_elem
                + 2 * (max_bs * self.num_head * self.max_num_partitions) * 4,
                dtype=torch.uint8,
                device=self.device,
            )

        self.scale = float(1.0 / (self.head_dim**0.5))
        self.k_scale = self.v_scale = torch.tensor([1.0], dtype=torch.float32).to(
            self.device
        )

        self.logits_soft_cap = 0.0

        self.forward_metadata: ForwardMetadata = None

        if self.use_mla:
            _valid_heads = self.num_head in (4, 8) or (
                self.num_head % 16 == 0 and 16 <= self.num_head <= 128
            )
            assert _valid_heads, (
                f"Aiter MLA supports num_head of 4, 8, or multiples of 16 "
                f"in [16, 128].\n"
                f"Provided {self.num_head} number of heads.\n"
                "Try adjusting tensor_parallel_size value."
            )
            self.num_head_padded = 16 if self.num_head < 16 else self.num_head
            self.head_repeat_factor = 16 // self.num_head if self.num_head < 16 else 1

            self.enable_dp_attention = is_dp_attention_enabled()
            self.qo_indptr_ = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
            global _use_mla_ps_kernel, fast_mode, intra_batch_mode

            # current mla_decode_fwd only support fake-nps in self.num_head == 16
            # so all num_head size does not use qh16 kernel to simulate
            # it should not use fake-nps (fast_mode = False, intra_batch_mode = True)
            # it will cause gpu-fault or accuracy issue
            if self.num_head == 32 or self.num_head == 128:
                fast_mode = True
                intra_batch_mode = False

            # current persist a16w16 mla_decode kernel does not support head_num = 128
            # need to fall back to non-persist
            # only use mla_ps_kernel when fp8 kv_cache
            # for non-fp8 kv_cache on tp8, use non-persist kernel to avoid performance degradation
            # head_num=16 (tp8 perf issue), head_num=128 (unsupported, like tp1 or --enable-dp-attention with tp8-dp8)
            if (
                self.num_head_padded == 16 or self.num_head_padded == 128
            ) and self.kv_cache_dtype is not fp8_dtype:
                _use_mla_ps_kernel = False
                fast_mode = False
                intra_batch_mode = False

            self.max_split_per_batch = 32 if _use_mla_ps_kernel else None

            if self.num_draft_tokens is None and _use_mla_ps_kernel:
                self.max_split_per_batch = 64

            self.fix_max_split_per_batch = self.max_split_per_batch

    def _get_aiter_paged_ragged_kv_cache_dtype(self) -> str:
        """``kv_cache_dtype`` string for ``paged_attention_ragged`` (aiter ``pa/pa_ragged.py``).

        **Behavior change:** we no longer upcast FP8 KV to the activations dtype for this decode path.
        Paged K/V stay in native FP8 storage; we pass ``\"fp8_e4m3\"`` so the kernel dequants on read
        (``k_scale`` / ``v_scale``) instead of widening the cache to bf16/fp16 for ``\"auto\"``.

        **Context (short):** aiter accepts ``auto`` / ``fp8`` / ``fp8_e4m3`` only (not ``fp8_e5m2``).
        On HIP, ``configure_kv_cache_dtype`` maps CLI ``fp8_e5m2`` and ``fp8_e4m3`` to ``fp8_dtype``;
        return ``\"fp8_e4m3\"`` when ``self.kv_cache_dtype == fp8_dtype``, else ``\"auto\"``.
        """
        if self.kv_cache_dtype != fp8_dtype:
            return "auto"
        return "fp8_e4m3"

    def make_mla_decode_meta_data_buffer(self, max_seqlen_qo, batch_size):
        nhead = self.num_head_padded
        dtype = self.kv_cache_dtype

        if self.enable_dp_attention:
            gpu = torch.cuda.current_device()
            device_properties = torch.cuda.get_device_properties(gpu)
            cu_num = device_properties.multi_processor_count
            self.max_split_per_batch = min(
                (cu_num + batch_size - 1) // batch_size, self.fix_max_split_per_batch
            )

        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            batch_size,
            max_seqlen_qo,
            nhead,
            dtype,
            dtype,
            is_sparse=False,
            fast_mode=fast_mode,
            num_kv_splits=self.max_split_per_batch,
            intra_batch_mode=intra_batch_mode,
        )

        # aiter implementation
        # the tensor's meaning please refer aiter/ops/attention.py
        work_metadata = torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device="cuda"
        )
        work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_type, device="cuda"
        )
        work_info_set = torch.empty(
            work_info_set_size,
            dtype=work_info_set_type,
            device="cuda",
        )
        reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device="cuda"
        )
        reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device="cuda"
        )
        reduce_partial_map = torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_type, device="cuda"
        )

        return (
            work_metadata,
            work_indptr,
            work_info_set,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
        )

    def make_mla_meta_data(
        self,
        qo_indptr,
        kv_indptr,
        kv_last_page_len,
        work_metadata,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_q_len,
        fast_mode,
        max_split_per_batch,
        intra_batch_mode,
    ):

        nhead_kv = 1
        page_size = self.page_size
        dtype = self.kv_cache_dtype

        meta = get_mla_metadata_v1(
            qo_indptr,
            kv_indptr,
            kv_last_page_len,
            self.num_head_padded // nhead_kv,
            nhead_kv,
            False,
            work_metadata,
            work_info_set,
            work_indptr,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            kv_granularity=max(page_size, 16),
            max_seqlen_qo=max_q_len,
            uni_seqlen_qo=max_q_len,
            fast_mode=fast_mode,
            max_split_per_batch=max_split_per_batch,
            intra_batch_mode=intra_batch_mode,
            dtype_q=dtype,
            dtype_kv=dtype,
        )

    def make_mla_prefill_ps_meta_data_buffer(
        self, batch_size: int, max_qlen: int, qlen_granularity: int
    ):
        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_size, work_info_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_ps_metadata_info_v1(
            batch_size=batch_size,
            num_head_k=self.num_kv_head,
            max_qlen=max_qlen,
            qlen_granularity=qlen_granularity,
        )

        device = self.device
        work_metadata_ptrs = torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device=device
        )
        work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_type, device=device
        )
        work_info = torch.empty(work_info_size, dtype=work_info_type, device=device)
        reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device=device
        )
        reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device=device
        )
        reduce_partial_map = torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_type, device=device
        )

        return (
            work_metadata_ptrs,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
        )

    def make_mla_prefill_ps_meta_data(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        seq_lens: torch.Tensor,
        work_metadata: torch.Tensor,
        work_indptr: torch.Tensor,
        work_info: torch.Tensor,
        reduce_indptr: torch.Tensor,
        reduce_final_map: torch.Tensor,
        reduce_partial_map: torch.Tensor,
        is_causal: bool = True,
    ):
        gqa_ratio = self.num_head // self.num_kv_head
        num_heads_k = self.num_kv_head
        tile_q = 256
        qhead_granularity = gqa_ratio
        qlen_granularity = tile_q // qhead_granularity
        kvlen_granularity = max(128, self.page_size)
        block_size = self.page_size

        qo_indptr_cpu = qo_indptr.to("cpu", dtype=torch.int32)
        kv_indptr_cpu = kv_indptr.to("cpu", dtype=torch.int32)
        seq_lens_cpu = seq_lens.to("cpu", dtype=torch.int32)

        get_ps_metadata_v1(
            qo_indptr_cpu,
            kv_indptr_cpu,
            seq_lens_cpu,
            gqa_ratio,
            num_heads_k,
            work_metadata,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            qhead_granularity=qhead_granularity,
            qlen_granularity=qlen_granularity,
            kvlen_granularity=kvlen_granularity,
            block_size=block_size,
            is_causal=is_causal,
        )

    # for page size > 1 useful conversion function
    def _transform_table_1_to_real(self, page_table: torch.Tensor) -> torch.Tensor:
        page_size = self.page_size
        if page_size == 1:
            return page_table
        max_seqlen_k = page_table.shape[1]
        strided_indices = torch.arange(
            0, max_seqlen_k, page_size, device=page_table.device, dtype=torch.int32
        )
        return page_table[:, strided_indices] // page_size

    def _build_unified_page_table_from_spec(
        self,
        spec_info,
        bs: int,
        dest_buf: Optional[torch.Tensor] = None,
        swa_dest_buf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert ragged (token-level) kv_indices from spec_info into a 2D
        block-level page_table of shape (bs, max_num_blocks_per_seq).
        unified_attention expects max_seqlen_k = page_table.shape[1] *
        page_size to be a captured constant, so rows are sized to the
        backend-level max_num_blocks_per_seq regardless of seqused_k.
        """
        kv_indptr = spec_info.kv_indptr
        kv_flat = spec_info.kv_indices
        page_size = self.page_size
        max_blocks = (self.max_context_len + page_size - 1) // page_size

        swa_slot_mapping = None
        swa_page_table = None

        if dest_buf is not None:
            # The scatter kernel fills [0, num_blocks) and loads past that use
            # other=0, so the tail is 0-filled. Under graph replay rows > bs
            # are stale but unified_attention only walks rows [0, bs).
            page_table = dest_buf
        else:
            page_table = torch.zeros(
                bs, max_blocks, dtype=torch.int32, device=self.device
            )

        if self.use_sliding_window_kv_pool:
            swa_slot_mapping = self.token_to_kv_pool.full_to_swa_index_mapping.long()

            if swa_dest_buf is not None:
                swa_page_table = swa_dest_buf
            else:
                swa_page_table = torch.zeros(
                    bs, max_blocks, dtype=torch.int32, device=self.device
                )

        BLOCK_SIZE = 1024
        grid = (bs, triton.cdiv(max(max_blocks, 1), BLOCK_SIZE))
        scatter_ragged_to_page_table_kernel[grid](
            kv_flat,
            kv_indptr,
            page_table,
            page_table.stride(0),
            swa_page_table,
            swa_slot_mapping,
            PAGE_SIZE=page_size,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_SWA=(swa_slot_mapping is not None),
        )

        return page_table, swa_page_table

    def _build_verify_unified_metadata(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        draft_num: int,
        page_table_dest: Optional[torch.Tensor] = None,
        swa_page_table_dest: Optional[torch.Tensor] = None,
    ):
        """Build the 2D block page_table + qo_indptr for EAGLE target_verify
        through unified_attention. Assumes the new draft K/V have already been
        written by set_kv_buffer, so req_to_token[rp, :seq_lens[i]+draft_num]
        covers both the prefix and the freshly committed draft tokens. Returns
        (page_table, qo_indptr, max_q_len=draft_num).
        """
        device = seq_lens.device
        qo_indptr = self.qo_indptr[: bs + 1]
        qo_indptr[: bs + 1] = torch.arange(
            0,
            (1 + bs) * draft_num,
            step=draft_num,
            dtype=torch.int32,
            device=device,
        )

        page_size = self.page_size
        max_blocks = (self.max_context_len + page_size - 1) // page_size

        swa_slot_mapping = None
        swa_page_table = None

        if page_table_dest is not None:
            page_table = page_table_dest
        else:
            page_table = torch.zeros(bs, max_blocks, dtype=torch.int32, device=device)

        if self.use_sliding_window_kv_pool:
            swa_slot_mapping = self.token_to_kv_pool.full_to_swa_index_mapping.long()

            if swa_page_table_dest is not None:
                swa_page_table = swa_page_table_dest
            else:
                swa_page_table = torch.zeros(
                    bs, max_blocks, dtype=torch.int32, device=device
                )

        BLOCK_SIZE = 1024
        grid = (bs, triton.cdiv(max(max_blocks, 1), BLOCK_SIZE))
        scatter_req_to_token_to_page_table_kernel[grid](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            page_table,
            self.req_to_token.stride(0),
            page_table.stride(0),
            swa_page_table,
            swa_slot_mapping,
            DRAFT_NUM=draft_num,
            PAGE_SIZE=page_size,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_SWA=(swa_slot_mapping is not None),
        )

        return page_table, qo_indptr, draft_num, swa_page_table

    def _resolve_v2_num_draft_tokens(
        self,
        extend_seq_lens: Optional[torch.Tensor] = None,
        extend_seq_lens_cpu: Optional[list[int]] = None,
    ) -> int:
        """Resolve fixed per-request extend length for DRAFT_EXTEND_V2."""
        num_draft_tokens = self.num_draft_tokens
        if num_draft_tokens is None:
            if extend_seq_lens is not None and extend_seq_lens.numel() > 0:
                # Avoid list scans in hot path when tensor lengths are already available.
                num_draft_tokens = int(extend_seq_lens[0].item())
            elif extend_seq_lens_cpu:
                num_draft_tokens = max(extend_seq_lens_cpu)
            else:
                raise ValueError(
                    "DRAFT_EXTEND_V2 requires speculative_num_draft_tokens or "
                    "non-empty extend_seq_lens/extend_seq_lens_cpu."
                )

        num_draft_tokens = int(num_draft_tokens)
        if extend_seq_lens is not None and extend_seq_lens.numel() > 0:
            if not torch.all(extend_seq_lens == num_draft_tokens):
                raise ValueError(
                    "DRAFT_EXTEND_V2 expects fixed extend length per request; got "
                    f"extend_seq_lens={extend_seq_lens}, expected all == {num_draft_tokens}."
                )
        if extend_seq_lens_cpu and any(
            x != num_draft_tokens for x in extend_seq_lens_cpu
        ):
            raise ValueError(
                "DRAFT_EXTEND_V2 expects fixed extend length per request; got "
                f"{extend_seq_lens_cpu}, expected all == {num_draft_tokens}."
            )
        return num_draft_tokens

    def _get_kv_indices_scratch(
        self, required_tokens: int, device: torch.device
    ) -> torch.Tensor:
        if (
            self._kv_indices_scratch is None
            or self._kv_indices_scratch.device != device
            or self._kv_indices_scratch.numel() < required_tokens
        ):
            self._kv_indices_scratch = torch.empty(
                required_tokens, dtype=torch.int32, device=device
            )
        return self._kv_indices_scratch[:required_tokens]

    def _set_uniform_qo_indptr(
        self, bs: int, tokens_per_req: int, device: torch.device
    ) -> torch.Tensor:
        qo_indptr = self.qo_indptr[: bs + 1]
        qo_indptr[: bs + 1] = torch.arange(
            0,
            bs * tokens_per_req + 1,
            step=tokens_per_req,
            dtype=torch.int32,
            device=device,
        )
        return qo_indptr

    def _ensure_spec_v2_topk_supported(self):
        if self.topk > 1:
            raise NotImplementedError(
                "AiterAttnBackend SPEC_V2 path currently supports topk <= 1 only. "
                f"Got topk={self.topk}."
            )

    def _mla_decode_fwd_with_head_pad(
        self,
        q: torch.Tensor,
        k_buffer_flat: torch.Tensor,
        layer,
        **kwargs,
    ):
        """Wrap mla_decode_fwd with head-dimension padding for num_head < 16.

        When head_repeat_factor > 1 (i.e. num_head is 4 or 8), q is
        repeat-interleaved to reach num_head_padded (16) before the kernel
        call, and the corresponding output columns are sliced back afterward.
        q / o must already be shaped (..., num_head, head_dim).
        """
        if self.head_repeat_factor > 1:
            q_in = q.repeat_interleave(self.head_repeat_factor, dim=1)
            o = q.new_empty(
                (q.shape[0], self.num_head_padded, layer.v_head_dim),
                dtype=self.input_dtype,
            )
            mla_decode_fwd(q_in, k_buffer_flat, o, **kwargs)
            return o[:, :: self.head_repeat_factor, :]
        else:
            o = q.new_empty(
                (q.shape[0], layer.tp_q_head_num, layer.v_head_dim),
                dtype=self.input_dtype,
            )
            mla_decode_fwd(q, k_buffer_flat, o, **kwargs)
            return o

    def mla_fp8_prefill_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
    ):
        total_q = q.shape[0]
        nhead = layer.tp_q_head_num
        v_head_dim = layer.v_head_dim

        if q.dtype != fp8_dtype:
            q = q.to(fp8_dtype)
        if k.dtype != fp8_dtype:
            k = k.to(fp8_dtype)
        if v.dtype != fp8_dtype:
            v = v.to(fp8_dtype)
        one_scale = torch.ones((), dtype=torch.float32, device=q.device)

        tile_q = 256
        reduce_indptr = self.forward_metadata.reduce_indptr
        reduce_final_map = self.forward_metadata.reduce_final_map
        reduce_partial_map = self.forward_metadata.reduce_partial_map

        logits = torch.empty(
            (reduce_partial_map.size(0) * tile_q, nhead, v_head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        attn_lse = torch.empty(
            (reduce_partial_map.size(0) * tile_q, nhead),
            dtype=torch.float32,
            device=q.device,
        )
        final_lse = torch.empty(
            (total_q, nhead),
            dtype=torch.float32,
            device=q.device,
        )
        output = q.new_empty(
            (total_q, nhead, v_head_dim),
            dtype=self.input_dtype,
        )

        mla_prefill_ps_asm_fwd(
            q,
            k,
            v,
            self.forward_metadata.qo_indptr,
            self.forward_metadata.kv_indptr,
            self.forward_metadata.fp8_prefill_kv_indices,
            self.forward_metadata.work_indptr,
            self.forward_metadata.work_info_set,
            self.forward_metadata.max_q_len,
            layer.scaling,
            True,
            logits,
            attn_lse,
            output,
            one_scale,
            one_scale,
            one_scale,
        )
        mla_reduce_v1(
            logits,
            attn_lse,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            tile_q,
            output,
            final_lse,
        )
        return output

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for aiter attention backend."""

        bs = forward_batch.batch_size
        kv_indptr = self.kv_indptr
        spec_info = forward_batch.spec_info
        qo_indptr = None
        kv_last_page_len = None
        max_q_len = None
        max_kv_len = None

        work_metadata = None
        work_indptr = None
        work_info_set = None
        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None

        num_kv_splits = None
        swa_page_table = None
        max_kv_len = forward_batch.seq_lens_cpu.max().item()

        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None or forward_batch.forward_mode.is_idle():
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]

                if not self.use_triton_unified_attention:
                    kv_indices = self._get_kv_indices_scratch(
                        forward_batch.seq_lens_sum, forward_batch.seq_lens.device
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
                    max_q_len = 1
                    page_size = self.page_size
                    max_num_blocks_per_seq = (max_kv_len + page_size - 1) // page_size
                    kv_indices = torch.zeros(
                        bs, max_kv_len, dtype=torch.int32, device=self.device
                    )

                    create_flashmla_kv_indices_triton[(bs,)](
                        self.req_to_token,
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                        max_kv_len,
                        1,
                    )

                    if self.use_sliding_window_kv_pool:
                        swa_page_table = (
                            self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                kv_indices
                            )
                        )

                        kv_indices = self._transform_table_1_to_real(kv_indices)
                        swa_page_table = self._transform_table_1_to_real(swa_page_table)
                    elif self.page_size > 1:
                        kv_indices = self._transform_table_1_to_real(kv_indices)

                    qo_indptr = self.qo_indptr_unified_decode[: bs + 1]

            else:
                if self.use_triton_unified_attention and not self.use_mla:
                    bs = spec_info.kv_indptr.shape[0] - 1
                    kv_indices, swa_page_table = (
                        self._build_unified_page_table_from_spec(spec_info, bs)
                    )
                    max_q_len = 1
                    qo_indptr = self.qo_indptr_unified_decode[: bs + 1]
                    kv_indptr = None
                else:
                    kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                    bs = kv_indptr.shape[0] - 1

            if self.use_mla:
                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(self.kv_last_page_len[:bs], dim=0)
                kv_last_page_len = self.kv_last_page_len[:bs]
                max_q_len = 1

                if _use_mla_ps_kernel:
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_decode_meta_data_buffer(max_q_len, bs)

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        work_metadata,
                        work_info_set,
                        work_indptr,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
                run_graph=False,
                swa_page_table=swa_page_table,
            )

        elif forward_batch.forward_mode.is_draft_extend_v2():
            # EAGLE V2: DRAFT_EXTEND_V2 mode - extend draft KV cache with all predicted tokens
            self._ensure_spec_v2_topk_supported()
            if self.use_mla:
                device = forward_batch.seq_lens.device
                num_draft_tokens = self._resolve_v2_num_draft_tokens()
                qo_indptr = self._set_uniform_qo_indptr(bs, num_draft_tokens, device)

                kv_indptr = self.kv_indptr[: bs + 1]
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)

                kv_indices = self._get_kv_indices_scratch(
                    forward_batch.seq_lens_sum, device
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

                if _use_mla_ps_kernel:
                    max_seqlen_qo = num_draft_tokens
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_decode_meta_data_buffer(max_seqlen_qo, bs)

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        self.kv_last_page_len[:bs],
                        work_metadata,
                        work_info_set,
                        work_indptr,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        max_seqlen_qo,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    self.kv_last_page_len[:bs],
                    num_draft_tokens,
                    forward_batch.seq_lens_cpu.max().item(),
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                    run_graph=False,
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
                )
        elif forward_batch.forward_mode.is_draft_extend():
            # EAGLE V1: DRAFT_EXTEND mode - uses spec_info.num_accept_tokens
            if self.use_mla:
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.seq_lens_sum,
                        self.req_to_token,
                    )
                )

                if _use_mla_ps_kernel:
                    max_seqlen_qo = max(forward_batch.extend_seq_lens_cpu)
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_decode_meta_data_buffer(max_seqlen_qo, bs)

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        self.kv_last_page_len[:bs],
                        work_metadata,
                        work_info_set,
                        work_indptr,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        max_seqlen_qo,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    # self.mla_indices_updater_prefill.kv_last_page_len,
                    self.kv_last_page_len[:bs],
                    max(forward_batch.extend_seq_lens_cpu),
                    forward_batch.seq_lens_cpu.max().item(),
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                    run_graph=False,
                )
            else:
                # Non-MLA draft_extend: use triton extend kernel with causal masking
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.seq_lens_sum,
                        self.req_to_token,
                    )
                )
                kv_indices = kv_indices.to(torch.int64)
                draft_max_extend_len = torch.max(spec_info.num_accept_tokens).item()

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    None,
                    draft_max_extend_len,
                    None,
                    custom_mask=custom_mask,
                    mask_indptr=None,
                    max_extend_len=draft_max_extend_len,
                )
        elif forward_batch.forward_mode.is_target_verify():
            if self.use_mla:
                draft_num = spec_info.draft_token_num
                kv_lens = forward_batch.seq_lens + draft_num
                kv_lens_sum = forward_batch.seq_lens_sum + draft_num * bs
                device = forward_batch.seq_lens.device

                qo_indptr = self.qo_indptr[: bs + 1]
                qo_indptr[: bs + 1] = torch.arange(
                    0,
                    (1 + bs) * draft_num,
                    step=draft_num,
                    dtype=torch.int32,
                    device=device,
                )
                kv_indptr = self.kv_indptr[: bs + 1]
                kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
                kv_indices = self._get_kv_indices_scratch(
                    kv_lens_sum,
                    device,
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

                # if self.kv_cache_dtype == fp8_dtype:
                if _use_mla_ps_kernel:
                    max_seqlen_qo = draft_num
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_decode_meta_data_buffer(max_seqlen_qo, bs)

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        self.kv_last_page_len[:bs],
                        work_metadata,
                        work_info_set,
                        work_indptr,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        max_seqlen_qo,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    # self.mla_indices_updater_prefill.kv_last_page_len,
                    self.kv_last_page_len[:bs],
                    draft_num,
                    None,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                    run_graph=False,
                )
            else:
                bs = len(forward_batch.req_pool_indices)
                draft_num = spec_info.draft_token_num

                if self._use_unified_verify:
                    page_table, qo_indptr, max_q_len, swa_page_table = (
                        self._build_verify_unified_metadata(
                            bs,
                            forward_batch.seq_lens,
                            forward_batch.req_pool_indices,
                            draft_num,
                        )
                    )
                    max_kv_len = page_table.shape[1] * self.page_size
                    self.forward_metadata = ForwardMetadata(
                        None,  # kv_indptr unused in unified-verify path
                        page_table,  # 2D block page_table stored in kv_indices
                        qo_indptr,
                        None,
                        max_q_len,
                        max_kv_len,
                        max_extend_len=max_q_len,
                        swa_page_table=swa_page_table,
                    )
                else:
                    qo_indptr = torch.arange(
                        0,
                        (1 + bs) * draft_num,
                        step=draft_num,
                        dtype=torch.int32,
                        device=self.device,
                    )

                    kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
                    kv_indptr = kv_indptr[: bs + 1]

                    kv_indices = torch.empty(
                        kv_indptr[-1], dtype=torch.int64, device=self.device
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

                    custom_mask = spec_info.custom_mask
                    seq_mask_len = draft_num * (forward_batch.seq_lens + draft_num)
                    mask_indptr = self.mask_indptr
                    mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
                    mask_indptr = mask_indptr[: bs + 1]

                    self.forward_metadata = ForwardMetadata(
                        kv_indptr,
                        kv_indices,
                        qo_indptr,
                        None,
                        draft_num,
                        None,
                        custom_mask=custom_mask,
                        mask_indptr=mask_indptr,
                        max_extend_len=draft_num,
                    )
        else:
            prefix_lens = forward_batch.extend_prefix_lens

            if self.is_multimodal:
                extend_no_prefix = False
            else:
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            if self.use_mla:
                self.mla_indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    forward_batch.extend_seq_lens,
                    max(forward_batch.extend_seq_lens_cpu),
                    forward_batch.seq_lens_cpu.max().item(),
                    spec_info=None,
                )

                max_q_len = self.mla_indices_updater_prefill.max_q_len
                qo_indptr = self.mla_indices_updater_prefill.qo_indptr
                kv_indptr = self.mla_indices_updater_prefill.kv_indptr

                work_metadata = None
                work_indptr = None
                work_info_set = None
                reduce_indptr = None
                reduce_final_map = None
                reduce_partial_map = None
                fp8_prefill_kv_indices = None

                if _use_fp8_prefill_attn:
                    tile_q = 256
                    qlen_granularity = tile_q // (self.num_head // self.num_kv_head)
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_prefill_ps_meta_data_buffer(
                        bs, max_q_len, qlen_granularity
                    )

                    self.make_mla_prefill_ps_meta_data(
                        qo_indptr,
                        kv_indptr,
                        forward_batch.seq_lens,
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        is_causal=True,
                    )

                    total_s = forward_batch.seq_lens_sum
                    fp8_prefill_kv_indices = torch.arange(
                        total_s, device=self.device, dtype=torch.int32
                    )

                self.forward_metadata = ForwardMetadata(
                    self.mla_indices_updater_prefill.kv_indptr,
                    self.mla_indices_updater_prefill.kv_indices,
                    qo_indptr,
                    self.kv_last_page_len[:bs],
                    max_q_len,
                    self.mla_indices_updater_prefill.max_kv_len,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    fp8_prefill_kv_indices=fp8_prefill_kv_indices,
                )
            else:
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    prefix_lens,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=None,
                )

                if self.use_sliding_window_kv_pool:
                    swa_page_table = (
                        self.token_to_kv_pool.translate_loc_from_full_to_swa(
                            self.indices_updater_prefill.kv_indices
                        )
                    )

                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    max(forward_batch.extend_seq_lens_cpu),
                    forward_batch.seq_lens_cpu.max().item(),
                    swa_page_table=swa_page_table,
                )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        # PR #20978 pads max_bs beyond pool_size for higher cuda-graph
        # coverage. Reallocate indptr buffers so they fit the padded max_bs.
        # See: https://github.com/sgl-project/sglang/pull/20978
        if max_bs + 1 > self.kv_indptr.shape[0]:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=self.device
            )
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=self.device
            )
            self.mask_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=self.device
            )
            if hasattr(self, "qo_indptr_"):
                self.qo_indptr_ = torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=self.device
                )

        self.cuda_graph_kv_last_page_len = torch.ones(
            max_bs, dtype=torch.int32, device=self.device
        )
        if kv_indices_buf is None:
            max_num_blocks_per_seq = (
                self.max_context_len + self.page_size - 1
            ) // self.page_size
            self.cuda_graph_kv_indices = torch.zeros(
                (max_bs * max_num_blocks_per_seq),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device=self.device,
            )

        # if self.use_mla and (_use_mla_ps_kernel or self.kv_cache_dtype == fp8_dtype):
        if self.use_mla and _use_mla_ps_kernel:
            # for persistent mla_decode_fwd
            max_seqlen_qo = (
                1 if self.num_draft_tokens is None else self.num_draft_tokens
            )

            (
                self.work_metadata,
                self.work_indptr,
                self.work_info_set,
                self.reduce_indptr,
                self.reduce_final_map,
                self.reduce_partial_map,
            ) = self.make_mla_decode_meta_data_buffer(max_seqlen_qo, max_bs)

        else:
            self.work_metadata = None
            self.work_indptr = None
            self.work_info_set = None

            self.reduce_indptr = None
            self.reduce_final_map = None
            self.reduce_partial_map = None

        if self.use_sliding_window_kv_pool:
            max_num_blocks_per_seq = (
                self.max_context_len + self.page_size - 1
            ) // self.page_size
            self.cuda_graph_swa_page_table = torch.zeros(
                (max_bs, max_num_blocks_per_seq),
                dtype=torch.int32,
                device=self.device,
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

        num_kv_splits = None
        # num_kv_splits_indptr = None

        work_metadata = None
        work_info_set = None
        work_indptr = None

        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None

        swa_page_table = None

        max_kv_len = torch.max(seq_lens).item()

        if forward_mode.is_decode_or_idle():
            qo_indptr = None
            kv_last_page_len = None
            max_q_len = None

            if spec_info is None or (
                self.use_triton_unified_attention and not self.use_mla
            ):
                max_num_blocks_per_seq = (
                    self.max_context_len + self.page_size - 1
                ) // self.page_size

                if not self.use_triton_unified_attention:
                    kv_indptr = self.kv_indptr
                    kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                    kv_indptr = kv_indptr[: bs + 1]
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
                    max_q_len = 1
                    kv_indices = self.cuda_graph_kv_indices.view(
                        -1, max_num_blocks_per_seq
                    )

                    if self.use_sliding_window_kv_pool:
                        swa_page_table = self.cuda_graph_swa_page_table

                    if spec_info is not None:
                        self._build_unified_page_table_from_spec(
                            spec_info,
                            bs,
                            dest_buf=kv_indices,
                            swa_dest_buf=swa_page_table,
                        )
                    else:
                        page_indices = self.req_to_token[
                            req_pool_indices[:bs], :max_kv_len
                        ]

                        if self.use_sliding_window_kv_pool:
                            swa_page_indices = (
                                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                    page_indices
                                )
                            )

                            page_indices = self._transform_table_1_to_real(page_indices)
                            swa_page_indices = self._transform_table_1_to_real(
                                swa_page_indices
                            )

                            new_rows = swa_page_indices.shape[0]
                            new_cols = swa_page_indices.shape[1]

                            kv_indices[:new_rows, :new_cols].copy_(page_indices)
                            swa_page_table = self.cuda_graph_swa_page_table
                            swa_page_table[:new_rows, :new_cols].copy_(swa_page_indices)
                        elif self.page_size > 1:
                            page_indices = self._transform_table_1_to_real(page_indices)
                            new_rows = page_indices.shape[0]
                            new_cols = page_indices.shape[1]
                            kv_indices[:new_rows, :new_cols].copy_(page_indices)

                    qo_indptr = self.qo_indptr_unified_decode[: bs + 1]

                    kv_indptr = None
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

            if self.use_mla:
                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(
                    self.cuda_graph_kv_last_page_len[:bs], dim=0
                )
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = 1

                if _use_mla_ps_kernel:
                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
                swa_page_table=swa_page_table,
            )

        elif forward_mode.is_target_verify():
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            if self.use_mla:
                kv_lens = seq_lens + self.num_draft_tokens
            else:
                kv_lens = seq_lens
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
            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = self.num_draft_tokens

            if self.use_mla:
                if _use_mla_ps_kernel:
                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    max_kv_len,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                )
            else:
                if self._use_unified_verify:
                    max_num_blocks_per_seq = (
                        self.max_context_len + self.page_size - 1
                    ) // self.page_size
                    page_table = self.cuda_graph_kv_indices.view(
                        -1, max_num_blocks_per_seq
                    )[:bs]

                    swa_page_table = None

                    if self.use_sliding_window_kv_pool:
                        swa_page_table = self.cuda_graph_swa_page_table.view(
                            -1, max_num_blocks_per_seq
                        )[:bs]

                    _page_table, _qo_indptr, _max_q_len, _swa_page_table = (
                        self._build_verify_unified_metadata(
                            bs,
                            seq_lens,
                            req_pool_indices,
                            self.num_draft_tokens,
                            page_table_dest=page_table,
                            swa_page_table_dest=swa_page_table,
                        )
                    )
                    max_kv_len = max_num_blocks_per_seq * self.page_size
                    self.forward_metadata = ForwardMetadata(
                        None,
                        _page_table,
                        _qo_indptr,
                        kv_last_page_len,
                        _max_q_len,
                        max_kv_len,
                        max_extend_len=_max_q_len,
                        swa_page_table=_swa_page_table,
                    )
                else:
                    custom_mask = self.cuda_graph_custom_mask
                    custom_mask[: spec_info.custom_mask.shape[0]] = (
                        spec_info.custom_mask
                    )
                    seq_mask_len = max_q_len * (seq_lens + max_q_len)
                    mask_indptr = self.mask_indptr
                    mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
                    mask_indptr = mask_indptr[: bs + 1]

                    self.forward_metadata = ForwardMetadata(
                        kv_indptr,
                        kv_indices,
                        qo_indptr,
                        kv_last_page_len,
                        max_q_len,
                        max_kv_len,
                        custom_mask=custom_mask,
                        mask_indptr=mask_indptr,
                        max_extend_len=max_q_len,
                    )
        elif forward_mode.is_draft_extend_v2():
            # EAGLE V2: Uses fixed num_draft_tokens per batch
            self._ensure_spec_v2_topk_supported()
            num_tokens_per_bs = self._resolve_v2_num_draft_tokens()
            qo_indptr = self._set_uniform_qo_indptr(bs, num_tokens_per_bs, self.device)
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

            if self.use_mla and _use_mla_ps_kernel:
                num_kv_splits = self.max_split_per_batch

                self.make_mla_meta_data(
                    qo_indptr,
                    kv_indptr,
                    kv_last_page_len,
                    self.work_metadata,
                    self.work_info_set,
                    self.work_indptr,
                    self.reduce_indptr,
                    self.reduce_final_map,
                    self.reduce_partial_map,
                    max_q_len,
                    fast_mode=fast_mode,
                    max_split_per_batch=num_kv_splits,
                    intra_batch_mode=intra_batch_mode,
                )

                work_metadata = self.work_metadata
                work_info_set = self.work_info_set
                work_indptr = self.work_indptr

                reduce_indptr = self.reduce_indptr
                reduce_final_map = self.reduce_final_map
                reduce_partial_map = self.reduce_partial_map

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
            )
        elif forward_mode.is_draft_extend():
            # EAGLE V1: Uses speculative_num_steps + 1
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

            if self.use_mla:
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = num_tokens_per_bs

                if _use_mla_ps_kernel:
                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    max_kv_len,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                )
            else:
                # Non-MLA draft_extend cuda graph: use triton extend kernel
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    None,
                    num_tokens_per_bs,
                    None,
                    custom_mask=None,
                    mask_indptr=None,
                    max_extend_len=num_tokens_per_bs,
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
    ):

        num_kv_splits = None
        # num_kv_splits_indptr = None

        work_metadata = None
        work_info_set = None
        work_indptr = None

        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None

        swa_page_table = None
        max_kv_len = seq_lens_cpu.max().item()

        if forward_mode.is_decode_or_idle():
            qo_indptr = None
            kv_last_page_len = None
            max_q_len = None

            if spec_info is None or (
                self.use_triton_unified_attention and not self.use_mla
            ):
                max_num_blocks_per_seq = (
                    self.max_context_len + self.page_size - 1
                ) // self.page_size

                if not self.use_triton_unified_attention:
                    kv_indptr = self.kv_indptr
                    kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                    kv_indptr = kv_indptr[: bs + 1]
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
                    max_q_len = 1
                    kv_indices = self.cuda_graph_kv_indices.view(
                        -1, max_num_blocks_per_seq
                    )

                    if self.use_sliding_window_kv_pool:
                        swa_page_table = self.cuda_graph_swa_page_table

                    if spec_info is not None:
                        self._build_unified_page_table_from_spec(
                            spec_info,
                            bs,
                            dest_buf=kv_indices,
                            swa_dest_buf=swa_page_table,
                        )
                    else:
                        page_indices = self.req_to_token[
                            req_pool_indices[:bs], :max_kv_len
                        ]

                        if self.use_sliding_window_kv_pool:
                            swa_page_indices = (
                                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                    page_indices
                                )
                            )

                            page_indices = self._transform_table_1_to_real(page_indices)
                            swa_page_indices = self._transform_table_1_to_real(
                                swa_page_indices
                            )

                            new_rows = swa_page_indices.shape[0]
                            new_cols = swa_page_indices.shape[1]

                            kv_indices[:new_rows, :new_cols].copy_(page_indices)
                            swa_page_table = self.cuda_graph_swa_page_table
                            swa_page_table[:new_rows, :new_cols].copy_(swa_page_indices)
                        elif self.page_size > 1:
                            page_indices = self._transform_table_1_to_real(page_indices)
                            new_rows = page_indices.shape[0]
                            new_cols = page_indices.shape[1]
                            kv_indices[:new_rows, :new_cols].copy_(page_indices)

                    qo_indptr = self.qo_indptr_unified_decode[: bs + 1]

                    kv_indptr = None
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

            if self.use_mla:
                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(
                    self.cuda_graph_kv_last_page_len[:bs], dim=0
                )
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = 1

                if _use_mla_ps_kernel:
                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
                swa_page_table=swa_page_table,
                # num_kv_splits_indptr=num_kv_splits_indptr,
            )

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
            if self.use_mla:
                kv_lens = seq_lens + self.num_draft_tokens
            else:
                kv_lens = seq_lens
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
            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = self.num_draft_tokens

            if self.use_mla:
                if _use_mla_ps_kernel:
                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    max_kv_len,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                )
            else:
                if self._use_unified_verify:
                    max_num_blocks_per_seq = (
                        self.max_context_len + self.page_size - 1
                    ) // self.page_size
                    page_table = self.cuda_graph_kv_indices.view(
                        -1, max_num_blocks_per_seq
                    )[:bs]

                    swa_page_table = None

                    if self.use_sliding_window_kv_pool:
                        swa_page_table = self.cuda_graph_swa_page_table.view(
                            -1, max_num_blocks_per_seq
                        )[:bs]

                    _page_table, _qo_indptr, _max_q_len, _swa_page_table = (
                        self._build_verify_unified_metadata(
                            bs,
                            seq_lens,
                            req_pool_indices,
                            self.num_draft_tokens,
                            page_table_dest=page_table,
                            swa_page_table_dest=swa_page_table,
                        )
                    )

                    max_kv_len_unified = max_num_blocks_per_seq * self.page_size
                    self.forward_metadata = ForwardMetadata(
                        None,
                        _page_table,
                        _qo_indptr,
                        kv_last_page_len,
                        _max_q_len,
                        max_kv_len_unified,
                        max_extend_len=_max_q_len,
                        swa_page_table=_swa_page_table,
                    )
                else:
                    custom_mask = self.cuda_graph_custom_mask
                    custom_mask[: spec_info.custom_mask.shape[0]] = (
                        spec_info.custom_mask
                    )
                    seq_mask_len = max_q_len * (seq_lens + max_q_len)
                    mask_indptr = self.mask_indptr[: bs + 1]
                    mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)

                    self.forward_metadata = ForwardMetadata(
                        kv_indptr,
                        kv_indices,
                        qo_indptr,
                        kv_last_page_len,
                        max_q_len,
                        max_kv_len,
                        custom_mask=custom_mask,
                        mask_indptr=mask_indptr,
                        max_extend_len=max_q_len,
                    )
        elif forward_mode.is_draft_extend_v2():
            # EAGLE V2: Fixed num_draft_tokens per batch
            self._ensure_spec_v2_topk_supported()
            seq_lens = seq_lens[:bs]
            num_tokens_per_bs = self._resolve_v2_num_draft_tokens()
            extend_lens = torch.full(
                (bs,), num_tokens_per_bs, dtype=torch.int32, device=seq_lens.device
            )

            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
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

            if self.use_mla and _use_mla_ps_kernel:
                num_kv_splits = self.max_split_per_batch

                self.make_mla_meta_data(
                    qo_indptr,
                    kv_indptr,
                    kv_last_page_len,
                    self.work_metadata,
                    self.work_info_set,
                    self.work_indptr,
                    self.reduce_indptr,
                    self.reduce_final_map,
                    self.reduce_partial_map,
                    max_q_len,
                    fast_mode=fast_mode,
                    max_split_per_batch=num_kv_splits,
                    intra_batch_mode=intra_batch_mode,
                )

                work_metadata = self.work_metadata
                work_info_set = self.work_info_set
                work_indptr = self.work_indptr

                reduce_indptr = self.reduce_indptr
                reduce_final_map = self.reduce_final_map
                reduce_partial_map = self.reduce_partial_map

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
            )
        elif forward_mode.is_draft_extend():
            # EAGLE V1: Uses spec_info.num_accept_tokens
            num_tokens_per_bs = self.speculative_num_steps + 1
            seq_lens = seq_lens[:bs]
            extend_lens = spec_info.num_accept_tokens[:bs]
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
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

            if self.use_mla and _use_mla_ps_kernel:
                num_kv_splits = self.max_split_per_batch

                self.make_mla_meta_data(
                    qo_indptr,
                    kv_indptr,
                    kv_last_page_len,
                    self.work_metadata,
                    self.work_info_set,
                    self.work_indptr,
                    self.reduce_indptr,
                    self.reduce_final_map,
                    self.reduce_partial_map,
                    max_q_len,
                    fast_mode=fast_mode,
                    max_split_per_batch=num_kv_splits,
                    intra_batch_mode=intra_batch_mode,
                )

                work_metadata = self.work_metadata
                work_info_set = self.work_info_set
                work_indptr = self.work_indptr

                reduce_indptr = self.reduce_indptr
                reduce_final_map = self.reduce_final_map
                reduce_partial_map = self.reduce_partial_map

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
            )

        else:
            raise ValueError("Invalid forward mode")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1 if self.num_draft_tokens is None else self.num_draft_tokens

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        # AITER verify path does not require post-draft buffer patching currently.
        # This override prevents overlap-plan stream mode from failing with the
        # base class NotImplementedError.
        pass

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        self.logits_soft_cap = layer.logit_cap

        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        k_descale = None
        v_descale = None
        if self.kv_cache_dtype == fp8_dtype:
            k_descale = layer.k_scale if layer.k_scale is not None else self.k_scale
            v_descale = layer.v_scale if layer.v_scale is not None else self.k_scale

        if k is not None:
            assert v is not None
            if save_kv_cache:
                # Only use SWA-specific kv cache write (reshape_and_cache_flash) when
                # both unified attention and sliding window kv pool are active.
                # Non-SWA models (e.g. Qwen3-VL) enabled via SGLANG_USE_AITER_UNIFIED_ATTN
                # use standard set_kv_buffer, as they lack SWA-specific attributes
                # like full_to_swa_index_mapping.
                if (
                    self.use_triton_unified_attention
                    and self.use_sliding_window_kv_pool
                ):
                    token_to_kv_pool = forward_batch.token_to_kv_pool
                    k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                        layer.layer_id
                    )
                    slot_mapping_swa = token_to_kv_pool.full_to_swa_index_mapping

                    launch_reshape_and_cache_flash(
                        k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                        v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                        k_cache.view(
                            -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                        ),
                        v_cache.view(
                            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                        ),
                        cache_loc,
                        (
                            slot_mapping_swa.long()
                            if layer.sliding_window_size > 0
                            else None
                        ),
                        k_scale=k_descale,
                        v_scale=v_descale,
                    )
                elif self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
                else:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, k_descale, v_descale
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
                and not forward_batch.forward_mode.is_draft_extend_v2()
            ):
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
                if kv_indices.shape[0] == 0 or extend_no_prefix:
                    if _use_fp8_prefill_attn:
                        output = self.mla_fp8_prefill_attn(
                            q,
                            k,
                            v,
                            layer,
                        )
                    else:
                        output = flash_attn_varlen_func(
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
                    return output
                elif layer.qk_head_dim != (kv_lora_rank + qk_rope_head_dim):
                    K_Buffer = torch.index_select(K_Buffer, 0, kv_indices)
                    kvc, k_pe = torch.split(
                        K_Buffer, [kv_lora_rank, qk_rope_head_dim], dim=-1
                    )

                    if self.kv_cache_dtype == fp8_dtype:
                        dtype = q.dtype

                        kvc = kvc.to(dtype)
                        k_pe = k_pe.to(dtype)

                    if (
                        _use_fp8_prefill_attn
                        and layer.kv_b_proj.weight.dtype == torch.uint8
                    ):
                        # MXFP4 weights + FP8 prefill: fuse GEMM, nope/v split, and k_pe cat
                        # into a single kernel (fused_gemm_afp4wfp4_split_cat) that writes k and v
                        # directly in FP8, avoiding a separate elementwise cast
                        k, v = layer.kv_b_proj(
                            (
                                kvc.squeeze(1),
                                k_pe.expand(-1, layer.tp_k_head_num, -1),
                                qk_nope_head_dim,
                                layer.v_head_dim,
                                fp8_dtype,
                            )
                        )[0]
                    else:
                        kv = layer.kv_b_proj(kvc.contiguous())[0]

                        kv = kv.view(
                            -1, layer.tp_k_head_num, qk_nope_head_dim + layer.v_head_dim
                        )
                        k, v = torch.split(
                            kv, [qk_nope_head_dim, layer.v_head_dim], dim=-1
                        )
                        k = torch.cat(
                            [
                                k,
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

                    if _use_fp8_prefill_attn:
                        return self.mla_fp8_prefill_attn(q, k, v, layer)
                    else:
                        return flash_attn_varlen_func(
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
                work_metadata = self.forward_metadata.work_metadata
                work_indptr = self.forward_metadata.work_indptr
                work_info_set = self.forward_metadata.work_info_set

                reduce_indptr = self.forward_metadata.reduce_indptr
                reduce_final_map = self.forward_metadata.reduce_final_map
                reduce_partial_map = self.forward_metadata.reduce_partial_map

                num_kv_splits = self.forward_metadata.num_kv_splits

                o = self._mla_decode_fwd_with_head_pad(
                    q,
                    K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                    layer,
                    qo_indptr=self.forward_metadata.qo_indptr,
                    kv_indptr=self.forward_metadata.kv_indptr,
                    kv_indices=self.forward_metadata.kv_indices,
                    kv_last_page_lens=self.forward_metadata.kv_last_page_len,
                    max_seqlen_q=self.forward_metadata.max_q_len,
                    sm_scale=layer.scaling,
                    logit_cap=layer.logit_cap,
                    work_meta_data=work_metadata,
                    work_indptr=work_indptr,
                    work_info_set=work_info_set,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    q_scale=k_descale,
                    kv_scale=k_descale,
                    intra_batch_mode=intra_batch_mode,
                    num_kv_splits=num_kv_splits,
                )
                return o
            elif (
                forward_batch.forward_mode.is_draft_extend()
                or forward_batch.forward_mode.is_draft_extend_v2()
            ):
                work_metadata = self.forward_metadata.work_metadata
                work_indptr = self.forward_metadata.work_indptr
                work_info_set = self.forward_metadata.work_info_set

                reduce_indptr = self.forward_metadata.reduce_indptr
                reduce_final_map = self.forward_metadata.reduce_final_map
                reduce_partial_map = self.forward_metadata.reduce_partial_map

                num_kv_splits = self.forward_metadata.num_kv_splits

                if self.forward_metadata.run_graph is not True:
                    bs, q_pad, q_mask = pad_sequence_with_mask(
                        q.view(q.shape[0], -1),
                        qo_indptr[:-1],
                        forward_batch.extend_seq_lens,
                        self.forward_metadata.max_q_len,
                    )
                    o = self._mla_decode_fwd_with_head_pad(
                        q_pad.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        layer,
                        qo_indptr=self.forward_metadata.qo_indptr,
                        kv_indptr=self.forward_metadata.kv_indptr,
                        kv_indices=self.forward_metadata.kv_indices,
                        kv_last_page_lens=self.forward_metadata.kv_last_page_len,
                        max_seqlen_q=self.forward_metadata.max_q_len,
                        sm_scale=layer.scaling,
                        logit_cap=layer.logit_cap,
                        work_meta_data=work_metadata,
                        work_indptr=work_indptr,
                        work_info_set=work_info_set,
                        reduce_indptr=reduce_indptr,
                        reduce_final_map=reduce_final_map,
                        reduce_partial_map=reduce_partial_map,
                        q_scale=k_descale,
                        kv_scale=k_descale,
                        intra_batch_mode=intra_batch_mode,
                        num_kv_splits=num_kv_splits,
                    )

                    total_valid_q = int(qo_indptr[-1].item())
                    return o[:total_valid_q]
                else:
                    o = self._mla_decode_fwd_with_head_pad(
                        q,
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        layer,
                        qo_indptr=self.forward_metadata.qo_indptr,
                        kv_indptr=self.forward_metadata.kv_indptr,
                        kv_indices=self.forward_metadata.kv_indices,
                        kv_last_page_lens=self.forward_metadata.kv_last_page_len,
                        max_seqlen_q=self.forward_metadata.max_q_len,
                        sm_scale=layer.scaling,
                        logit_cap=layer.logit_cap,
                        work_meta_data=work_metadata,
                        work_indptr=work_indptr,
                        work_info_set=work_info_set,
                        reduce_indptr=reduce_indptr,
                        reduce_final_map=reduce_final_map,
                        reduce_partial_map=reduce_partial_map,
                        q_scale=k_descale,
                        kv_scale=k_descale,
                        intra_batch_mode=intra_batch_mode,
                        num_kv_splits=num_kv_splits,
                    )
                    return o
            else:
                raise ValueError(
                    f"Invalid forward mode for MLA prefill: {forward_batch.forward_mode=}"
                )
        else:
            if (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend()
            ):
                if layer.qk_head_dim != layer.v_head_dim:
                    o = q.new_empty(
                        (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                    )
                else:
                    o = torch.empty_like(q)

                # target_verify goes through unified_attention when topk == 1
                # (the linear draft chain gives a pure causal mask). MLA and
                # draft_extend still use the legacy extend_attention_fwd path.
                if (
                    self._use_unified_verify
                    and forward_batch.forward_mode.is_target_verify()
                ):
                    k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                        layer.layer_id
                    )
                    page_table = self.forward_metadata.kv_indices
                    max_kv_len = page_table.shape[1] * self.page_size

                    window_size = (-1, -1)

                    if (
                        layer.sliding_window_size is not None
                        and layer.sliding_window_size > -1
                    ):
                        window_size = (layer.sliding_window_size - 1, 0)
                        if self.forward_metadata.swa_page_table is not None:
                            page_table = self.forward_metadata.swa_page_table

                    q_unified = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
                    k_unified = k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                    )
                    v_unified = v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                    )
                    if layer.tp_k_head_num == 1 and layer.tp_q_head_num > 1:
                        # Qwen3.5 can replicate one KV head across multiple TP ranks.
                        # Present the local KV head as per-Q-head stride-0 views so
                        # target_verify uses the same local head mapping as the model.
                        k_unified = k_unified.expand(-1, -1, layer.tp_q_head_num, -1)
                        v_unified = v_unified.expand(-1, -1, layer.tp_q_head_num, -1)

                    # The seq_lens + draft_num add has to run INSIDE the graph
                    # region; a host-side pre-add would allocate a new tensor
                    # each replay and break the captured pointer.
                    unified_attention(
                        q=q_unified,
                        k=k_unified,
                        v=v_unified,
                        out=o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                        cu_seqlens_q=self.forward_metadata.qo_indptr,
                        seqused_k=forward_batch.seq_lens + self.num_draft_tokens,
                        max_seqlen_q=self.forward_metadata.max_q_len,
                        max_seqlen_k=max_kv_len,
                        softmax_scale=layer.scaling,
                        causal=True,
                        window_size=window_size,
                        block_table=page_table,
                        softcap=layer.logit_cap,
                        q_descale=None,
                        k_descale=k_descale,
                        v_descale=v_descale,
                        sinks=sinks,
                    )
                    return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

                self.extend_attention_fwd(
                    q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k.contiguous(),
                    v.contiguous(),
                    o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                    forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                    self.forward_metadata.qo_indptr,
                    self.forward_metadata.kv_indptr,
                    self.forward_metadata.kv_indices,
                    self.forward_metadata.custom_mask,
                    True,  # causal
                    self.forward_metadata.mask_indptr,
                    self.forward_metadata.max_extend_len,
                    1.0,  # k_scale
                    1.0,  # v_scale
                    layer.scaling,
                    logit_cap=layer.logit_cap,
                )
                return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )

            bs0 = forward_batch.batch_size + 1

            # To keep the mha_batch_prefill_func function parameters
            # declare the necessary parameter and assign None as default value
            q_descale = None

            # TODO kkhuang-amd need to remove it when mha_batch_prefill_func support fp8-kv
            if self.kv_cache_dtype == fp8_dtype:
                q = q.to(fp8_dtype)
                q_descale = layer.k_scale if layer.k_scale is not None else self.k_scale

            window_size = (-1, -1)
            page_table = self.forward_metadata.kv_indices

            if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
                window_size = (layer.sliding_window_size, -1)
                if self.forward_metadata.swa_page_table is not None:
                    page_table = self.forward_metadata.swa_page_table

            o = mha_batch_prefill_func(
                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k_cache,
                v_cache,
                self.qo_indptr[:bs0],
                self.forward_metadata.kv_indptr[:bs0],
                page_table,
                self.forward_metadata.max_q_len,
                self.forward_metadata.max_kv_len,
                causal=True,
                logits_soft_cap=self.logits_soft_cap,
                alibi_slopes=None,
                return_lse=False,
                return_attn_probs=False,
                window_size=window_size,
                sink_ptr=sinks,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
            )

            # The fp8bf16 aiter prefill kernel returns bf16 even when the
            # model computes in fp16. Cast back so the attention output keeps
            # the same dtype as the rest of the model activations.
            if o.dtype != self.input_dtype:
                o = o.to(self.input_dtype)

            return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        k_descale = None
        v_descale = None
        if self.kv_cache_dtype == fp8_dtype:
            k_descale = layer.k_scale if layer.k_scale is not None else self.k_scale
            v_descale = layer.v_scale if layer.v_scale is not None else self.k_scale

        if save_kv_cache:
            # Only use SWA-specific kv cache write (reshape_and_cache_flash) when
            # both unified attention and sliding window kv pool are active.
            # Non-SWA models (e.g. Qwen3-VL) enabled via SGLANG_USE_AITER_UNIFIED_ATTN
            # use standard set_kv_buffer, as they lack SWA-specific attributes
            # like full_to_swa_index_mapping.
            if self.use_triton_unified_attention and self.use_sliding_window_kv_pool:
                token_to_kv_pool = forward_batch.token_to_kv_pool
                k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                    layer.layer_id
                )
                slot_mapping_swa = token_to_kv_pool.full_to_swa_index_mapping

                launch_reshape_and_cache_flash(
                    k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                    k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                    ),
                    v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                    ),
                    forward_batch.out_cache_loc,
                    slot_mapping_swa.long() if layer.sliding_window_size > 0 else None,
                    k_scale=k_descale,
                    v_scale=v_descale,
                )
            elif self.use_triton_unified_attention and self.kv_cache_dtype == fp8_dtype:
                # [PATCH] FP8 non-SWA: use launch_reshape_and_cache_flash to
                # fuse bf16→fp8 cast + paged write in one Triton kernel,
                # eliminating separate float8_copy + store_kvcache overhead.
                token_to_kv_pool = forward_batch.token_to_kv_pool
                k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
                launch_reshape_and_cache_flash(
                    k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                    k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                    ),
                    v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                    ),
                    forward_batch.out_cache_loc,
                )
            else:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )

        if self.use_mla:
            k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

            work_metadata = self.forward_metadata.work_metadata
            work_indptr = self.forward_metadata.work_indptr
            work_info_set = self.forward_metadata.work_info_set

            reduce_indptr = self.forward_metadata.reduce_indptr
            reduce_final_map = self.forward_metadata.reduce_final_map
            reduce_partial_map = self.forward_metadata.reduce_partial_map

            num_kv_splits = self.forward_metadata.num_kv_splits

            o = self._mla_decode_fwd_with_head_pad(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k_buffer.view(-1, 1, 1, layer.qk_head_dim),
                layer,
                qo_indptr=self.forward_metadata.qo_indptr,
                kv_indptr=self.forward_metadata.kv_indptr,
                kv_indices=self.forward_metadata.kv_indices,
                kv_last_page_lens=self.forward_metadata.kv_last_page_len,
                max_seqlen_q=self.forward_metadata.max_q_len,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                work_meta_data=work_metadata,
                work_indptr=work_indptr,
                work_info_set=work_info_set,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                q_scale=k_descale,
                kv_scale=k_descale,
                intra_batch_mode=intra_batch_mode,
                num_kv_splits=num_kv_splits,
            )
        else:
            self.logits_soft_cap = layer.logit_cap

            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )

            if layer.qk_head_dim != layer.v_head_dim:
                o = q.new_empty(
                    (q.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                    dtype=self.input_dtype,
                )
            else:
                o = torch.empty_like(q, dtype=self.input_dtype)

            if self.use_triton_unified_attention:
                bs = forward_batch.batch_size
                window_size = (-1, -1)
                page_table = self.forward_metadata.kv_indices

                if (
                    layer.sliding_window_size is not None
                    and layer.sliding_window_size > -1
                ):
                    window_size = (layer.sliding_window_size - 1, 0)
                    if self.forward_metadata.swa_page_table is not None:
                        page_table = self.forward_metadata.swa_page_table

                max_kv_len = page_table.shape[1] * self.page_size

                unified_attention(
                    q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k=k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                    ),
                    v=v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                    ),
                    out=o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    cu_seqlens_q=self.forward_metadata.qo_indptr,
                    seqused_k=forward_batch.seq_lens,
                    max_seqlen_q=self.forward_metadata.max_q_len,
                    max_seqlen_k=max_kv_len,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=window_size,
                    block_table=page_table,
                    softcap=0,
                    q_descale=None,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    sinks=sinks,
                )
            else:
                # Drop FP8 KV upcast: keep paged cache in native FP8 and use ``fp8_e4m3`` for
                # in-kernel dequant in ``paged_attention_ragged``. (HIP maps CLI e5m2/e4m3 to
                # ``fp8_dtype``; aiter has no ``fp8_e5m2`` string.)
                aiter_kv_str = self._get_aiter_paged_ragged_kv_cache_dtype()

                paged_attention_ragged(
                    o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    self.workspace_buffer,
                    q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k_cache.view(-1, 1, layer.tp_k_head_num, layer.qk_head_dim),
                    v_cache.view(-1, 1, layer.tp_v_head_num, layer.v_head_dim),
                    self.scale,
                    self.forward_metadata.kv_indptr,
                    self.forward_metadata.kv_indices,
                    self.kv_last_page_len,
                    1,
                    self.max_num_partitions,
                    None,
                    aiter_kv_str,
                    "NHD",
                    self.logits_soft_cap,
                    self.k_scale,
                    self.v_scale,
                    None,
                    _AITER_PARTITION_SIZE_ROCM,
                )

        return o


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

            token_num = kv_indptr[-1]
            kv_indices[token_num:] = kv_indices[0]

            extend_lens = seq_lens - prefix_lens

            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
        else:
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
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
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
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
                    topk=topk,
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
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)
