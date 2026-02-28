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
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
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
except ImportError:
    print(
        "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
    )

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.utils import pad_sequence_with_mask
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
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


global_workspace_buffer = None

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

        self.input_dtype = model_runner.model_config.dtype

        self.page_size = model_runner.server_args.page_size

        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)

        self.device = model_runner.device
        self.is_multimodal = model_runner.model_config.is_multimodal
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
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
        self.mask_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int64, device=model_runner.device
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

        self.scale = float(1.0 / (self.head_dim**0.5))
        self.k_scale = self.v_scale = torch.tensor([1.0], dtype=torch.float32).to(
            self.device
        )

        self.logits_soft_cap = 0.0

        self.forward_metadata: ForwardMetadata = None

        if self.use_mla:
            self.enable_dp_attention = is_dp_attention_enabled()
            self.qo_indptr_ = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
            global _use_mla_ps_kernel, fast_mode, intra_batch_mode

            if self.num_head == 32:
                fast_mode = True
                intra_batch_mode = False

            # current persist a16w16 mla_decode kernel does not support head_num = 128
            # need to fall back to non-persist
            # only use mla_ps_kernel when fp8 kv_cache
            # for non-fp8 kv_cache on tp8, use non-persist kernel to avoid performance degradation
            # head_num=16 (tp8 perf issue), head_num=128 (unsupported, like tp1 or --enable-dp-attention with tp8-dp8)
            if (
                self.num_head == 16 or self.num_head == 128
            ) and self.kv_cache_dtype is not fp8_dtype:
                _use_mla_ps_kernel = False
                fast_mode = False
                intra_batch_mode = False

            self.max_split_per_batch = 32 if _use_mla_ps_kernel else None

            if self.num_draft_tokens is None and _use_mla_ps_kernel:
                self.max_split_per_batch = 64

            self.fix_max_split_per_batch = self.max_split_per_batch

    def make_mla_decode_meta_data_buffer(self, max_seqlen_qo, batch_size):
        nhead = self.num_head
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
            self.num_head // nhead_kv,
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

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for triton attention backend."""

        bs = forward_batch.batch_size
        kv_indptr = self.kv_indptr
        spec_info = forward_batch.spec_info
        qo_indptr = None
        kv_last_page_len = None
        max_q_len = None

        work_metadata = None
        work_indptr = None
        work_info_set = None
        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None

        num_kv_splits = None
        # num_kv_splits_indptr = None

        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None or forward_batch.forward_mode.is_idle():
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

        elif forward_batch.forward_mode.is_draft_extend():
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
                draft_max_extend_len = torch.max(spec_info.accept_length).item()

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
                # Non-MLA target_verify: use triton extend kernel with custom mask
                bs = len(forward_batch.req_pool_indices)
                draft_num = spec_info.draft_token_num

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
                    forward_batch.extend_seq_lens.max().item(),
                    forward_batch.seq_lens.max().item(),
                    spec_info=None,
                )

                max_q_len = self.mla_indices_updater_prefill.max_q_len
                qo_indptr = self.mla_indices_updater_prefill.qo_indptr

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
                        qo_indptr,
                        forward_batch.seq_lens,
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        is_causal=True,
                    )

                    total_s = int(forward_batch.extend_seq_lens.sum())
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
                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    self.indices_updater_prefill.max_q_len,
                    self.indices_updater_prefill.max_kv_len,
                )

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

        if forward_mode.is_decode_or_idle():
            qo_indptr = None
            kv_last_page_len = None
            max_q_len = None

            if spec_info is None:
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
                kv_indptr[-1].item(),
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
                # num_kv_splits_indptr=num_kv_splits_indptr,
            )

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

                # if self.kv_cache_dtype == fp8_dtype:
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
                    kv_indptr[-1].item(),
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                    # num_kv_splits_indptr=num_kv_splits_indptr,
                )
            else:
                # Non-MLA target_verify cuda graph: use triton extend kernel metadata
                draft_num = self.num_draft_tokens
                qo_indptr = self.qo_indptr[: bs + 1]
                qo_indptr[: bs + 1] = torch.arange(
                    0,
                    (1 + bs) * draft_num,
                    step=draft_num,
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

                custom_mask = self.cuda_graph_custom_mask
                custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
                seq_mask_len = draft_num * (seq_lens + draft_num)
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
                    kv_indptr[-1].item(),
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

        if forward_mode.is_decode_or_idle():
            qo_indptr = None
            kv_last_page_len = None
            max_q_len = None

            if spec_info is None:
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
                    kv_indptr[-1].item(),
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
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
            if not self.use_mla:
                # Non-MLA: update custom_mask and mask_indptr for triton extend kernel
                custom_mask = self.cuda_graph_custom_mask
                custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
                seq_mask_len = self.num_draft_tokens * (
                    seq_lens + self.num_draft_tokens
                )
                mask_indptr = self.mask_indptr[: bs + 1]
                mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)

            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = self.num_draft_tokens

            # if self.kv_cache_dtype == fp8_dtype:
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
                kv_indptr[-1].item(),
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
                # num_kv_splits_indptr=num_kv_splits_indptr,
            )

        elif forward_mode.is_draft_extend():
            num_tokens_per_bs = self.speculative_num_steps + 1
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
                kv_indptr[-1].item(),
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
                # num_kv_splits_indptr=num_kv_splits_indptr,
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

        if k is not None:
            assert v is not None
            if save_kv_cache:
                if self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
                else:
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
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
                if kv_indices.shape[0] == 0 or extend_no_prefix:
                    if _use_fp8_prefill_attn:
                        total_s = q.shape[0]
                        nhead = layer.tp_q_head_num
                        v_head_dim = layer.v_head_dim

                        # q is cast here (after RoPE).
                        # k/v are already FP8 for MXFP4 main model (fused kv_b_proj),
                        # but need casting for FP8/BF16 weights (e.g. MTP draft model).
                        if q.dtype != fp8_dtype:
                            q = q.to(fp8_dtype)
                        if k.dtype != fp8_dtype:
                            k = k.to(fp8_dtype)
                        if v.dtype != fp8_dtype:
                            v = v.to(fp8_dtype)
                        one_scale = torch.ones((), dtype=torch.float32, device=q.device)

                        kv_indptr_asm = qo_indptr
                        kv_indices_asm = self.forward_metadata.fp8_prefill_kv_indices

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
                            (total_s, nhead),
                            dtype=torch.float32,
                            device=q.device,
                        )
                        output = q.new_empty(
                            (total_s, nhead, v_head_dim),
                            dtype=self.input_dtype,
                        )

                        mla_prefill_ps_asm_fwd(
                            q,
                            k,
                            v,
                            qo_indptr,
                            kv_indptr_asm,
                            kv_indices_asm,
                            self.forward_metadata.work_indptr,
                            self.forward_metadata.work_info_set,
                            max_q_len,
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
                o = q.new_empty(
                    (q.shape[0], layer.tp_q_head_num, layer.v_head_dim),
                    dtype=self.input_dtype,
                )

                work_metadata = self.forward_metadata.work_metadata
                work_indptr = self.forward_metadata.work_indptr
                work_info_set = self.forward_metadata.work_info_set

                reduce_indptr = self.forward_metadata.reduce_indptr
                reduce_final_map = self.forward_metadata.reduce_final_map
                reduce_partial_map = self.forward_metadata.reduce_partial_map

                num_kv_splits = self.forward_metadata.num_kv_splits

                mla_decode_fwd(
                    q,
                    K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                    o,
                    self.forward_metadata.qo_indptr,
                    self.forward_metadata.kv_indptr,
                    self.forward_metadata.kv_indices,
                    self.forward_metadata.kv_last_page_len,
                    self.forward_metadata.max_q_len,
                    sm_scale=layer.scaling,
                    logit_cap=layer.logit_cap,
                    work_meta_data=work_metadata,
                    work_indptr=work_indptr,
                    work_info_set=work_info_set,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    q_scale=layer.k_scale,
                    kv_scale=layer.k_scale,
                    intra_batch_mode=intra_batch_mode,
                    num_kv_splits=num_kv_splits,
                )
                return o
            elif forward_batch.forward_mode.is_draft_extend():

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
                    o = q.new_empty(
                        (
                            bs * self.forward_metadata.max_q_len,
                            layer.tp_q_head_num,
                            layer.v_head_dim,
                        ),
                        dtype=self.input_dtype,
                    )
                    mla_decode_fwd(
                        q_pad.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        o,
                        self.forward_metadata.qo_indptr,
                        self.forward_metadata.kv_indptr,
                        self.forward_metadata.kv_indices,
                        self.forward_metadata.kv_last_page_len,
                        self.forward_metadata.max_q_len,
                        sm_scale=layer.scaling,
                        logit_cap=layer.logit_cap,
                        work_meta_data=work_metadata,
                        work_indptr=work_indptr,
                        work_info_set=work_info_set,
                        reduce_indptr=reduce_indptr,
                        reduce_final_map=reduce_final_map,
                        reduce_partial_map=reduce_partial_map,
                        q_scale=layer.k_scale,
                        kv_scale=layer.k_scale,
                        intra_batch_mode=intra_batch_mode,
                        num_kv_splits=num_kv_splits,
                    )

                    return o[q_mask]
                else:
                    o = q.new_empty(
                        (q.shape[0], layer.tp_q_head_num, layer.v_head_dim),
                        dtype=self.input_dtype,
                    )

                    mla_decode_fwd(
                        q,
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        o,
                        self.forward_metadata.qo_indptr,
                        self.forward_metadata.kv_indptr,
                        self.forward_metadata.kv_indices,
                        self.forward_metadata.kv_last_page_len,
                        self.forward_metadata.max_q_len,
                        sm_scale=layer.scaling,
                        logit_cap=layer.logit_cap,
                        work_meta_data=work_metadata,
                        work_indptr=work_indptr,
                        work_info_set=work_info_set,
                        reduce_indptr=reduce_indptr,
                        reduce_final_map=reduce_final_map,
                        reduce_partial_map=reduce_partial_map,
                        q_scale=layer.k_scale,
                        kv_scale=layer.k_scale,
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
                # Use triton extend kernel which supports custom masks and causal masking
                if layer.qk_head_dim != layer.v_head_dim:
                    o = q.new_empty(
                        (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                    )
                else:
                    o = torch.empty_like(q)

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
                    layer.scaling,
                    logit_cap=layer.logit_cap,
                )
                return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )

            bs0 = forward_batch.batch_size + 1

            # TODO kkhuang-amd need to remove it when mha_batch_prefill_func support fp8-kv
            if self.kv_cache_dtype == fp8_dtype:
                dtype = q.dtype
                k_cache = k_cache.to(dtype)
                v_cache = v_cache.to(dtype)

            o = mha_batch_prefill_func(
                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k_cache,
                v_cache,
                self.qo_indptr[:bs0],
                self.forward_metadata.kv_indptr[:bs0],
                self.forward_metadata.kv_indices,
                self.forward_metadata.max_q_len,
                self.forward_metadata.max_kv_len,
                causal=True,
                logits_soft_cap=self.logits_soft_cap,
                alibi_slopes=None,
                return_lse=False,
                return_attn_probs=False,
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

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty(
                (q.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                dtype=self.input_dtype,
            )
        else:
            o = torch.empty_like(q, dtype=self.input_dtype)

        if save_kv_cache:

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

            mla_decode_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k_buffer.view(-1, 1, 1, layer.qk_head_dim),
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                self.forward_metadata.qo_indptr,
                self.forward_metadata.kv_indptr,
                self.forward_metadata.kv_indices,
                self.forward_metadata.kv_last_page_len,
                self.forward_metadata.max_q_len,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                work_meta_data=work_metadata,
                work_indptr=work_indptr,
                work_info_set=work_info_set,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                q_scale=layer.k_scale,
                kv_scale=layer.k_scale,
                intra_batch_mode=intra_batch_mode,
                num_kv_splits=num_kv_splits,
            )
        else:
            self.logits_soft_cap = layer.logit_cap

            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )

            # TODO kkhuang-amd need to remove it when paged_attention_ragged support fp8-kv
            if self.kv_cache_dtype == fp8_dtype:
                dtype = q.dtype

                k_cache = k_cache.to(dtype)
                v_cache = v_cache.to(dtype)

            paged_attention_ragged(
                o.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
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
                "auto",
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

            self.max_kv_len = torch.max(paged_kernel_lens).item()

            extend_lens = seq_lens - prefix_lens
            self.max_q_len = torch.max(extend_lens).item()

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
                seq_lens_cpu=None,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)
