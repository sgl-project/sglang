from __future__ import annotations

from sglang.srt.runtime_context import (
    get_parallel,
    get_schedule,
    get_spec,
)

"""
end to end attention solution with aiter kernels
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

import torch
import triton

from sglang.kernels.ops.attention.utils import (
    assert_buffer_fits,
    create_flashinfer_kv_indices_triton,
    create_flashmla_kv_indices_triton,
    get_num_kv_index_blocks_flashmla,
)
from sglang.kernels.ops.kvcache.aiter_unified_attention import (
    scatter_ragged_to_page_table_kernel,
    scatter_req_to_token_to_page_table_kernel,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import (
    is_dp_attention_enabled,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_utils import (
    draft_kv_indices_buffer_width,
    draft_kv_indices_used_len,
    generate_draft_decode_kv_indices,
)
from sglang.srt.utils import is_gfx95_supported

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

try:
    from aiter import (
        flash_attn_varlen_fp8_pertensor_func,
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

    from sglang.kernels.ops.attention.unified_attention_3d_mtp import (
        unified_attention_3d_mtp_func,
    )
except ImportError:
    print(
        "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
    )

from sglang.kernels.ops.attention.utils import (
    launch_reshape_and_cache_flash,
    pad_sequence_with_mask,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    fp8_dtype,
    scaled_fp8_quant,
)
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.environ import envs
from sglang.srt.layers.attention.aiter_mla_gluon import (
    log_mla_gluon_capability,
    mla_gluon_decode,
    prefer_mla_gluon_decode,
)
from sglang.srt.layers.attention.aiter_utils import (
    forward_decode_vectorized_5d,
    forward_extend_vectorized_5d,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

# Use aiter mla persist design for fp8-kv cache
_use_mla_ps_kernel = get_bool_env_var("SGLANG_AITER_MLA_PERSIST", "True")

# Use fp8 prefill only on gfx95
_use_fp8_prefill_attn = (
    get_bool_env_var("SGLANG_AITER_FP8_PREFILL_ATTN", "True") and is_gfx95_supported()
)

# (v_head_dim -> query head counts) that aiter's mla_reduce_v1 has an
# instantiation. This map is copied from MLA_REDUCE_ROUTER in
# aiter/csrc/kernels/mla/reduce.cu. See https://github.com/ROCm/aiter/blob/7915f53a4225b3f9cb632a97a23369dbebcf1be0/csrc/kernels/mla/reduce.cu#L923
_MLA_REDUCE_V1_HEADS = {
    64: frozenset({64}),
    128: frozenset({1, 2, 4, 8, 10, 16, 32, 40, 64, 128}),
    512: frozenset({8, 16, 32, 48, 64, 80, 96, 112, 128}),
}


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
    # full->SWA translated out_cache_loc (SWA KV-store write target)
    swa_out_cache_loc: Optional[torch.Tensor] = None


_AITER_PARTITION_SIZE_ROCM = 256


def _asm_context_prefill_gather_indices(
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_slots: int,
    forward_mode=None,
):
    """KV-pool slots to gather for the ASM context-chunk prefill.

    kv_indptr/kv_indices are token-level for every page_size:
    AiterIndicesUpdaterPrefill sets kv_indptr = cumsum(seq_lens) and writes one
    kv_indices entry per token, so token t of sequence i lives in pool slot
    kv_indices[kv_indptr[i] + t]. There is no page arithmetic to apply here.

    Returns (tok_idx, cu_seqlens_k), or None if the metadata disagrees with
    seq_lens (mixed/spec batches) and the caller must use the paged kernel.
    """
    bs = kv_indptr.numel() - 1
    device = kv_indices.device
    kv_indptr = kv_indptr.to(torch.long)
    seq_lens = seq_lens.to(device=device, dtype=torch.long)
    # kvlen must not exceed the tokens this batch actually has in kv_indices,
    # otherwise the gather runs off the end of the table.
    seq_lens = torch.minimum(seq_lens, kv_indptr[1:] - kv_indptr[:bs])
    total_k = int(seq_lens.sum().item())
    cu_k = torch.zeros(bs + 1, dtype=torch.long, device=device)
    torch.cumsum(seq_lens, 0, out=cu_k[1:])
    seq_ids = torch.repeat_interleave(torch.arange(bs, device=device), seq_lens)
    pos_in_seq = torch.arange(total_k, device=device) - cu_k[seq_ids]
    kv_slot = kv_indptr[seq_ids] + pos_in_seq
    if total_k and int(kv_slot.max().item()) >= kv_indices.numel():
        logger.warning(
            "[asm-context-prefill] metadata mismatch, falling back:"
            " mode=%s bs=%s kv_slot_max=%s kv_indices=%s seq_lens=%s kv_indptr=%s",
            forward_mode,
            bs,
            int(kv_slot.max().item()),
            kv_indices.numel(),
            seq_lens.tolist(),
            kv_indptr.tolist(),
        )
        return None
    tok_idx = kv_indices[kv_slot].to(torch.long)
    if total_k and int(tok_idx.max().item()) >= num_kv_slots:
        logger.warning(
            "[asm-context-prefill] gather index out of pool, falling back:"
            " mode=%s bs=%s tok_idx_max=%s num_kv_slots=%s",
            forward_mode,
            bs,
            int(tok_idx.max().item()),
            num_kv_slots,
        )
        return None
    return tok_idx, cu_k


class AiterAttnBackend(AttentionBackend):

    # kv_indptr/qo_indptr are preallocated at (req pool + 1); an extend batch
    # can never carry more seqs than the pool.
    extend_dummy_seqs_capped_by_req_pool: bool = True

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        topk: int = 1,
    ):
        super().__init__()
        # Lazy import to avoid the initialization of cuda context
        from sglang.kernels.ops.attention.extend_attention import (
            extend_attention_fwd,
        )

        self.input_dtype = model_runner.model_config.dtype

        self.page_size = get_schedule().page_size

        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)

        self.device = model_runner.device
        self.is_multimodal = model_runner.model_config.is_multimodal
        self.num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.speculative_num_steps = get_spec().speculative_num_steps
        self.topk = topk
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size
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

        # The asm fp8 prefill reduces through mla_reduce_v1, which only has
        # instantiations for the head shapes in _MLA_REDUCE_V1_HEADS; anything
        # else is tiled up to one it does carry, or falls back to
        # flash_attn_varlen_func when the table has nothing to reach.
        self.fp8_prefill_num_head = (
            self.check_fp8_prefill_num_head(
                num_head=self.num_head,
                num_kv_head=self.num_kv_head,
                v_head_dim=self.v_head_dim,
            )
            if self.use_mla
            else None
        )
        self.use_fp8_prefill_attn = (
            _use_fp8_prefill_attn and self.fp8_prefill_num_head is not None
        )
        # Padding is only offered at GQA ratio 1, so the kv side takes the same
        # delta and the ratio the PS metadata is built for stays put.
        self.fp8_prefill_num_kv_head = self.num_kv_head + (
            (self.fp8_prefill_num_head or self.num_head) - self.num_head
        )
        if self.use_fp8_prefill_attn and self.fp8_prefill_num_head != self.num_head:
            logger.info(
                f"aiter asm fp8 MLA prefill pads {self.num_head} query heads to "
                f"{self.fp8_prefill_num_head}; mla_reduce_v1 has no "
                f"{self.num_head}-head instantiation at head_dim {self.v_head_dim}."
            )

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

        # Pool refs — captured at construction so they survive deletion of the
        # corresponding ForwardBatch fields.
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool

        # sliding window attention
        self.use_sliding_window_kv_pool = (
            isinstance(model_runner.token_to_kv_pool, SWAKVPool)
            and model_runner.token_to_kv_pool.swa_layer_nums > 0
        )

        # Detect SHUFFLE 5D ("vectorized") KV cache layout. When active
        # we (a) skip the launch_reshape_and_cache_flash shortcut and always go
        # through `set_kv_buffer` (which dispatches to the 5D Triton writer),
        # and (b) route the decode attention through pa_decode_gluon (see the
        # corresponding branch in forward_decode), since unified_attention's
        # 4D `.view(-1, page, H, D)` cannot be applied to a 5D pool.
        def _pool_is_vec5d(pool):
            if isinstance(pool, SWAKVPool):
                return getattr(pool.full_kv_pool, "kv_cache_layout", "nhd") == (
                    "vectorized_5d"
                )
            return getattr(pool, "kv_cache_layout", "nhd") == "vectorized_5d"

        self.kv_cache_is_vectorized_5d = _pool_is_vec5d(model_runner.token_to_kv_pool)

        if self.use_sliding_window_kv_pool:
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
            _mla_low_head_repeat = (4, 8)
            _mla_low_head_zero_pad = (12,)
            _valid_heads = (
                self.num_head in _mla_low_head_repeat
                or self.num_head in _mla_low_head_zero_pad
                or (self.num_head % 16 == 0 and 16 <= self.num_head <= 128)
            )
            may_run_mla_decode = self.may_run_mla_decode_kernel(
                decode_attention_backend=model_runner.decode_attention_backend_str,
                speculative_algorithm=get_spec().speculative_algorithm,
                speculative_attention_mode=get_spec().speculative_attention_mode,
            )
            # _mla_decode_fwd_with_head_pad brings any count below 16 up to it,
            # by repetition when it divides 16 and by tiling otherwise.
            _pad_heads_to_16 = self.num_head < 16
            assert _valid_heads or _pad_heads_to_16 or not may_run_mla_decode, (
                f"Aiter MLA supports num_head of 4, 8, 12, or multiples of 16 "
                f"in [16, 128].\n"
                f"Provided {self.num_head} number of heads.\n"
                "Try adjusting tensor_parallel_size value, or run decode on "
                "another backend (--decode-attention-backend)."
            )
            self.num_head_padded = 16 if self.num_head < 16 else self.num_head
            if self.num_head in _mla_low_head_repeat:
                self.head_pad_mode = "repeat"
                self.head_repeat_factor = 16 // self.num_head
            elif self.num_head in _mla_low_head_zero_pad:
                self.head_pad_mode = "zero"
                self.head_repeat_factor = 1
            else:
                self.head_pad_mode = "none"
                self.head_repeat_factor = 1

            self.enable_dp_attention = is_dp_attention_enabled()
            self.qo_indptr_ = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
            global _use_mla_ps_kernel, fast_mode, intra_batch_mode

            # current mla_decode_fwd only support fake-nps in self.num_head == 16
            # so all num_head size does not use qh16 kernel to simulate
            # it should not use fake-nps (fast_mode = False, intra_batch_mode = True)
            # it will cause gpu-fault or accuracy issue
            if self.num_head in (32, 64, 128):
                fast_mode = True
                intra_batch_mode = False

            # current persist a16w16 mla_decode kernel does not support head_num = 128
            # need to fall back to non-persist
            # only use mla_ps_kernel when fp8 kv_cache
            # for non-fp8 kv_cache on tp8, use non-persist kernel to avoid performance degradation
            # head_num=16 (tp8 perf issue), head_num=128 (unsupported, like tp1 or --enable-dp-attention with tp8-dp8)
            # Native 16-head persist is slow on TP8; keep disabled unless zero-pad
            # (e.g. Kimi K3 h12 -> qh16) where persist ASM is the fast path.
            if (
                (self.num_head_padded == 16 and self.head_pad_mode != "zero")
                or self.num_head_padded == 128
            ) and self.kv_cache_dtype is not fp8_dtype:
                _use_mla_ps_kernel = False
                fast_mode = False
                intra_batch_mode = False
            # Zero-pad topology (h12->qh16): prefer Gluon decode over PS kernel.
            if self.head_pad_mode == "zero" and self.kv_cache_dtype == fp8_dtype:
                # Disable ps only when gluon kernel is selected to avoid falling
                # back to incorrect aiter kernel
                if prefer_mla_gluon_decode(
                    head_pad_mode=self.head_pad_mode,
                    num_head=self.num_head,
                    kv_cache_dtype=self.kv_cache_dtype,
                ):
                    _use_mla_ps_kernel = False
                    fast_mode = False
                    intra_batch_mode = False
                log_mla_gluon_capability(logger)

            self.max_split_per_batch = 32 if _use_mla_ps_kernel else None

            if self.num_draft_tokens is None and _use_mla_ps_kernel:
                self.max_split_per_batch = 64

            self.fix_max_split_per_batch = self.max_split_per_batch

    def pad_heads(self, x: torch.Tensor, padded: int) -> torch.Tensor:
        num_head = x.shape[1]
        reps = -(-padded // num_head)  # ceil(padded / num_head)
        return x.repeat(1, reps, 1)[:, :padded, :].contiguous()

    def check_fp8_prefill_num_head(
        self, *, num_head: int, num_kv_head: int, v_head_dim: int
    ) -> Optional[int]:
        """Check _MLA_REDUCE_V1_HEADS to get head count to run the asm fp8
        mla prefill, return None if it is invalid so we will fall it back to
        aiter fa implementation.
        """
        supported = _MLA_REDUCE_V1_HEADS.get(v_head_dim, frozenset())
        if num_head in supported:
            return num_head
        if num_head != num_kv_head:
            return None
        larger = [h for h in supported if h > num_head]
        return min(larger) if larger else None

    def may_run_mla_decode_kernel(
        self,
        *,
        decode_attention_backend: Optional[str],
        speculative_algorithm: Optional[str],
        speculative_attention_mode: str,
    ) -> bool:
        """Decode whether aiter backend will invoke mla_decode_fwd"""
        if decode_attention_backend == "aiter":
            return True
        return (
            speculative_algorithm is not None
            and speculative_attention_mode == "prefill"
        )

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
            num_head_k=self.fp8_prefill_num_kv_head,
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
        gqa_ratio = self.fp8_prefill_num_head // self.fp8_prefill_num_kv_head
        num_heads_k = self.fp8_prefill_num_kv_head
        tile_q = 256
        qhead_granularity = gqa_ratio
        qlen_granularity = tile_q // qhead_granularity
        kvlen_granularity = 128
        block_size = 1

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

        repeat (4/8): tile q heads to 16, slice back to num_head.
        zero (12): pad four zero-valued dummy heads to 16 (vLLM #50371 style).
        q / o must already be shaped (..., num_head, head_dim).
        """
        num_head = layer.tp_q_head_num
        if self.head_pad_mode == "repeat" or (
            self.head_pad_mode == "none" and self.num_head_padded != self.num_head
        ):
            q_in = self.pad_heads(q, self.num_head_padded)
            o = q.new_empty(
                (q.shape[0], self.num_head_padded, layer.v_head_dim),
                dtype=self.input_dtype,
            )
            mla_decode_fwd(q_in, k_buffer_flat, o, **kwargs)
            return o[:, : self.num_head, :]
        if self.head_pad_mode == "zero":
            q_in = q.new_zeros(
                (q.shape[0], self.num_head_padded, q.shape[-1]),
                dtype=q.dtype,
            )
            q_in[:, :num_head, :] = q
            o = q.new_empty(
                (q.shape[0], self.num_head_padded, layer.v_head_dim),
                dtype=self.input_dtype,
            )
            mla_decode_fwd(q_in, k_buffer_flat, o, **kwargs)
            return o[:, :num_head, :]
        o = q.new_empty(
            (q.shape[0], num_head, layer.v_head_dim),
            dtype=self.input_dtype,
        )
        mla_decode_fwd(q, k_buffer_flat, o, **kwargs)
        return o

    def _zero_pad_mla_q_heads(
        self, q: torch.Tensor, layer: RadixAttention
    ) -> torch.Tensor:
        """Zero-pad q heads num_head -> num_head_padded (12 -> 16) for the
        aiter MLA prefill kernels. Input/return are 3-D (T, H, D); the extra
        heads compute garbage outputs that are sliced away after the kernel."""
        q3 = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        q_pad = q3.new_zeros((q3.shape[0], self.num_head_padded, q3.shape[-1]))
        q_pad[:, : layer.tp_q_head_num, :] = q3
        return q_pad

    def _resolve_fp8_kv_scale_float(self, layer: RadixAttention, k_descale) -> float:
        cached = getattr(layer, "_aiter_kv_scale_float", None)
        if cached is not None:
            return cached
        if k_descale is None:
            val = 1.0
        elif isinstance(k_descale, torch.Tensor):
            val = float(k_descale.item())
        else:
            val = float(k_descale)
        layer._aiter_kv_scale_float = val
        return val

    def _resolve_mla_gluon_min_kv_seq_len(self, forward_batch: ForwardBatch) -> int:
        try:
            if torch.cuda.is_current_stream_capturing():
                return int(self.max_context_len)
        except Exception:
            pass
        if forward_batch.seq_lens_cpu is not None:
            return int(forward_batch.seq_lens_cpu.max())
        seq_lens = forward_batch.seq_lens
        if seq_lens is None or seq_lens.numel() == 0:
            return 1
        return int(seq_lens.max().item())

    def _forward_mla_decode(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        k_descale,
    ):
        k_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        q_mla = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        max_q_len = self.forward_metadata.max_q_len or 1

        if (
            prefer_mla_gluon_decode(
                head_pad_mode=getattr(self, "head_pad_mode", "none"),
                num_head=getattr(self, "num_head", layer.tp_q_head_num),
                kv_cache_dtype=self.kv_cache_dtype,
            )
            and max_q_len == 1
        ):
            kv_scale = self._resolve_fp8_kv_scale_float(layer, k_descale)
            min_kv_seq_len = self._resolve_mla_gluon_min_kv_seq_len(forward_batch)
            gluon_out = mla_gluon_decode(
                q=q_mla,
                k_buffer=k_buffer,
                layer=layer,
                kv_indices=self.forward_metadata.kv_indices,
                kv_indptr=self.forward_metadata.kv_indptr,
                seq_lens=forward_batch.seq_lens,
                sm_scale=layer.scaling,
                kv_scale=kv_scale,
                min_kv_seq_len=min_kv_seq_len,
            )
            if gluon_out is not None:
                return gluon_out

        work_metadata = self.forward_metadata.work_metadata
        work_indptr = self.forward_metadata.work_indptr
        work_info_set = self.forward_metadata.work_info_set
        reduce_indptr = self.forward_metadata.reduce_indptr
        reduce_final_map = self.forward_metadata.reduce_final_map
        reduce_partial_map = self.forward_metadata.reduce_partial_map
        num_kv_splits = self.forward_metadata.num_kv_splits

        return self._mla_decode_fwd_with_head_pad(
            q_mla,
            k_buffer.view(-1, 1, 1, layer.qk_head_dim),
            layer,
            qo_indptr=self.forward_metadata.qo_indptr,
            kv_indptr=self.forward_metadata.kv_indptr,
            kv_indices=self.forward_metadata.kv_indices,
            kv_last_page_lens=self.forward_metadata.kv_last_page_len,
            max_seqlen_q=max_q_len,
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
        # mla_reduce_v1 dispatches on the head count, so a model it has no
        # instantiation for runs on the next one up, with the query heads tiled
        # to fill and the extra output columns sliced back off.
        head_pad = self.fp8_prefill_num_head - nhead
        if head_pad:
            q = self.pad_heads(q, self.fp8_prefill_num_head)
            k = self.pad_heads(k, self.fp8_prefill_num_head)
            v = self.pad_heads(v, self.fp8_prefill_num_head)
            nhead = self.fp8_prefill_num_head

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
            # Prefill PS metadata has no split cap; 0 keeps AITER's default reduce sizing.
            0,
            output,
            final_lse,
        )
        return output[:, : layer.tp_q_head_num, :] if head_pad else output

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        seq_lens_cpu = (
            forward_batch.seq_lens.cpu() if in_capture else forward_batch.seq_lens_cpu
        )
        verify_tokens_per_req = (
            forward_batch.input_ids.shape[0] // forward_batch.batch_size
            if forward_batch.forward_mode.is_target_verify()
            else None
        )
        self._apply_cuda_graph_metadata(
            bs=forward_batch.batch_size,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            seq_lens_sum=None if in_capture else forward_batch.seq_lens_sum,
            forward_mode=forward_batch.forward_mode,
            spec_info=forward_batch.spec_info,
            seq_lens_cpu=seq_lens_cpu,
            verify_tokens_per_req=verify_tokens_per_req,
        )

        # Refill the SWA write-target buffer from the live out_cache_loc and
        # bind it onto the metadata before replay (_apply rebuilds it each call).
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            n = forward_batch.out_cache_loc.shape[0]
            self.cuda_graph_swa_out_cache_loc[n:].zero_()
            self.cuda_graph_swa_out_cache_loc[:n].copy_(
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc
                )
            )
            self.forward_metadata.swa_out_cache_loc = self.cuda_graph_swa_out_cache_loc[
                :n
            ]

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
        swa_out_cache_loc = None
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            swa_out_cache_loc = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                forward_batch.out_cache_loc
            )
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

                    create_flashmla_kv_indices_triton[
                        (bs, get_num_kv_index_blocks_flashmla(max_kv_len, 1))
                    ](
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
                        # AITER attention kernels require int32 page indices;
                        # full_to_swa_index_mapping is stored as int64.
                        swa_page_table = (
                            self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                kv_indices
                            ).to(torch.int32)
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
                swa_out_cache_loc=swa_out_cache_loc,
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
                kv_indices, kv_indptr, qo_indptr, _ = (
                    forward_batch.spec_info.generate_attn_arg_prefill(
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
                    None,
                    forward_batch.spec_info.num_tokens_per_req,
                    max_kv_len,
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
                        swa_out_cache_loc=swa_out_cache_loc,
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

                # fp8 PS-ASM prefill memory-faults on gfx950 for 12-head
                # (zero-pad) models; keep it off and use flash-attn fallback.
                if self.use_fp8_prefill_attn and self.head_pad_mode != "zero":
                    tile_q = 256
                    qlen_granularity = tile_q // (
                        self.fp8_prefill_num_head // self.fp8_prefill_num_kv_head
                    )
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
                    # AITER attention kernels (e.g. mha_batch_prefill_func)
                    # require int32 page indices; full_to_swa_index_mapping is
                    # stored as int64.
                    swa_page_table = (
                        self.token_to_kv_pool.translate_loc_from_full_to_swa(
                            self.indices_updater_prefill.kv_indices
                        ).to(torch.int32)
                    )

                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    max(forward_batch.extend_seq_lens_cpu),
                    forward_batch.seq_lens_cpu.max().item(),
                    swa_page_table=swa_page_table,
                    swa_out_cache_loc=swa_out_cache_loc,
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
            # Non-unified AITER CUDA graph paths fill this buffer with flat
            # token-level kv_indices via create_flashinfer_kv_indices_triton
            # (kv_indptr = cumsum(seq_lens)).  Even when the allocator is
            # page-based, these writes are per-token, so page-sized allocation
            # would under-allocate by page_size when page_size > 1.
            # TODO(aiter, page_size>1): root fix is to make page_size>1
            # actually engage the attention kernel (`forward_decode` still
            # calls paged_attention_ragged with view(-1, 1, ...) and
            # block_size=1). That requires a per-page indices kernel + all
            # metadata sites + paged_attention_ragged call site + FP8 KV
            # coordination, after which this allocation can revert to
            # per-page (gated on use_mla).
            # Reserve draft slack: MLA target_verify writes seq_len +
            # num_draft_tokens per row; without it a near-full sequence
            # overflows the buffer. Mirrors dsa / flashmla.
            draft_slack = self.num_draft_tokens or 0
            buffer_numel = max_bs * (
                max_num_blocks_per_seq * self.page_size + draft_slack
            )
            self.cuda_graph_kv_indices = torch.zeros(
                (buffer_numel,),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        if self.use_triton_unified_attention:
            # Keep a distinct page-table buffer for unified attention.  Sharing
            # cuda_graph_kv_indices with non-unified token indices makes
            # page-table width ambiguous after the token buffer is expanded.
            max_num_blocks_per_seq = (
                self.max_context_len + self.page_size - 1
            ) // self.page_size
            self.cuda_graph_page_table = torch.zeros(
                (max_bs, max_num_blocks_per_seq),
                dtype=torch.int32,
                device=self.device,
            )

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
            # SWA write-target buffer; refilled and bound onto forward_metadata
            # in init_forward_metadata_out_graph before each replay.
            self.cuda_graph_swa_out_cache_loc = torch.zeros(
                (max_num_tokens,),
                dtype=torch.int64,
                device=self.device,
            )

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        verify_tokens_per_req: Optional[int],
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
        max_kv_len = (
            seq_lens_cpu.max().item()
            if seq_lens_cpu is not None
            else torch.max(seq_lens).item()
        )

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
                    kv_indices = self.cuda_graph_page_table

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
                            # AITER attention kernels require int32 page indices;
                            # full_to_swa_index_mapping is stored as int64.
                            swa_page_indices = (
                                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                    page_indices
                                ).to(torch.int32)
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
            assert verify_tokens_per_req is not None
            # MLA uses a fixed draft length (num_draft_tokens); the non-MLA
            # unified path derives it per batch from input_ids.
            tokens_per_req = (
                self.num_draft_tokens if self.use_mla else verify_tokens_per_req
            )
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * tokens_per_req,
                step=tokens_per_req,
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
            # seq_lens_sum is None at capture (dummy seq_lens); only check on replay.
            if seq_lens_sum is not None:
                kv_indices_used = seq_lens_sum + (
                    self.num_draft_tokens * bs if self.use_mla else 0
                )
                assert_buffer_fits(
                    kv_indices_used,
                    kv_indices.numel(),
                    "aiter target_verify kv_indices",
                    bs=bs,
                    seq_lens_sum=seq_lens_sum,
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
            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]

            if self.use_mla:
                max_q_len = self.num_draft_tokens
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
                max_q_len = verify_tokens_per_req
                if self._use_unified_verify:
                    max_num_blocks_per_seq = (
                        self.max_context_len + self.page_size - 1
                    ) // self.page_size
                    page_table = self.cuda_graph_page_table[:bs]

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
                            verify_tokens_per_req,
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
            num_tokens_per_req = self._resolve_v2_num_draft_tokens()
            extend_lens = torch.full(
                (bs,), num_tokens_per_req, dtype=torch.int32, device=seq_lens.device
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
            max_q_len = num_tokens_per_req

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

    def _use_fused_fp8_kv_write(self, layer: RadixAttention) -> bool:
        # Fused write reuses K's num_heads/head_dim for V, so it needs FP8,
        # non-MLA, non-SWA, and matching K/V head count + head_dim.
        return (
            self.kv_cache_dtype == fp8_dtype
            and not self.use_mla
            and not self.use_sliding_window_kv_pool
            and layer.tp_k_head_num == layer.tp_v_head_num
            and layer.qk_head_dim == layer.v_head_dim
        )

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
            v_descale = layer.v_scale if layer.v_scale is not None else self.v_scale

        if k is not None:
            assert v is not None
            if save_kv_cache:
                # 5D pool cannot be reshaped to the 4D paged view used by
                # launch_reshape_and_cache_flash; always route through
                # set_kv_buffer which dispatches to the SHUFFLE 5D writer.
                if self.kv_cache_is_vectorized_5d:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                        k,
                        v,
                        k_descale,
                        v_descale,
                    )
                # Only use SWA-specific kv cache write (reshape_and_cache_flash) when
                # both unified attention and sliding window kv pool are active.
                # Non-SWA models (e.g. Qwen3-VL) enabled via SGLANG_USE_AITER_UNIFIED_ATTN
                # use standard set_kv_buffer, as they lack SWA-specific attributes
                # like full_to_swa_index_mapping.
                elif (
                    self.use_triton_unified_attention
                    and self.use_sliding_window_kv_pool
                ):
                    token_to_kv_pool = self.token_to_kv_pool
                    k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(
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
                    self.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
                elif self._use_fused_fp8_kv_write(layer):
                    # FP8: fuse bf16->fp8 cast + paged write in one kernel.
                    k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(
                        layer.layer_id
                    )
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
                        k_scale=k_descale,
                        v_scale=v_descale,
                    )
                else:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                        k,
                        v,
                        k_descale,
                        v_descale,
                    )

        if self.use_mla:
            max_q_len = self.forward_metadata.max_q_len
            max_kv_len = self.forward_metadata.max_kv_len
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            qo_indptr = self.forward_metadata.qo_indptr
            K_Buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
            V_Buffer = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
            kv_lora_rank = V_Buffer.shape[-1]
            qk_rope_head_dim = K_Buffer.shape[-1] - kv_lora_rank
            qk_nope_head_dim = k.shape[-1] - qk_rope_head_dim
            assert len(q.shape) == 3
            assert len(k.shape) == 3
            assert len(v.shape) == 3

            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend_v2()
            ):
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
                if kv_indices.shape[0] == 0 or extend_no_prefix:
                    if self.use_fp8_prefill_attn and self.head_pad_mode != "zero":
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
                        self.use_fp8_prefill_attn
                        and self.head_pad_mode != "zero"
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

                    if self.use_fp8_prefill_attn and self.head_pad_mode != "zero":
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
                    if self.head_pad_mode == "zero":
                        # 12 heads/rank (Kimi-K3 TP8): zero-pad q to 16 heads,
                        # run the aiter MLA prefill kernel, slice 12 back.
                        q_in = self._zero_pad_mla_q_heads(q, layer)
                        o = q.new_empty(
                            (q.shape[0], self.num_head_padded, layer.v_head_dim)
                        )
                    else:
                        q_in = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
                        if layer.qk_head_dim != layer.v_head_dim:
                            o = q.new_empty(
                                (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                            )
                        else:
                            o = torch.empty_like(q)

                    mla_prefill_fwd(
                        q_in,
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        (
                            o.view(-1, self.num_head_padded, layer.v_head_dim)
                            if self.head_pad_mode == "zero"
                            else o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                        ),
                        qo_indptr,
                        kv_indptr,
                        kv_indices,
                        self.forward_metadata.kv_last_page_len,
                        self.forward_metadata.max_q_len,
                        layer.scaling,
                        layer.logit_cap,
                    )
                    K_Buffer = K_Buffer.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
                    if self.head_pad_mode == "zero":
                        return (
                            o[:, : layer.tp_q_head_num, :]
                            .contiguous()
                            .view(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                        )
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
            elif forward_batch.forward_mode.is_draft_extend_v2():
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
            if forward_batch.forward_mode.is_target_verify():
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
                    k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(
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
                    # Shape gate, not a model gate: the kernel is tuned for a 16:1
                    # GQA ratio (block_m=32 -> block_q=2 packs two draft tokens per
                    # tile), and head_dim 256 is the only validated size. The kv-head
                    # count itself is free (kv_head_idx = program_id(1)), but K and V
                    # must share it -- the kernel has a single kv-head grid dim.
                    # Qwen3.5-397B-A17B at TP1/TP2 is the only config known to
                    # match today; any model with the same shapes qualifies.
                    # TODO(yichiche): relax the head_dim gate once other sizes are
                    # measured -- the wrapper passes HEAD_SIZE_PADDED unpadded, so
                    # only powers of 2 work (128/256 OK, 192 is not).
                    num_queries_per_kv = layer.tp_q_head_num // layer.tp_k_head_num
                    use_unified_attention_3d_mtp = (
                        is_gfx95_supported()
                        and 1 < self.forward_metadata.max_q_len <= 4
                        and max_kv_len > 512
                        and num_queries_per_kv == 16
                        and layer.tp_k_head_num == layer.tp_v_head_num
                        and layer.qk_head_dim == 256
                        and layer.v_head_dim == 256
                        and self.page_size == 16
                        and q_unified.dtype == torch.bfloat16
                        and k_unified.dtype == fp8_dtype
                        and window_size == (-1, -1)
                        and not layer.logit_cap
                        and sinks is None
                    )
                    if use_unified_attention_3d_mtp:
                        unified_attention_3d_mtp_func(
                            q=q_unified,
                            k=k_unified,
                            v=v_unified,
                            out=o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                            cu_seqlens_q=self.forward_metadata.qo_indptr,
                            seqused_k=(
                                forward_batch.seq_lens + self.forward_metadata.max_q_len
                            ),
                            max_seqlen_q=self.forward_metadata.max_q_len,
                            max_seqlen_k=max_kv_len,
                            softmax_scale=layer.scaling,
                            block_table=page_table,
                            k_descale=k_descale,
                            v_descale=v_descale,
                        )
                        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

                    # GQA-packing fix: do NOT expand the single KV head to
                    # tp_q_head_num. Passing K/V with the true kv-head count (exactly
                    # like forward_decode) lets unified_attention derive
                    # num_queries_per_kv = tp_q_head_num (the GQA group) and pack all Q
                    # heads against one KV load. The old stride-0 .expand() made the
                    # wrapper see num_kv_heads=tp_q_head_num -> num_queries_per_kv=1 ->
                    # full MHA tiling (~7x more KV traffic at long context; trace
                    # signature num_query_heads_16/num_queries_per_kv_1). GQA head
                    # mapping here is identical to the proven decode path.

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
                    self.token_to_kv_pool.get_key_buffer(layer.layer_id),
                    self.token_to_kv_pool.get_value_buffer(layer.layer_id),
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

            # draft_extend (EAGLE-v2 KV catch-up) is decode-shaped: short Q
            # (accepted tokens) against long paged KV, GQA. The default
            # mha_batch_prefill_func FMHA has no split-KV, so at short Q it is
            # occupancy-starved (~0.2% HBM BW, ~400x over the memory floor).
            # Route it through unified_attention (GQA-packed + split-KV), exactly
            # like target_verify, so it runs near the memory floor.
            if (
                self._use_unified_verify
                and forward_batch.forward_mode.is_draft_extend_v2()
                and envs.SGLANG_AITER_UNIFIED_DRAFT_EXTEND.get()
            ):
                bs = forward_batch.batch_size
                if layer.qk_head_dim != layer.v_head_dim:
                    o = q.new_empty(
                        (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                    )
                else:
                    o = torch.empty_like(q)
                k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
                page_table, swa_page_table = self._build_unified_page_table_from_spec(
                    self.forward_metadata, bs
                )
                pt = page_table
                de_window = (-1, -1)
                if (
                    layer.sliding_window_size is not None
                    and layer.sliding_window_size > -1
                ):
                    de_window = (layer.sliding_window_size - 1, 0)
                    if swa_page_table is not None:
                        pt = swa_page_table
                kv_indptr = self.forward_metadata.kv_indptr
                # seqused_k MUST be int64 (kv_indptr is int32, so the diff is
                # int32 and has to be widened). unified_attention derives the
                # per-tile KV addresses from this dtype: with an int32
                # seqused_k the whole K/V offset chain stays 32-bit and wraps
                # once a per-layer KV buffer reaches 2 GiB, silently returning
                # NaN. The verify (seq_lens + max_q_len) and decode
                # (seq_lens) call sites pass int64 for the same reason.
                seqused_k = (kv_indptr[1 : bs + 1] - kv_indptr[:bs]).to(torch.int64)
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
                    seqused_k=seqused_k,
                    max_seqlen_q=self.forward_metadata.max_q_len,
                    max_seqlen_k=pt.shape[1] * self.page_size,
                    softmax_scale=layer.scaling,
                    causal=True,
                    window_size=de_window,
                    block_table=pt,
                    softcap=layer.logit_cap,
                    q_descale=None,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    sinks=sinks,
                )
                return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

            bs0 = forward_batch.batch_size + 1
            q_descale = None

            window_size = (-1, -1)
            if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
                window_size = (layer.sliding_window_size, -1)

            # Context-chunk prefill (extend batches WITH a prefix) via the
            # gfx950 ASM fp8 varlen fmha. The ck_tile paged batch_prefill runs
            # at ~15% FP8 MFU at these shapes while the ASM kernel is ~3.5x
            # faster; gathering the paged fp8 KV into a contiguous varlen
            # buffer costs only ~20 us per layer at 70k context. The no-prefix
            # first chunk already takes the ASM branch below.
            if (
                is_gfx95_supported()
                and forward_batch.forward_mode.is_extend()
                and forward_batch.extend_prefix_lens_cpu is not None
                and any(forward_batch.extend_prefix_lens_cpu)
                and window_size == (-1, -1)
                and sinks is None
                and self.logits_soft_cap == 0.0
                and layer.qk_head_dim == 256
                and layer.v_head_dim == 256
                and self.kv_cache_dtype == fp8_dtype
                and not self.kv_cache_is_vectorized_5d
                and self.forward_metadata.max_kv_len is not None
            ):
                bs = forward_batch.batch_size
                k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
                gathered = _asm_context_prefill_gather_indices(
                    self.forward_metadata.kv_indptr[: bs + 1],
                    self.forward_metadata.kv_indices,
                    forward_batch.seq_lens[:bs],
                    self.token_to_kv_pool.get_key_buffer(layer.layer_id).shape[0],
                    forward_batch.forward_mode,
                )
                if gathered is not None:
                    tok_idx, cu_k = gathered
                    hk = layer.tp_k_head_num * layer.qk_head_dim
                    hv = layer.tp_v_head_num * layer.v_head_dim
                    # uint8 view: index_select is not implemented for fp8.
                    k_gather = (
                        k_cache.view(-1, hk)
                        .view(torch.uint8)
                        .index_select(0, tok_idx)
                        .view(k_cache.dtype)
                        .view(-1, layer.tp_k_head_num, layer.qk_head_dim)
                    )
                    v_gather = (
                        v_cache.view(-1, hv)
                        .view(torch.uint8)
                        .index_select(0, tok_idx)
                        .view(v_cache.dtype)
                        .view(-1, layer.tp_v_head_num, layer.v_head_dim)
                    )
                    fp8_q_descale = (
                        layer.k_scale if layer.k_scale is not None else self.k_scale
                    )
                    o = flash_attn_varlen_fp8_pertensor_func(
                        q.contiguous()
                        .view(-1, layer.tp_q_head_num, layer.qk_head_dim)
                        .to(fp8_dtype),
                        k_gather,
                        v_gather,
                        fp8_q_descale.reshape(1),
                        k_descale.reshape(1),
                        v_descale.reshape(1),
                        self.qo_indptr[:bs0],
                        cu_k.to(torch.int32),
                        self.forward_metadata.max_q_len,
                        int(self.forward_metadata.max_kv_len),
                        softmax_scale=layer.scaling,
                        causal=True,
                    )
                    if o.dtype != self.input_dtype:
                        o = o.to(self.input_dtype)
                    return o.view(-1, layer.tp_q_head_num * layer.qk_head_dim)

            if (
                is_gfx95_supported()
                and forward_batch.forward_mode.is_extend()
                and forward_batch.extend_prefix_lens_cpu is not None
                and not any(forward_batch.extend_prefix_lens_cpu)
                and window_size == (-1, -1)
                and sinks is None
                and self.logits_soft_cap == 0.0
                and layer.qk_head_dim == 256
                and layer.v_head_dim == 256
                and self.kv_cache_dtype == fp8_dtype
            ):
                q_c = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                k_c = k.contiguous().view(-1, layer.tp_k_head_num, layer.head_dim)
                v_c = v.contiguous().view(-1, layer.tp_v_head_num, layer.v_head_dim)
                fp8_q_descale = (
                    layer.k_scale if layer.k_scale is not None else self.k_scale
                )
                o = flash_attn_varlen_fp8_pertensor_func(
                    q_c.to(fp8_dtype),
                    k_c.to(fp8_dtype),
                    v_c.to(fp8_dtype),
                    fp8_q_descale.reshape(1),
                    k_descale.reshape(1),
                    v_descale.reshape(1),
                    self.qo_indptr[:bs0],
                    self.qo_indptr[:bs0],
                    self.forward_metadata.max_q_len,
                    self.forward_metadata.max_q_len,
                    softmax_scale=layer.scaling,
                    causal=True,
                )
                if o.dtype != self.input_dtype:
                    o = o.to(self.input_dtype)
                return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

            if self.kv_cache_is_vectorized_5d:
                return forward_extend_vectorized_5d(
                    self,
                    q,
                    k,
                    v,
                    layer,
                    forward_batch,
                    bs0,
                    window_size,
                    sinks,
                )

            # NHD path — original aiter paged batch_prefill.
            # TODO kkhuang-amd need to remove it when mha_batch_prefill_func support fp8-kv
            if self.kv_cache_dtype == fp8_dtype:
                q = q.to(fp8_dtype)
                q_descale = layer.k_scale if layer.k_scale is not None else self.k_scale

            k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)

            page_table = self.forward_metadata.kv_indices
            if (
                layer.sliding_window_size is not None
                and layer.sliding_window_size > -1
                and self.forward_metadata.swa_page_table is not None
            ):
                page_table = self.forward_metadata.swa_page_table

            extra_kwargs = {}
            attn_out = getattr(forward_batch, "_attn_output", None)
            if attn_out is not None and q.dtype != fp8_dtype:
                extra_kwargs["out"] = attn_out.view(
                    -1, layer.tp_q_head_num, layer.head_dim
                )

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
                **extra_kwargs,
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
            v_descale = layer.v_scale if layer.v_scale is not None else self.v_scale

        if save_kv_cache:
            # SHUFFLE 5D pool path — see forward_extend for rationale.
            if self.kv_cache_is_vectorized_5d:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(
                        forward_batch.out_cache_loc,
                        self.forward_metadata.swa_out_cache_loc,
                    ),
                    k,
                    v,
                    k_descale,
                    v_descale,
                )
            # Only use SWA-specific kv cache write (reshape_and_cache_flash) when
            # both unified attention and sliding window kv pool are active.
            # Non-SWA models (e.g. Qwen3-VL) enabled via SGLANG_USE_AITER_UNIFIED_ATTN
            # use standard set_kv_buffer, as they lack SWA-specific attributes
            # like full_to_swa_index_mapping.
            elif self.use_triton_unified_attention and self.use_sliding_window_kv_pool:
                token_to_kv_pool = self.token_to_kv_pool
                k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
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
            elif self.use_mla:
                # MLA pool has its own set_kv_buffer (no scale args).
                self.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )
            elif self._use_fused_fp8_kv_write(layer):
                # FP8: fuse bf16->fp8 cast + paged write in one kernel.
                token_to_kv_pool = self.token_to_kv_pool
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
                    k_scale=k_descale,
                    v_scale=v_descale,
                )
            else:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(
                        forward_batch.out_cache_loc,
                        self.forward_metadata.swa_out_cache_loc,
                    ),
                    k,
                    v,
                    k_descale,
                    v_descale,
                )

        if self.use_mla:
            o = self._forward_mla_decode(q, layer, forward_batch, k_descale)
            return o.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)
        else:
            self.logits_soft_cap = layer.logit_cap

            k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)

            if layer.qk_head_dim != layer.v_head_dim:
                o = q.new_empty(
                    (q.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                    dtype=self.input_dtype,
                )
            else:
                o = torch.empty_like(q, dtype=self.input_dtype)

            if self.kv_cache_is_vectorized_5d:
                # SHUFFLE 5D pool: pa_decode_gluon for full + SWA layers
                # (see :func:`aiter_utils.forward_decode_vectorized_5d`
                # for the dispatch rationale).
                forward_decode_vectorized_5d(
                    self, q, layer, forward_batch, k_cache, v_cache, o, sinks
                )
            elif self.use_triton_unified_attention:
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
                q_descale = None
                if self.kv_cache_dtype == fp8_dtype:
                    q_descale = (
                        layer.k_scale if layer.k_scale is not None else self.k_scale
                    )
                    q, _ = scaled_fp8_quant(q, q_descale)

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
                    q_descale=q_descale,
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
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size
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
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )
        self.device = model_runner.device
        # Cached variables for generate_draft_decode_kv_indices
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.page_size = get_schedule().page_size

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
            self.req_to_token_pool.req_to_token,
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
                : draft_kv_indices_used_len(seq_lens_sum, self.topk, bs, i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices_width = draft_kv_indices_buffer_width(
            forward_batch.batch_size, self.topk, self.max_context_len
        )
        kv_indices = torch.empty(
            (self.speculative_num_steps, kv_indices_width),
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
        kv_indices_width = draft_kv_indices_buffer_width(
            max_bs, self.topk, self.max_context_len
        )
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, kv_indices_width),
            dtype=torch.int32,
            device=self.device,
        )
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, kv_indices_buf=self.cuda_graph_kv_indices[i]
            )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        inner_fb = build_inner_fb_view(
            forward_batch,
            bs=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
        )

        def call_fn(i, _forward_batch):
            self.attn_backends[i].init_forward_metadata_out_graph(
                inner_fb, in_capture=in_capture
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        for attn_backend in self.attn_backends:
            attn_backend.init_forward_metadata_in_graph(forward_batch)
