from __future__ import annotations

from sglang.srt.runtime_context import get_parallel, get_spec

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
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dcp import (
    dcp_enabled,
    get_attention_dcp_rank,
    get_attention_dcp_world_size,
    update_local_kv_lens_for_dcp,
)
from sglang.srt.layers.dcp.planner import plan_dcp_decode_metadata
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
except ImportError:
    print(
        "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
    )

from sglang.kernels.ops.attention.utils import (
    launch_reshape_and_cache_flash,
    pad_sequence_with_mask,
)
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.configs.model_config import AttentionArch
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
    # dcp_g_kv_indptr keeps the GLOBAL per-request kv_indptr so that it can
    # mask on global positions g(j)=j*W+r. See mla_decode_fwd(cp_world_size,...).
    dcp_g_kv_indptr: Optional[torch.Tensor] = None
    dcp_cp_world_size: int = 1
    dcp_cp_rank: int = 0
    # Per-rank PAGE table + shard length (tokens) for the Triton DCP decode,
    # built once per forward instead of per layer. See build_dcp_page_table.
    dcp_block_table: Optional[torch.Tensor] = None
    dcp_local_kv_lens: Optional[torch.Tensor] = None
    # DCP target-verify flattens the gamma+1 window into one single-token row
    # per query token, so stage A needs its own [bs * q_len + 1] arange
    # cu_seqlens_q (qo_indptr above stays the per-request window boundaries).
    dcp_verify_qo_indptr: Optional[torch.Tensor] = None
    # Per-ROW TOKEN table for the Gluon DCP target-verify stage A. mla_gluon
    # pages at 1 token, so it cannot read dcp_block_table (pages of page_size);
    # only built when SGLANG_USE_AITER_GLUON_MLA_DCP is on.
    dcp_verify_token_table: Optional[torch.Tensor] = None


_AITER_PARTITION_SIZE_ROCM = 256

# KV page size for the DCP extend path. mla_prefill_fwd sets TILE_SIZE == block_size
# (the MLA kernel asserts NUM_BLOCKS_GATHER_PER_TILE == 1), and on gfx950 a
# 16384x16384 chunk measured 5156 ms at 1, 55.8 ms at 16, 39.5 ms at 64. This is
# a staging-buffer page size, independent of the KV pool's --page-size.
_DCP_PREFILL_PAGE_SIZE = 64

# The aiter MLA reduce and sglang's DCP merge (correct_attn_out) both work in
# base-2, so any LSE produced here has to be rebased from natural log.
_LOG2E = 1.4426950408889634


def dense_causal_mla_attn_base2(
    q: torch.Tensor,
    k_window: torch.Tensor,
    scaling: float,
    bs: int,
    q_len: int,
    kv_lora_rank: int,
):
    """Dense causal MLA attention over one in-hand ``q_len``-token window.

    Used as stage B of the DCP target-verify decomposition. The window is tiny
    (``gamma + 1`` queries over the same ``gamma + 1`` keys), so a batched matmul
    beats a paged kernel here -- and it sidesteps two kernel limitations that only
    bite inside the window: mla_decode_fwd derives its causal mask from the LOCAL shard
    index (which under DCP is not the global position), and its segmented reduce
    would need to mask the segments the causal mask empties.

    Args:
        q: [bs * q_len, num_heads, kv_lora_rank + qk_rope_head_dim]
        k_window: [bs * q_len, 1, kv_lora_rank + qk_rope_head_dim] latent KV of
            the window tokens, in query order. ``v`` is its first
            ``kv_lora_rank`` columns (MLA absorb form).

    Returns (out [bs * q_len, num_heads, kv_lora_rank] float32,
    lse [bs * q_len, num_heads] float32 base-2).
    """
    num_heads = q.shape[1]
    qb = q.view(bs, q_len, num_heads, -1)
    kb = k_window.view(bs, q_len, -1)
    # Scores in the input dtype (matching the kernels), softmax/LSE in fp32.
    scores = torch.einsum("bihd,bjd->bhij", qb, kb).float() * scaling
    causal = torch.ones(q_len, q_len, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    lse_e = torch.logsumexp(scores, dim=-1)  # [bs, num_heads, q_len], natural log
    probs = torch.exp(scores - lse_e.unsqueeze(-1))
    vb = kb[..., :kv_lora_rank].float()
    out = torch.einsum("bhij,bjd->bihd", probs, vb)  # [bs, q_len, H, kv_lora_rank]
    return (
        out.reshape(bs * q_len, num_heads, kv_lora_rank),
        (lse_e * _LOG2E).permute(0, 2, 1).reshape(bs * q_len, num_heads),
    )


def lse_combine_base2(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    out_dtype: torch.dtype,
):
    """Merge two partial attentions over DISJOINT key sets, given base-2 LSEs.

    ``out_*`` are [tokens, heads, dim] and ``lse_*`` are [tokens, heads]. A
    partial that saw no keys carries ``lse = -inf`` (what dcp_mla_reduce
    writes for an empty shard); its weight is forced to 0 rather than NaN.
    """
    m = torch.maximum(lse_a, lse_b)
    w_a = torch.nan_to_num(torch.exp2(lse_a - m), nan=0.0, posinf=0.0, neginf=0.0)
    w_b = torch.nan_to_num(torch.exp2(lse_b - m), nan=0.0, posinf=0.0, neginf=0.0)
    denom = w_a + w_b
    out = out_a.float() * w_a.unsqueeze(-1) + out_b.float() * w_b.unsqueeze(-1)
    out = out / denom.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
    lse = torch.where(denom == 0.0, float("-inf"), m + torch.log2(denom))
    return out.to(out_dtype), lse


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

        self.page_size = model_runner.server_args.page_size

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
            self.dcp_world_size = get_attention_dcp_world_size()
            # Under DCP the decode path runs an aiter MLA kernel
            # (_mla_decode_fwd_dcp), which tiles the query heads and so
            # serves any local head count. The {4,8,16k} constraint below
            # applies to the non-DCP path only, so skip it under DCP
            # (e.g. Kimi-K3 has 12 local heads at tp8).
            _valid_heads = self.num_head in (4, 8) or (
                self.num_head % 16 == 0 and 16 <= self.num_head <= 128
            )
            assert self.dcp_world_size > 1 or _valid_heads, (
                f"Aiter MLA supports num_head of 4, 8, or multiples of 16 "
                f"in [16, 128].\n"
                f"Provided {self.num_head} number of heads.\n"
                "Try adjusting tensor_parallel_size value."
            )
            self.num_head_padded = 16 if self.num_head < 16 else self.num_head
            self.head_repeat_factor = 16 // self.num_head if self.num_head < 16 else 1

            # when DCP is on, it all-gathers Q across the dcp group, so the
            # compute op as well as its persist metadata operates on the gathered
            # head count num_head * dcp_world_size, not the local num_head.
            # No op for non-dcp case
            _gathered_num_head = self.num_head * self.dcp_world_size
            self.mla_kernel_num_head_padded = (
                16 if _gathered_num_head < 16 else _gathered_num_head
            )

            self.enable_dp_attention = is_dp_attention_enabled()
            self.qo_indptr_ = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
            global _use_mla_ps_kernel, fast_mode, intra_batch_mode

            # current mla_decode_fwd only support fake-nps in num_head == 16
            # so all num_head size does not use qh16 kernel to simulate
            # it should not use fake-nps (fast_mode = False, intra_batch_mode = True)
            # it will cause gpu-fault or accuracy issue.
            if self.mla_kernel_num_head_padded in (32, 64, 128):
                fast_mode = True
                intra_batch_mode = False

            # current persist a16w16 mla_decode kernel does not support head_num = 128
            # need to fall back to non-persist
            # only use mla_ps_kernel when fp8 kv_cache
            # for non-fp8 kv_cache on tp8, use non-persist kernel to avoid performance degradation
            # head_num=16 (tp8 perf issue), head_num=128 (unsupported, like tp1 or --enable-dp-attention with tp8-dp8)
            if (
                self.mla_kernel_num_head_padded in (16, 128)
                and self.kv_cache_dtype is not fp8_dtype
            ):
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
        # Under DCP this is the gathered head count (num_head * dcp_world_size);
        # equals num_head_padded when DCP is off.
        nhead = self.mla_kernel_num_head_padded
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

        # Only thread the round-robin CP flag when DCP is on, so the non-DCP path
        # stays agnostic to the installed aiter's get_mla_metadata_v1 signature.
        cprr_kwargs = {"is_cp_round_robin": True} if dcp_enabled() else {}

        # For dcp, kv_indptr here is already localized to this rank's shard
        meta = get_mla_metadata_v1(
            qo_indptr,
            kv_indptr,
            kv_last_page_len,
            self.mla_kernel_num_head_padded // nhead_kv,
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
            **cprr_kwargs,
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
        return_lse: bool = False,
        **kwargs,
    ):
        """Wrap mla_decode_fwd with head-dimension padding for num_head < 16.

        When head_repeat_factor > 1 (i.e. num_head is 4 or 8), q is
        repeat-interleaved to reach num_head_padded (16) before the kernel
        call, and the corresponding output columns are sliced back afterward.
        q / o must already be shaped (..., num_head, head_dim).

        When ``return_lse`` is set (DCP case), also returns the
        per-(token, head) log-sum-exp so the caller can merge partial outputs
        across dcp ranks.
        """
        # Head padding is driven by this call's ACTUAL query-head count
        # (layer.tp_q_head_num), not the backend's static self.num_head: under DCP
        # the dcp-group-gathered layer has num_local_heads * dcp_world_size heads,
        # which is usually already >= 16 (no padding) even when num_local_heads < 16.
        n_heads = layer.tp_q_head_num
        repeat_factor = (16 // n_heads) if n_heads < 16 else 1
        if repeat_factor > 1:
            q_in = q.repeat_interleave(repeat_factor, dim=1)
            o = q.new_empty(
                (q.shape[0], n_heads * repeat_factor, layer.v_head_dim),
                dtype=self.input_dtype,
            )
            # Only opt into the (out, lse) tuple return under DCP; the non-DCP
            # path keeps the legacy single-return contract so it stays agnostic
            # to the installed aiter's mla_decode_fwd signature.
            if return_lse:
                _, lse = mla_decode_fwd(
                    q_in, k_buffer_flat, o, return_lse=True, **kwargs
                )
                return o[:, ::repeat_factor, :], lse[:, ::repeat_factor]
            mla_decode_fwd(q_in, k_buffer_flat, o, **kwargs)
            return o[:, ::repeat_factor, :]
        else:
            o = q.new_empty(
                (q.shape[0], n_heads, layer.v_head_dim),
                dtype=self.input_dtype,
            )
            if return_lse:
                _, lse = mla_decode_fwd(q, k_buffer_flat, o, return_lse=True, **kwargs)
                return o, lse
            mla_decode_fwd(q, k_buffer_flat, o, **kwargs)
            return o

    def _dcp_graph_max_local_kv_len(self) -> int:
        """Fixed upper bound on this rank's local (round-robin) KV shard length,
        used as ``max_seqlen_kv`` during cuda-graph capture/replay so the
        segment count (and every derived buffer shape) is static. A rank owns the
        positions ``pos % W == rank``, so its shard holds at most
        ``ceil(max_context_len / W)`` tokens."""
        w = max(self.dcp_world_size, 1)
        return (self.max_context_len + w - 1) // w

    def _mla_decode_fwd_dcp(self, q, k_buffer, layer, k_descale):
        """DCP decode over this rank's round-robin KV shard.

        Two interchangeable kernels, same (out, base-2 lse) contract:

        * ``_mla_decode_fwd_dcp_triton`` (default) -- aiter's Triton MLA kernel,
          reached through ``aiter.ops.triton.attention.mla.mla_decode_fwd``.
        * ``_mla_decode_fwd_dcp_gluon`` -- aiter's Gluon MLA kernel
          (``aiter.ops.triton.gluon.mla_gluon``), gated on
          SGLANG_USE_AITER_GLUON_MLA_DCP because it needs triton >= 3.7.

        NOTE the two are NOT the same kernel family, despite both having been
        called "gluon" here historically: on gfx950 ``mla_decode_fwd`` picks its
        Gluon implementation only under ``IS_DEVICE_ARCH_GFX12`` and otherwise
        launches ``_triton_kernels.attention.mla._mla_decode_fwd_kernel``. The
        verify and prefill helpers go through the same entry point and are
        likewise Triton on this arch.
        """
        if envs.SGLANG_USE_AITER_GLUON_MLA_DCP.get():
            return self._mla_decode_fwd_dcp_gluon(q, layer)
        return self._mla_decode_fwd_dcp_triton(q, k_buffer, layer, k_descale)

    def _mla_decode_fwd_dcp_gluon(self, q, layer):
        """DCP decode on aiter's Gluon MLA kernel.

        Takes the shard in ``mla_gluon``'s varlen form -- the per-rank token
        indices and request boundaries that ``_plan_dcp_decode_metadata``
        already localized in place (``kv_values // dcp_world_size`` are physical
        rows of this rank's pool). That needs neither the paged block table nor
        the segment reduce the Triton path requires.

        ``min_kv_seq_len=1`` pins NUM_KV_SPLITS to 1, which keeps the grid and
        every buffer shape static (cuda-graph safe) and is the regime this
        kernel wins biggest in -- it is flat in batch where the Triton path is
        linear. It costs throughput only at small batch with a long shard,
        where splitting would fill the machine; sizing splits from a static
        per-graph bound is the follow-up.

        Returns (out, lse) with out [tokens, num_gathered_heads, kv_lora_rank]
        and lse [tokens, num_gathered_heads] in base-2.
        """
        from aiter.ops.triton.gluon.mla_gluon import mla_gluon

        fm = self.forward_metadata
        bs = fm.kv_indptr.shape[0] - 1
        num_heads = layer.tp_q_head_num  # gathered heads = num_local_heads * dcp
        kv_lora_rank = layer.v_head_dim

        q3 = q.view(bs, num_heads, layer.qk_head_dim)
        kv_c = self.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1, layer.qk_head_dim
        )
        out = q.new_empty((bs, num_heads, kv_lora_rank), dtype=self.input_dtype)

        _, lse = mla_gluon(
            q3[:, :, :kv_lora_rank],
            q3[:, :, kv_lora_rank:],
            kv_c,
            out,
            fm.kv_indices,
            fm.kv_indptr[: bs + 1],
            layer.scaling,
            use_2d_view=False,
            min_kv_seq_len=1,
            return_lse=True,
        )
        # mla_gluon returns a natural-log lse [bs, qlen, nhead]; the cross-rank
        # merge (cp_lse_ag_out_rs_mla) takes base-2.
        return out, lse.view(bs, num_heads) * _LOG2E

    def _mla_decode_fwd_dcp_triton(self, q, k_buffer, layer, k_descale):
        """DCP decode via aiter's Triton MLA kernel over this rank's round-robin
        KV shard.

        The kernel tiles the query heads, so it serves any gathered head count
        (e.g. Kimi-K3's 96 at tp8 dcp8).
        Round-robin global-position causal masking is unnecessary for pure decode
        (q_len == 1): every shard entry is a past token, so this rank attends its
        whole shard, and partials are merged across ranks by the caller
        (cp_lse_ag_out_rs_mla) using the base-2 lse returned here.

        Returns (out, lse) with out [tokens, num_gathered_heads, kv_lora_rank] and
        lse [tokens, num_gathered_heads] in base-2.
        """
        from aiter.ops.triton.attention.mla import mla_decode_fwd

        from sglang.kernels.ops.attention.dcp_mla_reduce import (
            dcp_mla_reduce,
        )

        fm = self.forward_metadata
        # kv_indptr was localized to this rank's round-robin shard by
        # _plan_dcp_decode_metadata during init_forward_metadata, which also
        # built the per-rank PAGE table (once per forward, not per layer).
        kv_indptr = fm.kv_indptr
        bs = kv_indptr.shape[0] - 1
        num_heads = layer.tp_q_head_num  # gathered heads = num_local_heads * dcp
        kv_lora_rank = layer.v_head_dim
        qk_rope_head_dim = layer.qk_head_dim - kv_lora_rank

        seqused_k = fm.dcp_local_kv_lens
        block_tables = fm.dcp_block_table
        # Under cuda-graph capture/replay a GPU->CPU sync (.item()) is illegal, and
        # a per-step max would make the kernel's NUM_SEGMENTS (and thus the segm-buffer
        # shapes) vary between capture and replay. Pin max_local to a fixed
        # per-graph upper bound (worst-case local shard = ceil(ctx_len / W)) so
        # every shape is static; per-token correctness still comes from the exact
        # per-request seqused_k masking inside the kernel and the reduce.
        if fm.run_graph:
            max_local = self._dcp_graph_max_local_kv_len()
        else:
            max_local = int(seqused_k.max().item())

        # cu_seqlens_q for decode is [0, 1, ..., bs] (1 query token per request),
        # which is exactly the MLA qo_indptr already built in the metadata.
        cu_seqlens_q = fm.qo_indptr[: bs + 1]
        q3 = q.view(bs, num_heads, layer.qk_head_dim)
        kv_buffer = k_buffer.view(-1, self.page_size, 1, layer.qk_head_dim)
        out = q.new_empty((bs, num_heads, kv_lora_rank), dtype=self.input_dtype)

        segm_output, segm_max, segm_expsum = mla_decode_fwd(
            q3,
            kv_buffer,
            out,
            cu_seqlens_q,
            seqused_k,
            max_local,
            block_tables,
            layer.scaling,
            kv_lora_rank,
            qk_rope_head_dim,
            True,  # causal
            k_descale,
            k_descale,
            skip_reduce=True,
        )
        # segment-reduce the shard partials to (out_local, lse2_local) [base-2].
        return dcp_mla_reduce(
            segm_output,
            segm_max,
            segm_expsum,
            seqused_k,
            self.page_size,
            self.input_dtype,
        )

    def _mla_verify_fwd_dcp(self, q, k_window, layer, k_descale):
        """DCP target-verify, on the Gluon MLA kernel when it is enabled.

        Both variants use the same two-stage split (see the Triton one below);
        only stage A's kernel differs. Stage B is a dense torch matmul either
        way. The Gluon variant needs the per-row TOKEN table, which is only
        built when the flag is on -- fall back if it is missing (e.g. a graph
        captured before the flag was read).
        """
        if (
            envs.SGLANG_USE_AITER_GLUON_MLA_DCP.get()
            and self.forward_metadata.dcp_verify_token_table is not None
        ):
            return self._mla_verify_fwd_dcp_gluon(q, k_window, layer)
        return self._mla_verify_fwd_dcp_triton(q, k_window, layer, k_descale)

    def _mla_verify_fwd_dcp_gluon(self, q, k_window, layer):
        """DCP target-verify stage A on aiter's Gluon MLA kernel.

        The metadata already flattened the ``bs x q_len`` window into
        ``bs * q_len`` single-token rows, so this is a plain decode for
        mla_gluon: q is 3-D (qlen 1) and every row attends its whole committed
        shard, bounded by its own ``cache_seqlens``. Rows of one request repeat
        that request's shard, which is what ``dcp_verify_token_table`` holds.

        ``min_kv_seq_len=1`` pins NUM_KV_SPLITS to 1 for the same
        capture-stability reason as the decode path.
        """
        from aiter.ops.triton.gluon.mla_gluon import mla_gluon

        fm = self.forward_metadata
        q_len = fm.max_q_len
        num_heads = layer.tp_q_head_num
        kv_lora_rank = layer.v_head_dim
        seqused_k = fm.dcp_local_kv_lens
        n_rows = seqused_k.shape[0]
        bs = n_rows // q_len

        q3 = q.view(n_rows, num_heads, layer.qk_head_dim)
        out_a = q.new_empty((n_rows, num_heads, kv_lora_rank), dtype=self.input_dtype)
        _, lse_a = mla_gluon(
            q3[:, :, :kv_lora_rank],
            q3[:, :, kv_lora_rank:],
            self.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
                -1, layer.qk_head_dim
            ),
            out_a,
            fm.dcp_verify_token_table,
            seqused_k,
            layer.scaling,
            use_2d_view=True,
            min_kv_seq_len=1,
            return_lse=True,
        )
        lse_a = lse_a.view(n_rows, num_heads) * _LOG2E

        if fm.dcp_cp_rank != 0:
            return out_a, lse_a

        out_b, lse_b = dense_causal_mla_attn_base2(
            q, k_window, layer.scaling, bs, q_len, kv_lora_rank
        )
        return lse_combine_base2(out_a, lse_a, out_b, lse_b, self.input_dtype)

    def _mla_verify_fwd_dcp_triton(self, q, k_window, layer, k_descale):
        """DCP target-verify, split at the verify-window boundary.

        A verify step queries ``q_len = gamma + 1`` tokens per request against
        the committed KV plus the window itself. Splitting the key set there
        avoids the one thing mla_decode_fwd cannot do under DCP -- masking on the GLOBAL
        position ``g(j) = j * W + r`` rather than the local shard index ``j``:

        * stage A, the COMMITTED shard: every entry precedes every window token,
          so no causal masking is needed at all. The metadata flattens the batch
          into ``bs * q_len`` single-token rows, which puts the kernel in its
          ``num_tokens_per_seq == 1`` regime where the mask degenerates to
          ``j < seq_len`` -- exactly "attend this rank's whole shard".
        * stage B, the in-hand window: dense and not yet sharded, so a plain
          causal ``q_len x q_len`` attention is correct on its own.

        Stage B is folded in on ONE rank only, so that after the caller's
        cross-rank merge (cp_lse_ag_out_rs_mla) the window is counted exactly
        once; the other ranks simply return their shard partial.

        Returns (out [bs * q_len, H_gathered, kv_lora_rank],
        lse [bs * q_len, H_gathered]) with lse in base-2, ready for that merge.
        """
        from aiter.ops.triton.attention.mla import mla_decode_fwd

        from sglang.kernels.ops.attention.dcp_mla_reduce import (
            dcp_mla_reduce,
        )

        fm = self.forward_metadata
        q_len = fm.max_q_len
        num_heads = layer.tp_q_head_num  # gathered heads = num_local_heads * dcp
        kv_lora_rank = layer.v_head_dim
        qk_rope_head_dim = layer.qk_head_dim - kv_lora_rank

        # Both were expanded to one row per window token when the metadata was
        # planned, so the shard lengths and page table are already per-row.
        seqused_k = fm.dcp_local_kv_lens
        n_rows = seqused_k.shape[0]
        bs = n_rows // q_len

        # Same reasoning as the decode path: a per-step max would make the kernel's
        # NUM_SEGMENTS vary between capture and replay, and .item() is illegal
        # under capture. Per-token correctness comes from seqused_k regardless.
        if fm.run_graph:
            max_local = self._dcp_graph_max_local_kv_len()
        else:
            max_local = int(seqused_k.max().item())

        k_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        out_a = q.new_empty((n_rows, num_heads, kv_lora_rank), dtype=self.input_dtype)
        segm_output, segm_max, segm_expsum = mla_decode_fwd(
            q.view(n_rows, num_heads, layer.qk_head_dim),
            k_buffer.view(-1, self.page_size, 1, layer.qk_head_dim),
            out_a,
            fm.dcp_verify_qo_indptr[: n_rows + 1],
            seqused_k,
            max_local,
            fm.dcp_block_table,
            layer.scaling,
            kv_lora_rank,
            qk_rope_head_dim,
            True,  # causal (degenerate at num_tokens_per_seq == 1)
            k_descale,
            k_descale,
            skip_reduce=True,
        )
        out_a, lse_a = dcp_mla_reduce(
            segm_output,
            segm_max,
            segm_expsum,
            seqused_k,
            self.page_size,
            self.input_dtype,
        )

        if fm.dcp_cp_rank != 0:
            return out_a, lse_a

        out_b, lse_b = dense_causal_mla_attn_base2(
            q, k_window, layer.scaling, bs, q_len, kv_lora_rank
        )
        return lse_combine_base2(out_a, lse_a, out_b, lse_b, self.input_dtype)

    def _mla_prefill_fwd_dcp(self, q, layer, k_descale, forward_batch):
        """DCP prefill (extend), on the Gluon MLA kernel when it can serve the
        batch and on aiter's Triton absorb-prefill kernel otherwise.

        ``mla_gluon`` has no ``qo_indptr``: its MTP mode takes q as
        ``[bs, qlen, nhead, dim]``, so one compiled kernel serves any qlen but
        every request in the batch must contribute the SAME number of extend
        tokens. A prefill batch is ragged in general, so this dispatches per
        batch rather than per config, and the Triton path stays the fallback.
        """
        if envs.SGLANG_USE_AITER_GLUON_MLA_DCP.get():
            extend_lens = forward_batch.extend_seq_lens_cpu
            if extend_lens and len(set(extend_lens)) == 1:
                return self._mla_prefill_fwd_dcp_gluon(
                    q, layer, forward_batch, q_len=extend_lens[0]
                )
        return self._mla_prefill_fwd_dcp_triton(q, layer, k_descale, forward_batch)

    def _mla_prefill_fwd_dcp_gluon(self, q, layer, forward_batch, q_len: int):
        """DCP prefill (extend) on aiter's Gluon MLA kernel, MTP mode.

        Same assembled-KV setup as the Triton path below, but taken in
        ``mla_gluon``'s varlen form (``dcp_kv_indices`` + ``dcp_kv_indptr``), so
        it needs neither the page repack nor a paged block table. MTP masking is
        ``q_pos attends KV[0, seq_len - q_len + q_pos]``, which is exactly the
        extend window: ``seq_len`` is the request's assembled prefix + extend and
        ``q_len`` its extend count.
        """
        from aiter.ops.triton.gluon.mla_gluon import mla_gluon

        dcp_meta = forward_batch.attn_dcp_metadata
        kv_indptr = dcp_meta.dcp_kv_indptr
        bs = kv_indptr.shape[0] - 1
        num_heads = layer.tp_q_head_num
        kv_lora_rank = layer.v_head_dim

        q4 = q.view(bs, q_len, num_heads, layer.qk_head_dim)
        out = q.new_empty(
            (q.shape[0], num_heads * kv_lora_rank), dtype=self.input_dtype
        )
        mla_gluon(
            q4[..., :kv_lora_rank],
            q4[..., kv_lora_rank:],
            dcp_meta.dcp_kv_buffer.view(-1, layer.qk_head_dim),
            out.view(bs, q_len, num_heads, kv_lora_rank),
            dcp_meta.dcp_kv_indices,
            kv_indptr[: bs + 1],
            layer.scaling,
            use_2d_view=False,
            min_kv_seq_len=1,
            return_lse=False,
        )
        return out

    def _mla_prefill_fwd_dcp_triton(self, q, layer, k_descale, forward_batch):
        """DCP prefill (extend) via aiter's MLA absorb-prefill kernel over
        the assembled full-sequence KV in ``attn_dcp_metadata.dcp_kv_buffer``.

        Under DCP the KV pool is physically round-robin sharded (owner rule
        pos % W == rank, physical = pos // W), so the full per-request sequence
        cannot be read from this rank's local cache. The model layer
        (forward_absorb_prepare -> all_gather_kv_cache_for_mla_extend) assembles
        the full KV — gathered prefix + in-hand new tokens — into dcp_kv_buffer,
        indexed by dcp_kv_indices with per-request boundaries dcp_kv_indptr.

        Kimi-K3 MLA is NoPE with 12 local heads at tp8; the ASM mla_prefill_fwd
        and flash_attn_varlen absorb path can't serve those dims. mla_prefill_fwd tiles
        heads. Q is NOT gathered for extend (each rank computes full attention for
        its local heads over the full KV — no cross-rank merge).
        """
        from aiter.ops.triton.attention.mla import mla_prefill_fwd

        from sglang.kernels.ops.attention.dcp_mla_reduce import (
            pack_dcp_kv_into_pages,
        )

        dcp_meta = forward_batch.attn_dcp_metadata
        kv_buffer = dcp_meta.dcp_kv_buffer  # [seq_lens_sum, 1, qk_head_dim]
        kv_indptr = dcp_meta.dcp_kv_indptr  # [bs + 1] full-seq boundaries
        kv_indices = dcp_meta.dcp_kv_indices  # indices into dcp_kv_buffer
        bs = kv_indptr.shape[0] - 1
        num_heads = layer.tp_q_head_num
        kv_lora_rank = layer.v_head_dim
        qk_rope_head_dim = layer.qk_head_dim - kv_lora_rank

        # NOTE: the .item() below is a GPU->CPU sync, which would be illegal
        # under cuda-graph capture -- but this path is uncapturable anyway
        # (pack_dcp_kv_into_pages allocates per batch) and prefill cuda graphs
        # are disabled for K3 DCP. Do NOT try to assert that via
        # ForwardMetadata.run_graph: it defaults to True and is only set to
        # False explicitly on the eager decode/verify paths, so on the prefill
        # metadata it is always True and says nothing about capture state.
        seqused_k = (kv_indptr[1 : bs + 1] - kv_indptr[:bs]).to(torch.int32)
        max_kv = int(seqused_k.max().item())
        # mla_prefill_fwd takes its KV tile straight from the paged block size, so feeding
        # the assembled buffer as block_size 1 collapses each tile to a single
        # token: a 16384x16384 chunk measured 5156 ms that way vs 39 ms at block
        # size 64 on gfx950 (~131x). Repack into per-request page-aligned pages.
        paged_kv, block_tables = pack_dcp_kv_into_pages(
            kv_buffer, kv_indptr, kv_indices, bs, _DCP_PREFILL_PAGE_SIZE
        )

        out = q.new_empty(
            (q.shape[0], num_heads * kv_lora_rank), dtype=self.input_dtype
        )
        mla_prefill_fwd(
            q.view(-1, num_heads, layer.qk_head_dim),
            paged_kv,
            out.view(-1, num_heads, kv_lora_rank),
            self.forward_metadata.qo_indptr,  # extend query-token boundaries
            seqused_k,
            max_kv,
            block_tables,
            layer.scaling,
            kv_lora_rank,
            qk_rope_head_dim,
            True,  # causal
            k_descale,
            k_descale,
        )
        return out

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
            # Prefill PS metadata has no split cap; 0 keeps AITER's default reduce sizing.
            0,
            output,
            final_lse,
        )
        return output

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
            encoder_lens=forward_batch.encoder_lens,
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

        # dcp metadata
        dcp_g_kv_indptr = None
        dcp_cp_world_size = 1
        dcp_cp_rank = 0
        dcp_block_table = None
        dcp_local_kv_lens = None
        dcp_verify_qo_indptr = None
        dcp_verify_token_table = None
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

                    if (
                        self.use_mla
                        and dcp_enabled()
                        and not forward_batch.forward_mode.is_idle()
                    ):
                        dcp_g_kv_indptr = kv_indptr.clone()
                        dcp_cp_world_size = get_attention_dcp_world_size()
                        dcp_cp_rank = get_attention_dcp_rank()
                        kv_lens = forward_batch.seq_lens[:bs].to(torch.int32).clone()
                        self._plan_dcp_decode_metadata(
                            kv_indptr,
                            kv_indices,
                            kv_lens,
                            forward_batch.seq_lens_cpu,
                            bs,
                        )
                        (
                            dcp_block_table,
                            dcp_local_kv_lens,
                        ) = self._build_dcp_decode_page_table(
                            kv_indptr,
                            forward_batch.req_pool_indices,
                            bs,
                            (max_kv_len + self.dcp_world_size - 1)
                            // self.dcp_world_size,
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

                # DCP decode runs the aiter MLA kernel (builds its own block-table
                # metadata in forward_decode), so skip the cprr persist metadata.
                if _use_mla_ps_kernel and dcp_cp_world_size == 1:
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
                dcp_g_kv_indptr=dcp_g_kv_indptr,
                dcp_cp_world_size=dcp_cp_world_size,
                dcp_cp_rank=dcp_cp_rank,
                dcp_block_table=dcp_block_table,
                dcp_local_kv_lens=dcp_local_kv_lens,
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
        elif forward_batch.forward_mode.is_target_verify():
            if self.use_mla:
                draft_num = spec_info.draft_token_num
                device = forward_batch.seq_lens.device
                # Under DCP the committed prefix and the gamma+1 window are
                # attended as two separate stages, so the shard plan must cover
                # the COMMITTED length only: the window tokens this rank owns
                # were already written to the cache by the model layer and would
                # otherwise be counted twice (once in each stage).
                verify_dcp = self.use_mla and dcp_enabled()
                if verify_dcp:
                    kv_lens = forward_batch.seq_lens.to(torch.int32).clone()
                    kv_lens_sum = forward_batch.seq_lens_sum
                else:
                    kv_lens = forward_batch.seq_lens + draft_num
                    kv_lens_sum = forward_batch.seq_lens_sum + draft_num * bs

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

                if verify_dcp:
                    dcp_g_kv_indptr = kv_indptr.clone()
                    dcp_cp_world_size = get_attention_dcp_world_size()
                    dcp_cp_rank = get_attention_dcp_rank()
                    # seq_lens_cpu is NOT usable here: the DSPARK verify path
                    # pre-adds the window to it while the device seq_lens still
                    # holds the committed length, so the CPU fast path would
                    # plan a shard that is draft_num tokens too long.
                    self._plan_dcp_decode_metadata(
                        kv_indptr,
                        kv_indices,
                        kv_lens,
                        None,
                        bs,
                    )
                    (
                        dcp_block_table,
                        dcp_local_kv_lens,
                        dcp_verify_qo_indptr,
                    ) = self._build_dcp_verify_page_table(
                        kv_indptr,
                        forward_batch.req_pool_indices,
                        bs,
                        draft_num,
                        (max_kv_len + self.dcp_world_size - 1) // self.dcp_world_size,
                    )
                    if envs.SGLANG_USE_AITER_GLUON_MLA_DCP.get():
                        dcp_verify_token_table = self._build_dcp_verify_token_table(
                            kv_indptr,
                            kv_indices,
                            bs,
                            draft_num,
                            (max_kv_len + self.dcp_world_size - 1)
                            // self.dcp_world_size,
                        )

                # if self.kv_cache_dtype == fp8_dtype:
                if _use_mla_ps_kernel and not verify_dcp:
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
                    dcp_g_kv_indptr=dcp_g_kv_indptr,
                    dcp_cp_world_size=dcp_cp_world_size,
                    dcp_cp_rank=dcp_cp_rank,
                    dcp_block_table=dcp_block_table,
                    dcp_local_kv_lens=dcp_local_kv_lens,
                    dcp_verify_qo_indptr=dcp_verify_qo_indptr,
                    dcp_verify_token_table=dcp_verify_token_table,
                )
            else:
                draft_num = forward_batch.input_ids.shape[0] // bs
                bs = len(forward_batch.req_pool_indices)

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

    def _dcp_graph_local_len_bounds(self, bs: int) -> tuple[int, int]:
        """Static (max, total) upper bounds on this rank's local shard lengths
        for a cuda-graph batch of ``bs`` rows, as accepted by
        ``plan_dcp_decode_metadata(static_local_len_bounds=...)``. Every row is
        bounded by the worst-case shard, so the batch total is ``bs`` times it.
        """
        max_local = self._dcp_graph_max_local_kv_len()
        return max_local, bs * max_local

    def _plan_dcp_decode_metadata(
        self,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_lens_gpu: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        bs: int,
        static_local_len_bounds: Optional[tuple[int, int]] = None,
    ):
        """Localize kv_indptr / kv_indices to this rank's DCP round-robin shard
        (in place). When ``seq_lens_cpu`` is available, size the plan from a CPU
        local-length array (``init_metadata_replay=True``) so no per-step GPU->CPU
        sync is needed; when only ``static_local_len_bounds`` is available, size it
        from those bounds, which is equally sync-free.
        """
        if static_local_len_bounds is not None:
            plan_dcp_decode_metadata(
                kv_lens_gpu,
                kv_indptr,
                kv_indices,
                init_metadata_replay=False,
                fast_decode_kwargs={},
                bs=bs,
                static_local_len_bounds=static_local_len_bounds,
            )
        elif seq_lens_cpu is not None:
            kv_len_arr_cpu = seq_lens_cpu[:bs].to(torch.int32).clone()
            update_local_kv_lens_for_dcp(kv_len_arr_cpu)
            plan_dcp_decode_metadata(
                kv_lens_gpu,
                kv_indptr,
                kv_indices,
                init_metadata_replay=True,
                fast_decode_kwargs={"kv_len_arr_cpu": kv_len_arr_cpu},
                bs=bs,
            )
        else:
            plan_dcp_decode_metadata(
                kv_lens_gpu,
                kv_indptr,
                kv_indices,
                init_metadata_replay=False,
                fast_decode_kwargs={},
                bs=bs,
            )

    def _build_dcp_decode_page_table(
        self,
        kv_indptr: torch.Tensor,
        req_pool_indices: torch.Tensor,
        bs: int,
        max_local_kv_len: int,
        out: Optional[torch.Tensor] = None,
        out_lens: Optional[torch.Tensor] = None,
    ):
        """Per-rank page table + shard lengths for the Triton DCP decode.

        Built once per forward (the kernel call is per layer, and rebuilding the
        table 24x per step is pure overhead). ``kv_indptr`` must already be
        localized by ``_plan_dcp_decode_metadata``, so its per-request diffs are
        this rank's shard lengths in tokens (the kernel's ``seqused_k``).

        Both outputs are written into caller-provided buffers on the cuda-graph
        path: this runs OUT of the graph, so a freshly allocated tensor would be
        invisible to the captured kernels, which keep the capture-time pointer.
        """
        from sglang.kernels.ops.attention.dcp_mla_reduce import (
            build_dcp_page_table,
        )

        lens = (kv_indptr[1 : bs + 1] - kv_indptr[:bs]).to(torch.int32)
        if out_lens is not None:
            out_lens.copy_(lens)
            local_kv_lens = out_lens
        else:
            local_kv_lens = lens
        max_pages = (max_local_kv_len + self.page_size - 1) // self.page_size
        block_table = build_dcp_page_table(
            self.req_to_token,
            req_pool_indices,
            local_kv_lens,
            bs,
            max_pages,
            self.page_size,
            self.dcp_world_size,
            get_attention_dcp_rank(),
            out=out,
        )
        return block_table, local_kv_lens

    def _build_dcp_verify_page_table(
        self,
        kv_indptr: torch.Tensor,
        req_pool_indices: torch.Tensor,
        bs: int,
        q_len: int,
        max_local_kv_len: int,
        out: Optional[torch.Tensor] = None,
        out_lens: Optional[torch.Tensor] = None,
        qo_indptr: Optional[torch.Tensor] = None,
    ):
        """Per-ROW page table, shard lengths and cu_seqlens_q for the Triton DCP
        target-verify stage A.

        Stage A flattens the ``bs x q_len`` verify window into ``bs * q_len``
        single-token rows, which is what puts the kernel in its
        ``num_tokens_per_seq == 1`` regime where the causal mask degenerates to
        "attend the whole shard" (see _mla_verify_fwd_dcp_triton). Every row of a
        request shares that request's committed shard, so the decode table and
        lengths just repeat ``q_len`` times -- expanded once per forward rather
        than per layer.

        On the cuda-graph path the caller supplies capture-stable buffers: this
        runs OUT of the graph, so freshly allocated outputs would be invisible to
        the captured kernels (they keep the capture-time pointer).
        """
        block_table, local_kv_lens = self._build_dcp_decode_page_table(
            kv_indptr, req_pool_indices, bs, max_local_kv_len
        )
        n_rows = bs * q_len
        if out is None:
            out = block_table.new_empty((n_rows, block_table.shape[1]))
        out.view(bs, q_len, -1).copy_(block_table.unsqueeze(1).expand(bs, q_len, -1))
        if out_lens is None:
            out_lens = local_kv_lens.new_empty((n_rows,))
        out_lens.view(bs, q_len).copy_(local_kv_lens.unsqueeze(1).expand(bs, q_len))
        if qo_indptr is None:
            qo_indptr = torch.arange(
                n_rows + 1, dtype=torch.int32, device=kv_indptr.device
            )
        return out, out_lens, qo_indptr

    def _build_dcp_verify_token_table(
        self,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        bs: int,
        q_len: int,
        max_local_kv_len: int,
        out: Optional[torch.Tensor] = None,
    ):
        """Per-ROW TOKEN table for the Gluon DCP target-verify stage A.

        Same expansion as the page table above -- every row of a request shares
        that request's committed shard and length -- but one column per TOKEN,
        because ``mla_gluon`` fixes PAGE_SIZE at 1. Built from the ragged
        (kv_indptr, kv_indices) that ``_plan_dcp_decode_metadata`` already
        localized, so it costs one scatter per forward, not per layer.
        """
        from sglang.kernels.ops.attention.dcp_mla_reduce import build_dcp_block_table

        per_req = build_dcp_block_table(kv_indptr, kv_indices, bs, max_local_kv_len)
        n_rows = bs * q_len
        if out is None:
            out = per_req.new_empty((n_rows, per_req.shape[1]))
        out.view(bs, q_len, -1).copy_(per_req.unsqueeze(1).expand(bs, q_len, -1))
        return out

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
        if self.use_mla and dcp_enabled():
            self.cuda_graph_dcp_g_kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=self.device
            )
            # Page table for the Triton DCP decode: capture-stable shape, filled
            # out-of-graph in _apply_cuda_graph_metadata. Width is the worst-case
            # local shard (ceil(ctx_len / W)) in pages.
            max_local_pages = (
                self._dcp_graph_max_local_kv_len() + self.page_size - 1
            ) // self.page_size
            self.cuda_graph_dcp_block_table = torch.zeros(
                (max_bs, max_local_pages), dtype=torch.int32, device=self.device
            )
            self.cuda_graph_dcp_local_kv_lens = torch.zeros(
                (max_bs,), dtype=torch.int32, device=self.device
            )
            if self.num_draft_tokens:
                # Target-verify flattens each request's gamma+1 window into that
                # many single-token rows (see _build_dcp_verify_page_table), so
                # it needs max_bs * num_draft_tokens rows rather than max_bs.
                # cu_seqlens_q is a plain arange, so one buffer serves every bs.
                n_verify_rows = max_bs * self.num_draft_tokens
                self.cuda_graph_dcp_verify_block_table = torch.zeros(
                    (n_verify_rows, max_local_pages),
                    dtype=torch.int32,
                    device=self.device,
                )
                self.cuda_graph_dcp_verify_local_kv_lens = torch.zeros(
                    (n_verify_rows,), dtype=torch.int32, device=self.device
                )
                self.cuda_graph_dcp_verify_qo_indptr = torch.arange(
                    n_verify_rows + 1, dtype=torch.int32, device=self.device
                )
                # One column per TOKEN rather than per page, so this is
                # page_size times wider than the block table above. Only the
                # Gluon path reads it; skip the allocation otherwise.
                self.cuda_graph_dcp_verify_token_table = (
                    torch.zeros(
                        (n_verify_rows, self._dcp_graph_max_local_kv_len()),
                        dtype=torch.int32,
                        device=self.device,
                    )
                    if envs.SGLANG_USE_AITER_GLUON_MLA_DCP.get()
                    else None
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
        encoder_lens: Optional[torch.Tensor],
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

        # DCP metadata which will be populated for MLA decode when dcp enabled
        dcp_g_kv_indptr = None
        dcp_cp_world_size = 1
        dcp_cp_rank = 0
        dcp_block_table = None
        dcp_local_kv_lens = None
        dcp_verify_qo_indptr = None
        dcp_verify_token_table = None

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

                    if self.use_mla and dcp_enabled() and not forward_mode.is_idle():
                        self.cuda_graph_dcp_g_kv_indptr[: bs + 1] = kv_indptr
                        dcp_g_kv_indptr = self.cuda_graph_dcp_g_kv_indptr[: bs + 1]
                        dcp_cp_world_size = get_attention_dcp_world_size()
                        dcp_cp_rank = get_attention_dcp_rank()
                        kv_lens = seq_lens[:bs].to(torch.int32).clone()
                        self._plan_dcp_decode_metadata(
                            kv_indptr,
                            kv_indices,
                            kv_lens,
                            seq_lens_cpu,
                            bs,
                        )
                        # Runs out-of-graph, so filling the capture-stable page
                        # table buffer here is legal.
                        (
                            dcp_block_table,
                            dcp_local_kv_lens,
                        ) = self._build_dcp_decode_page_table(
                            kv_indptr,
                            req_pool_indices,
                            bs,
                            self._dcp_graph_max_local_kv_len(),
                            out=self.cuda_graph_dcp_block_table[:bs],
                            out_lens=self.cuda_graph_dcp_local_kv_lens[:bs],
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

                # DCP decode runs the aiter MLA kernel (builds its own block-table
                # metadata in forward_decode), so the cprr persist metadata is
                # unused on this path and is skipped. Mirrors the eager guard in
                # init_forward_metadata.
                if _use_mla_ps_kernel and dcp_cp_world_size == 1:
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
                dcp_g_kv_indptr=dcp_g_kv_indptr,
                dcp_cp_world_size=dcp_cp_world_size,
                dcp_cp_rank=dcp_cp_rank,
                dcp_block_table=dcp_block_table,
                dcp_local_kv_lens=dcp_local_kv_lens,
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
            # Under DCP the committed prefix and the gamma+1 window are attended
            # as two separate stages, so plan the shard over the COMMITTED length
            # only -- the window tokens this rank owns are already in the cache
            # and would otherwise be counted in both stages.
            verify_dcp = self.use_mla and dcp_enabled()
            if self.use_mla and not verify_dcp:
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

            if verify_dcp:
                self.cuda_graph_dcp_g_kv_indptr[: bs + 1] = kv_indptr
                dcp_g_kv_indptr = self.cuda_graph_dcp_g_kv_indptr[: bs + 1]
                dcp_cp_world_size = get_attention_dcp_world_size()
                dcp_cp_rank = get_attention_dcp_rank()
                # seq_lens_cpu is NOT usable here: the DSPARK verify path
                # pre-adds the window to it while the device seq_lens still holds
                # the committed length, so the CPU fast path would plan a shard
                # draft_num tokens too long. The exact device sizing costs two
                # GPU->CPU syncs on EVERY replay, so size the plan from the
                # static per-graph bounds instead -- sync-free by construction.
                # (Measured as a wash at bs=1 and bs~44: the overlap scheduler
                # keeps the CPU ahead, so the sync waited on finished work.)
                self._plan_dcp_decode_metadata(
                    kv_indptr,
                    kv_indices,
                    seq_lens[:bs].to(torch.int32).clone(),
                    None,
                    bs,
                    static_local_len_bounds=self._dcp_graph_local_len_bounds(bs),
                )
                n_rows = bs * self.num_draft_tokens
                (
                    dcp_block_table,
                    dcp_local_kv_lens,
                    dcp_verify_qo_indptr,
                ) = self._build_dcp_verify_page_table(
                    kv_indptr,
                    req_pool_indices,
                    bs,
                    self.num_draft_tokens,
                    self._dcp_graph_max_local_kv_len(),
                    out=self.cuda_graph_dcp_verify_block_table[:n_rows],
                    out_lens=self.cuda_graph_dcp_verify_local_kv_lens[:n_rows],
                    qo_indptr=self.cuda_graph_dcp_verify_qo_indptr[: n_rows + 1],
                )
                if self.cuda_graph_dcp_verify_token_table is not None:
                    dcp_verify_token_table = self._build_dcp_verify_token_table(
                        kv_indptr,
                        kv_indices,
                        bs,
                        self.num_draft_tokens,
                        self._dcp_graph_max_local_kv_len(),
                        out=self.cuda_graph_dcp_verify_token_table[:n_rows],
                    )

            if self.use_mla:
                max_q_len = self.num_draft_tokens
                if _use_mla_ps_kernel and not verify_dcp:
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
                    dcp_g_kv_indptr=dcp_g_kv_indptr,
                    dcp_cp_world_size=dcp_cp_world_size,
                    dcp_cp_rank=dcp_cp_rank,
                    dcp_block_table=dcp_block_table,
                    dcp_local_kv_lens=dcp_local_kv_lens,
                    dcp_verify_qo_indptr=dcp_verify_qo_indptr,
                    dcp_verify_token_table=dcp_verify_token_table,
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
                    if dcp_enabled():
                        # DCP: the KV pool is physically round-robin sharded. Use
                        # set_mla_kv_buffer, whose kernel takes the RAW virtual
                        # out_cache_loc and INTERNALLY owner-filters
                        # (loc % dcp_size == dcp_rank) and shards (loc // dcp_size).
                        # Pass the raw cache_loc and the FULL (unmasked, undivided) k
                        # split into (k_nope, k_pe): pre-dividing/pre-masking here
                        # double-applies the kernel's own filter and corrupts the
                        # write. (set_kv_buffer's MLA DCP path owner-filters but does
                        # NOT //dcp_size, so it is wrong for this sharded pool.)
                        kv_lora_rank = v.shape[-1]
                        self.token_to_kv_pool.set_mla_kv_buffer(
                            layer,
                            cache_loc,
                            k[..., :kv_lora_rank],
                            k[..., kv_lora_rank:],
                        )
                    else:
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

            if (
                forward_batch.forward_mode.is_target_verify()
                and self.forward_metadata.dcp_cp_world_size > 1
            ):
                # Two-stage DCP verify. Dispatched before the dims below because
                # under DCP the model layer writes the KV shard itself and hands
                # us the window's k/v directly (v is k's first kv_lora_rank
                # columns), not a full-sequence k to measure head dims from.
                return self._mla_verify_fwd_dcp(q, k, layer, k_descale)

            qk_nope_head_dim = k.shape[-1] - qk_rope_head_dim
            assert len(q.shape) == 3
            assert len(k.shape) == 3
            assert len(v.shape) == 3

            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend_v2()
            ):
                if dcp_enabled():
                    # DCP absorb-prefill over the assembled full-seq KV
                    # (dcp_kv_buffer), since the local cache is round-robin sharded.
                    return self._mla_prefill_fwd_dcp(q, layer, k_descale, forward_batch)
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
                        seqused_k=(
                            forward_batch.seq_lens + self.forward_metadata.max_q_len
                        ),
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

            bs0 = forward_batch.batch_size + 1
            q_descale = None

            window_size = (-1, -1)
            if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
                window_size = (layer.sliding_window_size, -1)

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
                return o.view(-1, layer.tp_q_head_num * layer.head_dim)

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
            k_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)

            if self.forward_metadata.dcp_cp_world_size > 1:
                # DCP decode: run an aiter MLA decode over this rank's
                # round-robin KV shard and return (out, lse2). The kernel
                # tiles query heads, so any gathered head count works, and its
                # base-2 lse feeds cp_lse_ag_out_rs_mla for the cross-rank merge.
                return self._mla_decode_fwd_dcp(q, k_buffer, layer, k_descale)

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
