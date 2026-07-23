from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.flash_attention import flash_attn_varlen_func
from sglang.kernels.ops.attention.metadata import get_num_kv_splits_triton
from sglang.kernels.ops.kvcache.kv_indices import (
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.configs.hybrid_arch import (
    hybrid_gdn_config,
    kimi_linear_config,
    linear_attn_model_spec,
)
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.quantized_kv_prefill import (
    _apply_oscar_rotation,
    _pool_uses_oscar_rotation,
    apply_inverse_v_rotation,
    apply_segmented_hadamard_transform,
    dequantize_prefix_kv,
    prepare_quantized_extend_qkv,
)
from sglang.srt.layers.dcp import (
    cp_lse_ag_out_rs_mha,
    create_triton_kv_indices_for_dcp_triton,
    get_dcp_lens,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVWriteLoc, MHATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool
from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.spec_utils import (
    draft_kv_indices_buffer_width,
    draft_kv_indices_used_len,
    generate_draft_decode_kv_indices,
)
from sglang.srt.utils import (
    get_bool_env_var,
    get_device_core_count,
    get_int_env_var,
    is_cuda,
    is_gfx95_supported,
    is_gfx942_supported,
    is_xpu,
    next_power_of_2,
)

_is_cuda = is_cuda()
_is_gfx942 = is_gfx942_supported()
_is_xpu = is_xpu()

if _is_cuda:
    from sgl_kernel.utils import is_arch_support_pdl

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


_MLA_DECODE_MIN_BLOCK_KV = 32


def _mla_decode_kv_splits_cap(
    base_max_kv_splits: int, sm_count: int, max_context_len: int
) -> int:
    if sm_count <= 0:
        return base_max_kv_splits
    sm_cap = next_power_of_2(sm_count)
    ctx_cap = next_power_of_2(triton.cdiv(max_context_len, _MLA_DECODE_MIN_BLOCK_KV))
    return max(base_max_kv_splits, min(sm_cap, ctx_cap))


@triton.jit
def _count_mixed_hp_lens_kernel(
    req_to_token_ptr,  # int32 [num_req_slots, max_ctx]
    req_pool_indices_ptr,  # int64 [bs]
    seq_lens_ptr,  # int32 [bs]
    hp_lens_ptr,  # int32 [bs]
    rtt_stride_row,
    HP_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Count per-request HP lengths without dense mask materialization.

    This keeps the req-pool indirection fused with the tier classification so
    ``_build_mixed_kv_indices`` never has to materialize a gathered ``rows``
    tensor or per-token boolean masks. The quant tier length is derived from
    ``seq_len - hp_len`` on the Python side.
    """
    req = tl.program_id(0)
    req_pool_idx = tl.load(req_pool_indices_ptr + req).to(tl.int64)
    seq_len = tl.load(seq_lens_ptr + req).to(tl.int32)

    hp_count = tl.zeros((), dtype=tl.int32)
    num_loops = tl.cdiv(seq_len, BLOCK_SIZE)
    for i in range(num_loops):
        offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offs < seq_len
        slot = tl.load(
            req_to_token_ptr + req_pool_idx * rtt_stride_row + offs.to(tl.int64),
            mask=valid,
            other=0,
        ).to(tl.int64)
        hp_count += tl.sum((valid & (slot >= HP_OFFSET)).to(tl.int32), axis=0)

    tl.store(hp_lens_ptr + req, hp_count)


@triton.jit
def _scatter_mixed_kv_indices_kernel(
    req_to_token_ptr,  # int32 [num_req_slots, max_ctx]
    req_pool_indices_ptr,  # int64 [bs]
    seq_lens_ptr,  # int32 or int64 [bs] -- cast inside
    hp_kv_indptr_ptr,  # int32 [bs + 1]   already cumsum'd
    quant_kv_indptr_ptr,  # int32 [bs + 1]   already cumsum'd
    hp_kv_indices_ptr,  # int64 [*] destination, pre-sized
    quant_kv_indices_ptr,  # int64 [*] destination, pre-sized
    rtt_stride_row,
    HP_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Slot-id-classified scatter into hp/quant index buffers, one block per req.

    For each request i we walk ``req_to_token[req_pool_indices[i], 0..seq_len)``
    in ``BLOCK_SIZE`` chunks. Each lane decides whether its slot id is HP
    (``slot >= HP_OFFSET``) or quant, then contributes to within-block exclusive
    prefix sums that act as scatter offsets into the pre-cumsum'd
    ``hp_kv_indptr`` / ``quant_kv_indptr`` tier-local layout. No masked-select,
    no Python bs-loop, and no D2H sync: stride and offset arithmetic is all on
    device with shapes known statically.
    """
    req = tl.program_id(0)
    req_pool_idx = tl.load(req_pool_indices_ptr + req).to(tl.int64)
    seq_len = tl.load(seq_lens_ptr + req).to(tl.int32)
    hp_base = tl.load(hp_kv_indptr_ptr + req).to(tl.int64)
    quant_base = tl.load(quant_kv_indptr_ptr + req).to(tl.int64)

    # Running counters for the chunked scatter. Triton tracks these as scalar
    # SSA values that accumulate across the Python-side for loop below.
    hp_running = tl.zeros((), dtype=tl.int32)
    quant_running = tl.zeros((), dtype=tl.int32)

    num_loops = tl.cdiv(seq_len, BLOCK_SIZE)
    for i in range(num_loops):
        offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offs < seq_len
        slot = tl.load(
            req_to_token_ptr + req_pool_idx * rtt_stride_row + offs.to(tl.int64),
            mask=valid,
            other=0,
        ).to(tl.int64)
        # HP slot ids start at exactly ``HP_OFFSET`` (page 0 is a valid HP
        # page), so the boundary is ``>=`` not ``>``. The unified pool /
        # allocator (``unified_kv_pool._split_global_locs``,
        # ``unified_kv_allocator.free``) and the GPU flush kernel
        # (``gpu_flush_int2``) all classify by ``>=``; using ``>`` here would
        # misclassify HP slot id ``HP_OFFSET`` as quant and read OOB from
        # the quant buffer.
        is_hp = valid & (slot >= HP_OFFSET)
        is_quant = valid & (
            slot < HP_OFFSET
        )  # == valid & ~is_hp; explicit to avoid ~bool dtype quirks

        hp_inc = is_hp.to(tl.int32)
        quant_inc = is_quant.to(tl.int32)

        # tl.cumsum gives an inclusive prefix; subtract the lane value to get
        # the exclusive prefix (= rank of this lane among HP/quant entries
        # within this block).
        hp_rank = tl.cumsum(hp_inc, axis=0) - hp_inc
        quant_rank = tl.cumsum(quant_inc, axis=0) - quant_inc

        tl.store(
            hp_kv_indices_ptr + hp_base + (hp_running + hp_rank).to(tl.int64),
            slot - HP_OFFSET,
            mask=is_hp,
        )
        tl.store(
            quant_kv_indices_ptr
            + quant_base
            + (quant_running + quant_rank).to(tl.int64),
            slot,
            mask=is_quant,
        )

        hp_running += tl.sum(hp_inc, axis=0)
        quant_running += tl.sum(quant_inc, axis=0)


def logit_capping_mod(logit_capping_method, logit_cap):
    # positive logit_cap -> tanh cap
    if logit_capping_method == "tanh":
        return logit_cap
    else:
        raise ValueError()


@dataclass
class ForwardMetadata:
    attn_logits: torch.Tensor
    attn_lse: torch.Tensor
    max_extend_len: int
    num_kv_splits: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor
    custom_mask: torch.Tensor
    mask_indptr: torch.Tensor
    # Sliding window
    window_kv_indptr: torch.Tensor
    window_kv_indices: torch.Tensor
    window_num_kv_splits: torch.Tensor
    window_kv_offsets: torch.Tensor
    # Separate attn_logits for SWA layers when v_head_dim differs
    swa_attn_logits: Optional[torch.Tensor] = None
    # full->SWA translated out_cache_loc (SWA KV-store write target)
    swa_out_cache_loc: Optional[torch.Tensor] = None
    # PHYSICAL full-attn write target for the unified pool (eager: translated tensor;
    # cuda-graph: capture-stable buffer view). None for non-unified pools.
    out_cache_loc_full_physical: Optional[torch.Tensor] = None
    # Per-tier indptr/indices for the unified single-launch mixed int2 path.
    mixed_hp_kv_indptr: Optional[torch.Tensor] = None
    mixed_hp_kv_indices: Optional[torch.Tensor] = None
    mixed_quant_kv_indptr: Optional[torch.Tensor] = None
    mixed_quant_kv_indices: Optional[torch.Tensor] = None
    # Single combined stage-1 scratch: HP splits in the first hp_max slots,
    # quant splits in the next quant_max slots. Stage-2 reduces both in one
    # launch.
    mixed_attn_logits: Optional[torch.Tensor] = None
    mixed_attn_lse: Optional[torch.Tensor] = None
    # Per-tier split counts populated by get_num_kv_splits_triton.
    mixed_hp_num_kv_splits: Optional[torch.Tensor] = None
    mixed_quant_num_kv_splits: Optional[torch.Tensor] = None


class TritonAttnBackend(AttentionBackend):
    # CUDA-graph replay rebuilds metadata from preallocated kv_indptr/kv_indices
    # buffers; it never reads seq_lens_cpu / seq_lens_sum.
    needs_cpu_seq_lens: bool = False

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
    ):
        # Lazy import to avoid the initialization of cuda context
        from sglang.kernels.ops.attention.decode_attention import (
            decode_attention_fwd,
            decode_attention_fwd_int2_unified,
            decode_attention_fwd_quantized,
        )
        from sglang.kernels.ops.attention.extend_attention import (
            build_unified_kv_indices,
            extend_attention_fwd,
            extend_attention_fwd_unified,
        )
        from sglang.kernels.ops.attention.verify_splitkv import (
            verify_splitkv_fwd,
        )

        super().__init__()

        self.decode_attention_fwd = torch.compiler.disable(decode_attention_fwd)
        self.decode_attention_fwd_quantized = torch.compiler.disable(
            decode_attention_fwd_quantized
        )
        self.decode_attention_fwd_int2_unified = torch.compiler.disable(
            decode_attention_fwd_int2_unified
        )
        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)
        self.extend_attention_fwd_unified = torch.compiler.disable(
            extend_attention_fwd_unified
        )
        self.build_unified_kv_indices = torch.compiler.disable(build_unified_kv_indices)
        # Split-KV EAGLE-verify kernel; enabled below once topk is known (valid only at topk == 1).
        self.verify_splitkv_fwd = torch.compiler.disable(verify_splitkv_fwd)

        # Parse args
        self.skip_prefill = skip_prefill
        max_bs = model_runner.req_to_token_pool.size
        self.sliding_window_size = model_runner.sliding_window_size
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
        self.use_sliding_window_kv_pool = isinstance(self.token_to_kv_pool, SWAKVPool)
        # Lets the Triton wrappers specialize on PAGE_SIZE; page_size=1 is
        # byte-identical to the slot-based envelope.
        self.page_size = getattr(model_runner, "page_size", 1) or 1
        # Unified pool v2p hook (None = no-op): req_to_token holds VIRTUAL ids but
        # kernels need PHYSICAL. Applied eagerly so the captured graph has no translate.
        self._translate_kv_loc = getattr(
            self.token_to_kv_pool_allocator, "translate_kv_loc", None
        )
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        # Split-KV verify is bit-equivalent only for a pure-causal chain (topk==1)
        # and is gfx95-only; else fall back to extend_attention_fwd.
        self.use_verify_splitkv = (
            is_gfx95_supported()
            and envs.SGLANG_ENABLE_SPLITKV_VERIFY.get()
            and self.topk == 1
        )
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        self.dcp_size = get_parallel().attn_dcp_size
        self.dcp_rank = get_parallel().attn_dcp_rank
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        ) * self.dcp_size
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size
        )
        # The decode kernel's "// Lv" stride trick requires attn_logits.shape[-1]
        # to exactly match the layer's v_head_dim, so hybrid SWA models with
        # differing SWA/full v_head_dim need a second buffer for SWA layers.
        full_v_head_dim = model_runner.model_config.v_head_dim
        swa_v_head_dim = model_runner.model_config.swa_v_head_dim
        if self.sliding_window_size is not None and swa_v_head_dim != full_v_head_dim:
            self.v_head_dim = full_v_head_dim
            self.swa_v_head_dim = swa_v_head_dim
        elif (
            hybrid_gdn_config(model_runner.model_config) is not None
            or kimi_linear_config(model_runner.model_config) is not None
            or linear_attn_model_spec(model_runner.model_config) is not None
        ):
            # For hybrid linear models, layer_id = 0 may not be full attention
            self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
            self.swa_v_head_dim = None
        else:
            self.v_head_dim = getattr(
                model_runner.token_to_kv_pool,
                "v_head_dim",
                model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1],
            )
            self.swa_v_head_dim = None
        self.max_context_len = model_runner.model_config.context_len
        self.enable_mixed_kv = (
            getattr(model_runner.token_to_kv_pool, "mixed_kv_enabled", None) is not None
            and model_runner.token_to_kv_pool.mixed_kv_enabled()
            and not self.use_mla
            and self.sliding_window_size is None
            and self.swa_v_head_dim is None
        )
        self.mixed_hp_prefix_tokens = (
            model_runner.token_to_kv_pool.hp_prefix_tokens
            if self.enable_mixed_kv
            else 0
        )
        self.mixed_hp_recent_tokens = (
            model_runner.token_to_kv_pool.hp_recent_tokens
            if self.enable_mixed_kv
            else 0
        )
        self.mixed_hp_global_offset = (
            model_runner.token_to_kv_pool.hp_global_offset
            if self.enable_mixed_kv
            else 0
        )
        # Mixed-KV decode uses a fixed HP split count because the HP window is
        # bounded by ``hp_prefix + hp_recent + flush_interval - 1`` tokens.
        # ``SGLANG_MIXED_KV_HP_MAX_SPLITS`` is therefore the direct per-request
        # HP cap for the unified int2 decode path.
        self.max_hp_kv_splits = (
            envs.SGLANG_MIXED_KV_HP_MAX_SPLITS.get() if self.enable_mixed_kv else 0
        )
        # Output dtype for per-tier intermediate buffers in the mixed-KV path.
        self.model_dtype = model_runner.dtype
        self.device = model_runner.device
        self.device_core_count = get_device_core_count(model_runner.gpu_id)
        self.static_kv_splits = get_bool_env_var(
            "SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS", "false"
        )
        self.max_kv_splits = model_runner.server_args.triton_attention_num_kv_splits
        if self.use_mla and not _is_xpu:
            self.max_kv_splits = _mla_decode_kv_splits_cap(
                self.max_kv_splits,
                self.device_core_count,
                self.max_context_len,
            )
            if _is_gfx942:
                # gfx942's 304 CUs round up to 512 splits, doubling the persistent
                # fp32 attn_logits buffer to ~4 GiB on Kimi-K2.6 and faulting in
                # ROCm graph replay; pin to 256 to match validated gfx950 behavior.
                self.max_kv_splits = min(self.max_kv_splits, 256)
        if _is_cuda:
            self.use_pdl = is_arch_support_pdl()
        else:
            self.use_pdl = False

        self.allow_bidirectional_attention_in_extend = (
            cuda_graph_fully_disabled()
            and model_runner.server_args.chunked_prefill_size == -1
        )

        self.enable_deterministic = (
            model_runner.server_args.enable_deterministic_inference
        )

        if self.enable_deterministic:
            # Fixed split tile size for batch invariance
            self.split_tile_size = get_int_env_var(
                "SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE", 256
            )
            self.static_kv_splits = False
        else:
            self.split_tile_size = (
                model_runner.server_args.triton_attention_split_tile_size
            )

        if self.split_tile_size is not None:
            self.max_kv_splits = (
                self.max_context_len + self.split_tile_size - 1
            ) // self.split_tile_size

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        # TODO(Jianan Ji): verify behavior when kv_indptr_buf is provided and sliding window is enabled
        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        # Sliding window may need a second buffer for interleaved attention types
        self.window_kv_indptr = None
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if kv_indptr_buf is None:
                self.window_kv_indptr = torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
            else:
                self.window_kv_indptr = torch.zeros_like(kv_indptr_buf)

        if not self.skip_prefill:
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device
            )

            self.mask_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device
            )

        self.forward_metadata: ForwardMetadata = None

        self.cuda_graph_custom_mask = None
        # Tree-mask scratch is fetched from the target backend only.
        self.is_draft_runner = model_runner.is_draft_worker

    def get_num_kv_splits(
        self,
        num_kv_splits: torch.Tensor,
        seq_lens: torch.Tensor,
        max_kv_splits: Optional[int] = None,
    ):
        """Fill ``num_kv_splits`` with a per-sequence split count.

        ``max_kv_splits`` overrides the per-call upper bound (defaults to
        ``self.max_kv_splits``). The mixed-KV path uses the override to cap
        the HP-side split count independently of the quant/primary side.
        """
        if max_kv_splits is None:
            max_kv_splits = self.max_kv_splits
        num_token, num_seq = num_kv_splits.shape[0], seq_lens.shape[0]
        # NOTE(alcanderian): Considering speculative_decodeing,
        # num_kv_splits.shape[0] will be topk * real_num_token.
        # And the real_num_token is num_seq in decoding phase.
        num_group = num_token // num_seq

        assert (
            num_group * num_seq == num_token
        ), f"num_seq({num_seq}), num_token({num_token}), something goes wrong!"

        if (
            self.static_kv_splits or self.device_core_count <= 0
        ) and not self.enable_deterministic:
            num_kv_splits.fill_(max_kv_splits)
            return

        if self.split_tile_size is not None and self.enable_deterministic:
            if num_group > 1:
                expanded_seq_lens = seq_lens.repeat_interleave(num_group)
            else:
                expanded_seq_lens = seq_lens

            num_kv_splits[:] = torch.clamp(
                (expanded_seq_lens + self.split_tile_size - 1) // self.split_tile_size,
                max=max_kv_splits,
            )
            return

        if num_seq < 256:
            SCHEDULE_SEQ = 256
        else:
            SCHEDULE_SEQ = triton.next_power_of_2(num_seq)

        get_num_kv_splits_triton[(1,)](
            num_kv_splits,
            seq_lens,
            num_seq,
            num_group,
            self.num_head,
            self.num_kv_head,
            max_kv_splits,
            self.device_core_count,
            MAX_NUM_SEQ=SCHEDULE_SEQ,
        )

    def _build_mixed_kv_indices(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        hp_kv_indptr: torch.Tensor,
        hp_kv_indices: torch.Tensor,
        quant_kv_indptr: torch.Tensor,
        quant_kv_indices: torch.Tensor,
        bs: int,
    ):
        """Classify each token's slot id as HP vs quant and scatter into the
        caller-provided per-tier index buffers.

        Sync-free on the decode hot path. Previously this routine ran a
        ``for i in range(bs)`` Python loop with ``rows[hp_mask[i]]``
        masked-selects whose output shape is data-dependent -- each
        masked-select forces a cudaStreamSynchronize so PyTorch can learn the
        size. That was the single biggest CPU-critical-path blocker in
        mixed-KV decode after the flush pipeline was fused.

        The replacement:
          * ``hp_kv_indptr`` is built from a Triton HP-length counting kernel
            that streams ``req_to_token`` through the req-pool indirection --
            no dense gather/mask materialization, no sync.
          * ``quant_kv_indptr`` is derived from the full sequence lengths minus
            the HP prefix sum, so there is no separate quant-length pass.
          * The per-(req, pos) scatter into ``hp_kv_indices`` /
            ``quant_kv_indices`` happens inside a single triton kernel
            (``_scatter_mixed_kv_indices_kernel``) that walks each request's
            ``[0, seq_len)`` range in ``BLOCK_SIZE`` chunks and uses
            ``tl.cumsum`` for within-block ranks. No Python bs-loop, no
            masked-select, no sync.
        """
        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs].to(torch.int64)
        # Cast seq_lens to int32 once; both mixed-KV Triton kernels want
        # int32. Keeps the conversion off the hot path's per-step alloc trail.
        seq_lens_i32 = seq_lens.to(torch.int32)
        hp_lens = torch.empty_like(seq_lens_i32)
        # Count directly from ``req_to_token`` so the hot path no longer
        # materializes a dense gathered ``rows`` tensor or boolean masks.
        _count_mixed_hp_lens_kernel[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens_i32,
            hp_lens,
            self.req_to_token.stride(0),
            HP_OFFSET=int(self.mixed_hp_global_offset),
            BLOCK_SIZE=512,
            num_warps=2,
            num_stages=1,
        )

        # indptr = exclusive prefix sum of per-req lengths. ``cumsum`` + slice
        # assignment are shape-static so no D2H read is forced. The leading
        # ``[0]`` element stays at zero from the buffer's ``torch.zeros``
        # allocation; assigning a Python scalar there would force a sync H2D
        # copy that blocks the CPU on prior decode work, recreating the
        # ~1.5 ms inter-step bubble.
        hp_kv_indptr[1 : bs + 1] = torch.cumsum(hp_lens, dim=0)
        quant_kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens_i32, dim=0)
        quant_kv_indptr[1 : bs + 1] -= hp_kv_indptr[1 : bs + 1]

        # Single triton launch scatters the tier-classified slot ids directly
        # into the pre-sized destination buffers. BLOCK_SIZE here is the
        # per-request chunk size; picking 512 matches
        # ``create_flashinfer_kv_indices_triton`` and balances occupancy
        # against the ``tl.cumsum`` reduction depth.
        _scatter_mixed_kv_indices_kernel[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens_i32,
            hp_kv_indptr,
            quant_kv_indptr,
            hp_kv_indices,
            quant_kv_indices,
            self.req_to_token.stride(0),
            HP_OFFSET=int(self.mixed_hp_global_offset),
            BLOCK_SIZE=512,
            num_warps=2,
            num_stages=1,
        )

    def _forward_extend_quantized_dense(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        causal: bool,
        pre_rotated_q: Optional[torch.Tensor] = None,
        pre_rotated_k: Optional[torch.Tensor] = None,
        pre_rotated_v: Optional[torch.Tensor] = None,
        need_v_inverse_override: Optional[bool] = None,
    ):
        kv_pool = self.token_to_kv_pool
        q3 = (
            pre_rotated_q
            if pre_rotated_q is not None
            else q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        )
        k3 = pre_rotated_k if pre_rotated_k is not None else k.contiguous()
        v3 = pre_rotated_v if pre_rotated_v is not None else v.contiguous()
        if need_v_inverse_override is None:
            q3, k3, v3, need_v_inverse = prepare_quantized_extend_qkv(
                kv_pool,
                layer,
                q3,
                k3,
                v3,
                q_already_hadamard_transformed=pre_rotated_q is not None,
                kv_already_hadamard_transformed=(
                    pre_rotated_k is not None and pre_rotated_v is not None
                ),
            )
        else:
            need_v_inverse = need_v_inverse_override

        prefix_k, prefix_v = dequantize_prefix_kv(
            kv_pool,
            layer.layer_id,
            self.forward_metadata.kv_indices,
            q3.dtype,
        )

        unified_k_parts = []
        unified_v_parts = []
        unified_k_lens = []
        prefix_indptr = self.forward_metadata.kv_indptr
        extend_start_loc = forward_batch.extend_start_loc
        for i, extend_len in enumerate(forward_batch.extend_seq_lens_cpu):
            prefix_start = int(prefix_indptr[i].item())
            prefix_end = int(prefix_indptr[i + 1].item())
            extend_start = int(extend_start_loc[i].item())
            extend_end = extend_start + int(extend_len)
            req_k = torch.cat(
                [prefix_k[prefix_start:prefix_end], k3[extend_start:extend_end]], dim=0
            )
            req_v = torch.cat(
                [prefix_v[prefix_start:prefix_end], v3[extend_start:extend_end]], dim=0
            )
            unified_k_parts.append(req_k)
            unified_v_parts.append(req_v)
            unified_k_lens.append(req_k.shape[0])

        unified_k = torch.cat(unified_k_parts, dim=0) if unified_k_parts else k3[:0]
        unified_v = torch.cat(unified_v_parts, dim=0) if unified_v_parts else v3[:0]
        cu_seqlens_q = self.forward_metadata.qo_indptr.to(torch.int32)
        cu_seqlens_k = torch.empty(
            (len(unified_k_lens) + 1,), dtype=torch.int32, device=self.device
        )
        cu_seqlens_k[0] = 0
        cu_seqlens_k[1:] = torch.cumsum(
            torch.tensor(unified_k_lens, dtype=torch.int32, device=self.device), dim=0
        )

        result = flash_attn_varlen_func(
            q=q3,
            k=unified_k,
            v=unified_v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max(forward_batch.extend_seq_lens_cpu),
            max_seqlen_k=max(unified_k_lens) if unified_k_lens else 0,
            softmax_scale=layer.scaling,
            causal=causal,
            window_size=(-1, -1),
            softcap=logit_capping_mod(layer.logit_capping_method, layer.logit_cap),
        )
        result = apply_inverse_v_rotation(result, kv_pool, layer, need_v_inverse)
        o.copy_(result.view_as(o))
        return o

    def _dcp_lens(self, lens: torch.Tensor, start: Optional[torch.Tensor] = None):
        return get_dcp_lens(lens, self.dcp_size, self.dcp_rank, start)

    def _dcp_kv_indices(
        self,
        req_pool_indices: torch.Tensor,
        lens: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: Optional[torch.Tensor] = None,
        kv_start_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Build per-DCP-rank sharded KV indptr/indices. eager passes kv_indices=None
        # (fresh tensor); cuda-graph passes an address-stable buffer to fill in place.
        dcp_lens = self._dcp_lens(lens, kv_start_idx)
        kv_indptr[1 : len(req_pool_indices) + 1] = torch.cumsum(dcp_lens, dim=0)
        kv_indptr = kv_indptr[: len(req_pool_indices) + 1]
        if kv_indices is None:
            kv_indices = torch.empty(
                int(dcp_lens.sum().item()), dtype=torch.int64, device=self.device
            )
        create_triton_kv_indices_for_dcp_triton[(len(req_pool_indices),)](
            self.req_to_token,
            req_pool_indices,
            dcp_lens,
            kv_indptr,
            kv_start_idx,
            kv_indices,
            self.req_to_token.stride(0),
            self.dcp_size,
            self.dcp_rank,
        )
        return kv_indptr, kv_indices, dcp_lens

    def _fill_kv_indptr_and_indices(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        kv_indices: torch.Tensor,
    ) -> torch.Tensor:
        kv_indptr = self.kv_indptr[: bs + 1]
        kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            kv_indptr,
            None,
            kv_indices,
            self.req_to_token.stride(0),
        )
        return kv_indptr

    def _update_decode_kv_buffers(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ):
        """Fill KV (and SWA) cuda-graph buffers for decode/idle mode.

        Returns ``(kv_indptr, window_kv_indptr, window_kv_lens, num_kv_splits_lens)``
        where ``window_kv_lens`` is ``None`` when sliding-window is disabled and
        ``num_kv_splits_lens`` is the per-request length used to size kv splits
        (per-DCP-rank length clamped to >=1 when DCP is enabled, full seq_lens
        otherwise).
        """
        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs]
        if self.dcp_size > 1:
            # DCP: per-rank sharded; write into the same cuda-graph buffers
            # _build_cuda_graph_forward_metadata reads back.
            _, _, dcp_seq_lens = self._dcp_kv_indices(
                req_pool_indices,
                seq_lens,
                self.kv_indptr,
                self.cuda_graph_kv_indices,
                None,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            num_kv_splits_lens = dcp_seq_lens.clamp_min(1)
        else:
            kv_indptr = self._fill_kv_indptr_and_indices(
                bs, seq_lens, req_pool_indices, self.cuda_graph_kv_indices
            )
            # Unified pool: VIRTUAL ids written here are translated to PHYSICAL in
            # init_forward_metadata_out_graph (replay-prep) so the captured graph
            # carries zero translate nodes.
            num_kv_splits_lens = seq_lens
        window_kv_indptr = self.window_kv_indptr
        window_kv_lens = None
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            # Unified pool: leave the window VIRTUAL too (translated alongside the
            # full kv_indices later); baseline SWA keeps the eager window translate.
            window_kv_indptr, _, window_kv_lens, _ = update_sliding_window_buffer(
                self.window_kv_indptr,
                self.req_to_token,
                self.sliding_window_size,
                seq_lens,
                req_pool_indices,
                bs,
                token_to_kv_pool=self.token_to_kv_pool,
                window_kv_indices=self.cuda_graph_window_kv_indices,
                skip_full_to_swa_translation=(self._translate_kv_loc is not None),
            )
        return kv_indptr, window_kv_indptr, window_kv_lens, num_kv_splits_lens

    def _update_target_verify_buffers(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        spec_info,
    ):
        """Fill all cuda-graph buffers for target_verify mode."""
        qo_indptr = self.qo_indptr[: bs + 1]
        qo_indptr[: bs + 1] = torch.arange(
            0,
            (1 + bs) * self.num_draft_tokens,
            step=self.num_draft_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        kv_indptr = self._fill_kv_indptr_and_indices(
            bs, seq_lens, req_pool_indices, self.cuda_graph_kv_indices
        )
        window_kv_indptr = self.window_kv_indptr
        window_kv_indices = None
        window_num_kv_splits = None
        window_kv_offsets = None
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            window_kv_indices = self.cuda_graph_window_kv_indices
            window_num_kv_splits = self.cuda_graph_window_num_kv_splits
            window_kv_offsets = self.cuda_graph_window_kv_offsets
            window_kv_indptr, window_kv_indices, _, window_kv_offsets[:bs] = (
                update_sliding_window_buffer(
                    self.window_kv_indptr,
                    self.req_to_token,
                    self.sliding_window_size,
                    seq_lens[:bs],
                    req_pool_indices,
                    bs,
                    token_to_kv_pool=self.token_to_kv_pool,
                    window_kv_indices=window_kv_indices,
                )
            )
        custom_mask = self.cuda_graph_custom_mask
        if (
            spec_info is not None
            and getattr(spec_info, "custom_mask", None) is not None
        ):
            custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
        else:
            custom_mask = None
        seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
        mask_indptr = self.mask_indptr[: bs + 1]
        mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
        return (
            qo_indptr,
            kv_indptr,
            custom_mask,
            mask_indptr,
            window_kv_indptr,
            window_kv_indices,
            window_num_kv_splits,
            window_kv_offsets,
        )

    def _update_draft_extend_buffers(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        """Fill QO + KV cuda-graph buffers for draft_extend mode."""
        seq_lens = seq_lens[:bs]
        # V2 draft-extend fills num_draft_tokens per req; num_steps+1 only equals
        # that when topk == 1.
        num_tokens_per_req = (
            self.num_draft_tokens
            if forward_mode.is_draft_extend_v2()
            else self.speculative_num_steps + 1
        )
        qo_indptr = self.qo_indptr[: bs + 1]
        qo_indptr[: bs + 1] = torch.arange(
            0,
            bs * num_tokens_per_req + 1,
            step=num_tokens_per_req,
            dtype=torch.int32,
            device=self.device,
        )
        # DRAFT_EXTEND_V2: kv_indptr/kv_indices cover only the prefix (extend K/V go
        # separately). Capture warmup lacks extend_seq_lens_tensor -> fall back to
        # zeros; clamp at 0 so padded rows (seq_lens==fill 1) don't go negative.
        if (
            spec_info is not None
            and getattr(spec_info, "extend_seq_lens_tensor", None) is not None
        ):
            extend_seq_lens = spec_info.extend_seq_lens_tensor[:bs].to(torch.int32)
        else:
            extend_seq_lens = torch.zeros(bs, dtype=torch.int32, device=seq_lens.device)
        kv_lens = torch.clamp(seq_lens - extend_seq_lens, min=0).to(torch.int32)
        kv_indptr = self._fill_kv_indptr_and_indices(
            bs, kv_lens, req_pool_indices, self.cuda_graph_kv_indices
        )
        return qo_indptr, kv_indptr, num_tokens_per_req

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        bs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        forward_mode = forward_batch.forward_mode
        spec_info = forward_batch.spec_info

        if in_capture:
            assert forward_batch.encoder_lens is None, "Not supported"
            # Multi-step spec decode: kv buffers come from spec_info, not the
            # cuda-graph pool, so replay is not involved.
            if forward_mode.is_decode_or_idle() and spec_info is not None:
                self.forward_metadata = ForwardMetadata(
                    attn_logits=self.cuda_graph_attn_logits,
                    attn_lse=self.cuda_graph_attn_lse,
                    max_extend_len=None,
                    num_kv_splits=self.cuda_graph_num_kv_splits,
                    kv_indptr=spec_info.kv_indptr,
                    kv_indices=spec_info.kv_indices,
                    qo_indptr=None,
                    custom_mask=None,
                    mask_indptr=None,
                    window_kv_indptr=self.window_kv_indptr,
                    window_kv_indices=None,
                    window_num_kv_splits=None,
                    window_kv_offsets=None,
                    swa_attn_logits=self.cuda_graph_swa_attn_logits,
                )
                return

            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                forward_mode=forward_mode,
                spec_info=spec_info,
            )
            out_cache_loc_full_physical = self._translate_cuda_graph_shared_pool_locs(
                forward_batch, bs
            )
            swa_out_cache_loc = self._fill_cuda_graph_swa_out_cache_loc(forward_batch)
            self.forward_metadata = self._build_cuda_graph_forward_metadata(
                bs,
                forward_mode,
                spec_info,
                swa_out_cache_loc,
                out_cache_loc_full_physical,
            )
        else:
            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                forward_mode=forward_mode,
                spec_info=spec_info,
            )
            # Metadata view is reused from capture; just refill the buffers.
            self._translate_cuda_graph_shared_pool_locs(forward_batch, bs)
            self._fill_cuda_graph_swa_out_cache_loc(forward_batch)

    def _fill_cuda_graph_swa_out_cache_loc(
        self, forward_batch: ForwardBatch
    ) -> Optional[torch.Tensor]:
        """Refill the SWA write-target buffer from live out_cache_loc, returning the
        [:n] view (None for non-SWA / multi-step draft) so the captured store reads
        fresh slots on replay."""
        if not self.use_sliding_window_kv_pool:
            return None
        out_cache_loc = forward_batch.out_cache_loc
        if (
            out_cache_loc is None
            or out_cache_loc.shape[0] > self.cuda_graph_swa_out_cache_loc.shape[0]
        ):
            return None
        n = out_cache_loc.shape[0]
        self.cuda_graph_swa_out_cache_loc[n:].zero_()
        self.cuda_graph_swa_out_cache_loc[:n].copy_(
            self.token_to_kv_pool.translate_loc_from_full_to_swa(out_cache_loc)
        )
        return self.cuda_graph_swa_out_cache_loc[:n]

    def _translate_cuda_graph_shared_pool_locs(
        self, forward_batch: ForwardBatch, bs: int
    ) -> Optional[torch.Tensor]:
        """Unified pool: eager v2p translate of the cuda-graph read+write LOC buffers,
        run BEFORE graph.replay() reading the live post-compaction v2p, so the
        captured graph carries zero translate nodes. No-op for non-unified pools.

        Read buffers (full kv_indices, SWA window) are translated IN PLACE; the
        full-attn WRITE loc is RETURNED as the [:n] view of the backend-owned
        out_cache_loc_full_physical buffer. Eager .item() bounds are fine here
        (out-of-graph), so no in-graph translate variant is needed.
        """
        if self._translate_kv_loc is None:
            return None
        # seq_lens_sum is the reliable "mirror present" signal: it is
        # None-preserving into the replay view, unlike seq_lens_cpu (always a
        # non-None but stale slice for gpu_only batches). None -> fall back to a
        # per-step D2H `.item()` on the indptr.
        have_cpu_mirror = forward_batch.seq_lens_sum is not None
        # Full-attention read path. kv_indptr[bs] == seq_lens_sum.
        n_kv = (
            forward_batch.seq_lens_sum
            if have_cpu_mirror
            else int(self.kv_indptr[bs].item())
        )
        if n_kv > 0:
            self.cuda_graph_kv_indices[:n_kv] = self._translate_kv_loc(
                self.cuda_graph_kv_indices[:n_kv]
            )
        # SWA window read path. window_kv_indptr[bs] == sum(min(seq_len, window)).
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if have_cpu_mirror:
                n_win = int(
                    forward_batch.seq_lens_cpu[:bs]
                    .clamp(max=self.sliding_window_size)
                    .sum()
                )
            else:
                n_win = int(self.window_kv_indptr[bs].item())
            if n_win > 0:
                self.cuda_graph_window_kv_indices[:n_win] = (
                    self.token_to_kv_pool.translate_loc_from_full_to_swa(
                        self.cuda_graph_window_kv_indices[:n_win]
                    )
                )
        # Full-attention write path: translate out_cache_loc -> physical into the
        # capture-stable buffer and RETURN the [:n] view.
        out_cache_loc = forward_batch.out_cache_loc
        n = out_cache_loc.shape[0]
        # Zero the padded tail first: a smaller replay batch leaves [n:] holding
        # stale ids that the captured store would write; send them to slot 0 (sink).
        self.cuda_graph_out_cache_loc_full_physical[n:].zero_()
        self.cuda_graph_out_cache_loc_full_physical[:n].copy_(
            self._translate_kv_loc(out_cache_loc)
        )
        return self.cuda_graph_out_cache_loc_full_physical[:n]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for triton attention backend."""

        bs = forward_batch.batch_size
        window_kv_indptr = self.window_kv_indptr
        window_kv_indices = None
        window_num_kv_splits = None
        window_kv_offsets = None
        swa_attn_logits = None
        spec_info = forward_batch.spec_info
        mixed_hp_kv_indptr = None
        mixed_hp_kv_indices = None
        mixed_quant_kv_indptr = None
        mixed_quant_kv_indices = None
        mixed_attn_logits = None
        mixed_attn_lse = None
        mixed_hp_num_kv_splits = None
        mixed_quant_num_kv_splits = None

        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None or spec_info.kv_indptr is None:
                # kv_indptr is None for draft-extend's idle batch; build from seq_lens.
                if self.dcp_size > 1:
                    # DCP: per-rank sharded KV indices, else each rank reads the
                    # whole KV instead of its owner shard.
                    kv_indptr, kv_indices, _ = self._dcp_kv_indices(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        self.kv_indptr,
                    )
                else:
                    # gpu_only: seq_lens_sum may be None; over-allocate is safe (ragged write).
                    seq_lens_sum = forward_batch.seq_lens_sum
                    if seq_lens_sum is None:
                        seq_lens_sum = bs * self.max_context_len
                    kv_indices = torch.empty(
                        seq_lens_sum, dtype=torch.int64, device=self.device
                    )
                    kv_indptr = self._fill_kv_indptr_and_indices(
                        bs,
                        forward_batch.seq_lens,
                        forward_batch.req_pool_indices,
                        kv_indices,
                    )
                    if self._translate_kv_loc is not None:
                        kv_indices = self._translate_kv_loc(kv_indices)
                if self.enable_mixed_kv:
                    mixed_hp_kv_indptr = torch.zeros(
                        (bs + 1,), dtype=torch.int32, device=self.device
                    )
                    mixed_quant_kv_indptr = torch.zeros(
                        (bs + 1,), dtype=torch.int32, device=self.device
                    )
                    mixed_hp_kv_indices = torch.empty(
                        forward_batch.seq_lens_sum,
                        dtype=torch.int64,
                        device=self.device,
                    )
                    mixed_quant_kv_indices = torch.empty(
                        forward_batch.seq_lens_sum,
                        dtype=torch.int64,
                        device=self.device,
                    )
                    total_splits = self.max_kv_splits + self.max_hp_kv_splits
                    # Single combined stage-1 scratch. LSE is pre-filled with
                    # -inf so the tier-agnostic stage-2 can skip unused splits.
                    mixed_attn_logits = torch.empty(
                        (bs, self.num_head, total_splits, self.v_head_dim),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    mixed_attn_lse = torch.full(
                        (bs, self.num_head, total_splits),
                        float("-inf"),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    mixed_hp_num_kv_splits = torch.full(
                        (bs,),
                        self.max_hp_kv_splits,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    mixed_quant_num_kv_splits = torch.empty(
                        (bs,), dtype=torch.int32, device=self.device
                    )
                    self._build_mixed_kv_indices(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        mixed_hp_kv_indptr,
                        mixed_hp_kv_indices,
                        mixed_quant_kv_indptr,
                        mixed_quant_kv_indices,
                        bs,
                    )
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_kv_indptr, window_kv_indices, window_kv_lens, _ = (
                        update_sliding_window_buffer(
                            self.window_kv_indptr,
                            self.req_to_token,
                            self.sliding_window_size,
                            forward_batch.seq_lens,
                            forward_batch.req_pool_indices,
                            bs,
                            self.device,
                            self.token_to_kv_pool,
                        )
                    )
                    window_num_kv_splits = torch.empty(
                        (bs,), dtype=torch.int32, device=self.device
                    )
                    self.get_num_kv_splits(window_num_kv_splits, window_kv_lens)
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                bs = kv_indptr.shape[0] - 1

            attn_logits = torch.empty(
                (bs, self.num_head, self.max_kv_splits, self.v_head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            if self.swa_v_head_dim is not None:
                swa_attn_logits = torch.empty(
                    (bs, self.num_head, self.max_kv_splits, self.swa_v_head_dim),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                swa_attn_logits = None
            attn_lse = torch.empty(
                (bs, self.num_head, self.max_kv_splits),
                dtype=torch.float32,
                device=self.device,
            )
            num_kv_splits = torch.empty((bs,), dtype=torch.int32, device=self.device)
            if self.enable_mixed_kv and mixed_quant_num_kv_splits is not None:
                # HP uses the fixed cap above; only the quant tier is
                # right-sized, and it uses the full sequence length as a cheap
                # planning proxy instead of per-tier mixed-KV counts.
                self.get_num_kv_splits(
                    mixed_quant_num_kv_splits, forward_batch.seq_lens
                )
            else:
                self.get_num_kv_splits(
                    num_kv_splits,
                    (
                        self._dcp_lens(forward_batch.seq_lens).clamp_min(1)
                        if self.dcp_size > 1
                        else forward_batch.seq_lens
                    ),
                )

            qo_indptr = None
            custom_mask = None
            mask_indptr = None
            max_extend_len = None
        elif forward_batch.forward_mode.is_target_verify():
            bs = len(forward_batch.req_pool_indices)
            qo_indptr = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            # gpu_only: seq_lens_sum may be None; over-allocate is safe (ragged write).
            seq_lens_sum = forward_batch.seq_lens_sum
            if seq_lens_sum is None:
                seq_lens_sum = bs * self.max_context_len
            kv_indices = torch.empty(
                seq_lens_sum, dtype=torch.int64, device=self.device
            )
            kv_indptr = self._fill_kv_indptr_and_indices(
                bs,
                forward_batch.seq_lens,
                forward_batch.req_pool_indices,
                kv_indices,
            )

            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                # window_kv_offsets gives the start position in custom mask
                (
                    window_kv_indptr,
                    window_kv_indices,
                    window_kv_lens,
                    window_kv_offsets,
                ) = update_sliding_window_buffer(
                    self.window_kv_indptr,
                    self.req_to_token,
                    self.sliding_window_size,
                    forward_batch.seq_lens,
                    forward_batch.req_pool_indices,
                    bs,
                    self.device,
                    self.token_to_kv_pool,
                )

            custom_mask = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (
                forward_batch.seq_lens + self.num_draft_tokens
            )
            mask_indptr = self.mask_indptr
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
            mask_indptr = mask_indptr[: bs + 1]
            max_extend_len = self.num_draft_tokens
            num_kv_splits = None
            attn_logits = None
            attn_lse = None

        else:
            if self.dcp_size > 1:
                kv_indptr, kv_indices, _ = self._dcp_kv_indices(
                    forward_batch.req_pool_indices,
                    forward_batch.extend_prefix_lens,
                    self.kv_indptr,
                )
            else:
                # gpu_only leaves _cpu unset; over-allocate is safe (ragged write).
                if forward_batch.extend_prefix_lens_cpu is not None:
                    kv_indices_len = sum(forward_batch.extend_prefix_lens_cpu)
                else:
                    kv_indices_len = bs * self.max_context_len
                kv_indices = torch.empty(
                    kv_indices_len,
                    dtype=torch.int64,
                    device=self.device,
                )
                kv_indptr = self._fill_kv_indptr_and_indices(
                    bs,
                    forward_batch.extend_prefix_lens,
                    forward_batch.req_pool_indices,
                    kv_indices,
                )
                if self._translate_kv_loc is not None:
                    kv_indices = self._translate_kv_loc(kv_indices)
            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                (
                    window_kv_indptr,
                    window_kv_indices,
                    window_kv_lens,
                    window_kv_offsets,
                ) = update_sliding_window_buffer(
                    self.window_kv_indptr,
                    self.req_to_token,
                    self.sliding_window_size,
                    forward_batch.extend_prefix_lens,
                    forward_batch.req_pool_indices,
                    bs,
                    self.device,
                    self.token_to_kv_pool,
                )

            qo_indptr = self.qo_indptr
            qo_indptr[1 : bs + 1] = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
            mask_indptr = None
            attn_logits = None
            attn_lse = None
            # Defensive GPU-max fallback when extend_seq_lens_cpu is absent.
            if forward_batch.extend_seq_lens_cpu is not None:
                max_extend_len = max(forward_batch.extend_seq_lens_cpu)
            else:
                max_extend_len = int(forward_batch.extend_seq_lens.max())
            num_kv_splits = None

        swa_out_cache_loc = None
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            swa_out_cache_loc = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                forward_batch.out_cache_loc
            )

        # Unified pool full-attention WRITE loc (virtual out_cache_loc -> physical),
        # carried in the metadata (-> KVWriteLoc.full_loc). None for non-unified pools.
        out_cache_loc_full_physical = None
        if (
            self._translate_kv_loc is not None
            and forward_batch.out_cache_loc is not None
        ):
            out_cache_loc_full_physical = self._translate_kv_loc(
                forward_batch.out_cache_loc
            )

        self.forward_metadata = ForwardMetadata(
            attn_logits,
            attn_lse,
            max_extend_len,
            num_kv_splits,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
            window_kv_indptr,
            window_kv_indices,
            window_num_kv_splits,
            window_kv_offsets,
            swa_attn_logits=swa_attn_logits,
            swa_out_cache_loc=swa_out_cache_loc,
            out_cache_loc_full_physical=out_cache_loc_full_physical,
            mixed_hp_kv_indptr=mixed_hp_kv_indptr,
            mixed_hp_kv_indices=mixed_hp_kv_indices,
            mixed_quant_kv_indptr=mixed_quant_kv_indptr,
            mixed_quant_kv_indices=mixed_quant_kv_indices,
            mixed_attn_logits=mixed_attn_logits,
            mixed_attn_lse=mixed_attn_lse,
            mixed_hp_num_kv_splits=mixed_hp_num_kv_splits,
            mixed_quant_num_kv_splits=mixed_quant_num_kv_splits,
        )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
        cuda_graph_num_kv_splits_buf: Optional[torch.Tensor] = None,
    ):
        self.cuda_graph_attn_logits = torch.zeros(
            (max_num_tokens, self.num_head, self.max_kv_splits, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        if self.swa_v_head_dim is not None:
            self.cuda_graph_swa_attn_logits = torch.zeros(
                (
                    max_num_tokens,
                    self.num_head,
                    self.max_kv_splits,
                    self.swa_v_head_dim,
                ),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.cuda_graph_swa_attn_logits = None
        self.cuda_graph_attn_lse = torch.zeros(
            (max_num_tokens, self.num_head, self.max_kv_splits),
            dtype=torch.float32,
            device=self.device,
        )

        if cuda_graph_num_kv_splits_buf is None:
            self.cuda_graph_num_kv_splits = torch.full(
                (max_num_tokens,),
                self.max_kv_splits,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.cuda_graph_num_kv_splits = cuda_graph_num_kv_splits_buf

        if kv_indices_buf is None:
            self.cuda_graph_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.int64,
                device=self.device,
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        if not self.skip_prefill and not self.is_draft_runner:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device=self.device,
            )

        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if kv_indices_buf is None:
                self.cuda_graph_window_kv_indices = torch.zeros(
                    (max_num_tokens * self.sliding_window_size),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                self.cuda_graph_window_kv_indices = torch.zeros_like(kv_indices_buf)

            self.cuda_graph_window_num_kv_splits = torch.full(
                (max_num_tokens,),
                self.max_kv_splits,
                dtype=torch.int32,
                device=self.device,
            )

            self.cuda_graph_window_kv_offsets = torch.zeros(
                (max_bs,),
                dtype=torch.int32,
                device=self.device,
            )

        if self.use_sliding_window_kv_pool:
            # SWA write-target buffer; refilled at replay from out_cache_loc.
            self.cuda_graph_swa_out_cache_loc = torch.zeros(
                (max_num_tokens,),
                dtype=torch.int64,
                device=self.device,
            )

        if self._translate_kv_loc is not None:
            # Unified pool full-attention write-target buffer, refilled at replay
            # (-> KVWriteLoc.full_loc). Capture-stable, mirrors cuda_graph_swa_out_cache_loc.
            self.cuda_graph_out_cache_loc_full_physical = torch.zeros(
                (max_num_tokens,),
                dtype=torch.int64,
                device=self.device,
            )

        if self.enable_mixed_kv:
            self.cuda_graph_mixed_hp_kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=self.device
            )
            self.cuda_graph_mixed_quant_kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=self.device
            )
            self.cuda_graph_mixed_hp_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.int64,
                device=self.device,
            )
            self.cuda_graph_mixed_quant_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.int64,
                device=self.device,
            )
            total_splits = self.max_kv_splits + self.max_hp_kv_splits
            # Single combined stage-1 scratch. LSE pre-filled to -inf so the
            # tier-agnostic stage-2 skips unused splits.
            self.cuda_graph_mixed_attn_logits = torch.zeros(
                (max_num_tokens, self.num_head, total_splits, self.v_head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            self.cuda_graph_mixed_attn_lse = torch.full(
                (max_num_tokens, self.num_head, total_splits),
                float("-inf"),
                dtype=torch.float32,
                device=self.device,
            )
            self.cuda_graph_mixed_hp_num_kv_splits = torch.full(
                (max_num_tokens,),
                self.max_hp_kv_splits,
                dtype=torch.int32,
                device=self.device,
            )
            self.cuda_graph_mixed_quant_num_kv_splits = torch.zeros(
                (max_num_tokens,), dtype=torch.int32, device=self.device
            )

    def _build_cuda_graph_forward_metadata(
        self,
        bs: int,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        swa_out_cache_loc: Optional[torch.Tensor] = None,
        out_cache_loc_full_physical: Optional[torch.Tensor] = None,
    ) -> ForwardMetadata:
        """Construct ForwardMetadata from the current cuda-graph buffer state.

        Called by capture after the buffer-update helpers have already run
        (either via replay or directly).  All fields reference the same
        self.cuda_graph_* tensors that the captured graph kernels will
        read — the Python object is rebuilt each capture, but the underlying
        GPU memory addresses are stable. ``swa_out_cache_loc`` is the
        pre-allocated SWA write-target buffer view (or None for non-SWA).
        """
        swa = self.sliding_window_size is not None and self.sliding_window_size > 0
        if forward_mode.is_decode_or_idle():
            return ForwardMetadata(
                attn_logits=self.cuda_graph_attn_logits,
                attn_lse=self.cuda_graph_attn_lse,
                max_extend_len=None,
                num_kv_splits=self.cuda_graph_num_kv_splits,
                kv_indptr=self.kv_indptr[: bs + 1],
                kv_indices=self.cuda_graph_kv_indices,
                qo_indptr=None,
                custom_mask=None,
                mask_indptr=None,
                window_kv_indptr=self.window_kv_indptr[: bs + 1] if swa else None,
                window_kv_indices=self.cuda_graph_window_kv_indices if swa else None,
                window_num_kv_splits=(
                    self.cuda_graph_window_num_kv_splits if swa else None
                ),
                window_kv_offsets=None,
                swa_attn_logits=self.cuda_graph_swa_attn_logits,
                swa_out_cache_loc=swa_out_cache_loc,
                out_cache_loc_full_physical=out_cache_loc_full_physical,
                mixed_hp_kv_indptr=(
                    self.cuda_graph_mixed_hp_kv_indptr[: bs + 1]
                    if self.enable_mixed_kv
                    else None
                ),
                mixed_hp_kv_indices=(
                    self.cuda_graph_mixed_hp_kv_indices
                    if self.enable_mixed_kv
                    else None
                ),
                mixed_quant_kv_indptr=(
                    self.cuda_graph_mixed_quant_kv_indptr[: bs + 1]
                    if self.enable_mixed_kv
                    else None
                ),
                mixed_quant_kv_indices=(
                    self.cuda_graph_mixed_quant_kv_indices
                    if self.enable_mixed_kv
                    else None
                ),
                mixed_attn_logits=(
                    self.cuda_graph_mixed_attn_logits if self.enable_mixed_kv else None
                ),
                mixed_attn_lse=(
                    self.cuda_graph_mixed_attn_lse if self.enable_mixed_kv else None
                ),
                mixed_hp_num_kv_splits=(
                    self.cuda_graph_mixed_hp_num_kv_splits
                    if self.enable_mixed_kv
                    else None
                ),
                mixed_quant_num_kv_splits=(
                    self.cuda_graph_mixed_quant_num_kv_splits
                    if self.enable_mixed_kv
                    else None
                ),
            )
        elif forward_mode.is_target_verify():
            custom_mask = (
                self.cuda_graph_custom_mask
                if spec_info is not None
                and getattr(spec_info, "custom_mask", None) is not None
                else None
            )
            return ForwardMetadata(
                attn_logits=None,
                attn_lse=None,
                max_extend_len=self.num_draft_tokens,
                num_kv_splits=None,
                kv_indptr=self.kv_indptr[: bs + 1],
                kv_indices=self.cuda_graph_kv_indices,
                qo_indptr=self.qo_indptr[: bs + 1],
                custom_mask=custom_mask,
                mask_indptr=self.mask_indptr[: bs + 1],
                window_kv_indptr=self.window_kv_indptr[: bs + 1] if swa else None,
                window_kv_indices=self.cuda_graph_window_kv_indices if swa else None,
                window_num_kv_splits=(
                    self.cuda_graph_window_num_kv_splits if swa else None
                ),
                window_kv_offsets=self.cuda_graph_window_kv_offsets if swa else None,
                swa_out_cache_loc=swa_out_cache_loc,
                out_cache_loc_full_physical=out_cache_loc_full_physical,
            )
        elif forward_mode.is_draft_extend_v2():
            return ForwardMetadata(
                attn_logits=None,
                attn_lse=None,
                # Must match the per-req query count (num_tokens_per_req) used to
                # build qo_indptr above, else the extend kernel grid is too small
                # for topk > 1 (num_draft_tokens > num_steps+1) and drops query
                # blocks.
                max_extend_len=(
                    self.num_draft_tokens
                    if forward_mode.is_draft_extend_v2()
                    else self.speculative_num_steps + 1
                ),
                num_kv_splits=None,
                kv_indptr=self.kv_indptr[: bs + 1],
                kv_indices=self.cuda_graph_kv_indices,
                qo_indptr=self.qo_indptr[: bs + 1],
                custom_mask=None,
                mask_indptr=None,
                window_kv_indptr=self.window_kv_indptr,
                window_kv_indices=None,
                window_num_kv_splits=None,
                window_kv_offsets=None,
                swa_out_cache_loc=swa_out_cache_loc,
                out_cache_loc_full_physical=out_cache_loc_full_physical,
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=} for CUDA Graph.")

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        """Shared capture+replay body for the cuda-graph init path.

        Public entry: :py:meth:`init_forward_metadata_out_graph`.
        """
        # NOTE: encoder_lens expected to be zeros or None
        if forward_mode.is_decode_or_idle():
            assert spec_info is None, "Multi-step cuda graph init is not done here."
            _, _, window_kv_lens, num_kv_splits_lens = self._update_decode_kv_buffers(
                bs, seq_lens, req_pool_indices
            )
            if self.enable_mixed_kv:
                self._build_mixed_kv_indices(
                    req_pool_indices,
                    seq_lens,
                    self.cuda_graph_mixed_hp_kv_indptr,
                    self.cuda_graph_mixed_hp_kv_indices,
                    self.cuda_graph_mixed_quant_kv_indptr,
                    self.cuda_graph_mixed_quant_kv_indices,
                    bs,
                )
                self.cuda_graph_mixed_hp_num_kv_splits[:bs] = self.max_hp_kv_splits
                # Quant tier is right-sized from the full sequence length (cheap
                # planning proxy); the unified attention wrapper re-fills LSE
                # with -inf every call so the shared scratch enters stage-2 in
                # a known state.
                self.get_num_kv_splits(
                    self.cuda_graph_mixed_quant_num_kv_splits[:bs], seq_lens[:bs]
                )
            self.get_num_kv_splits(
                self.cuda_graph_num_kv_splits[:bs], num_kv_splits_lens[:bs]
            )
            if window_kv_lens is not None:
                self.get_num_kv_splits(
                    self.cuda_graph_window_num_kv_splits[:bs], window_kv_lens[:bs]
                )
        elif forward_mode.is_target_verify():
            bs = len(req_pool_indices)
            self._update_target_verify_buffers(
                bs, seq_lens, req_pool_indices, spec_info
            )
        elif forward_mode.is_draft_extend_v2():
            self._update_draft_extend_buffers(
                bs, seq_lens, req_pool_indices, forward_mode, spec_info
            )
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph replay."
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def get_verify_buffers_to_fill_after_draft(self):
        """
        Return buffers for verify attention kernels that needs to be filled after draft.

        Typically, these are tree mask and position buffers.
        """
        return [self.cuda_graph_custom_mask, None]

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        pass

    def _set_kv_buffer(
        self,
        forward_batch: ForwardBatch,
        layer: RadixAttention,
        loc_info,
        k: torch.Tensor,
        v: torch.Tensor,
        k_scale=None,
        v_scale=None,
    ) -> None:
        # DCP writes to the local physical shard (loc = out_cache_loc //
        # dcp_size) through the masked path so each rank only stores the tokens
        # it owns. Non-DCP keeps the original write loc and plain set_kv_buffer.
        if self.dcp_size > 1:
            loc = forward_batch.out_cache_loc // self.dcp_size
            if (
                forward_batch.positions is not None
                and forward_batch.positions.numel() == loc.numel()
            ):
                dcp_kv_mask = forward_batch.positions % self.dcp_size == self.dcp_rank
            else:
                dcp_kv_mask = forward_batch.dcp_kv_mask
            kwargs = {"dcp_kv_mask": dcp_kv_mask}
        else:
            loc = loc_info
            kwargs = {}
        if k_scale is None and v_scale is None:
            self.token_to_kv_pool.set_kv_buffer(layer, loc, k, v, **kwargs)
        else:
            self.token_to_kv_pool.set_kv_buffer(
                layer, loc, k, v, k_scale, v_scale, **kwargs
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
        score_mod=None,
        aux_tensors=None,
    ):
        # TODO: reuse the buffer across layers
        attn_out = getattr(forward_batch, "_attn_output", None)
        if attn_out is not None:
            o = attn_out
        elif layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        kv_pool = self.token_to_kv_pool
        kv_dtype_is_int2 = (
            isinstance(kv_pool, (MHATokenToKVPool, UnifiedInt2HPKVPool))
            and kv_pool.dtype == "int2"
        )
        # Int2 prefill used to rotate K/V once for attention and again when
        # writing the KV cache. Pre-rotate here so both consumers can share
        # the same tensors; the write then passes
        # ``already_hadamard_transformed=True`` so the pool skips its own
        # rotation.
        use_quantized_dense_prefill = (
            kv_dtype_is_int2
            and not self.enable_deterministic
            and self.dcp_size == 1
            and (layer.sliding_window_size is None or layer.sliding_window_size <= -1)
            and self.forward_metadata.custom_mask is None
            and self.forward_metadata.window_kv_offsets is None
        )
        pre_rotated_q = None
        pre_rotated_k = None
        pre_rotated_v = None
        need_v_inverse = None

        if k is None and v is None:
            assert (
                not kv_dtype_is_int2
            ), "int2 KV cache cannot rebuild K/V from the packed pool buffers"
            pool = self.token_to_kv_pool
            cache_loc = forward_batch.out_cache_loc
            if isinstance(pool, SWAKVPool) and pool.layers_mapping[layer.layer_id][1]:
                cache_loc = pool.translate_loc_from_full_to_swa(cache_loc)
            k_buffer, v_buffer = pool.get_kv_buffer(layer.layer_id)
            k = k_buffer[cache_loc]
            v = v_buffer[cache_loc]
        elif k is None or v is None:
            raise ValueError("Both k and v should be None or not None")
        else:
            if use_quantized_dense_prefill:
                pre_rotated_q, pre_rotated_k, pre_rotated_v, need_v_inverse = (
                    prepare_quantized_extend_qkv(
                        kv_pool,
                        layer,
                        q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                        k.contiguous(),
                        v.contiguous(),
                    )
                )
            # Save KV cache first (must do this before unified kernel)
            if save_kv_cache:
                loc_info = KVWriteLoc(
                    forward_batch.out_cache_loc,
                    self.forward_metadata.swa_out_cache_loc,
                    full_loc=self.forward_metadata.out_cache_loc_full_physical,
                )
                if pre_rotated_k is not None and pre_rotated_v is not None:
                    # Int2: write the pre-rotated K/V so the pool keeps the
                    # rotated-domain representation without a second rotation.
                    kv_pool.set_kv_buffer(
                        layer,
                        loc_info,
                        pre_rotated_k,
                        pre_rotated_v,
                        layer.k_scale,
                        layer.v_scale,
                        already_hadamard_transformed=True,
                    )
                elif layer.k_scale is None:
                    self._set_kv_buffer(forward_batch, layer, loc_info, k, v)
                elif self.use_mla:
                    # For MLA, scale K manually before storing since MLATokenToKVPool
                    # doesn't accept scale parameters. Clone to protect k from mutation
                    # since it's used later in the attention kernel.
                    k_scaled = k.clone().div_(layer.k_scale)
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        loc_info,
                        k_scaled,
                        v,
                    )
                else:
                    self._set_kv_buffer(
                        forward_batch,
                        layer,
                        loc_info,
                        k.clone(),  # cloned to protect k,v from in-place mutation in set_kv_buffer
                        v.clone(),
                        layer.k_scale,
                        layer.v_scale,
                    )

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        causal = True
        if (
            layer.is_cross_attention
            or layer.attn_type == AttentionType.ENCODER_ONLY
            or (
                layer.attn_type == AttentionType.DECODER_BIDIRECTIONAL
                and self.allow_bidirectional_attention_in_extend
            )
        ):
            causal = False

        if self.dcp_size > 1:
            if score_mod is not None:
                raise NotImplementedError(
                    "DCP Triton extend does not support score_mod"
                )
            return self._forward_extend_dcp(
                q, k, v, layer, forward_batch, causal, logits_soft_cap, sinks
            )

        # Deterministic mode: use unified 1-stage kernel
        if self.enable_deterministic:
            return self._forward_extend_unified(
                q,
                o,
                layer,
                forward_batch,
                causal,
                logits_soft_cap,
                sinks,
                score_mod=score_mod,
                aux_tensors=aux_tensors,
            )

        if use_quantized_dense_prefill:
            return self._forward_extend_quantized_dense(
                q,
                k,
                v,
                o,
                layer,
                forward_batch,
                causal,
                pre_rotated_q=pre_rotated_q,
                pre_rotated_k=pre_rotated_k,
                pre_rotated_v=pre_rotated_v,
                need_v_inverse_override=need_v_inverse,
            )

        # Normal mode: use original 2-stage kernel
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            sliding_window_size = (
                layer.sliding_window_size
            )  # Needed for sliding window mask
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
            window_kv_offsets = self.forward_metadata.window_kv_offsets
        else:
            sliding_window_size = -1
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            window_kv_offsets = None

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        # Split-KV EAGLE-verify fast path (ROCm/Triton). On target-verify
        # (topk=1 causal chain), run the bandwidth-efficient split-KV kernel
        # instead of the serial-prefix extend kernel. verify_splitkv_fwd()
        # returns True if it ran (o written), or False for any case it cannot
        # serve bit-equivalently (its can_handle() gates on non-causal / sinks /
        # sliding-window / ragged / topk>1), so we fall through to
        # extend_attention_fwd below. Correctness is never at risk.
        if (
            self.use_verify_splitkv
            and score_mod is None
            and forward_batch.forward_mode.is_target_verify()
            and self.verify_splitkv_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k.contiguous(),
                v.contiguous(),
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                self.token_to_kv_pool.get_key_buffer(layer.layer_id),
                self.token_to_kv_pool.get_value_buffer(layer.layer_id),
                self.forward_metadata.qo_indptr,
                kv_indptr,
                kv_indices,
                self.forward_metadata.custom_mask,
                causal,
                self.forward_metadata.mask_indptr,
                self.forward_metadata.max_extend_len,
                k_descale,
                v_descale,
                layer.scaling,
                logit_cap=logits_soft_cap,
                sliding_window_size=sliding_window_size,
                sinks=sinks,
                window_kv_offsets=window_kv_offsets,
                xai_temperature_len=layer.xai_temperature_len,
                max_bs=self.req_to_token_pool.size,
            )
        ):
            return o

        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            self.forward_metadata.qo_indptr,
            kv_indptr,
            kv_indices,
            self.forward_metadata.custom_mask,
            causal,
            self.forward_metadata.mask_indptr,
            self.forward_metadata.max_extend_len,
            k_descale,
            v_descale,
            layer.scaling,
            logit_cap=logits_soft_cap,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_kv_offsets=window_kv_offsets,
            xai_temperature_len=layer.xai_temperature_len,
            page_size=self.page_size,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
        )
        return o

    def _forward_extend_dcp(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        causal: bool,
        logits_soft_cap: float,
        sinks: Optional[torch.Tensor],
    ):
        if sinks is not None:
            raise NotImplementedError("DCP Triton extend does not support sinks")
        if self.forward_metadata.custom_mask is not None:
            raise NotImplementedError("DCP Triton extend does not support custom masks")
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            raise NotImplementedError(
                "DCP Triton extend does not support sliding window"
            )

        group = get_parallel().dcp_group
        q_local = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim).contiguous()
        total_tokens, local_heads, _ = q_local.shape

        kv_indptr = self.forward_metadata.kv_indptr
        kv_indices = self.forward_metadata.kv_indices
        max_extend_len = self.forward_metadata.max_extend_len

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        k_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buffer = self.token_to_kv_pool.get_value_buffer(layer.layer_id)

        current_out = torch.zeros(
            (total_tokens, local_heads, layer.v_head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        current_lse = torch.full(
            (total_tokens, local_heads),
            -float("inf"),
            device=q.device,
            dtype=torch.float32,
        )

        # Current chunk K/V is still local before masked cache write, so it can
        # use the original extend kernel's current-token stage directly.
        if k.numel() > 0:
            empty_kv_indptr = torch.zeros_like(kv_indptr)
            self.extend_attention_fwd(
                q_local,
                k.contiguous(),
                v.contiguous(),
                current_out,
                k_buffer,
                v_buffer,
                self.forward_metadata.qo_indptr,
                empty_kv_indptr,
                kv_indices[:0],
                None,
                causal,
                None,
                max_extend_len,
                1.0,
                1.0,
                sm_scale=layer.scaling,
                logit_cap=logits_soft_cap,
                xai_temperature_len=layer.xai_temperature_len,
                lse_extend=current_lse,
                skip_prefix=True,
            )

        if kv_indices.numel() == 0:
            return current_out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim).to(
                q.dtype
            )

        # Prefix KV is sharded across DCP ranks, so compute each rank's
        # partial attention with all gathered query heads and merge by LSE.
        q_all = group.all_gather(q_local, dim=1).contiguous()
        total_heads = q_all.shape[1]
        prefix_out = torch.zeros(
            (total_tokens, total_heads, layer.v_head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        prefix_lse = torch.full(
            (total_tokens, total_heads),
            -float("inf"),
            device=q.device,
            dtype=torch.float32,
        )
        empty_k = k[:0].contiguous()
        empty_v = v[:0].contiguous()
        self.extend_attention_fwd(
            q_all,
            empty_k,
            empty_v,
            prefix_out,
            k_buffer,
            v_buffer,
            self.forward_metadata.qo_indptr,
            kv_indptr,
            kv_indices,
            None,
            False,
            None,
            max_extend_len,
            k_descale,
            v_descale,
            sm_scale=layer.scaling,
            logit_cap=logits_soft_cap,
            xai_temperature_len=layer.xai_temperature_len,
            lse_extend=prefix_lse,
            skip_extend=True,
        )

        prefix_out, prefix_lse = cp_lse_ag_out_rs_mha(
            prefix_out, prefix_lse, group, return_lse=True
        )
        final_lse = torch.logaddexp(prefix_lse, current_lse)
        prefix_scale = torch.exp(prefix_lse - final_lse).unsqueeze(-1)
        current_scale = torch.exp(current_lse - final_lse).unsqueeze(-1)
        prefix_scale = torch.nan_to_num(prefix_scale, nan=0.0, posinf=0.0, neginf=0.0)
        current_scale = torch.nan_to_num(current_scale, nan=0.0, posinf=0.0, neginf=0.0)
        out = prefix_out * prefix_scale + current_out * current_scale
        return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim).to(q.dtype)

    def _forward_extend_unified(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        causal: bool,
        logits_soft_cap: float,
        sinks: Optional[torch.Tensor],
        score_mod=None,
        aux_tensors=None,
    ):
        """
        Unified 1-stage extend attention for deterministic inference.
        Both prefix and extend KV are accessed through unified kv_indices.
        """
        bs = forward_batch.batch_size

        # Determine sliding window settings
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            sliding_window_size = layer.sliding_window_size
            # Note: for unified kernel, we use full kv_indptr (not window)
            prefix_kv_indptr = self.forward_metadata.window_kv_indptr
            prefix_kv_indices = self.forward_metadata.window_kv_indices
            # Compute window start positions (absolute position of first key in window)
            # window_start_pos = seq_len - window_len
            window_kv_lens = prefix_kv_indptr[1 : bs + 1] - prefix_kv_indptr[:bs]
            # Handle TARGET_VERIFY mode where extend_prefix_lens might not be set
            if forward_batch.extend_prefix_lens is not None:
                window_start_pos = (
                    forward_batch.extend_prefix_lens[:bs] - window_kv_lens
                )
            else:
                # Infer from spec_info: prefix_len = seq_len - draft_token_num
                if forward_batch.spec_info is not None and hasattr(
                    forward_batch.spec_info, "draft_token_num"
                ):
                    extend_prefix_lens = (
                        forward_batch.seq_lens[:bs]
                        - forward_batch.spec_info.draft_token_num
                    )
                    window_start_pos = extend_prefix_lens - window_kv_lens
                else:
                    window_start_pos = None
        else:
            sliding_window_size = -1
            prefix_kv_indptr = self.forward_metadata.kv_indptr
            prefix_kv_indices = self.forward_metadata.kv_indices
            window_start_pos = None

        extend_kv_indices = forward_batch.out_cache_loc
        pool = self.token_to_kv_pool
        if (
            layer.sliding_window_size is not None
            and layer.sliding_window_size > -1
            and isinstance(pool, SWAKVPool)
            and pool.layers_mapping[layer.layer_id][1]
        ):
            extend_kv_indices = pool.translate_loc_from_full_to_swa(extend_kv_indices)

        # Handle cases where extend_seq_lens or extend_start_loc might not be set
        # In speculative decoding, we can infer these from spec_info or compute them
        if forward_batch.extend_seq_lens is None:
            # TARGET_VERIFY mode: infer extend_seq_lens from spec_info
            if forward_batch.spec_info is not None and hasattr(
                forward_batch.spec_info, "draft_token_num"
            ):
                draft_token_num = forward_batch.spec_info.draft_token_num
                extend_seq_lens = torch.full(
                    (bs,), draft_token_num, dtype=torch.int32, device=self.device
                )
            else:
                raise RuntimeError(
                    "extend_seq_lens is None but cannot infer from spec_info. "
                    "This should not happen in TARGET_VERIFY mode."
                )
        else:
            extend_seq_lens = forward_batch.extend_seq_lens

        # Check extend_start_loc separately - it might be None even when extend_seq_lens is set
        if forward_batch.extend_start_loc is None:
            # Compute extend_start_loc from extend_seq_lens
            # extend_start_loc[i] = sum(extend_seq_lens[0:i])
            extend_start_loc = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=self.device),
                    torch.cumsum(extend_seq_lens[:-1], dim=0),
                ]
            )
        else:
            extend_start_loc = forward_batch.extend_start_loc

        unified_kv_indptr, unified_kv_indices, prefix_lens = (
            self.build_unified_kv_indices(
                prefix_kv_indptr,
                prefix_kv_indices,
                extend_start_loc,
                extend_seq_lens,
                extend_kv_indices,
                bs,
            )
        )

        # Convert prefix_lens to int32 for the kernel
        prefix_lens = prefix_lens.to(torch.int32)

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        # Call unified kernel
        self.extend_attention_fwd_unified(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            k_descale,
            v_descale,
            self.forward_metadata.qo_indptr,
            unified_kv_indptr,
            unified_kv_indices,
            prefix_lens,
            self.forward_metadata.max_extend_len,
            custom_mask=self.forward_metadata.custom_mask,
            mask_indptr=self.forward_metadata.mask_indptr,
            sm_scale=layer.scaling,
            logit_cap=logits_soft_cap,
            is_causal=causal,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_start_pos=window_start_pos,
            xai_temperature_len=layer.xai_temperature_len,
            page_size=self.page_size,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
        )

        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
        score_mod=None,
        aux_tensors=None,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        if save_kv_cache:
            if self.use_mla:
                if layer.k_scale is not None:
                    # MLATokenToKVPool doesn't accept scale parameters; k is unused
                    # after this point in decode, so scale in place.
                    k.div_(layer.k_scale)
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    v,
                )
            elif isinstance(self.token_to_kv_pool, UnifiedInt2HPKVPool):
                # Mixed HP+int2 decode write: loc holds HP-recent slot ids from
                # alloc_hp_recent; is_decode routes to the HP-only write path.
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                    is_decode=True,
                )
            else:
                self._set_kv_buffer(
                    forward_batch,
                    layer,
                    KVWriteLoc(
                        forward_batch.out_cache_loc,
                        self.forward_metadata.swa_out_cache_loc,
                        full_loc=self.forward_metadata.out_cache_loc_full_physical,
                    ),
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
        else:
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        # Select the correctly-sized attn_logits buffer for this layer.
        # The triton kernel's // Lv stride trick requires attn_logits.shape[-1]
        # to exactly match the layer's v_head_dim.
        attn_logits = self.forward_metadata.attn_logits
        if (
            self.forward_metadata.swa_attn_logits is not None
            and layer.v_head_dim == self.swa_v_head_dim
        ):
            attn_logits = self.forward_metadata.swa_attn_logits

        # Int2 quantized KV cache path (the only supported quant tier).
        kv_pool = self.token_to_kv_pool
        if isinstance(kv_pool, (MHATokenToKVPool, UnifiedInt2HPKVPool)) and (
            kv_pool.dtype == "int2"
        ):
            assert self.dcp_size == 1, "int2 KV cache does not support DCP"
            uses_oscar = _pool_uses_oscar_rotation(kv_pool)

            q_for_decode = q.contiguous().view(
                -1, layer.tp_q_head_num, layer.qk_head_dim
            )
            mixed_decode_metadata_available = (
                self.forward_metadata.mixed_hp_kv_indptr is not None
            )
            mixed_decode_enabled = (
                self.enable_mixed_kv
                and sinks is None
                and mixed_decode_metadata_available
            )
            if (
                self.enable_mixed_kv
                and mixed_decode_metadata_available
                and sinks is not None
            ):
                raise NotImplementedError(
                    "Mixed KV windows do not support sink tokens in Triton decode."
                )

            # Hard guarantee that the upstream gating actually held: if mixed
            # KV is enabled with an int2 pool, ``init_forward_metadata`` must
            # have built the per-tier indices. Falling through to the
            # non-mixed ``decode_attention_fwd_quantized`` path would treat
            # HP slot ids (>= HP_OFFSET) as quant slot ids and read OOB
            # garbage from the quant buffer. The known offenders are the
            # ``spec_info != None`` decode-or-idle paths (currently gated out
            # at server-args level); this assertion makes the gating
            # load-bearing at the kernel boundary so any future widening of
            # those upstream gates surfaces here loudly instead of silently
            # corrupting attention output.
            if self.enable_mixed_kv:
                assert mixed_decode_metadata_available, (
                    "Mixed-KV pool active but mixed decode metadata not built. "
                    "spec_info / non-decode-or-idle paths must not reach the "
                    "mixed-KV decode dispatch -- check upstream gating in "
                    "ServerArgs._unified_mixed_kv_active and the "
                    "KVCacheConfigurator int2 pool selection."
                )

            oscar_layer_idx = layer.layer_id - kv_pool.start_layer

            if uses_oscar:
                q_for_decode = _apply_oscar_rotation(
                    q_for_decode, kv_pool._R_k[oscar_layer_idx]
                )
            else:
                q_for_decode = apply_segmented_hadamard_transform(q_for_decode)
            if mixed_decode_enabled:
                bs = q_for_decode.shape[0]
                self.decode_attention_fwd_int2_unified(
                    q_for_decode,
                    kv_pool.get_hp_key_buffer(layer.layer_id),
                    kv_pool.get_hp_value_buffer(layer.layer_id),
                    kv_pool.get_raw_key_buffer(layer.layer_id),
                    kv_pool.get_raw_value_buffer(layer.layer_id),
                    kv_pool.get_key_scales_zeros(layer.layer_id),
                    kv_pool.get_value_scales_zeros(layer.layer_id),
                    o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    self.forward_metadata.mixed_hp_kv_indptr,
                    self.forward_metadata.mixed_hp_kv_indices,
                    self.forward_metadata.mixed_quant_kv_indptr,
                    self.forward_metadata.mixed_quant_kv_indices,
                    self.forward_metadata.mixed_attn_logits[:bs],
                    self.forward_metadata.mixed_attn_lse[:bs],
                    self.forward_metadata.mixed_hp_num_kv_splits[:bs],
                    self.forward_metadata.mixed_quant_num_kv_splits[:bs],
                    self.max_hp_kv_splits,
                    self.max_kv_splits,
                    layer.scaling,
                    logit_cap=logits_soft_cap,
                    sinks=sinks,
                    xai_temperature_len=layer.xai_temperature_len,
                )
            else:
                # Use optimized quantized attention kernel
                self.decode_attention_fwd_quantized(
                    q_for_decode,
                    kv_pool.get_raw_key_buffer(layer.layer_id),
                    kv_pool.get_raw_value_buffer(layer.layer_id),
                    kv_pool.get_key_scales_zeros(layer.layer_id),
                    kv_pool.get_value_scales_zeros(layer.layer_id),
                    o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    kv_indptr,
                    kv_indices,
                    self.forward_metadata.attn_logits,
                    self.forward_metadata.attn_lse,
                    self.forward_metadata.num_kv_splits,
                    self.max_kv_splits,
                    layer.scaling,
                    kv_pool.dtype,
                    logit_cap=logits_soft_cap,
                    sinks=sinks,
                    xai_temperature_len=layer.xai_temperature_len,
                )
            # int2: V is always rotated, so apply the inverse rotation to the
            # output. Oscar mode uses ``o @ R_v.T``; Hadamard mode re-applies
            # the segmented FWHT (self-inverse with 1/sqrt(N)).
            if uses_oscar:
                R_v = kv_pool._R_v[oscar_layer_idx]
                o3 = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                o3.copy_((o3.to(R_v.dtype) @ R_v.T).to(o3.dtype))
            else:
                o = apply_segmented_hadamard_transform(o)
            return o

        if self.dcp_size > 1:
            if score_mod is not None:
                raise NotImplementedError(
                    "DCP Triton decode does not support score_mod"
                )
            group = get_parallel().dcp_group
            with use_symmetric_memory(group):
                q_for_decode = q.view(
                    -1, layer.tp_q_head_num, layer.qk_head_dim
                ).contiguous()
            q_for_decode = group.all_gather(q_for_decode, dim=1).contiguous()
            o_for_decode = torch.empty(
                (q_for_decode.shape[0], q_for_decode.shape[1], layer.v_head_dim),
                dtype=torch.float32,
                device=q.device,
            )
            self.forward_metadata.attn_lse.fill_(-float("inf"))
            self.decode_attention_fwd(
                q_for_decode,
                self.token_to_kv_pool.get_key_buffer(layer.layer_id),
                self.token_to_kv_pool.get_value_buffer(layer.layer_id),
                o_for_decode,
                kv_indptr,
                kv_indices,
                attn_logits,
                self.forward_metadata.attn_lse,
                self.forward_metadata.num_kv_splits,
                self.max_kv_splits,
                layer.scaling,
                k_descale,
                v_descale,
                logit_cap=logits_soft_cap,
                sinks=sinks,
                xai_temperature_len=layer.xai_temperature_len,
            )
            local_lse = torch.logsumexp(
                self.forward_metadata.attn_lse[
                    : q_for_decode.shape[0], : q_for_decode.shape[1], :
                ],
                dim=-1,
            )
            o = cp_lse_ag_out_rs_mha(o_for_decode, local_lse, group)
            return o.reshape(-1, layer.tp_q_head_num * layer.v_head_dim).to(q.dtype)

        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            kv_indptr,
            kv_indices,
            attn_logits,
            self.forward_metadata.attn_lse,
            self.forward_metadata.num_kv_splits,
            self.max_kv_splits,
            layer.scaling,
            k_descale,
            v_descale,
            logit_cap=logits_soft_cap,
            sinks=sinks,
            xai_temperature_len=layer.xai_temperature_len,
            has_mla=self.use_mla,
            use_pdl=self.use_pdl,
            page_size=self.page_size,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
        )
        return o


class TritonMultiStepDraftBackend:
    """
    Wrap multiple triton attention backends as one for multiple consecutive
    draft decoding steps.
    """

    needs_cpu_seq_lens: bool = False

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.attn_backends: List[TritonAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                TritonAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
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
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: Optional[torch.Tensor],
        call_fn: int,
    ):
        if kv_indices_buffer is None:
            kv_indices_buffer = self.cuda_graph_kv_indices

        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum
        if seq_lens_sum is None:
            # seq_lens_sum here only slice-clamps a preallocated kv_indices buffer;
            # over-estimate is safe. Use a static UB to skip the per-iter .sum().item() D2H.
            seq_lens_sum = num_seqs * self.max_context_len

        generate_draft_decode_kv_indices[
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
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps),
            next_power_of_2(bs),
            self.page_size,
        )

        if call_fn is None:
            return

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
            dtype=torch.int64,
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
            dtype=torch.int64,
            device=self.device,
        )
        self.cuda_graph_num_kv_splits = torch.full(
            (max_num_tokens,),
            self.attn_backends[0].max_kv_splits,
            dtype=torch.int32,
            device=self.device,
        )

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs,
                max_num_tokens,
                kv_indices_buf=self.cuda_graph_kv_indices[i],
                cuda_graph_num_kv_splits_buf=self.cuda_graph_num_kv_splits,
            )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        if in_capture:
            inner_fb = build_inner_fb_view(
                forward_batch,
                bs=forward_batch.batch_size,
                forward_mode=ForwardMode.DECODE,
            )

            def call_fn(i, _forward_batch):
                self.attn_backends[i].init_forward_metadata_out_graph(
                    inner_fb, in_capture=True
                )

            self.common_template(forward_batch, None, call_fn)
        else:
            bs = forward_batch.batch_size
            self.common_template(forward_batch, None, None)

            # NOTE: Multi-step's attention backends use the slice of
            # - kv_indptr buffer (cuda graph and non-cuda graph)
            # - kv_indices buffer (cuda graph only)
            # So we don't need to assign the KV indices inside the attention backend.

            # Compute num_kv_splits only once
            num_token = bs * self.topk
            self.attn_backends[-1].get_num_kv_splits(
                self.attn_backends[-1].cuda_graph_num_kv_splits[:num_token],
                forward_batch.seq_lens[:bs],
            )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        for attn_backend in self.attn_backends:
            attn_backend.init_forward_metadata_in_graph(forward_batch)


def update_sliding_window_buffer(
    window_kv_indptr,
    req_to_token,
    sliding_window_size,
    seq_lens,
    req_pool_indices,
    bs,
    device=None,
    token_to_kv_pool=None,
    window_kv_indices=None,
    skip_full_to_swa_translation=False,
):
    """Fill window KV buffers for sliding-window attention.

    Pass ``window_kv_indices`` to write into a pre-allocated buffer (CUDA-graph
    path); omit it (or pass ``None``) to allocate a fresh tensor (eager path,
    requires ``device``).

    ``skip_full_to_swa_translation=True`` leaves ``window_kv_indices`` as VIRTUAL
    full-token ids (no eager full->swa translate). The unified-memory-pool cuda-graph
    builder passes this so the window translate is deferred to
    ``TritonAttnBackend._translate_cuda_graph_shared_pool_locs`` (run in
    ``init_forward_metadata_out_graph``, BEFORE ``graph.replay()``), which reads
    the live v2p and rewrites the static window buffer to swa-physical in place;
    baseline SWA leaves it False (eager).
    """
    window_kv_lens = torch.minimum(
        seq_lens,
        torch.tensor(sliding_window_size),
    )
    window_kv_indptr[1 : bs + 1] = torch.cumsum(window_kv_lens, dim=0)
    window_kv_indptr = window_kv_indptr[: bs + 1]
    if window_kv_indices is None:
        window_kv_indices = torch.empty(
            window_kv_indptr[-1], dtype=torch.int64, device=device
        )
    window_kv_start_idx = seq_lens - window_kv_lens
    create_flashinfer_kv_indices_triton[(bs,)](
        req_to_token,
        req_pool_indices,
        window_kv_lens,
        window_kv_indptr,
        window_kv_start_idx,
        window_kv_indices,
        req_to_token.stride(0),
    )
    if not skip_full_to_swa_translation and hasattr(
        token_to_kv_pool, "translate_loc_from_full_to_swa"
    ):
        kv_last_index = window_kv_indptr[-1]
        window_kv_indices[:kv_last_index] = (
            token_to_kv_pool.translate_loc_from_full_to_swa(
                window_kv_indices[:kv_last_index]
            )
        )
    return window_kv_indptr, window_kv_indices, window_kv_lens, window_kv_start_idx
