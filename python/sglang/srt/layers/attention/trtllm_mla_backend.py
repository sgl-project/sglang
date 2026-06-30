"""
Support attention backend for TRTLLM MLA kernels from flashinfer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch
import triton

from sglang.jit_kernel.fixup_zero_kv import fixup_zero_kv_rows
from sglang.srt.environ import envs
from sglang.srt.layers.attention.flashinfer_mla_backend import (
    FlashInferMLAAttnBackend,
    FlashInferMLAMultiStepDraftBackend,
)
from sglang.srt.layers.attention.triton_ops.kv_indices import (
    create_flashmla_kv_indices_triton,
    get_num_kv_index_blocks_flashmla,
    get_num_page_per_block_flashmla,
)
from sglang.srt.layers.attention.triton_ops.pad import (
    pad_draft_extend_query as pad_draft_extend_query_triton,
)
from sglang.srt.layers.attention.triton_ops.pad import (
    unpad_draft_extend_output as unpad_draft_extend_output_triton,
)
from sglang.srt.layers.attention.utils import (
    concat_mla_absorb_q_general,
    mla_quantize_and_rope_for_fp8,
)
from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_flashinfer_available, is_float4_e2m1fn_x2

if is_flashinfer_available():
    import flashinfer

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


@torch._dynamo.disable()
def _merge_state_v2_wrapper(o, s_a, o_exp, s_b):
    """``merge_state_v2`` wrapped in ``@torch._dynamo.disable`` so the EAGLE
    tree cascade's online-softmax merge is not traced into a CUDA-graph capture
    (mirrors FA3's ``merge_state_v2_wrapper`` in flashattention_backend.py)."""
    from sgl_kernel import merge_state_v2

    return merge_state_v2(o, s_a, o_exp, s_b)


# Constants
DEFAULT_WORKSPACE_SIZE_MB = 150  # Memory workspace size in MB

# Block constraint from flashinfer requirements
# From flashinfer.decode._check_trtllm_gen_mla_shape:
#   block_num % (128 / block_size) == 0
# This imposes that the total number of blocks must be divisible by
# (128 / block_size). We capture the 128 constant here so we can
# compute the LCM with other padding constraints.
TRTLLM_BLOCK_CONSTRAINT = 128


def _quantize_fp8_qkv(q, k, v, layer):
    q = q.to(torch.float8_e4m3fn)

    k_scale = getattr(layer, "k_scale_float", None)
    if k_scale is None:
        k_scale = 1.0
    if k_scale != 1.0:
        assert hasattr(layer, "k_scale"), "k_scale is not set"
        k_2d, _ = scaled_fp8_quant(
            k.reshape(-1, k.shape[-1]).contiguous(), layer.k_scale
        )
        k = k_2d.reshape(k.shape)
    else:
        k = k.to(torch.float8_e4m3fn)

    v_scale = getattr(layer, "v_scale_float", None)
    if v_scale is None:
        v_scale = 1.0
    if v_scale != 1.0:
        assert hasattr(layer, "v_scale"), "v_scale is not set"
        v_2d, _ = scaled_fp8_quant(
            v.reshape(-1, v.shape[-1]).contiguous(), layer.v_scale
        )
        v = v_2d.reshape(v.shape)
    else:
        v = v.to(torch.float8_e4m3fn)

    return q, k, v, k_scale, v_scale


global_zero_init_workspace_buffer = None
# cute-dsl needs its own workspace: it overwrites the buffer with split-KV
# partials, which corrupts the trtllm-gen multiCtasKv counters that rely on the
# zero-init buffer (they share it under attention-backend=cutedsl_mla, where
# draft-extend falls back to trtllm-gen) and deadlocks the reduction.
global_cute_dsl_workspace_buffer = None


@dataclass
class TRTLLMMLAPrefillMetadata:
    """Metadata for TRTLLM MLA prefill operations."""

    max_seq_len: int
    cum_seq_lens: torch.Tensor
    seq_lens: torch.Tensor
    fallback_to_flashinfer_impl: bool = False


@dataclass
class TRTLLMMLADecodeMetadata:
    """Metadata for TRTLLM MLA decode operations."""

    block_kv_indices: Optional[torch.Tensor] = None
    max_seq_len_k: Optional[int] = None
    max_seq_len_q: Optional[int] = None
    sum_seq_lens_q: Optional[int] = None
    cu_seqlens_q: Optional[torch.Tensor] = None
    seq_lens_q: Optional[torch.Tensor] = None
    seq_lens_k: Optional[torch.Tensor] = None


@dataclass
class TRTLLMMLATreeVerifyMetadata:
    """Persistent CUDA-graph buffers for the EAGLE tree-verify (topk>1) cascade.

    The captured ``forward_extend`` reads the live ``spec_info.custom_mask`` /
    ``out_cache_loc`` only in eager mode. Under CUDA graph those data-dependent
    reads would bake stale tensor addresses at capture and corrupt the tree on
    replay, so ``init_forward_metadata_out_graph`` extracts the tree adjacency
    and the per-query draft cache slots into these persistent buffers (allocated
    once in ``init_cuda_graph_state``) on every capture and replay, and the
    captured forward reads only these stable tensors.
    """

    # [max_bs, num_draft_tokens, num_draft_tokens] bool tree adjacency.
    tree_mask: Optional[torch.Tensor] = None
    # [max_bs, num_draft_tokens] int64 cache slots of the draft tokens.
    draft_slots: Optional[torch.Tensor] = None


@dataclass
class TRTLLMMLADraftStepMetadata:
    """Per-step tree-expanded draft-decode metadata for the topk>1 cascade.

    Built once per draft pass (eager: ``_build_draft_step_meta``; CUDA graph:
    ``_fill_draft_step_meta_graph`` into persistent buffers). Read by the
    captured/eager ``forward_decode`` cascade.

    The query during a per-step draft decode has batch ``bs*topk`` (the tree
    frontier: request r, branch t at row ``r*topk + t``). Pass 1 attends each
    branch's full request prefix (prefix page table repeated topk); pass 2
    attends the branch's own linear chain of draft tokens for steps
    ``0..step_id``.
    """

    block_kv_indices_prefix: Optional[torch.Tensor] = None  # [bs*topk, max_blocks]
    seq_lens_prefix: Optional[torch.Tensor] = None  # [bs*topk] int32 prefix lens
    max_seq_len_prefix: Optional[int] = None
    draft_slots: Optional[torch.Tensor] = None  # [bs*topk, step_id+1] int64 slots
    bs: Optional[int] = None
    topk: Optional[int] = None


class TRTLLMMLABackend(FlashInferMLAAttnBackend):
    """TRTLLM MLA attention kernel from flashinfer."""

    # trtllm-gen kernels rebuild metadata from preallocated buffers and never
    # read seq_lens_cpu / seq_lens_sum; opt out of the D2H sync.
    needs_cpu_seq_lens: bool = False

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
        backend: str = "trtllm-gen",
    ):
        super().__init__(
            model_runner,
            skip_prefill,
            kv_indptr_buf,
            q_indptr_decode_buf,
        )

        config = model_runner.model_config

        # Model parameters
        self.num_q_heads = config.num_attention_heads // get_parallel().attn_tp_size
        self.num_kv_heads = config.get_num_kv_heads(get_parallel().attn_tp_size)
        self.num_local_heads = config.num_attention_heads // get_parallel().attn_tp_size

        # MLA-specific dimensions
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim

        # Runtime parameters
        self.backend = backend
        self.scaling = config.scaling
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.page_size = model_runner.page_size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        # Workspace allocation
        self.workspace_size = DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024
        if self.backend == "cute-dsl":
            # Separate buffer from trtllm-gen (see note above); safe to share
            # among cute-dsl instances.
            global global_cute_dsl_workspace_buffer
            if global_cute_dsl_workspace_buffer is None:
                global_cute_dsl_workspace_buffer = torch.zeros(
                    self.workspace_size,
                    dtype=torch.int8,
                    device=model_runner.device,
                )
            self.workspace_buffer = global_cute_dsl_workspace_buffer
        else:
            global global_zero_init_workspace_buffer
            if global_zero_init_workspace_buffer is None:
                global_zero_init_workspace_buffer = torch.zeros(
                    self.workspace_size,
                    dtype=torch.int8,
                    device=model_runner.device,
                )
            self.workspace_buffer = global_zero_init_workspace_buffer

        # CUDA graph state
        self.decode_cuda_graph_metadata = {}
        self.decode_cuda_graph_kv_indices = None
        self.padded_q_buffer = None
        self.unpad_output_buffer = None
        self.forward_prefill_metadata: Optional[TRTLLMMLAPrefillMetadata] = None
        self.forward_decode_metadata: Union[TRTLLMMLADecodeMetadata, None] = None

        self.disable_chunked_prefix_cache = (
            get_global_server_args().disable_chunked_prefix_cache
        )

        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.cuda_graph_custom_mask = None

        # EAGLE tree drafting (speculative_eagle_topk > 1) runs a 2-pass cascade
        # on the cuteDSL MLA decode kernel in both the verify (forward_extend)
        # and draft (forward_decode) paths. Cache the topk>1 check once here so
        # the per-batch forward never recomputes it. topk==1 (chain) keeps the
        # existing single-pass path unchanged.
        speculative_eagle_topk = model_runner.server_args.speculative_eagle_topk or 1
        self._tree_topk = speculative_eagle_topk
        self._is_tree_verify = speculative_eagle_topk > 1
        # Persistent CUDA-graph buffers for the tree-verify cascade, allocated in
        # init_cuda_graph_state and filled out-of-graph in
        # init_forward_metadata_out_graph. None in eager mode (live extraction).
        self.tree_verify_cuda_graph_metadata: Optional[TRTLLMMLATreeVerifyMetadata] = (
            None
        )
        self._tree_verify_graph_meta: Optional[TRTLLMMLATreeVerifyMetadata] = None

        # Stamped by TRTLLMMLAMultiStepDraftBackend on its per-step backends.
        # step_id is None on the verify backend (not a draft per-step backend).
        self.draft_step_id: Optional[int] = None
        self.draft_step_topk: int = 1
        self.draft_step_num_steps: int = 1
        self.draft_step_metadata: Optional[TRTLLMMLADraftStepMetadata] = None
        self.draft_step_cuda_graph_buffers: Optional[dict] = None

    def _calc_padded_blocks(self, max_seq_len: int) -> int:
        """
        Calculate padded block count that satisfies both TRT-LLM and Triton constraints.

        Args:
            max_seq_len: Maximum sequence length in tokens

        Returns:
            Number of blocks padded to satisfy all constraints
        """
        blocks = triton.cdiv(max_seq_len, self.page_size)

        # Apply dual constraints (take LCM to satisfy both):
        # 1. TRT-LLM: block_num % (128 / page_size) == 0
        # 2. Triton: number of pages per block
        trtllm_constraint = TRTLLM_BLOCK_CONSTRAINT // self.page_size
        triton_constraint = get_num_page_per_block_flashmla(self.page_size)
        constraint_lcm = math.lcm(trtllm_constraint, triton_constraint)

        if blocks % constraint_lcm != 0:
            blocks = triton.cdiv(blocks, constraint_lcm) * constraint_lcm
        return blocks

    def _create_block_kv_indices(
        self,
        batch_size: int,
        max_blocks: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create block KV indices tensor using Triton kernel.

        Args:
            batch_size: Batch size
            max_blocks: Maximum number of blocks per sequence
            req_pool_indices: Request pool indices
            seq_lens: Sequence lengths
            device: Target device

        Returns:
            Block KV indices tensor
        """
        block_kv_indices = torch.full(
            (batch_size, max_blocks), -1, dtype=torch.int32, device=device
        )

        create_flashmla_kv_indices_triton[
            (
                batch_size,
                get_num_kv_index_blocks_flashmla(max_blocks, self.page_size),
            )
        ](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            None,
            block_kv_indices,
            self.req_to_token.stride(0),
            max_blocks,
            PAGED_SIZE=self.page_size,
        )

        return block_kv_indices

    def _extract_tree_mask(
        self, forward_batch: ForwardBatch, bs: int, T: int
    ) -> torch.Tensor:
        """Extract the per-request draft-token x draft-token tree adjacency from
        the flattened ``spec_info.custom_mask``.

        The mask is laid out (eagle_info.py: ``paged_kernel_lens_sum * T +
        T*T*batch_size``) as, per request r, a row-major ``[T, prefix_len_r + T]``
        block: row i (verify query i) has ``prefix_len_r + T`` columns; the first
        ``prefix_len_r`` are prefix KV, the trailing T are the draft-token x
        draft-token tree adjacency. Requests are concatenated (ragged, prefix
        lens differ). The cascade only needs the trailing TxT sub-block (pass 1
        handles the full prefix unconditionally). Mirrors FA3's
        ``mask_extraction_indices`` (flashattention_backend.py).

        Returns a bool tensor ``[bs, T, T]`` where ``mask[r, i, j]`` is True iff
        verify query i of request r attends draft token j. Fully vectorized (no
        ``.item()`` / python loop) so it is CUDA-graph capture-safe.
        """
        custom_mask = forward_batch.spec_info.custom_mask
        # In is_target_verify, forward_batch.seq_lens still holds the pre-draft
        # prefix length per request, so seq_lens == prefix.
        prefix_lens = forward_batch.seq_lens.to(torch.int64)  # [bs]
        device = custom_mask.device

        block_cols = prefix_lens + T  # [bs] columns per row of request r
        block_sizes = block_cols * T  # [bs] flat elems for request r
        block_starts = torch.zeros(bs, dtype=torch.int64, device=device)
        if bs > 1:
            block_starts[1:] = torch.cumsum(block_sizes, dim=0)[:-1]

        # Flat index of query i / draft j of request r is:
        #   block_starts[r] + i*(prefix_r + T) + prefix_r + j.
        i_idx = torch.arange(T, device=device).view(1, T, 1)  # [1,T,1]
        j_idx = torch.arange(T, device=device).view(1, 1, T)  # [1,1,T]
        idx = (
            block_starts.view(bs, 1, 1)
            + i_idx * block_cols.view(bs, 1, 1)
            + prefix_lens.view(bs, 1, 1)
            + j_idx
        )  # [bs,T,T] int64
        # Under CUDA graph the batch is padded (bs > real) and the ragged offset
        # for padded rows can exceed custom_mask; clamp so padded rows read a
        # valid (dummy) bit (their tree mask is unused -- outputs discarded).
        idx = idx.reshape(-1).clamp_(0, custom_mask.numel() - 1)
        return custom_mask[idx].view(bs, T, T).to(torch.bool)

    def _forward_tree_verify(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        metadata: TRTLLMMLADecodeMetadata,
    ) -> torch.Tensor:
        """EAGLE tree-verify (topk>1) 2-pass cascade on the cuteDSL MLA decode
        kernel.

        ``q`` arrives already merged/quantized as ``[bs*T, H, D]`` (the caller
        replicated the existing forward_extend query prep). The KV cache for the
        draft tokens has already been written by the caller.

        Pass 1 (prefix): a single NON-CAUSAL fold over the request prefix, so all
        T draft tokens attend the full prefix and the prefix is read once
        (cute_dsl_mla_decode causal=False, depends on flashinfer #3771). Pass 2
        (tree): a bf16 einsum over only the T draft tokens, masked by the tree
        adjacency. Merge via online softmax (merge_state_v2). This mirrors FA3's
        cascade (flashattention_backend.py: shared prefix + masked expand +
        merge_state_v2_wrapper).
        """
        from flashinfer.cute_dsl.attention import cute_dsl_mla_decode

        spec_info = forward_batch.spec_info
        bs = forward_batch.batch_size
        T = spec_info.draft_token_num
        H = layer.tp_q_head_num
        D = layer.head_dim
        Dv = layer.v_head_dim
        N = bs * T

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache_3d = k_cache.view(-1, self.page_size, self.kv_cache_dim)
        kv_flat = k_cache.view(-1, self.kv_cache_dim)

        q4 = q.view(bs, T, H, D)
        bmm1_scale = self._compute_decode_bmm1_scale(layer)

        # cute-dsl needs its own workspace (the fold decode returns LSE; trtllm-gen
        # does not support return_lse). Prefer the dedicated cute-dsl workspace.
        ws = global_cute_dsl_workspace_buffer
        if ws is None:
            ws = self.workspace_buffer

        # ---- Pass 1: PREFIX -- single non-causal fold (reads each prefix once) ----
        # Clamp CUDA-graph padded rows (seq_len==0 / block_table==-1) to valid
        # dummies; the cute kernel device-asserts on a 0-length / -1-page row and
        # the padded outputs are discarded after the merge.
        prefix_lens = forward_batch.seq_lens.to(torch.int32).clamp(min=1)  # [bs]
        bt_prefix = metadata.block_kv_indices.clamp(min=0)  # [bs, max_blocks]
        o_pre, lse_pre = cute_dsl_mla_decode(
            query=q4,
            kv_cache=kv_cache_3d,
            workspace_buffer=ws,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=bt_prefix,
            seq_lens=prefix_lens,
            max_seq_len=int(metadata.max_seq_len_k),
            softmax_scale=bmm1_scale,
            causal=False,  # all T queries attend the full prefix (exact)
            return_lse=True,
        )
        o_pre_n = o_pre.reshape(N, H, Dv)
        lse_pre_n = lse_pre.reshape(N, H).float()  # [N, H] natural-log

        # ---- Pass 2: TREE -- tree-masked attention over the T draft tokens ----
        # The draft set is only T tokens, so compute the per-query tree-masked
        # attention directly (einsum); exact, static-shape (graph-safe), no
        # scratch gather. bf16 matmuls (tensor cores, fp32 accum); softmax/lse
        # stay fp32. bmm1_scale carries q_scale*k_scale*softmax so the float
        # matmul matches the kernel's matmul in pass 1.
        meta = self._tree_verify_graph_meta
        if meta is not None:
            tree = meta.tree_mask[:bs]  # [bs, T, T] bool (persistent CG buffer)
            ocl = meta.draft_slots[:bs].reshape(-1).clamp(0, kv_flat.shape[0] - 1)
        else:
            tree = self._extract_tree_mask(forward_batch, bs, T)  # eager, live
            ocl = (
                forward_batch.out_cache_loc.view(bs, T)
                .reshape(-1)
                .clamp(0, kv_flat.shape[0] - 1)
            )
        draft_kv = kv_flat[ocl].view(bs, T, self.kv_cache_dim).to(torch.bfloat16)
        q_v = q4.to(torch.bfloat16)  # [bs, T(q), H, D]
        scores = (
            torch.einsum("bqhd,bkd->bqhk", q_v, draft_kv).float() * bmm1_scale
        )  # [bs, Tq, H, Tk] fp32
        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~tree[:, :, None, :], neg)
        lse_tree_n = torch.logsumexp(scores, dim=-1).reshape(N, H)  # natural-log
        w = torch.softmax(scores, dim=-1)
        o_tree_n = (
            torch.einsum("bqhk,bkd->bqhd", w.to(torch.bfloat16), draft_kv[..., :Dv])
            .reshape(N, H, Dv)
            .to(o_pre_n.dtype)
        )

        # ---- Merge (online softmax). merge_state_v2 wants v[N,H,Dv] + lse[N,H]
        #      fp32 natural-log; both passes already match (no transpose). ----
        merged, _ = _merge_state_v2_wrapper(o_pre_n, lse_pre_n, o_tree_n, lse_tree_n)
        return merged.reshape(N, H * Dv)

    def _forward_tree_draft(
        self, query: torch.Tensor, layer: RadixAttention
    ) -> torch.Tensor:
        """EAGLE tree-draft (topk>1) per-step decode cascade on the cuteDSL MLA
        decode kernel.

        ``query`` arrives already merged/quantized as ``[bs*topk, H, D]`` (one
        query per tree branch, q_len==1). The branch's draft-token KV has already
        been written by the caller. Reads the per-step metadata built by
        ``TRTLLMMLAMultiStepDraftBackend.init_forward_metadata`` /
        ``_fill_draft_step_meta_graph``.

        Pass 1 (prefix): each branch attends its request's full prefix (the
        prefix page table repeated topk). Pass 2 (chain): each branch attends its
        own linear chain of <=step_id+1 draft tokens (a bf16 einsum, no tree mask
        -- the branches diverge only at the root). Merge via online softmax. This
        is the simpler sibling of the verify cascade (FA3
        flashattention_backend.py: shared prefix + per-branch expand + merge).
        """
        meta = self.draft_step_metadata
        H = layer.tp_q_head_num
        D = layer.head_dim
        Dv = layer.v_head_dim
        N = query.shape[0]  # bs*topk
        assert (
            N == meta.bs * meta.topk
        ), f"draft query batch {N} != bs*topk {meta.bs}*{meta.topk}"
        q4 = query.view(N, 1, H, D)  # one query per branch (q_len == 1)

        bmm1_scale = self._compute_decode_bmm1_scale(layer)
        ws = global_cute_dsl_workspace_buffer
        if ws is None:
            ws = self.workspace_buffer

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)
        kv_flat = k_cache.view(-1, self.kv_cache_dim)  # [total_slots, dim]

        # ---- Pass 1: PREFIX (each branch attends its request's full prefix) ----
        # Clamp CUDA-graph padded rows (seq_len==0 / block_table==-1) to dummies.
        bt_prefix = meta.block_kv_indices_prefix.clamp(min=0)
        sl_prefix = meta.seq_lens_prefix.clamp(min=1)
        o_pre, lse_pre = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q4,
            kv_cache=kv_cache,
            workspace_buffer=ws,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=bt_prefix,
            seq_lens=sl_prefix,
            max_seq_len=int(meta.max_seq_len_prefix),
            bmm1_scale=bmm1_scale,
            return_lse=True,
            backend="cute-dsl",
        )
        o_pre_n = o_pre.reshape(N, H, Dv)
        lse_pre_n = lse_pre.reshape(N, H).float()  # [N, H] natural-log

        # ---- Pass 2: DRAFT PATH -- attention over the branch's chain of draft
        #      tokens (a linear chain, no tree mask among them). bf16 einsum
        #      (exact, static-shape, graph-safe). ----
        draft_slots = meta.draft_slots  # [N, decode_length] int64 cache slots
        decode_length = draft_slots.shape[1]
        ds = draft_slots.reshape(-1).to(torch.long).clamp(0, kv_flat.shape[0] - 1)
        chain_kv = (
            kv_flat[ds].view(N, decode_length, self.kv_cache_dim).to(torch.bfloat16)
        )  # [N, dl, kv_dim]
        q_n = q4.reshape(N, H, D).to(torch.bfloat16)  # [N, H, D]
        scores = (
            torch.einsum("nhd,nkd->nhk", q_n, chain_kv).float() * bmm1_scale
        )  # [N, H, dl] fp32
        lse_tree_n = torch.logsumexp(scores, dim=-1)  # [N, H] natural-log
        w = torch.softmax(scores, dim=-1)
        o_tree_n = torch.einsum(
            "nhk,nkd->nhd", w.to(torch.bfloat16), chain_kv[..., :Dv]
        ).float()  # [N, H, Dv]

        # ---- Merge (online softmax). lse [N,H] fp32 natural-log; no transpose. ----
        merged, _ = _merge_state_v2_wrapper(
            o_pre_n, lse_pre_n, o_tree_n.to(o_pre_n.dtype), lse_tree_n.float()
        )
        return merged.reshape(N, H * Dv)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        """Initialize CUDA graph state for TRTLLM MLA."""

        max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)

        self.decode_cuda_graph_kv_indices = torch.full(
            (max_bs, max_blocks_per_seq), -1, dtype=torch.int32, device=self.device
        )
        num_tokens_per_bs = max_num_tokens // max_bs

        if is_float4_e2m1fn_x2(self.data_type):
            # Buffer for padded query: (max_bs, max_draft_tokens, num_q_heads, v_head_dim)
            self.store_dtype = torch.uint8
            self.padded_q_buffer = torch.zeros(
                (max_bs, num_tokens_per_bs // 2, self.num_q_heads, self.kv_cache_dim),
                dtype=self.store_dtype,
                device=self.device,
            )

            # Buffer for unpadded output: (max_num_tokens, num_q_heads, v_head_dim)
            self.unpad_output_buffer = torch.zeros(
                (max_num_tokens // 2, self.num_q_heads, 512),
                dtype=self.store_dtype,
                device=self.device,
            )
        else:
            # Buffer for padded query: (max_bs, max_draft_tokens, num_q_heads, v_head_dim)
            self.padded_q_buffer = torch.zeros(
                (max_bs, num_tokens_per_bs, self.num_q_heads, self.kv_cache_dim),
                dtype=self.data_type,
                device=self.device,
            )

            # Buffer for unpadded output: (max_num_tokens, num_q_heads, v_head_dim)
            self.unpad_output_buffer = torch.zeros(
                (max_num_tokens, self.num_q_heads, 512),
                dtype=self.data_type,
                device=self.device,
            )

        if self.num_draft_tokens and not self.skip_prefill:
            # Worst-case FULL_MASK tree-mask scratch (bool); build_tree writes it
            # in-place so the gpu_only path needs no seq_lens_sum.
            self.cuda_graph_custom_mask = torch.zeros(
                max_num_tokens * (self.max_context_len + self.num_draft_tokens),
                dtype=torch.bool,
                device=self.device,
            )

        # EAGLE tree-verify (topk>1) persistent buffers. The verify backend has
        # skip_prefill=False and is not a draft per-step backend. Filled
        # out-of-graph in init_forward_metadata_out_graph; read by the captured
        # _forward_tree_verify via _tree_verify_graph_meta.
        if (
            self._is_tree_verify
            and not self.skip_prefill
            and self.draft_step_id is None
            and self.num_draft_tokens
        ):
            T = self.num_draft_tokens
            self.tree_verify_cuda_graph_metadata = TRTLLMMLATreeVerifyMetadata(
                tree_mask=torch.zeros(
                    (max_bs, T, T), dtype=torch.bool, device=self.device
                ),
                draft_slots=torch.zeros(
                    (max_bs, T), dtype=torch.int64, device=self.device
                ),
            )

        # EAGLE tree-draft (topk>1) per-step persistent buffers. Stamped per-step
        # draft backends only. step_id is fixed per backend (step i decodes
        # length i+1). Filled in-place each capture/replay by
        # _fill_draft_step_meta_graph; read by the captured forward_decode.
        if self.draft_step_id is not None and self.draft_step_topk > 1:
            topk = self.draft_step_topk
            decode_length = self.draft_step_id + 1
            max_blocks = self._calc_padded_blocks(self.max_context_len)
            n = max_bs * topk
            self.draft_step_cuda_graph_buffers = {
                "block_kv_indices_prefix": torch.full(
                    (n, max_blocks), -1, dtype=torch.int32, device=self.device
                ),
                "seq_lens_prefix": torch.zeros(
                    (n,), dtype=torch.int32, device=self.device
                ),
                "draft_slots": torch.zeros(
                    (n, decode_length), dtype=torch.int64, device=self.device
                ),
            }

        super().init_cuda_graph_state(max_bs, max_num_tokens, kv_indices_buf)

    def get_verify_buffers_to_fill_after_draft(self):
        return [self.cuda_graph_custom_mask, None]

    def _init_cuda_graph_metadata(
        self,
        bs: int,
        num_tokens: int,
        forward_mode: ForwardMode,
        seq_lens: torch.Tensor,
        device: torch.device,
    ):
        """Allocate persistent metadata buffers for CUDA graph capture."""
        metadata = TRTLLMMLADecodeMetadata()

        if forward_mode.is_target_verify():
            metadata.seq_lens_k = torch.zeros((bs,), dtype=torch.int32, device=device)
        elif forward_mode.is_draft_extend_v2():
            num_tokens_per_bs = self.num_draft_tokens
            metadata.max_seq_len_q = num_tokens_per_bs
            metadata.sum_seq_lens_q = num_tokens_per_bs * bs
            metadata.cu_seqlens_q = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                num_tokens_per_bs,
                dtype=torch.int32,
                device=device,
            )
            metadata.seq_lens_q = torch.full(
                (bs,), num_tokens_per_bs, dtype=torch.int32, device=device
            )
            metadata.seq_lens_k = torch.zeros((bs,), dtype=torch.int32, device=device)

        # Capture with full width so future longer sequences are safe during replay.
        max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)
        block_kv_indices = self.decode_cuda_graph_kv_indices[:bs, :max_blocks_per_seq]
        metadata.block_kv_indices = block_kv_indices
        metadata.max_seq_len_k = self.max_context_len

        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
    ):
        """Shared decode / target-verify / draft-extend capture+replay body.

        Public entry: :py:meth:`init_forward_metadata_out_graph` (which routes
        the non-decode-family modes to the FlashInferMLA parent).
        """
        metadata = self.decode_cuda_graph_metadata[bs]

        if forward_mode.is_target_verify():
            seq_lens = seq_lens[:bs] + self.num_draft_tokens
            metadata.seq_lens_k.copy_(seq_lens)
        elif forward_mode.is_draft_extend_v2():
            num_tokens_per_bs = self.num_draft_tokens
            metadata.max_seq_len_q = num_tokens_per_bs
            metadata.sum_seq_lens_q = num_tokens_per_bs * bs
            seq_lens = seq_lens[:bs]
            metadata.seq_lens_k.copy_(seq_lens)

        # Update block indices for new sequences.
        create_flashmla_kv_indices_triton[
            (
                bs,
                get_num_kv_index_blocks_flashmla(
                    metadata.block_kv_indices.shape[1], self.page_size
                ),
            )
        ](
            self.req_to_token,
            req_pool_indices[:bs],
            seq_lens,
            None,
            metadata.block_kv_indices,
            self.req_to_token.stride(0),
            metadata.block_kv_indices.shape[1],
            PAGED_SIZE=self.page_size,
        )

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        """Get the fill value for sequence lengths in CUDA graph."""
        return 1

    def init_mha_chunk_metadata(self, forward_batch: ForwardBatch) -> None:
        has_prefix = any(forward_batch.extend_prefix_lens_cpu)
        fallback_to_flashinfer_impl = (
            self.disable_chunked_prefix_cache and has_prefix
        ) or is_in_tc_piecewise_cuda_graph()
        if fallback_to_flashinfer_impl:
            super().init_mha_chunk_metadata(
                forward_batch, disable_flashinfer_ragged=True
            )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        forward_mode = forward_batch.forward_mode

        if (
            not forward_mode.is_decode_or_idle()
            and not forward_mode.is_target_verify()
            and not forward_mode.is_draft_extend_v2()
        ):
            return super().init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )

        bs = forward_batch.batch_size
        if in_capture:
            num_tokens = forward_batch.positions.numel()
            self._init_cuda_graph_metadata(
                bs,
                num_tokens,
                forward_mode,
                forward_batch.seq_lens,
                forward_batch.seq_lens.device,
            )
            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                forward_mode=forward_mode,
            )
        else:
            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                forward_mode=forward_mode,
            )

        # EAGLE tree-verify (topk>1): extract the tree adjacency + draft slots
        # out-of-graph into the persistent buffers and point the captured forward
        # at them. The live spec_info.custom_mask / out_cache_loc read inside the
        # captured forward would bake stale addresses on replay.
        if (
            self._is_tree_verify
            and forward_mode.is_target_verify()
            and self.tree_verify_cuda_graph_metadata is not None
        ):
            T = self.num_draft_tokens
            meta = self.tree_verify_cuda_graph_metadata
            # seq_lens is padded to bs, so _extract_tree_mask uses bs. out_cache_loc
            # is the caller's REAL tensor (length raw_bs*T, NOT padded): fill the
            # real rows and zero the padded tail (padded outputs are discarded).
            meta.tree_mask[:bs].copy_(self._extract_tree_mask(forward_batch, bs, T))
            ocl = forward_batch.out_cache_loc
            raw_bs = ocl.numel() // T
            meta.draft_slots[:raw_bs].copy_(ocl.view(raw_bs, T).to(torch.int64))
            if raw_bs < bs:
                meta.draft_slots[raw_bs:bs].zero_()
            self._tree_verify_graph_meta = meta

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize the metadata for a forward pass."""
        # Eager path: the tree-verify cascade uses live spec_info extraction, not
        # the persistent CUDA-graph buffers.
        self._tree_verify_graph_meta = None
        # Delegate to parent for non-decode modes.
        if (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend_v2()
        ):
            # For extend batch with prefix length > 0, fallback to ragged kernel implemented in flashinfer MLA backend
            # when chunked prefix cache is disabled.
            # Also fallback to flashinfer MLA backend when in piecewise cuda graph, since it only supports MLA forward mode.
            has_prefix = any(forward_batch.extend_prefix_lens_cpu)
            fallback_to_flashinfer_impl = (
                self.disable_chunked_prefix_cache and has_prefix
            ) or is_in_tc_piecewise_cuda_graph()
            if fallback_to_flashinfer_impl:
                super().init_forward_metadata(forward_batch)

            seq_lens = forward_batch.seq_lens - forward_batch.extend_prefix_lens
            cum_seq_lens_q = torch.cat(
                (
                    torch.zeros(
                        1, dtype=torch.int32, device=forward_batch.seq_lens.device
                    ),
                    torch.cumsum(seq_lens, dim=0),
                )
            ).int()
            max_seq_len = max(forward_batch.extend_seq_lens_cpu)
            self.forward_prefill_metadata = TRTLLMMLAPrefillMetadata(
                max_seq_len,
                cum_seq_lens_q,
                seq_lens,
                fallback_to_flashinfer_impl,
            )
        elif (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            bs = forward_batch.batch_size
            self.forward_decode_metadata = TRTLLMMLADecodeMetadata()
            # This is necessary because the backend instance persists across forward passes,
            # and forward_prefill_metadata from a previous regular extend call could still be set.
            if (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend_v2()
            ):
                self.forward_prefill_metadata = None
            # Get maximum sequence length.
            if getattr(forward_batch, "seq_lens_cpu", None) is not None:
                max_seq = forward_batch.seq_lens_cpu.max().item()
            else:
                max_seq = forward_batch.seq_lens.max().item()

            seq_lens = forward_batch.seq_lens

            if forward_batch.forward_mode.is_target_verify():
                max_seq = max_seq + self.num_draft_tokens
                seq_lens = seq_lens + self.num_draft_tokens
                self.forward_decode_metadata.seq_lens_k = seq_lens.to(torch.int32)
            elif forward_batch.forward_mode.is_draft_extend_v2():
                sum_seq_lens_q = sum(forward_batch.extend_seq_lens_cpu)
                max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(
                        forward_batch.extend_seq_lens, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                # see NOTE(draft_extend seq_len handling)
                seq_lens = seq_lens - forward_batch.extend_seq_lens + max_seq_len_q

                self.forward_decode_metadata.max_seq_len_q = max_seq_len_q
                self.forward_decode_metadata.sum_seq_lens_q = sum_seq_lens_q
                self.forward_decode_metadata.cu_seqlens_q = cu_seqlens_q
                self.forward_decode_metadata.seq_lens_q = forward_batch.extend_seq_lens
                self.forward_decode_metadata.seq_lens_k = seq_lens.to(torch.int32)

            max_seqlen_pad = self._calc_padded_blocks(max_seq)
            block_kv_indices = self._create_block_kv_indices(
                bs,
                max_seqlen_pad,
                forward_batch.req_pool_indices,
                seq_lens,
                seq_lens.device,
            )

            self.forward_decode_metadata.block_kv_indices = block_kv_indices
            self.forward_decode_metadata.max_seq_len_k = int(max_seq)
            self.forward_decode_metadata.batch_size = bs

            forward_batch.decode_trtllm_mla_metadata = self.forward_decode_metadata
        else:
            return super().init_forward_metadata(forward_batch)

    def pad_draft_extend_query(
        self,
        q: torch.Tensor,
        padded_q: torch.Tensor,
        seq_lens_q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        """Pad draft extended query using Triton kernel."""
        return pad_draft_extend_query_triton(
            q,
            padded_q,
            seq_lens_q,
            cu_seqlens_q,
        )

    def unpad_draft_extend_output(
        self,
        raw_out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seq_lens_q: torch.Tensor,
        sum_seq_lens_q: int,
    ) -> torch.Tensor:
        """Unpad draft extended output using Triton kernel."""
        return unpad_draft_extend_output_triton(
            raw_out,
            cu_seqlens_q,
            seq_lens_q,
            sum_seq_lens_q,
            self.unpad_output_buffer,
        )

    def _compute_decode_bmm1_scale(self, layer: RadixAttention) -> float:
        """BMM1 scale q_scale * k_scale * softmax_scale. k_scale only
        applies when the KV cache stores FP8."""
        q_scale = 1.0
        if self.data_type == torch.float8_e4m3fn:
            k_scale = (
                layer.k_scale_float
                if getattr(layer, "k_scale_float", None) is not None
                else 1.0
            )
        else:
            if getattr(layer, "k_scale_float", None) is not None:
                logger.warning_once(
                    "Checkpoint has k_scale but KV cache dtype is not FP8. "
                    "Ignoring k_scale for BMM1 (k_scale=%.4f, kv_dtype=%s).",
                    layer.k_scale_float,
                    self.data_type,
                )
            k_scale = 1.0
        return q_scale * k_scale * layer.scaling

    def _run_decode_kernel(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        layer: RadixAttention,
    ) -> torch.Tensor:
        """Hook for subclasses to swap the decode/spec-verify kernel."""

        # Scale computation for TRTLLM MLA kernel BMM1 operation:
        # The final BMM1 scale is computed as: q_scale * k_scale * softmax_scale
        # Scale components:
        # - q_scale: Query scaling factor (set to 1.0 for both FP16/FP8 paths)
        # - k_scale: Key scaling factor from model checkpoint. Only applied when KV cache
        #   stores FP8-quantized values, to compensate for the quantization scaling.
        #   For BF16/FP16 KV cache, k_scale must be 1.0 since values are unscaled.
        # - softmax_scale: Attention softmax scaling = 1/sqrt(head_dim), pre-computed as layer.scaling
        bmm1_scale = self._compute_decode_bmm1_scale(layer)
        seq_lens_i32 = (
            seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
        )
        extra_kwargs = {"backend": self.backend} if self.backend != "trtllm-gen" else {}
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens_i32,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
            **extra_kwargs,
        )

    def _run_prefill_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        batch_size: int,
        cum_seq_lens_q: torch.Tensor,
        max_q_len: int,
        seq_lens_kv: torch.Tensor,
        cum_seq_lens_kv: torch.Tensor,
        max_kv_len: int,
        is_causal: bool,
        return_lse: bool,
        out_buffer: torch.Tensor,
        o_sf_scale: float = 1.0,
    ):
        """Hook for subclasses to swap the ragged prefill kernel. Q/K/V arrive
        in model-native dtype; subclasses do any kernel-specific quantization.
        Returns the output tensor or (output, lse) if return_lse."""
        q_scale = k_scale = v_scale = 1.0
        if self.data_type == torch.float8_e4m3fn:
            q, k, v, k_scale, v_scale = _quantize_fp8_qkv(q, k, v, layer)
        return flashinfer.prefill.trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=self.workspace_buffer,
            batch_size=batch_size,
            window_left=-1,
            enable_pdl=False,
            max_q_len=max_q_len,
            bmm1_scale=q_scale * k_scale * layer.scaling,
            bmm2_scale=v_scale,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            seq_lens=seq_lens_kv,
            max_kv_len=max_kv_len,
            is_causal=is_causal,
            return_lse=return_lse,
            o_sf_scale=o_sf_scale,
            out=out_buffer,
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR.get(),
        )

    def forward_decode(
        self,
        q: torch.Tensor,  # q_nope
        k: torch.Tensor,  # k_nope
        v: torch.Tensor,  # not used in this backend
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run forward for decode using TRTLLM MLA kernel."""
        merge_query = q_rope is not None
        if self.data_type == torch.float8_e4m3fn:
            # For FP8 path, we quantize the query and rope parts and merge them into a single tensor
            # Note: rope application in deepseek_v2.py:forward_absorb_prepare is skipped for FP8 decode path of this trtllm_mla backend
            assert all(
                x is not None for x in [q_rope, k_rope, cos_sin_cache]
            ), "For FP8 path and using flashinfer.rope.mla_rope_quantize we need all of q_rope, k_rope and cos_sin_cache to be not None."
            q, k, k_rope = mla_quantize_and_rope_for_fp8(
                q,
                q_rope,
                k.squeeze(1),
                k_rope.squeeze(1),
                forward_batch.positions,
                cos_sin_cache,
                is_neox,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            )
            merge_query = False

        # Save KV cache if requested
        if save_kv_cache:
            assert (
                k is not None and k_rope is not None
            ), "For populating trtllm_mla kv cache, both k_nope and k_rope should be not None."
            self.token_to_kv_pool.set_mla_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )

        # Prepare query tensor inline
        if merge_query:
            # For FP16 path, we merge the query and rope parts into a single tensor
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope_reshaped = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            query = concat_mla_absorb_q_general(q_nope, q_rope_reshaped)
        else:
            # For FP8 path, we already have the query and rope parts merged because of the quantize_and_rope_for_fp8 function
            query = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # Apply llama 4 scaling if provided
        if llama_4_scaling is not None:
            query = query.to(self.q_data_type) * llama_4_scaling
            query = query.to(self.data_type)

        # EAGLE tree-draft (topk>1): the per-step draft decode query has batch
        # bs*topk (the tree frontier). The stock metadata builds bs-row block
        # tables, so the cascade reads its own bs*topk metadata instead. Each
        # branch is a linear chain (no tree mask among draft tokens). Chain draft
        # (topk==1) keeps the single-pass path below.
        if (
            self.draft_step_topk > 1
            and forward_batch.forward_mode.is_decode_or_idle()
            and self.draft_step_metadata is not None
        ):
            return self._forward_tree_draft(query, layer)

        # Ensure query has shape [bs, acc_q_len, num_q_heads, head_dim] when seq_len 1
        if query.dim() == 3:
            query = query.unsqueeze(1)

        # Prepare KV cache inline
        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

        # Get metadata
        metadata = (
            getattr(forward_batch, "decode_trtllm_mla_metadata", None)
            or self.forward_decode_metadata
        )

        # Backstop: metadata was built pre-pad (marked) and DP padding then
        # grew the batch. The marker path deliberately does not re-plan
        # post-pad (DSA can't rebuild on a padded batch, see #27091), so this
        # local re-plan catches the size mismatch.
        batch_size = getattr(metadata, "batch_size", None)
        if batch_size is not None and batch_size < forward_batch.batch_size:
            self.init_forward_metadata(forward_batch)
            metadata = forward_batch.decode_trtllm_mla_metadata

        raw_out = self._run_decode_kernel(
            query=query,
            kv_cache=kv_cache,
            block_tables=metadata.block_kv_indices,
            seq_lens=forward_batch.seq_lens,
            max_seq_len=metadata.max_seq_len_k,
            layer=layer,
        )

        # Reshape output directly without slicing
        output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        return output

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if (
            self.forward_prefill_metadata is not None
            and self.forward_prefill_metadata.fallback_to_flashinfer_impl
        ):
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, q_rope, k_rope
            )

        # TODO refactor to avoid code duplication
        merge_query = q_rope is not None
        if (
            self.data_type == torch.float8_e4m3fn
        ) and forward_batch.forward_mode.is_target_verify():
            # For FP8 path, we quantize the query and rope parts and merge them into a single tensor
            # Note: rope application in deepseek_v2.py:forward_absorb_prepare is skipped for FP8 decode path of this trtllm_mla backend
            assert all(
                x is not None for x in [q_rope, k_rope, cos_sin_cache]
            ), "For FP8 path and using flashinfer.rope.mla_rope_quantize we need all of q_rope, k_rope and cos_sin_cache to be not None."
            q, k, k_rope = mla_quantize_and_rope_for_fp8(
                q,
                q_rope,
                k.squeeze(1),
                k_rope.squeeze(1),
                forward_batch.positions,
                cos_sin_cache,
                is_neox,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            )
            merge_query = False

        # Save KV cache if requested
        if save_kv_cache:
            assert (
                k is not None and k_rope is not None
            ), "For populating trtllm_mla kv cache, both k_nope and k_rope should be not None."
            self.token_to_kv_pool.set_mla_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )

        # TODO refactor to avoid code duplication
        # Prepare query tensor inline
        if merge_query:
            # For FP16 path, we merge the query and rope parts into a single tensor
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope_reshaped = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            q = concat_mla_absorb_q_general(q_nope, q_rope_reshaped)

        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # Apply llama 4 scaling if provided
        if llama_4_scaling is not None:
            q = q.to(self.q_data_type) * llama_4_scaling
            q = q.to(self.data_type)

        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            metadata = (
                getattr(forward_batch, "decode_trtllm_mla_metadata", None)
                or self.forward_decode_metadata
            )

            # Backstop: metadata was built pre-pad (marked) and DP padding
            # then grew the batch. The marker path deliberately does not
            # re-plan post-pad (DSA can't rebuild on a padded batch, see
            # #27091), so this local re-plan catches the size mismatch.
            batch_size = getattr(metadata, "batch_size", None)
            if batch_size is not None and batch_size < forward_batch.batch_size:
                self.init_forward_metadata(forward_batch)
                metadata = forward_batch.decode_trtllm_mla_metadata

            # Ensure query has shape [bs, num_draft_tokens, num_q_heads, head_dim]
            bs = forward_batch.batch_size

            k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
            kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

            q = q.to(self.data_type)

            # EAGLE tree-verify (topk>1): the single-pass decode kernel applies a
            # causal-among-Q mask (query i attends every draft token j<=i), which
            # is correct for a linear chain (topk==1) but wrong for a tree, where
            # query i must attend only its tree ancestors. Run the 2-pass cascade
            # instead. Chain verify (topk==1) keeps the single-pass path below.
            if self._is_tree_verify and forward_batch.forward_mode.is_target_verify():
                return self._forward_tree_verify(q, layer, forward_batch, metadata)

            if forward_batch.forward_mode.is_target_verify():
                max_seq_len = (
                    metadata.max_seq_len_k + forward_batch.spec_info.draft_token_num
                )
                # For target_verify, all sequences have the same number of draft tokens
                q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
                needs_unpad = False
            else:
                # draft_extend: handle varying num_correct_drafts_per_req. If total_tokens % bs == 0,
                # we can directly reshape q; otherwise, pad to max_seq_len_q.
                total_tokens = q.shape[0]
                tokens_per_seq = total_tokens // bs if bs > 0 else 0
                can_direct_view = bs > 0 and (total_tokens % bs == 0)

                if can_direct_view:
                    max_seq_len = metadata.max_seq_len_k + tokens_per_seq
                    q = q.view(bs, tokens_per_seq, layer.tp_q_head_num, layer.head_dim)
                    needs_unpad = False
                else:
                    # Varying lengths: pad q to (bs, max_seq_len_q, ...)
                    actual_seq_lens_q = forward_batch.extend_seq_lens
                    actual_max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                    max_seq_len = metadata.max_seq_len_k + actual_max_seq_len_q

                    actual_cu_seqlens_q = torch.nn.functional.pad(
                        torch.cumsum(actual_seq_lens_q, dim=0, dtype=torch.int32),
                        (1, 0),
                    )

                    if self.padded_q_buffer is not None:
                        padded_q = self.padded_q_buffer[
                            :bs, :actual_max_seq_len_q, :, :
                        ].to(dtype=q.dtype)
                        padded_q.zero_()
                    else:
                        padded_q = torch.zeros(
                            (
                                bs,
                                actual_max_seq_len_q,
                                layer.tp_q_head_num,
                                layer.head_dim,
                            ),
                            dtype=q.dtype,
                            device=q.device,
                        )

                    q = self.pad_draft_extend_query(
                        q, padded_q, actual_seq_lens_q, actual_cu_seqlens_q
                    )
                    needs_unpad = True
                    unpad_seq_lens_q = actual_seq_lens_q
                    unpad_cu_seqlens_q = actual_cu_seqlens_q
                    unpad_sum_seq_lens_q = total_tokens

            assert kv_cache.dtype == self.data_type

            raw_out = self._run_decode_kernel(
                query=q,
                kv_cache=kv_cache,
                block_tables=metadata.block_kv_indices,
                seq_lens=metadata.seq_lens_k,
                max_seq_len=max_seq_len,
                layer=layer,
            )

            if needs_unpad:
                # Unpad the output for draft_extend mode with varying lengths
                # Use the actual values computed during padding, not from metadata
                output = self.unpad_draft_extend_output(
                    raw_out,
                    unpad_cu_seqlens_q,
                    unpad_seq_lens_q,
                    unpad_sum_seq_lens_q,
                )
                output = output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            else:
                output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            return output

        if k_rope is not None:
            k = torch.cat([k, k_rope], dim=-1)
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v = v.view(-1, layer.tp_k_head_num, layer.v_head_dim)

        # When chunked prefix cache is enabled, dispatch to different path for ragged attention.
        if forward_batch.attn_attend_prefix_cache:
            # MHA for chunked prefix kv cache when running model with MLA
            assert forward_batch.prefix_chunk_idx is not None
            assert forward_batch.prefix_chunk_cu_seq_lens is not None
            assert q_rope is None
            assert k_rope is None
            chunk_idx = forward_batch.prefix_chunk_idx

            out = torch.empty(
                q.shape[0],
                layer.tp_q_head_num,
                layer.v_head_dim,
                dtype=self.q_data_type,
                device=q.device,
            )
            result = self._run_prefill_kernel(
                q=q,
                k=k,
                v=v,
                layer=layer,
                batch_size=forward_batch.batch_size,
                cum_seq_lens_q=self.forward_prefill_metadata.cum_seq_lens,
                max_q_len=self.forward_prefill_metadata.max_seq_len,
                seq_lens_kv=forward_batch.prefix_chunk_seq_lens[chunk_idx],
                cum_seq_lens_kv=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                max_kv_len=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                is_causal=False,
                return_lse=True,
                out_buffer=out,
                o_sf_scale=-1.0,
            )

            # The TRT-LLM ragged attention cubin kernel does not correctly
            # handle rows with kv_len == 0: it leaves stale data in the
            # workspace softmaxStats buffer and may produce non-zero output
            # for those rows.  Fix up by forcing out=0 and lse=-inf for
            # zero-KV rows so that downstream merge_state ignores them.
            # Skip entirely when this chunk has no zero-KV rows (pure CPU
            # check, precomputed in prepare_chunked_prefix_cache_info).
            if forward_batch.prefix_chunk_has_zero_kv[chunk_idx]:
                out_tensor, lse_tensor = result
                fixup_zero_kv_rows(
                    out_tensor,
                    lse_tensor,
                    forward_batch.prefix_chunk_seq_lens[chunk_idx],
                    self.forward_prefill_metadata.cum_seq_lens,
                    self.forward_prefill_metadata.max_seq_len,
                )

            return result
        else:
            out = torch.empty(
                q.shape[0],
                q.shape[1],
                v.shape[2],
                device=q.device,
                dtype=self.q_data_type,
            )
            return self._run_prefill_kernel(
                q=q,
                k=k,
                v=v,
                layer=layer,
                batch_size=forward_batch.batch_size,
                cum_seq_lens_q=self.forward_prefill_metadata.cum_seq_lens,
                max_q_len=self.forward_prefill_metadata.max_seq_len,
                seq_lens_kv=self.forward_prefill_metadata.seq_lens,
                cum_seq_lens_kv=self.forward_prefill_metadata.cum_seq_lens,
                max_kv_len=self.forward_prefill_metadata.max_seq_len,
                is_causal=True,
                return_lse=forward_batch.mha_return_lse,
                out_buffer=out,
                o_sf_scale=1.0,
            )


class TRTLLMMLAMultiStepDraftBackend(FlashInferMLAMultiStepDraftBackend):
    """Multi-step draft backend for TRT-LLM MLA used by EAGLE.

    Supports EAGLE tree drafting (speculative_eagle_topk > 1): the per-step draft
    decode runs the 2-pass cascade on the cuteDSL MLA decode kernel (see
    ``TRTLLMMLABackend._forward_tree_draft``). The chain path (topk == 1) keeps
    the stock single-pass per-step decode.
    """

    # Per-step draft decode never reads seq_lens_cpu / seq_lens_sum; opt out so
    # decide_needs_cpu_seq_lens' OR over the backends stays False.
    needs_cpu_seq_lens: bool = False

    # Lift the FlashInferMLAMultiStepDraftBackend topk==1 guard: the per-step
    # cuteDSL MLA cascade below handles the tree (topk > 1) draft decode.
    supports_tree_topk: bool = True

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
        backend: str = "trtllm-gen",
    ):
        super().__init__(model_runner, topk, speculative_num_steps)

        # The topk>1 cascade needs return_lse (trtllm-gen rejects it), so force
        # the cuteDSL decode kernel on the per-step backends for tree drafting.
        if topk > 1:
            backend = "cute-dsl"

        for i in range(self.speculative_num_steps - 1):
            per_step = TRTLLMMLABackend(
                model_runner,
                skip_prefill=True,
                kv_indptr_buf=self.kv_indptr[i],
                q_indptr_decode_buf=self.q_indptr_decode,
                backend=backend,
            )
            # Stamp the per-step backend so its forward_decode runs the tree
            # cascade (step i decodes a chain of length i+1).
            per_step.draft_step_id = i
            per_step.draft_step_topk = topk
            per_step.draft_step_num_steps = speculative_num_steps
            self.attn_backends[i] = per_step

    def _build_draft_step_meta(
        self,
        per_step_backend: TRTLLMMLABackend,
        forward_batch: ForwardBatch,
        step_id: int,
    ) -> TRTLLMMLADraftStepMetadata:
        """Build the prefix page table (repeated topk) + draft-path slot table
        for draft ``step_id``. EAGER path (allocates fresh tensors).

        ``forward_batch`` here is pre-loop: ``batch_size == bs``, ``seq_lens ==
        [bs]`` prefix token counts, ``out_cache_loc == [bs*topk*num_steps]``.
        """
        topk = self.topk
        num_steps = self.speculative_num_steps
        bs = forward_batch.batch_size
        device = forward_batch.seq_lens.device

        seq_lens_prefix_bs = forward_batch.seq_lens.to(torch.int32)  # [bs] prefix
        max_seq = int(seq_lens_prefix_bs.max().item())
        max_blocks = per_step_backend._calc_padded_blocks(max_seq)
        block_kv_indices_bs = per_step_backend._create_block_kv_indices(
            bs,
            max_blocks,
            forward_batch.req_pool_indices,
            seq_lens_prefix_bs,
            device,
        )  # [bs, max_blocks]

        # Expand to the bs*topk query layout: row r*topk+t == request r, branch t
        # (spec input_ids == topk_index.flatten(), (bs, topk) row-major). The same
        # prefix page-table row is repeated across the topk branches.
        block_kv_indices_prefix = block_kv_indices_bs.repeat_interleave(topk, dim=0)
        seq_lens_prefix = seq_lens_prefix_bs.repeat_interleave(topk)  # [bs*topk]

        # Draft-path cache slots for steps 0..step_id. EAGLE lays out the draft
        # out_cache_loc as [bs*topk, num_steps] row-major (request r, branch t at
        # row r*topk+t); assert it so a future layout change fails loudly here
        # rather than silently mis-gathering draft KV.
        assert forward_batch.out_cache_loc.numel() == bs * topk * num_steps, (
            f"draft out_cache_loc numel {forward_batch.out_cache_loc.numel()} "
            f"!= bs*topk*num_steps {bs}*{topk}*{num_steps}"
        )
        out_cache_loc = forward_batch.out_cache_loc.view(bs * topk, num_steps)
        decode_length = step_id + 1
        draft_slots = out_cache_loc[:, :decode_length].contiguous()  # [bs*topk, dl]

        return TRTLLMMLADraftStepMetadata(
            block_kv_indices_prefix=block_kv_indices_prefix,
            seq_lens_prefix=seq_lens_prefix,
            max_seq_len_prefix=max_seq,
            draft_slots=draft_slots,
            bs=bs,
            topk=topk,
        )

    def _fill_draft_step_meta_graph(
        self,
        per_step_backend: TRTLLMMLABackend,
        forward_batch: ForwardBatch,
        step_id: int,
    ) -> None:
        """Fill the per-step persistent CUDA-graph buffers in place and point
        ``per_step_backend.draft_step_metadata`` at them.

        ``forward_batch`` is the EAGLE draft batch: ``batch_size == bs``,
        ``seq_lens == [bs]`` prefix token counts, ``out_cache_loc ==
        [bs*topk*num_steps]``. The buffers were allocated once in
        ``init_cuda_graph_state``; only the destination buffers are stable (the
        captured forward_decode bakes their base addresses), the small
        repeat_interleave inputs allocate freely (this runs out-of-graph).
        """
        topk = self.topk
        num_steps = self.speculative_num_steps
        b = per_step_backend
        bs = forward_batch.batch_size
        n = bs * topk

        bufs = b.draft_step_cuda_graph_buffers
        bt_buf = bufs["block_kv_indices_prefix"]  # [max_bs*topk, max_blocks]
        seqlens_buf = bufs["seq_lens_prefix"]  # [max_bs*topk]
        slots_buf = bufs["draft_slots"]  # [max_bs*topk, decode_length]
        max_blocks = bt_buf.shape[1]

        # prefix seq_lens, expanded to bs*topk, filled in place.
        seq_lens_expanded = forward_batch.seq_lens.to(torch.int32).repeat_interleave(
            topk
        )  # [n]
        seqlens_buf[:n].copy_(seq_lens_expanded)

        # prefix page table, expanded to bs*topk, filled in place. Reset the
        # active slice to the no-page sentinel first so stale rows from a larger
        # prior batch never leak into a smaller replay.
        bt_buf[:n].fill_(-1)
        req_pool_expanded = forward_batch.req_pool_indices.repeat_interleave(
            topk
        )  # [n]
        create_flashmla_kv_indices_triton[
            (n, get_num_kv_index_blocks_flashmla(max_blocks, b.page_size))
        ](
            b.req_to_token,
            req_pool_expanded,
            seqlens_buf[:n],
            None,
            bt_buf[:n],
            b.req_to_token.stride(0),
            max_blocks,
            PAGED_SIZE=b.page_size,
        )

        # draft-path cache slots for steps 0..step_id, filled in place. At replay
        # bs may be the padded bucket while out_cache_loc is still the caller's
        # real tensor of length raw_bs*topk*num_steps. Derive raw_bs, fill the
        # real rows, zero the padded tail (padded outputs are discarded).
        decode_length = step_id + 1
        assert forward_batch.out_cache_loc.numel() % (topk * num_steps) == 0, (
            f"draft out_cache_loc numel {forward_batch.out_cache_loc.numel()} "
            f"not divisible by topk*num_steps {topk}*{num_steps}"
        )
        raw_bs = forward_batch.out_cache_loc.numel() // (topk * num_steps)
        raw_n = raw_bs * topk
        out_cache_loc = forward_batch.out_cache_loc.view(raw_n, num_steps)
        slots_buf[:raw_n, :decode_length].copy_(out_cache_loc[:, :decode_length])
        if raw_n < n:
            slots_buf[raw_n:n, :decode_length].zero_()

        b.draft_step_metadata = TRTLLMMLADraftStepMetadata(
            # Stable buffer SLICES (same storage; the captured forward_decode
            # bakes the base address -- slicing [:n] keeps that address).
            block_kv_indices_prefix=bt_buf[:n],
            seq_lens_prefix=seqlens_buf[:n],
            max_seq_len_prefix=b.max_context_len,  # static bound (future longer replays)
            draft_slots=slots_buf[:n, :decode_length],
            bs=bs,
            topk=topk,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if self.topk <= 1:
            for i in range(self.speculative_num_steps - 1):
                self.attn_backends[i].init_forward_metadata(forward_batch)
            return
        # Tree draft (topk>1). forward_batch is pre-loop: out_cache_loc is the
        # full [bs*topk*num_steps] buffer. The cascade reads solely from
        # draft_step_metadata, so we build it directly (the stock per-step init
        # would only build a bs-row block table the cascade ignores).
        for i in range(self.speculative_num_steps - 1):
            b = self.attn_backends[i]
            b.draft_step_metadata = self._build_draft_step_meta(b, forward_batch, i)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        if self.topk > 1:
            # Tree draft. forward_batch is the EAGLE draft batch (batch_size==bs,
            # seq_lens==[bs] prefix, out_cache_loc==[bs*topk*num_steps]) at both
            # capture and replay (replay pads to the captured bs bucket), so the
            # [:n] buffer slices keep their baked addresses.
            for i in range(self.speculative_num_steps - 1):
                b = self.attn_backends[i]
                self._fill_draft_step_meta_graph(b, forward_batch, i)
            return

        if in_capture:
            return super().init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )
        inner_fb = build_inner_fb_view(
            forward_batch,
            bs=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
        )
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_out_graph(inner_fb)
