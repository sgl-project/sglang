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
from sglang.kernels.ops.attention.pad import (
    pad_draft_extend_query as pad_draft_extend_query_triton,
)
from sglang.kernels.ops.attention.pad import (
    unpad_draft_extend_output as unpad_draft_extend_output_triton,
)
from sglang.kernels.ops.attention.utils import (
    concat_mla_absorb_q_general,
    mla_quantize_and_rope_for_fp8,
)
from sglang.kernels.ops.kvcache.kv_indices import (
    create_flashmla_kv_indices_triton,
    get_num_kv_index_blocks_flashmla,
    get_num_page_per_block_flashmla,
)
from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.distributed.parallel_state import get_dcp_group
from sglang.srt.environ import envs
from sglang.srt.layers.attention.flashinfer_mla_backend import (
    FlashInferMLAAttnBackend,
    FlashInferMLAMultiStepDraftBackend,
)
from sglang.srt.layers.dcp import (
    dcp_a2a_exchange_packed,
    dcp_a2a_lse_reduce,
    dcp_a2a_lse_reduce_prepacked,
    dcp_enabled,
    dcp_lse_combine_triton,
    dcp_mask_pack_triton,
    dcp_pass2_causal_attn_triton,
    dcp_unpack_lse_combine,
    get_attention_dcp_rank,
    get_attention_dcp_world_size,
    get_dcp_lens,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import get_buffer, get_parallel, get_server_args
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_flashinfer_available, is_float4_e2m1fn_x2

if is_flashinfer_available():
    import flashinfer

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

logger = logging.getLogger(__name__)

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
    # DCP decode under CUDA graph: captured (constant) local max_seq_len for the
    # tokenspeed kernel. Set at graph capture (over-provisioned to the local
    # length of max_context_len) so forward_decode avoids a host sync (.item())
    # during replay. Left None on the eager path, where forward_decode computes
    # the exact local max directly.
    dcp_local_max_seq_len: Optional[int] = None
    # DCP verify (target_verify) pass-1: LOCAL block table over the rank's strided
    # PREFIX slice (excludes the T draft tokens, which are handled locally from the
    # k/k_rope inputs). dcp_prefix_local_max is the captured constant local prefix
    # max_seq_len for CUDA graph (None on the eager path).
    dcp_prefix_block_kv_indices: Optional[torch.Tensor] = None
    dcp_prefix_local_max: Optional[int] = None
    # DCP verify per-STEP hoists (layer-invariant; previously recomputed in
    # every layer's _forward_verify_dcp call = 61x per step): the rank-local
    # prefix lengths, the zero-owner mask, and the pass-2 block-table/seq
    # constants. Filled by init_forward_metadata (eager) or
    # _apply_cuda_graph_metadata (replay prep).
    dcp_local_prefix: Optional[torch.Tensor] = None
    dcp_zero_mask: Optional[torch.Tensor] = None
    dcp_draft_bt: Optional[torch.Tensor] = None
    dcp_draft_seq: Optional[torch.Tensor] = None


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
            self.workspace_buffer = get_buffer(
                "trtllm_mla_zero_workspace",
                lambda: torch.zeros(
                    self.workspace_size,
                    dtype=torch.int8,
                    device=model_runner.device,
                ),
            )

        # CUDA graph state
        self.decode_cuda_graph_metadata = {}
        self.decode_cuda_graph_kv_indices = None
        self.padded_q_buffer = None
        self.unpad_output_buffer = None
        self.forward_prefill_metadata: Optional[TRTLLMMLAPrefillMetadata] = None
        self.forward_decode_metadata: Union[TRTLLMMLADecodeMetadata, None] = None

        self.disable_chunked_prefix_cache = (
            get_server_args().disable_chunked_prefix_cache
        )

        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.cuda_graph_custom_mask = None
        # Tree-mask scratch is fetched from the target backend only.
        self.is_draft_runner = model_runner.is_draft_worker

        # DCP verify cascade: fused mask+pack / a2a-overlap / pass-1 no-cp /
        # Triton pass-2 A/B flags, cached once (read per layer otherwise).
        # See _forward_verify_dcp.
        self._dcp_fused_pack = envs.SGLANG_DCP_FUSED_PACK.get()
        self._dcp_a2a_overlap = envs.SGLANG_DCP_A2A_OVERLAP.get()
        self._dcp_pass1_no_cp = envs.SGLANG_DCP_PASS1_NO_CP.get()
        self._dcp_triton_pass2 = envs.SGLANG_DCP_TRITON_PASS2.get()
        self._dcp_single_call_verify = envs.SGLANG_DCP_SINGLE_CALL_VERIFY.get()
        # Dedicated side stream (+ fork/join events) for the overlapped verify
        # a2a; created lazily on first use, shared across CUDA graphs.
        self._dcp_comm_stream: Optional[torch.cuda.Stream] = None
        self._dcp_a2a_fork_ev: Optional[torch.cuda.Event] = None
        self._dcp_a2a_join_ev: Optional[torch.cuda.Event] = None
        # Extends the send/recv buffers' lifetime beyond the calling frame:
        # the side stream may still be consuming them when the locals die.
        self._dcp_a2a_keepalive = None

    def _get_dcp_comm_stream(self) -> torch.cuda.Stream:
        """Lazily create the DCP verify-a2a side stream and its fork/join
        events. One stream per backend, reused by every layer and every CUDA
        graph (events are re-recorded per fork/join pair, which is capture-safe
        — same pattern as memory_pool's alt_stream KV write)."""
        if self._dcp_comm_stream is None:
            self._dcp_comm_stream = torch.cuda.Stream()
            self._dcp_a2a_fork_ev = torch.cuda.Event()
            self._dcp_a2a_join_ev = torch.cuda.Event()
        return self._dcp_comm_stream

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
        if dcp_enabled():
            # DCP decode uses a LOCAL block table over (page_size*dcp_world) global
            # pages — fewer, differently-strided columns than the full buffer above.
            # Give it its OWN contiguous buffer: a column-slice of the full buffer
            # would have row stride = full width while the flashmla index kernel is
            # told shape[1], corrupting the table. Full width here == shape[1].
            eff_page = self.page_size * get_attention_dcp_world_size()
            npb = get_num_page_per_block_flashmla(eff_page)
            dcp_blocks = (
                triton.cdiv(
                    triton.cdiv(
                        self.max_context_len + (self.num_draft_tokens or 0), eff_page
                    ),
                    npb,
                )
                * npb
            )
            self.dcp_decode_cuda_graph_kv_indices = torch.full(
                (max_bs, dcp_blocks), -1, dtype=torch.int32, device=self.device
            )
            # Verify-DCP pass-1 prefix block table (its own contiguous buffer, same
            # width; prefix <= max_context_len). Separate from the decode buffer so
            # the decode and target-verify graphs never share index memory.
            self.dcp_verify_prefix_cuda_graph_kv_indices = torch.full(
                (max_bs, dcp_blocks), -1, dtype=torch.int32, device=self.device
            )
        num_tokens_per_req = max_num_tokens // max_bs

        if is_float4_e2m1fn_x2(self.data_type):
            # Buffer for padded query: (max_bs, max_draft_tokens, num_q_heads, v_head_dim)
            self.store_dtype = torch.uint8
            self.padded_q_buffer = torch.zeros(
                (max_bs, num_tokens_per_req // 2, self.num_q_heads, self.kv_cache_dim),
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
                (max_bs, num_tokens_per_req, self.num_q_heads, self.kv_cache_dim),
                dtype=self.data_type,
                device=self.device,
            )

            # Buffer for unpadded output: (max_num_tokens, num_q_heads, v_head_dim)
            self.unpad_output_buffer = torch.zeros(
                (max_num_tokens, self.num_q_heads, 512),
                dtype=self.data_type,
                device=self.device,
            )

        if self.num_draft_tokens and not self.skip_prefill and not self.is_draft_runner:
            # Worst-case FULL_MASK tree-mask scratch (bool); build_tree writes it
            # in-place so the gpu_only path needs no seq_lens_sum.
            self.cuda_graph_custom_mask = torch.zeros(
                max_num_tokens * (self.max_context_len + self.num_draft_tokens),
                dtype=torch.bool,
                device=self.device,
            )

        super().init_cuda_graph_state(max_bs, max_num_tokens, kv_indices_buf)

    def get_verify_buffers_to_fill_after_draft(self):
        return [self.cuda_graph_custom_mask, None]

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        # Nothing to redo after the draft: build_tree writes the tree mask
        # in-place into cuda_graph_custom_mask (exposed via
        # get_verify_buffers_to_fill_after_draft), and the plan-stream verify
        # metadata (seq_lens_k, block tables, DCP prefix tables) depends only
        # on seq_lens, not on draft output. Same contract as the triton and
        # trtllm_mha backends. Without this override, EAGLE-family verify
        # under the overlap plan stream hits the base NotImplementedError.
        pass

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
            num_tokens_per_req = self.num_draft_tokens
            metadata.max_seq_len_q = num_tokens_per_req
            metadata.sum_seq_lens_q = num_tokens_per_req * bs
            metadata.cu_seqlens_q = torch.arange(
                0,
                bs * num_tokens_per_req + 1,
                num_tokens_per_req,
                dtype=torch.int32,
                device=device,
            )
            metadata.seq_lens_q = torch.full(
                (bs,), num_tokens_per_req, dtype=torch.int32, device=device
            )
            metadata.seq_lens_k = torch.zeros((bs,), dtype=torch.int32, device=device)

        # Capture with full width so future longer sequences are safe during replay.
        if dcp_enabled() and forward_mode.is_decode_or_idle():
            # DCP decode: local block table over (page_size*dcp_world)-granular
            # global pages (local page N == global page N). Use the dedicated
            # contiguous DCP buffer at its full width so shape[1] == row stride
            # (the flashmla index kernel is told shape[1]). Capture a constant
            # local max_seq_len (upper bound = ceil(max_context_len / dcp_world))
            # so forward_decode avoids a host sync at replay.
            dcp_world = get_attention_dcp_world_size()
            block_kv_indices = self.dcp_decode_cuda_graph_kv_indices[:bs]
            metadata.dcp_local_max_seq_len = max(
                (self.max_context_len + dcp_world - 1) // dcp_world, 1
            )
        else:
            max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)
            block_kv_indices = self.decode_cuda_graph_kv_indices[
                :bs, :max_blocks_per_seq
            ]
        metadata.block_kv_indices = block_kv_indices
        metadata.max_seq_len_k = self.max_context_len

        if dcp_enabled() and forward_mode.is_target_verify():
            # Verify-DCP: capture the LOCAL prefix block table (its own contiguous
            # buffer, full width so shape[1] == row stride) + a constant local
            # prefix max so the cascade avoids a host sync at replay.
            dcp_world = get_attention_dcp_world_size()
            metadata.dcp_prefix_block_kv_indices = (
                self.dcp_verify_prefix_cuda_graph_kv_indices[:bs]
            )
            metadata.dcp_prefix_local_max = max(
                (self.max_context_len + (self.num_draft_tokens or 0) + dcp_world - 1)
                // dcp_world,
                1,
            )
            # Persistent per-step hoist buffers for the verify cascade. The
            # graph captures these tensors' ADDRESSES, so replay-prep must
            # write them IN PLACE (see _apply_cuda_graph_metadata); the pass-2
            # block table / seq-len constants are truly static per bs.
            metadata.dcp_local_prefix = torch.zeros(
                (bs,), dtype=torch.int32, device=device
            )
            metadata.dcp_zero_mask = torch.zeros((bs,), dtype=torch.bool, device=device)
            metadata.dcp_draft_bt = torch.arange(
                bs, dtype=torch.int32, device=device
            ).view(bs, 1)
            metadata.dcp_draft_seq = torch.full(
                (bs,), self.num_draft_tokens, dtype=torch.int32, device=device
            )

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
            # Intentional int64 -> int32 same-kind out= downcast.
            torch.add(seq_lens[:bs], self.num_draft_tokens, out=metadata.seq_lens_k)
            seq_lens = metadata.seq_lens_k
        elif forward_mode.is_draft_extend_v2():
            num_tokens_per_req = self.num_draft_tokens
            metadata.max_seq_len_q = num_tokens_per_req
            metadata.sum_seq_lens_q = num_tokens_per_req * bs
            seq_lens = seq_lens[:bs]
            metadata.seq_lens_k.copy_(seq_lens)

        # Update block indices for new sequences. DCP decode builds the LOCAL
        # block table over global pages of size page_size*dcp_world (seq_lens is
        # global here for decode); the resulting page indices address the rank's
        # compacted local pool (local page N == global page N).
        paged_size = self.page_size
        if dcp_enabled() and forward_mode.is_decode_or_idle():
            paged_size = self.page_size * get_attention_dcp_world_size()
        create_flashmla_kv_indices_triton[
            (
                bs,
                get_num_kv_index_blocks_flashmla(
                    metadata.block_kv_indices.shape[1], paged_size
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
            PAGED_SIZE=paged_size,
        )

        if dcp_enabled() and forward_mode.is_target_verify():
            # Verify-DCP pass-1: rebuild the LOCAL prefix block table over the
            # rank's strided slice (page_size*dcp_world global pages). seq_lens is
            # prefix + num_draft_tokens here (line above), so prefix = seq_lens_k - T.
            eff_page = self.page_size * get_attention_dcp_world_size()
            if self._dcp_single_call_verify:
                # B2 single-call: the dcp_prefix_* fields carry TOTAL (prefix+T)
                # semantics — table over seq_lens_k, local lens = local(prefix+T).
                prefix_lens = metadata.seq_lens_k[:bs].to(torch.int32)
            else:
                prefix_lens = (metadata.seq_lens_k[:bs] - self.num_draft_tokens).to(
                    torch.int32
                )
            create_flashmla_kv_indices_triton[
                (
                    bs,
                    get_num_kv_index_blocks_flashmla(
                        metadata.dcp_prefix_block_kv_indices.shape[1], eff_page
                    ),
                )
            ](
                self.req_to_token,
                req_pool_indices[:bs],
                prefix_lens,
                None,
                metadata.dcp_prefix_block_kv_indices,
                self.req_to_token.stride(0),
                metadata.dcp_prefix_block_kv_indices.shape[1],
                PAGED_SIZE=eff_page,
            )
            # Refresh the per-step hoist buffers IN PLACE (captured addresses).
            metadata.dcp_local_prefix.copy_(
                get_dcp_lens(
                    prefix_lens,
                    get_attention_dcp_world_size(),
                    get_attention_dcp_rank(),
                )
            )
            torch.eq(metadata.dcp_local_prefix, 0, out=metadata.dcp_zero_mask)

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        """Get the fill value for sequence lengths in CUDA graph."""
        return 1

    def init_mha_chunk_metadata(
        self, forward_batch: ForwardBatch, disable_flashinfer_ragged: bool = False
    ) -> None:
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

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize the metadata for a forward pass."""
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
            # Never read max_seq from the GPU tensor (.max().item() blocks the
            # host on the stream backlog); max_seq only sizes the block table /
            # scheduling hint, so the static context bound is a safe fallback.
            if getattr(forward_batch, "seq_lens_cpu", None) is not None:
                max_seq = forward_batch.seq_lens_cpu.max().item()
            else:
                max_seq = self.max_context_len

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

            if dcp_enabled() and forward_batch.forward_mode.is_decode_or_idle():
                # DCP decode: build a LOCAL block table over this rank's compacted
                # strided KV slice. The KV allocator uses (page_size*dcp_world)-
                # aligned global pages and stores each owned token at slot//world,
                # so the local physical page index equals slot//(page_size*world)
                # (local page N == global page N). Reinterpreting the flashmla
                # index kernel at PAGED_SIZE=page_size*dcp_world over the GLOBAL
                # positions yields exactly those local page indices. seq_lens here
                # is global; the kernel is fed LOCAL seq_lens in forward_decode.
                dcp_world = get_attention_dcp_world_size()
                eff_page = self.page_size * dcp_world
                npb = get_num_page_per_block_flashmla(eff_page)
                max_local_blocks = (
                    triton.cdiv(triton.cdiv(int(max_seq), eff_page), npb) * npb
                )
                block_kv_indices = torch.full(
                    (bs, max_local_blocks),
                    -1,
                    dtype=torch.int32,
                    device=seq_lens.device,
                )
                create_flashmla_kv_indices_triton[
                    (bs, get_num_kv_index_blocks_flashmla(max_local_blocks, eff_page))
                ](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    seq_lens,
                    None,
                    block_kv_indices,
                    self.req_to_token.stride(0),
                    max_local_blocks,
                    PAGED_SIZE=eff_page,
                )
            else:
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

            if dcp_enabled() and forward_batch.forward_mode.is_target_verify():
                # DCP verify pass-1 attends the PREFIX (positions before the T
                # draft tokens) over this rank's strided slice. Build a LOCAL
                # PREFIX block table the same way decode does, but over the GLOBAL
                # PREFIX lengths (forward_batch.seq_lens, i.e. seq_lens_k - T). The
                # drafts are handled locally in forward_extend from the k/k_rope
                # inputs, not from the sharded pool.
                dcp_world = get_attention_dcp_world_size()
                eff_page = self.page_size * dcp_world
                npb = get_num_page_per_block_flashmla(eff_page)
                if self._dcp_single_call_verify:
                    # B2 single-call: TOTAL (prefix+T) table + lens semantics.
                    prefix_max = max(int(max_seq), 1)
                    prefix_lens = (forward_batch.seq_lens + self.num_draft_tokens).to(
                        torch.int32
                    )
                else:
                    prefix_max = max(int(max_seq) - self.num_draft_tokens, 1)
                    prefix_lens = forward_batch.seq_lens.to(torch.int32)
                max_prefix_blocks = (
                    triton.cdiv(triton.cdiv(prefix_max, eff_page), npb) * npb
                )
                dcp_prefix_bt = torch.full(
                    (bs, max_prefix_blocks),
                    -1,
                    dtype=torch.int32,
                    device=seq_lens.device,
                )
                create_flashmla_kv_indices_triton[
                    (
                        bs,
                        get_num_kv_index_blocks_flashmla(max_prefix_blocks, eff_page),
                    )
                ](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    prefix_lens,
                    None,
                    dcp_prefix_bt,
                    self.req_to_token.stride(0),
                    max_prefix_blocks,
                    PAGED_SIZE=eff_page,
                )
                self.forward_decode_metadata.dcp_prefix_block_kv_indices = dcp_prefix_bt
                # Per-STEP hoists for the verify cascade (layer-invariant;
                # previously rebuilt 61x per step inside _forward_verify_dcp).
                md = self.forward_decode_metadata
                md.dcp_local_prefix = get_dcp_lens(
                    prefix_lens, dcp_world, get_attention_dcp_rank()
                ).to(torch.int32)
                md.dcp_zero_mask = md.dcp_local_prefix == 0
                md.dcp_draft_bt = torch.arange(
                    bs, dtype=torch.int32, device=seq_lens.device
                ).view(bs, 1)
                md.dcp_draft_seq = torch.full(
                    (bs,),
                    self.num_draft_tokens,
                    dtype=torch.int32,
                    device=seq_lens.device,
                )

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
        return_lse: bool = False,
        cp_world: int = 1,
        cp_rank: int = 0,
        causal_seqs: Optional[torch.Tensor] = None,
        causal_mask: Optional[bool] = None,
    ) -> torch.Tensor:
        """Hook for subclasses to swap the decode/spec-verify kernel.

        The DCP params (``return_lse``/``cp_world``/``cp_rank``/``causal_seqs``/
        ``causal_mask``) are honored only by subclasses whose kernel supports
        decode context parallelism (tokenspeed_mla). The base TRT-LLM kernel
        cannot emit the per-rank LSE the cross-rank merge needs, so DCP is
        rejected here; ``causal_mask`` is ignored (TRT-LLM applies its own).
        """
        if return_lse or cp_world > 1:
            raise NotImplementedError(
                "trtllm_mla decode kernel does not support decode context "
                "parallelism (needs per-rank LSE, which trtllm_batch_decode_mla "
                "cannot emit); use --attention-backend tokenspeed_mla for DCP."
            )

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

        if forward_batch.forward_mode.is_decode() and dcp_enabled():
            # DCP: this rank holds a compacted strided KV slice (block_kv_indices
            # were built over global pages in init_forward_metadata, see the DCP
            # branch there). Each rank runs full-head Q over its local shard and
            # returns base-2 LSE; forward_mla merges across ranks via
            # dcp_a2a_lse_reduce. seq_lens/max_seq_len are LOCAL, causal_seqs is
            # the per-request GLOBAL bound (kernel derives the local cutoff).
            dcp_world = get_attention_dcp_world_size()
            dcp_rank = get_attention_dcp_rank()
            global_seq_lens = forward_batch.seq_lens.to(torch.int32)
            local_seq_lens = get_dcp_lens(global_seq_lens, dcp_world, dcp_rank).to(
                torch.int32
            )
            # A request this rank owns no tokens for (local seq_len == 0, e.g. every
            # seq_len < dcp_world in the seq_len=1 decode warmup) makes the kernel
            # emit NaN/garbage + lse 0 for that row (it does not crash as long as
            # max_seq_len >= 1). Clamp max_seq_len to >=1, then mask those rows to a
            # zero partial (-inf base-2 LSE) so dcp_a2a_lse_reduce weights them out.
            # torch.where keeps this data-independent (no host sync / branch) so it
            # is CUDA-graph safe. At least rank 0 owns position 0, so every request
            # still has a real contributor across ranks.
            # Under CUDA graph, max_seq_len is a captured constant from metadata
            # (no host sync at replay); eager computes the exact local max.
            local_max = metadata.dcp_local_max_seq_len
            if local_max is None:
                local_max = (
                    int(local_seq_lens.max().item()) if local_seq_lens.numel() else 0
                )
            out, lse = self._run_decode_kernel(
                query=query,
                kv_cache=kv_cache,
                block_tables=metadata.block_kv_indices,
                seq_lens=local_seq_lens,
                max_seq_len=max(local_max, 1),
                layer=layer,
                return_lse=True,
                cp_world=dcp_world,
                cp_rank=dcp_rank,
                causal_seqs=global_seq_lens,
            )
            # out [B, q_len=1, H, kv_lora], lse [B, q_len=1, H].
            zero_owned = local_seq_lens == 0
            out = torch.where(zero_owned.view(-1, 1, 1, 1), out.new_zeros(()), out)
            lse = torch.where(
                zero_owned.view(-1, 1, 1), lse.new_full((), float("-inf")), lse
            )
            output = out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            # LSE [B, q_len=1, H] -> [B, H] for the cross-rank merge.
            lse = lse.reshape(output.shape[0], layer.tp_q_head_num)
            return output, lse

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

    def _forward_verify_dcp(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        k_rope: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        metadata: TRTLLMMLADecodeMetadata,
    ) -> torch.Tensor:
        """DCP target-verify via the validated 2-pass cascade.

        q: [bs, T, H_full, head_dim] fp8 (gathered heads, H_full =
        num_local_heads*dcp_world). k/k_rope: the T draft tokens' latent
        (fp8, [bs*T, 1, kv_lora_rank] + [bs*T, 1, qk_rope_head_dim]).

        Pass-1 folds the NON-causal prefix over this rank's strided slice
        (return_lse) and a2a-merges across ranks; pass-2 folds the CAUSAL
        draft chain LOCALLY (drafts arrive non-sharded via k/k_rope); the two
        partials are combined by base-2 online softmax. Returns the final
        [bs*T, H_local*kv_lora_rank]. See dcp-tokenspeed-contract.md §5 (probes
        883122 rel 0.011, 883458 rel 0.009).
        """
        dcp_world = get_attention_dcp_world_size()
        dcp_rank = get_attention_dcp_rank()
        bs, T, H_full = q.shape[0], q.shape[1], layer.tp_q_head_num
        H_local = H_full // dcp_world
        N = bs * T
        vd = layer.v_head_dim

        # ---- Pass-1: non-causal prefix fold over the local strided slice ----
        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)
        global_prefix = forward_batch.seq_lens.to(
            torch.int32
        )  # prefix = seq_lens_k - T
        # Layer-invariant values hoisted to per-step metadata (fallback keeps
        # the old inline path for callers without the hoist).
        if metadata.dcp_local_prefix is not None:
            local_prefix = metadata.dcp_local_prefix
        else:
            local_prefix = get_dcp_lens(global_prefix, dcp_world, dcp_rank).to(
                torch.int32
            )
        local_prefix_max = metadata.dcp_prefix_local_max
        if local_prefix_max is None:
            local_prefix_max = (
                int(local_prefix.max().item()) if local_prefix.numel() else 0
            )
        # Pass-1 is NON-causal, and in the tokenspeed fp8 decode kernel every
        # cp_world / K_causal(=causal_seqs) use sits inside a
        # `const_expr(is_causal)` branch (the per-token causal k_bound
        # arithmetic in mla_decode_fp8.py), so under causal_mask=False the cp
        # path is dead code: the kernel attends to exactly seq_lens[b] local
        # tokens either way (block table / seq_lens are already this rank's
        # LOCAL compacted slice; cp_world never touches KV indexing). Default
        # drops the cp args to compile the plain non-causal variant;
        # SGLANG_DCP_PASS1_NO_CP=0 restores the cp_world call for A/B.
        single_call = self._dcp_single_call_verify
        if single_call:
            # ---- B2 single-call CP-causal fold (SGLANG_DCP_SINGLE_CALL_VERIFY) ----
            # One fold over the ENTIRE local slice (owned draft latents included:
            # the owner-rule write at the top of forward_extend is same-stream
            # ordered before this read). The kernel resolves each q_tok's causal
            # bound in GLOBAL coordinates from causal_seqs = prefix + T, BEFORE
            # the cp divide. local_prefix / the block table carry TOTAL semantics
            # under this flag (see the gated metadata build sites).
            o1, lse1 = self._run_decode_kernel(
                query=q,
                kv_cache=kv_cache,
                block_tables=metadata.dcp_prefix_block_kv_indices,
                seq_lens=local_prefix,
                max_seq_len=max(local_prefix_max, 1),
                layer=layer,
                return_lse=True,
                cp_world=dcp_world,
                cp_rank=dcp_rank,
                causal_seqs=global_prefix + self.num_draft_tokens,
                causal_mask=True,
            )
        elif self._dcp_pass1_no_cp:
            pass1_cp_world, pass1_cp_rank, pass1_causal_seqs = 1, 0, None
        else:
            pass1_cp_world, pass1_cp_rank, pass1_causal_seqs = (
                dcp_world,
                dcp_rank,
                global_prefix,
            )
        if not single_call:
            o1, lse1 = self._run_decode_kernel(
                query=q,
                kv_cache=kv_cache,
                block_tables=metadata.dcp_prefix_block_kv_indices,
                seq_lens=local_prefix,
                max_seq_len=max(local_prefix_max, 1),
                layer=layer,
                return_lse=True,
                cp_world=pass1_cp_world,
                cp_rank=pass1_cp_rank,
                causal_seqs=pass1_causal_seqs,
                causal_mask=False,
            )
        # o1 [bs, T, H_full, vd], lse1 [bs, T, H_full]. Mask requests this rank
        # owns no prefix tokens for (local prefix len == 0).
        zero = (
            metadata.dcp_zero_mask
            if metadata.dcp_zero_mask is not None
            else local_prefix == 0
        )
        comm_backend = get_global_server_args().dcp_comm_backend
        # Fused mask+pack (SGLANG_DCP_FUSED_PACK=0 reverts): one Triton kernel
        # builds the a2a send buffer straight from the strided pass-1 outputs,
        # replacing 2x torch.where + 2x .contiguous() + the pack copies. Only
        # the a2a backend consumes the packed layout (fi_a2a packs its own).
        use_fused_pack = self._dcp_fused_pack and comm_backend == "a2a"
        overlap_a2a = use_fused_pack and self._dcp_a2a_overlap and not single_call
        recv_buf = None
        if use_fused_pack:
            send_buf = dcp_mask_pack_triton(o1, lse1, zero, dcp_world)
            if overlap_a2a:
                # Overlap (SGLANG_DCP_A2A_OVERLAP=0 serializes): launch the
                # exchange on a dedicated side stream so the NCCL a2a runs
                # concurrently with pass-2 below; the combine joins after.
                # Event fork/join is CUDA-graph-capture safe (same pattern as
                # memory_pool's alt_stream KV write). recv_buf is allocated on
                # the CURRENT stream so all captured allocations stay on the
                # capture stream; the side stream only touches existing
                # buffers, whose lifetime the backend keepalive extends.
                recv_buf = torch.empty_like(send_buf)
                comm_stream = self._get_dcp_comm_stream()
                self._dcp_a2a_fork_ev.record(torch.cuda.current_stream())
                comm_stream.wait_event(self._dcp_a2a_fork_ev)
                with torch.cuda.stream(comm_stream):
                    dcp_a2a_exchange_packed(send_buf, recv_buf, get_dcp_group())
                self._dcp_a2a_join_ev.record(comm_stream)
                self._dcp_a2a_keepalive = (send_buf, recv_buf)
                prefix_full = prefix_lse = None
            else:
                prefix_full, prefix_lse = dcp_a2a_lse_reduce_prepacked(
                    send_buf,
                    vd,
                    get_dcp_group(),
                    is_lse_base_on_e=False,
                    return_lse=True,
                )  # prefix_full [N, H_local, vd], prefix_lse [N, H_local]
        else:
            o1 = torch.where(zero.view(bs, 1, 1, 1), o1.new_zeros(()), o1)
            lse1 = torch.where(
                zero.view(bs, 1, 1), lse1.new_full((), float("-inf")), lse1
            )
            prefix_full, prefix_lse = dcp_a2a_lse_reduce(
                o1.reshape(N, H_full, vd).contiguous(),
                lse1.reshape(N, H_full).contiguous(),
                get_dcp_group(),
                is_lse_base_on_e=False,
                comm_backend=comm_backend,
                return_lse=True,
            )  # prefix_full [N, H_local, vd], prefix_lse [N, H_local]

        if single_call:
            # B2: the a2a-combined result IS the final output — the draft
            # tokens were inside the fold; no pass-2, no second combine.
            return prefix_full.reshape(N, H_local * vd)

        # ---- Pass-2: local causal-chain draft fold (drafts NOT sharded) ----
        q_local = q[
            :, :, dcp_rank * H_local : (dcp_rank + 1) * H_local, :
        ]  # [bs, T, H_local, head_dim] strided view
        if metadata.dcp_draft_seq is not None:
            draft_seq = metadata.dcp_draft_seq
        else:
            draft_seq = torch.full((bs,), T, dtype=torch.int32, device=q.device)
        if self._dcp_triton_pass2:
            # Tiny Triton causal kernel (SGLANG_DCP_TRITON_PASS2=0 reverts):
            # the T<=8-token chain fold is far below the tokenspeed kernel's
            # launch/tiling floor, and reading k/k_rope directly also drops the
            # per-layer draft page-pool build (cat + zeros + copy) and the
            # q_local .contiguous(). Scales mirror the tokenspeed backend's
            # _run_decode_kernel fp8-KV convention (softmax_scale =
            # layer.scaling * k_scale, output_scale = k_scale); out is bf16 and
            # lse fp32 base-2, the exact contract of the call it replaces.
            k_scale = getattr(layer, "k_scale_float", None)
            if k_scale is None:
                k_scale = 1.0
            o2, lse2 = dcp_pass2_causal_attn_triton(
                q_local,
                k.reshape(N, self.kv_lora_rank),
                k_rope.reshape(N, self.qk_rope_head_dim),
                draft_seq,
                softmax_scale=float(layer.scaling) * float(k_scale),
                output_scale=float(k_scale),
            )  # o2 [bs, T, H_local, vd], lse2 [bs, T, H_local]
        else:
            q_local = q_local.contiguous()
            draft_latent = torch.cat(
                [
                    k.reshape(N, self.kv_lora_rank),
                    k_rope.reshape(N, self.qk_rope_head_dim),
                ],
                dim=-1,
            )  # [N, kv_cache_dim] fp8
            draft_pool = draft_latent.new_zeros(bs, self.page_size, self.kv_cache_dim)
            draft_pool[:, :T, :] = draft_latent.reshape(bs, T, self.kv_cache_dim)
            draft_pool = draft_pool.unsqueeze(1)  # [bs, 1, page_size, kv_cache_dim]
            if metadata.dcp_draft_bt is not None:
                draft_bt = metadata.dcp_draft_bt
            else:
                draft_bt = torch.arange(bs, dtype=torch.int32, device=q.device).view(
                    bs, 1
                )
            o2, lse2 = self._run_decode_kernel(
                query=q_local,
                kv_cache=draft_pool,
                block_tables=draft_bt,
                seq_lens=draft_seq,
                max_seq_len=T,
                layer=layer,
                return_lse=True,
                cp_world=1,
                cp_rank=0,
                causal_seqs=None,
                causal_mask=True,
            )  # o2 [bs, T, H_local, vd], lse2 [bs, T, H_local]

        if overlap_a2a:
            # Join: pass-2 was issued while the side stream ran the a2a; gate
            # the recv-side unpack+combine (current stream) on its completion.
            torch.cuda.current_stream().wait_event(self._dcp_a2a_join_ev)
            prefix_full, prefix_lse = dcp_unpack_lse_combine(
                recv_buf,
                vd,
                is_lse_base_on_e=False,
                return_lse=True,
            )  # prefix_full [N, H_local, vd], prefix_lse [N, H_local]

        # ---- Combine prefix_full ⊕ pass-2 (base-2 online softmax) ----
        recv_out = torch.stack(
            [prefix_full, o2.reshape(N, H_local, vd)], dim=0
        )  # [2, N, H_local, vd]
        recv_lse = torch.stack(
            [prefix_lse, lse2.reshape(N, H_local)], dim=0
        )  # [2, N, H_local]
        final, _ = dcp_lse_combine_triton(recv_out, recv_lse, is_lse_base_on_e=False)
        return final.reshape(-1, H_local * vd)

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

            if forward_batch.forward_mode.is_target_verify():
                max_seq_len = (
                    metadata.max_seq_len_k + forward_batch.spec_info.draft_token_num
                )
                # For target_verify, all sequences have the same number of draft tokens
                q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
                needs_unpad = False
                if dcp_enabled():
                    # DCP verify: the single-fold kernel can't express the
                    # per-draft-token causal under a strided KV slice, so run the
                    # 2-pass cascade (non-causal DCP prefix fold + local causal
                    # draft fold + base-2 combine). Returns the final output.
                    return self._forward_verify_dcp(
                        q, k, k_rope, layer, forward_batch, metadata
                    )
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
    """Multi-step draft backend for TRT-LLM MLA used by EAGLE."""

    # Per-step draft decode never reads seq_lens_cpu / seq_lens_sum; opt out so
    # decide_needs_cpu_seq_lens' OR over the backends stays False.
    needs_cpu_seq_lens: bool = False

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
        backend: str = "trtllm-gen",
    ):
        super().__init__(model_runner, topk, speculative_num_steps)

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i] = TRTLLMMLABackend(
                model_runner,
                skip_prefill=True,
                kv_indptr_buf=self.kv_indptr[i],
                q_indptr_decode_buf=self.q_indptr_decode,
                backend=backend,
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

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
