"""
Attention backend for the flashinfer cute-dsl MLA decode kernels with decode
context parallelism (DCP).

Subclasses :class:`TRTLLMMLABackend` (``backend="cute-dsl"``) to reuse its MLA
data preparation, workspace, and prefill plumbing. The flashinfer cute-dsl
monolithic MLA decode kernel natively accepts cyclic DCP metadata
(``enable_dcp`` / ``cp_world`` / ``cp_rank`` / ``causal_seqlens_kv_global``) and
returns the rank-local ``(out, lse)`` needed by the cross-rank merge in
``deepseek_common/attention_forward_methods/forward_mla.py``.

Non-DCP (``dcp_size == 1``) decode falls through to the base cute-dsl path
unchanged. The DCP metadata helpers below are intentionally duplicated from
:mod:`tokenspeed_mla_backend` (they are kernel-agnostic) so that TokenSpeed
stays untouched; both should collapse into the base once the cute-dsl decode
path is stable (see the TODO in tokenspeed_mla_backend.py).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.dcp_kernels import (
    create_mla_kv_page_table_for_dcp,
)
from sglang.kernels.ops.attention.fixup_zero_kv import fixup_zero_kv_rows
from sglang.kernels.ops.attention.utils import (
    concat_mla_absorb_q_general,
    mla_quantize_and_rope_for_fp8,
    mla_quantize_without_rope_for_fp8,
)
from sglang.kernels.ops.kvcache.kv_indices import (
    get_num_kv_index_blocks_flashmla,
    get_num_page_per_block_flashmla,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLAMultiStepDraftBackend,
)
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_flashinfer_available

if is_flashinfer_available():
    import flashinfer

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# The flashinfer cute-dsl MLA decode kernel returns a natural-log (base-e) LSE,
# whereas sglang's DCP cross-rank merge (forward_mla: dcp_a2a_lse_reduce /
# cp_lse_ag_out_rs_mla) assumes the FlashInfer-MLA/FlashMLA base-2 convention
# (is_lse_base_on_e=False). Multiplying a natural-log LSE by log2(e) rebases it
# to base-2 (the softmax output is base-invariant; only the LSE value changes).
# CONFIRMED base-e (not base-2), so this rebase is required, not optional:
# the flashinfer-dcp-backport public-API unit test asserts the public
# trtllm_batch_decode_with_kv_cache_mla LSE against a torch.logsumexp
# (natural-log) reference at atol=1e-2 and passes (a base-2 LSE would be
# off by 1/ln2 ~= 44%). GPU job 467640:
# tests/attention/test_cute_dsl_mla_dcp*.py 27/27 + 17/17 pass.
_LSE_BASE2_FROM_NATURAL_LOG = math.log2(math.e)


class CuteDslMLABackend(TRTLLMMLABackend):
    """flashinfer cute-dsl MLA decode backend with decode context parallelism."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model_runner,
            skip_prefill,
            kv_indptr_buf,
            q_indptr_decode_buf,
            backend="cute-dsl",
        )

    # ------------------------------------------------------------------
    # DCP metadata (rank-local KV lengths + page table).
    # Duplicated from TokenspeedMLABackend — kernel-agnostic, keyed only on
    # dcp_size / dcp_rank / page_size / req_to_token.
    # ------------------------------------------------------------------
    def _get_dcp_local_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        parallel = get_parallel()
        if not parallel.dcp_enabled:
            return seq_lens
        return get_dcp_lens(seq_lens, parallel.dcp_size, parallel.dcp_rank).to(
            torch.int32
        )

    def _get_dcp_local_max_seq_len(self, max_seq_len: int) -> int:
        parallel = get_parallel()
        if not parallel.dcp_enabled:
            return max_seq_len
        local_max = max_seq_len // parallel.dcp_size + int(
            parallel.dcp_rank < max_seq_len % parallel.dcp_size
        )
        # A positive scheduling bound is required even when every sequence in a
        # padded graph row is empty on this rank.
        return max(local_max, 1)

    def _fill_dcp_block_kv_indices(
        self,
        block_kv_indices: torch.Tensor,
        req_pool_indices: torch.Tensor,
        local_seq_lens: torch.Tensor,
    ) -> None:
        parallel = get_parallel()
        pages_per_block = get_num_page_per_block_flashmla(self.page_size)
        create_mla_kv_page_table_for_dcp[
            (
                block_kv_indices.shape[0],
                get_num_kv_index_blocks_flashmla(
                    block_kv_indices.shape[1], self.page_size
                ),
            )
        ](
            self.req_to_token,
            req_pool_indices,
            local_seq_lens,
            block_kv_indices,
            self.req_to_token.stride(0),
            block_kv_indices.stride(0),
            PHYSICAL_PAGE_SIZE=self.page_size,
            DCP_SIZE=parallel.dcp_size,
            DCP_RANK=parallel.dcp_rank,
            PAGES_PER_BLOCK=pages_per_block,
        )

    def _create_block_kv_indices(
        self,
        batch_size: int,
        max_blocks: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if not get_parallel().dcp_enabled:
            return super()._create_block_kv_indices(
                batch_size,
                max_blocks,
                req_pool_indices,
                seq_lens,
                device,
            )
        block_kv_indices = torch.full(
            (batch_size, max_blocks), -1, dtype=torch.int32, device=device
        )
        self._fill_dcp_block_kv_indices(
            block_kv_indices,
            req_pool_indices,
            self._get_dcp_local_seq_lens(seq_lens),
        )
        return block_kv_indices

    def _init_cuda_graph_metadata(
        self,
        bs: int,
        num_tokens: int,
        forward_mode,
        seq_lens: torch.Tensor,
        device: torch.device,
    ):
        super()._init_cuda_graph_metadata(
            bs, num_tokens, forward_mode, seq_lens, device
        )
        if get_parallel().dcp_enabled:
            if forward_mode.is_target_verify():
                self.forward_decode_metadata.global_seq_lens_k = torch.zeros_like(
                    self.forward_decode_metadata.seq_lens_k
                )
            self.forward_decode_metadata.max_seq_len_k = (
                self._get_dcp_local_max_seq_len(
                    self.max_context_len
                    + (self.num_draft_tokens if forward_mode.is_target_verify() else 0)
                )
            )

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode,
    ):
        if not get_parallel().dcp_enabled:
            return super()._apply_cuda_graph_metadata(
                bs,
                req_pool_indices,
                seq_lens,
                forward_mode,
            )

        metadata = self.decode_cuda_graph_metadata[bs]
        if forward_mode.is_target_verify():
            torch.add(
                seq_lens[:bs],
                self.num_draft_tokens,
                out=metadata.global_seq_lens_k,
            )
            metadata.seq_lens_k.copy_(
                self._get_dcp_local_seq_lens(metadata.global_seq_lens_k)
            )
            local_seq_lens = metadata.seq_lens_k
        elif forward_mode.is_draft_extend_v2():
            num_tokens_per_req = self.num_draft_tokens
            metadata.max_seq_len_q = num_tokens_per_req
            metadata.sum_seq_lens_q = num_tokens_per_req * bs
            seq_lens = seq_lens[:bs]
            metadata.seq_lens_k.copy_(seq_lens)
            local_seq_lens = self._get_dcp_local_seq_lens(seq_lens)
        else:
            seq_lens = seq_lens[:bs]
            local_seq_lens = self._get_dcp_local_seq_lens(seq_lens)

        self._fill_dcp_block_kv_indices(
            metadata.block_kv_indices,
            req_pool_indices[:bs],
            local_seq_lens,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        if (
            get_parallel().dcp_enabled
            and self.forward_decode_metadata is not None
            and (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend_v2()
            )
        ):
            if forward_batch.forward_mode.is_target_verify():
                metadata = self.forward_decode_metadata
                metadata.global_seq_lens_k = metadata.seq_lens_k
                metadata.seq_lens_k = self._get_dcp_local_seq_lens(
                    metadata.global_seq_lens_k
                )
            self.forward_decode_metadata.max_seq_len_k = (
                self._get_dcp_local_max_seq_len(
                    self.forward_decode_metadata.max_seq_len_k
                )
            )

    # ------------------------------------------------------------------
    # Kernel + decode forward.
    # ------------------------------------------------------------------
    def _run_decode_kernel(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        layer: RadixAttention,
        *,
        causal_seqs: Optional[torch.Tensor] = None,
        cp_world: int = 1,
        cp_rank: int = 0,
        return_lse: bool = False,
    ):
        """Call the flashinfer cute-dsl MLA decode kernel.

        Without DCP (``cp_world <= 1``) this defers to the base cute-dsl path.
        With DCP, ``seq_lens`` are this rank's cyclic-local KV lengths and
        ``causal_seqs`` the global per-request KV lengths; the kernel returns a
        rank-local ``(out, lse)`` (LSE rebased to base-2 for the sglang merge).
        """
        if cp_world <= 1:
            return super()._run_decode_kernel(
                query, kv_cache, block_tables, seq_lens, max_seq_len, layer
            )
        if causal_seqs is None:
            raise ValueError(
                "causal_seqs (global per-request KV lengths) is required for DCP "
                "MLA decode."
            )
        bmm1_scale = self._compute_decode_bmm1_scale(layer)
        raw_out, lse = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=(
                seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
            ),
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
            backend="cute-dsl",
            enable_dcp=True,
            cp_world=cp_world,
            cp_rank=cp_rank,
            causal_seqlens_kv_global=(
                causal_seqs
                if causal_seqs.dtype == torch.int32
                else causal_seqs.to(torch.int32)
            ),
            return_lse=True,  # DCP requires the rank-local LSE for the merge
        )
        return raw_out, lse * _LSE_BASE2_FROM_NATURAL_LOG

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
    ):
        parallel = get_parallel()
        if not parallel.dcp_enabled:
            return super().forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope,
                k_rope,
                cos_sin_cache,
                is_neox,
                llama_4_scaling,
            )

        # Query / KV preparation mirrors the base cute-dsl decode (both FP16 and
        # FP8 KV), then swaps to the DCP kernel call + rank-local return.
        merge_query = q_rope is not None
        if self.data_type == torch.float8_e4m3fn:
            assert q_rope is not None and k_rope is not None
            if cos_sin_cache is None:
                q, k, k_rope = mla_quantize_without_rope_for_fp8(
                    q, q_rope, k.squeeze(1), k_rope.squeeze(1)
                )
            else:
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

        if save_kv_cache:
            assert k is not None and k_rope is not None
            self.token_to_kv_pool.set_mla_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )

        if merge_query:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope_reshaped = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            query = concat_mla_absorb_q_general(q_nope, q_rope_reshaped)
        else:
            query = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        if llama_4_scaling is not None:
            query = (query.to(self.q_data_type) * llama_4_scaling).to(self.data_type)
        if query.dim() == 3:
            query = query.unsqueeze(1)

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

        metadata = (
            getattr(forward_batch, "decode_trtllm_mla_metadata", None)
            or self.forward_decode_metadata
        )
        metadata_batch_size = getattr(metadata, "batch_size", None)
        if (
            metadata_batch_size is not None
            and metadata_batch_size < forward_batch.batch_size
        ):
            self.init_forward_metadata(forward_batch)
            metadata = forward_batch.decode_trtllm_mla_metadata

        global_seq_lens = forward_batch.seq_lens[: forward_batch.batch_size]
        local_seq_lens = self._get_dcp_local_seq_lens(global_seq_lens)
        raw_out, lse = self._run_decode_kernel(
            query=query,
            kv_cache=kv_cache,
            block_tables=metadata.block_kv_indices,
            seq_lens=local_seq_lens,
            max_seq_len=metadata.max_seq_len_k,
            layer=layer,
            causal_seqs=global_seq_lens,
            cp_world=parallel.dcp_size,
            cp_rank=parallel.dcp_rank,
            return_lse=True,
        )

        output = raw_out.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        lse = lse.view(-1, layer.tp_q_head_num)
        # Zero-KV rows (a request this rank owns no cyclic slice for) get a
        # neutral (out=0, lse=-inf) state so the cross-rank merge ignores them.
        fixup_zero_kv_rows(
            output,
            lse,
            local_seq_lens,
            self.q_indptr_decode[: forward_batch.batch_size + 1],
            1,
        )
        return output.flatten(1), lse


class CuteDslMLAMultiStepDraftBackend(TRTLLMMLAMultiStepDraftBackend):
    """Multi-step draft backend for cutedsl_mla used by EAGLE / DSPARK."""

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner, topk, speculative_num_steps)
        # Parent populates self.attn_backends with TRT-LLM instances; replace
        # them with cute-dsl instances sharing the parent's index buffers.
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i] = CuteDslMLABackend(
                model_runner,
                skip_prefill=True,
                kv_indptr_buf=self.kv_indptr[i],
                q_indptr_decode_buf=self.q_indptr_decode,
            )
