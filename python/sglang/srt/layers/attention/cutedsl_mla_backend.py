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
unchanged. The DCP metadata helpers live on :class:`TRTLLMMLABackend`; this
module only supplies the cute-dsl kernel call and its decode forward.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.fixup_zero_kv import fixup_zero_kv_rows
from sglang.kernels.ops.attention.utils import (
    concat_mla_absorb_q_general,
    mla_quantize_and_rope_for_fp8,
    mla_quantize_without_rope_for_fp8,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLAMultiStepDraftBackend,
)
from sglang.srt.layers.logits_processor import get_in_autotune_dummy_run
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_flashinfer_available

if is_flashinfer_available():
    import flashinfer

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


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
        rank-local ``(out, lse)``, the LSE in natural log.
        """
        if cp_world <= 1:
            return super()._run_decode_kernel(
                query,
                kv_cache,
                block_tables,
                seq_lens,
                max_seq_len,
                layer,
                causal_seqs=causal_seqs,
                cp_world=cp_world,
                cp_rank=cp_rank,
                return_lse=return_lse,
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
        return raw_out, lse

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
        if parallel.dcp_enabled and get_in_autotune_dummy_run():
            return self._dummy_dcp_decode_for_autotune(q, layer)
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
        query = None
        if self.data_type == torch.float8_e4m3fn:
            assert q_rope is not None and k_rope is not None
            if cos_sin_cache is None:
                if (
                    save_kv_cache
                    and self._fused_set_kv_concat_q_fp8
                    and not self.kv_index_translator.is_translating
                ):
                    # Static pool: out_cache_loc is already the physical loc.
                    # Fused: bf16->fp8 quantize + KV scatter + q concat in one
                    # launch; None when not covered.
                    query = self._set_kv_and_concat_q_fp8_fused(
                        layer=layer,
                        loc=forward_batch.out_cache_loc,
                        q=q,
                        q_rope=q_rope,
                        k=k,
                        k_rope=k_rope,
                    )
                if query is None:
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

        if query is None and save_kv_cache:
            assert k is not None and k_rope is not None
            self.token_to_kv_pool.set_mla_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )

        if query is not None:
            pass  # fused fp8 path already built the query and wrote KV
        elif merge_query:
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

        if metadata.seq_lens_k is not None and metadata.global_seq_lens_k is not None:
            # Hoisted path: int32 rank-local + global lens maintained once per
            # step by metadata init / graph replay-prep.
            local_seq_lens = metadata.seq_lens_k[: forward_batch.batch_size]
            global_seq_lens = metadata.global_seq_lens_k[: forward_batch.batch_size]
        else:
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
