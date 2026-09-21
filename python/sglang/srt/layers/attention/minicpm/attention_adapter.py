from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.kernels.ops.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.minicpm.sparse_utils import (
    MiniCPMSparseMetadata,
)
from sglang.srt.utils import is_flashinfer_available

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class MiniCPMFlashAttentionAdapter:
    def __init__(self, flash_attn_backend: FlashAttentionBackend):
        self.flash_attn_backend = flash_attn_backend

    def prepare_forward(
        self,
        metadata: MiniCPMSparseMetadata,
        *,
        is_prefill: bool,
        graph: bool,
    ) -> None:
        pass

    def init_cuda_graph_state(self, max_num_tokens: int) -> None:
        pass

    def forward(
        self,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        metadata: MiniCPMSparseMetadata,
        layer: RadixAttention,
        *,
        is_prefill: bool,
        k_descale: Optional[torch.Tensor] = None,
        v_descale: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks
        if k_descale is not None:
            kwargs["k_descale"] = k_descale
            kwargs["v_descale"] = v_descale

        return flash_attn_with_kvcache(
            q=q,
            k_cache=key_cache,
            v_cache=value_cache,
            page_table=metadata.sparse_page_table,
            cache_seqlens=metadata.sparse_cache_seqlens_int32,
            cu_seqlens_q=metadata.sparse_cu_seqlens_q,
            cu_seqlens_k_new=metadata.sparse_cu_seqlens_k,
            max_seqlen_q=(
                metadata.sparse_max_seq_len_q
                if is_prefill
                else metadata.base.max_seq_len_q
            ),
            softmax_scale=layer.scaling,
            causal=True,
            window_size=(-1, -1),
            softcap=layer.logit_cap,
            num_splits=self.flash_attn_backend.num_splits,
            ver=self.flash_attn_backend.fa_impl_ver,
            **kwargs,
        )


class MiniCPMFlashInferAdapter:
    def __init__(
        self,
        model_runner: ModelRunner,
        *,
        head_group_num: int,
        heads_per_group: int,
        head_dim: int,
        page_size: int,
        max_kv_tokens_per_row: int,
    ):
        if not is_flashinfer_available():
            raise RuntimeError("minicpm_flashinfer requires the flashinfer package.")

        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferAttnBackend,
        )

        self.device = model_runner.device
        self.head_group_num = head_group_num
        self.num_qo_heads = heads_per_group
        self.num_kv_heads = 1
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_kv_tokens_per_row = max_kv_tokens_per_row
        self.q_dtype = model_runner.dtype
        self.kv_dtype = model_runner.kv_cache_dtype

        max_sparse_bs = model_runner.req_to_token_pool.size * head_group_num
        self.kv_indptr = torch.zeros(
            max_sparse_bs + 1,
            dtype=torch.int32,
            device=self.device,
        )
        self.kv_indices = torch.zeros(
            max_sparse_bs * max_kv_tokens_per_row,
            dtype=torch.int32,
            device=self.device,
        )
        self.kv_last_page_len = torch.ones(
            max_sparse_bs,
            dtype=torch.int32,
            device=self.device,
        )
        self.rows = torch.arange(
            max_sparse_bs,
            dtype=torch.int32,
            device=self.device,
        )
        self.flashinfer_backend = FlashInferAttnBackend(
            model_runner,
            skip_prefill=False,
            kv_indptr_buf=self.kv_indptr,
            kv_last_page_len_buf=self.kv_last_page_len,
        )
        self.active_wrapper = None
        self.active_kv_indptr = None
        self.active_kv_indices = None
        self.active_rows = None
        self.prefill_planned = False

    def prepare_forward(
        self,
        metadata: MiniCPMSparseMetadata,
        *,
        is_prefill: bool,
        graph: bool,
    ) -> None:
        if is_prefill:
            self.prefill_planned = False
        else:
            self._prepare(
                metadata,
                is_prefill=False,
                graph=graph,
            )

    def init_cuda_graph_state(self, max_num_tokens: int) -> None:
        self.flashinfer_backend.init_cuda_graph_state(
            max_num_tokens,
            max_num_tokens,
            kv_indices_buf=self.kv_indices,
        )

    def _prepare(
        self,
        metadata: MiniCPMSparseMetadata,
        *,
        is_prefill: bool,
        graph: bool = False,
    ) -> None:
        cache_seqlens = metadata.sparse_cache_seqlens_int32
        sparse_bs = cache_seqlens.numel()
        if sparse_bs == 0:
            self.active_wrapper = None
            return

        if is_prefill:
            kv_indptr = metadata.sparse_cu_seqlens_k
            kv_indices = torch.empty(
                metadata.sparse_page_table.numel(),
                dtype=torch.int32,
                device=self.device,
            )
            kv_last_page_len = (cache_seqlens > 0).to(torch.int32)
            rows = torch.arange(
                sparse_bs,
                dtype=torch.int32,
                device=self.device,
            )
            wrapper = self.flashinfer_backend.prefill_wrappers_paged[0]
            wrapper.begin_forward(
                metadata.sparse_cu_seqlens_q,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                causal=True,
                q_data_type=self.q_dtype,
                kv_data_type=self.kv_dtype,
                non_blocking=True,
            )
        else:
            kv_indptr = self.kv_indptr[: sparse_bs + 1]
            kv_indptr[0] = 0
            torch.cumsum(cache_seqlens, dim=0, out=kv_indptr[1:])
            kv_indices = self.kv_indices[: sparse_bs * self.max_kv_tokens_per_row]
            kv_last_page_len = self.kv_last_page_len[:sparse_bs]
            kv_last_page_len.copy_((cache_seqlens > 0).to(torch.int32))
            rows = self.rows[:sparse_bs]
            if graph:
                graph_bs = sparse_bs // self.head_group_num
                wrapper = self.flashinfer_backend.get_cuda_graph_decode_wrappers(
                    bs=graph_bs,
                    num_tokens=sparse_bs,
                )[0]
            else:
                wrapper = self.flashinfer_backend.decode_wrappers[0]

            wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                q_data_type=self.q_dtype,
                kv_data_type=self.kv_dtype,
                non_blocking=True,
            )

        self.active_wrapper = wrapper
        self.active_kv_indptr = kv_indptr
        self.active_kv_indices = kv_indices
        self.active_rows = rows

    def forward(
        self,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        metadata: MiniCPMSparseMetadata,
        layer: RadixAttention,
        *,
        is_prefill: bool,
        k_descale: Optional[torch.Tensor] = None,
        v_descale: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if sinks is not None:
            raise NotImplementedError(
                "minicpm_flashinfer does not support attention sinks"
            )
        if is_prefill and not self.prefill_planned:
            self._prepare(metadata, is_prefill=True)
            self.prefill_planned = True

        cache_seqlens = metadata.sparse_cache_seqlens_int32
        sparse_bs = cache_seqlens.numel()
        create_flashinfer_kv_indices_triton[(sparse_bs,)](
            metadata.sparse_page_table,
            self.active_rows,
            cache_seqlens,
            self.active_kv_indptr,
            None,
            self.active_kv_indices,
            metadata.sparse_page_table.stride(0),
        )
        kwargs = {
            "sm_scale": layer.scaling,
            "logits_soft_cap": layer.logit_cap or None,
            "k_scale": layer.k_scale_float,
            "v_scale": layer.v_scale_float,
        }
        if is_prefill:
            return self.active_wrapper.forward(
                q,
                (key_cache, value_cache),
                causal=True,
                **kwargs,
            )
        return self.active_wrapper.forward(
            q,
            (key_cache, value_cache),
            **kwargs,
        )
