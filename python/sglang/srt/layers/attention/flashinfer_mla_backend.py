from __future__ import annotations

"""
Support attention backend for flashinfer MLA.
When radix cache is enabled, the backend only uses BatchMLAPaged wrapper when forwarding.
When radix cache is disabled, the backend uses BatchPrefill wrappers for prefilling (with or without prefix cache),
and uses BatchMLAPaged wrapper for decoding.
More details can be found in https://docs.flashinfer.ai/api/mla.html
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch

from sglang.global_config import global_config
from sglang.srt.layers.attention import AttentionBackend
from sglang.srt.layers.attention.flashinfer_backend import (
    create_flashinfer_kv_indices_triton,
    should_use_tensor_core,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_flashinfer_available

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInfo

if is_flashinfer_available():
    from flashinfer import (
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )
    from flashinfer.cascade import merge_state
    from flashinfer.mla import BatchMLAPagedAttentionWrapper


@dataclass
class DecodeMetadata:
    decode_wrapper: BatchMLAPagedAttentionWrapper


@dataclass
class PrefillMetadata:
    prefill_wrapper: Union[
        BatchPrefillWithPagedKVCacheWrapper, BatchMLAPagedAttentionWrapper
    ]
    use_ragged: bool


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None


class FlashInferMLAAttnBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        kv_indptr_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Parse constants
        self.max_context_len = model_runner.model_config.context_len

        global_config.enable_flashinfer_mla = True

        # Allocate buffers
        global global_workspace_buffer
        if global_workspace_buffer is None:
            global_workspace_buffer = torch.empty(
                global_config.flashinfer_workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        self.workspace_buffer = global_workspace_buffer

        max_bs = model_runner.req_to_token_pool.size
        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        self.qo_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )

        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )

        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )

        if not global_server_args_dict["disable_radix_cache"]:
            # use mla paged prefill
            self.prefill_wrapper_paged = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                backend="auto",
            )
        else:
            self.prefill_wrapper_paged = BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                backend="auto",
            )
        self.decode_wrapper = BatchMLAPagedAttentionWrapper(
            self.workspace_buffer, backend="auto"
        )

        # Create indices updater
        self.indices_updater_prefill = FlashInferMLAIndicesUpdaterPrefill(
            model_runner, self
        )
        self.indices_updater_decode = FlashInferMLAIndicesUpdaterDecode(
            model_runner, self
        )

        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None
        self.decode_cuda_graph_metadata = {}
        self.prefill_cuda_graph_metadata = {}

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                decode_wrapper=self.decode_wrapper,
            )
            self.forward_metadata = DecodeMetadata(self.decode_wrapper)
        else:
            prefix_lens = forward_batch.extend_prefix_lens
            use_ragged = global_server_args_dict["disable_radix_cache"]

            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                prefix_lens,
                prefill_wrapper_paged=self.prefill_wrapper_paged,
                use_ragged=use_ragged,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrapper_paged, use_ragged
            )

    def init_cuda_graph_state(
        self, max_bs: int, kv_indices_buf: Optional[torch.Tensor] = None
    ):
        if kv_indices_buf is None:
            cuda_graph_kv_indices = torch.zeros(
                (max_bs * self.max_context_len,),
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = kv_indices_buf

        self.cuda_graph_kv_indices = cuda_graph_kv_indices
        self.cuda_graph_custom_mask = torch.zeros(
            (max_bs * self.max_context_len),
            dtype=torch.uint8,
            device="cuda",
        )
        self.cuda_graph_qk_indptr = self.kv_indptr.clone()
        self.cuda_graph_qo_indptr = self.kv_indptr.clone()

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        if forward_mode.is_decode_or_idle():
            decode_wrapper = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                use_cuda_graph=True,
                qo_indptr=self.qo_indptr[: num_tokens + 1],
                kv_indptr=self.kv_indptr[: num_tokens + 1],
                kv_indices=self.cuda_graph_kv_indices,
                kv_len_arr=self.kv_last_page_len[:num_tokens],
                backend="auto",
            )

            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_decode.update(
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                decode_wrapper=decode_wrapper,
            )
            self.decode_cuda_graph_metadata[bs] = decode_wrapper
            self.forward_metadata = DecodeMetadata(decode_wrapper)
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
        spec_info: Optional[SpecInfo],
    ):
        if forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_sum,
                decode_wrapper=self.decode_cuda_graph_metadata[bs],
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        cache_loc = forward_batch.out_cache_loc
        logits_soft_cap = layer.logit_cap

        if not global_server_args_dict["disable_radix_cache"]:
            # use mla paged prefill
            prefill_wrapper_paged = self.forward_metadata.prefill_wrapper
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
            qall = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

            o = prefill_wrapper_paged.run(
                qall[:, :, : layer.v_head_dim],
                qall[:, :, layer.v_head_dim :],
                k_buf[:, :, : layer.v_head_dim],
                k_buf[:, :, layer.v_head_dim :],
            )
        else:
            # use mla ragged prefill
            o, _ = self.prefill_wrapper_ragged.forward_return_lse(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                k.view(-1, layer.tp_k_head_num, layer.head_dim),
                v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )

            # FIXME: Here should be another prefill_paged to call

            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    cache_loc,
                    k,
                    v,
                )

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        decode_wrapper = self.forward_metadata.decode_wrapper
        cache_loc = forward_batch.out_cache_loc

        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    cache_loc,
                    k,
                    v,
                )
        reshaped_q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        reshaped_k = k_buffer.view(-1, 1, layer.head_dim)
        o = decode_wrapper.run(
            reshaped_q[:, :, : layer.v_head_dim],
            reshaped_q[:, :, layer.v_head_dim :],
            reshaped_k[:, :, : layer.v_head_dim],
            reshaped_k[:, :, layer.v_head_dim :],
        )

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


class FlashInferMLAIndicesUpdaterDecode:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.scaling = model_runner.model_config.scaling
        self.data_type = model_runner.kv_cache_dtype
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrapper: BatchMLAPagedAttentionWrapper,
    ):
        decode_wrappers = decode_wrapper or self.decode_wrapper
        self.call_begin_forward(
            decode_wrapper,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            self.kv_indptr,
        )

    def call_begin_forward(
        self,
        wrapper: BatchMLAPagedAttentionWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        kv_indptr: torch.Tensor,
    ):
        bs = len(req_pool_indices)
        kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
        kv_indptr = kv_indptr[: bs + 1]
        kv_indices = torch.empty(
            paged_kernel_lens_sum, dtype=torch.int32, device="cuda"
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            kv_indptr,
            None,
            kv_indices,
            self.req_to_token.shape[1],
        )

        sm_scale = self.scaling
        q_indptr = torch.arange(0, bs + 1).to(0).int()
        kv_lens = paged_kernel_lens.to(torch.int32)
        wrapper.plan(
            q_indptr,
            kv_indptr,
            kv_indices,
            kv_lens,
            self.num_local_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            1,
            False,
            sm_scale,
            self.data_type,
            self.data_type,
        )


class FlashInferMLAIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.scaling = model_runner.model_config.scaling
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged

    def update(
        self,
        req_pool_indices: torch.Tnesor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrapper_paged: Union[
            BatchPrefillWithPagedKVCacheWrapper, BatchMLAPagedAttentionWrapper
        ],
        use_ragged: bool,
    ):
        if use_ragged:
            paged_kernel_lens = prefix_lens
            paged_kernel_lens_sum = paged_kernel_lens.sum().item()
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

        self.call_begin_forward(
            self.prefill_wrapper_ragged,
            prefill_wrapper_paged,
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            self.kv_indptr,
            self.qo_indptr,
            use_ragged,
        )

    def call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: Union[
            BatchPrefillWithPagedKVCacheWrapper, BatchMLAPagedAttentionWrapper
        ],
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
    ):
        bs = len(req_pool_indices)
        # Normal extend
        kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
        kv_indptr = kv_indptr[: bs + 1]
        kv_indices = torch.empty(
            paged_kernel_lens_sum,
            dtype=torch.int32,
            device=req_pool_indices.device,
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            kv_indptr,
            None,
            kv_indices,
            self.req_to_token.shape[1],
        )

        qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
        qo_indptr = qo_indptr[: bs + 1]
        sm_scale = self.scaling

        # extend part
        if use_ragged:
            wrapper_ragged.begin_forward(
                qo_indptr=qo_indptr,
                kv_indptr=qo_indptr,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,
                head_dim_vo=self.v_head_dim,
                q_data_type=self.q_data_type,
            )

        if not global_server_args_dict["disable_radix_cache"]:
            # mla paged prefill
            kv_len_arr = kv_indptr[1:] - kv_indptr[:-1]
            wrapper_paged.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_len_arr,
                self.num_qo_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                1,
                True,
                sm_scale,
                self.q_data_type,
                self.data_type,
            )

        # FIXME: Here should be some logic for prefill paged when not using radix cache？
