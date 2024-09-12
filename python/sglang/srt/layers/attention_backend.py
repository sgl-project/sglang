from __future__ import annotations

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer.cascade import merge_state
from flashinfer.decode import _grouped_size_compiled_for_decode_kernels

from sglang.global_config import global_config
from sglang.srt.layers.flashinfer_utils import update_flashinfer_indices
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode, InputMetadata

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class AttentionBackend(ABC):
    """The base class of attention backends"""

    @abstractmethod
    def init_forward_metadata(
        self, batch: ScheduleBatch, input_metadata: InputMetadata
    ):
        """Init the metadata for a forward pass."""
        raise NotImplementedError()

    def init_cuda_graph_state(self, max_bs: int):
        """Init the global shared states for cuda graph."""
        raise NotImplementedError()

    def init_forward_metadata_capture_cuda_graph(
        self, bs: int, req_pool_indices, seq_lens
    ):
        """Init the metadata for a forward pass for capturing a cuda graph."""
        raise NotImplementedError()

    def init_forward_metadata_replay_cuda_graph(
        self, bs: int, req_pool_indices, seq_lens
    ):
        """Init the metadata for a forward pass for replying a cuda graph."""
        raise NotImplementedError()

    def get_cuda_graph_seq_len_fill_value(self):
        raise NotImplementedError()

    def forward(self, q, k, v, layer: nn.Module, input_metadata: InputMetadata):
        """Run forward on an attention layer."""
        if input_metadata.forward_mode.is_decode():
            return self.forward_decode(q, k, v, layer, input_metadata)
        else:
            return self.forward_extend(q, k, v, layer, input_metadata)

    def forward_decode(self, q, k, v, layer: nn.Module, input_metadata: InputMetadata):
        raise NotImplementedError()

    def forward_extend(self, q, k, v, layer: nn.Module, input_metadata: InputMetadata):
        raise NotImplementedError()


class FlashInferAttnBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.model_runner = model_runner

        local_num_qo_heads = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )
        local_num_kv_heads = model_runner.model_config.get_num_kv_heads(
            model_runner.tp_size
        )
        if (
            not _grouped_size_compiled_for_decode_kernels(
                local_num_qo_heads, local_num_kv_heads
            )
            or local_num_qo_heads // local_num_kv_heads > 4
        ):
            self.decode_use_tensor_cores = True
        else:
            self.decode_use_tensor_cores = False

        self.workspace_buffer = torch.empty(
            global_config.flashinfer_workspace_size,
            dtype=torch.uint8,
            device="cuda",
        )

        if model_runner.sliding_window_size is None:
            self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer, "NHD"
            )
            self.prefill_wrapper_paged = BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer, "NHD"
            )
            self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                use_tensor_cores=self.decode_use_tensor_cores,
            )
        else:
            # Two wrappers: one for sliding window attention and one for full attention.
            # Using two wrappers is unnecessary in the current PR, but are prepared for future PRs
            self.prefill_wrapper_ragged = None
            self.prefill_wrapper_paged = []
            self.decode_wrapper = []
            for _ in range(2):
                self.prefill_wrapper_paged.append(
                    BatchPrefillWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")
                )
                self.decode_wrapper.append(
                    BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_tensor_cores=self.decode_use_tensor_cores,
                    )
                )

        self.forward_metadata = None
        self.cuda_graph_metadata = {}

    def init_forward_metadata(
        self, batch: ScheduleBatch, input_metadata: InputMetadata
    ):
        if input_metadata.forward_mode.is_decode():
            prefix_lens = None
            use_ragged = False
            total_num_tokens = None
        else:
            prefix_lens = input_metadata.extend_prefix_lens

            # Some heuristics to check whether to use ragged forward
            use_ragged = False
            if (
                int(torch.sum(input_metadata.seq_lens)) > 4096
                and self.model_runner.sliding_window_size is None
            ):
                use_ragged = True

            total_num_tokens = torch.sum(input_metadata.seq_lens).item()

        update_flashinfer_indices(
            input_metadata.forward_mode,
            self.model_runner,
            input_metadata.req_pool_indices,
            input_metadata.seq_lens,
            prefix_lens,
            use_ragged=use_ragged,
        )

        self.forward_metadata = (use_ragged, total_num_tokens, self.decode_wrapper)

    def init_cuda_graph_state(self, max_bs: int):
        self.cuda_graph_kv_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device="cuda"
        )
        self.cuda_graph_kv_indices = torch.zeros(
            (max_bs * self.model_runner.model_config.context_len,),
            dtype=torch.int32,
            device="cuda",
        )
        self.cuda_graph_kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device="cuda"
        )

        if self.model_runner.sliding_window_size is not None:
            self.cuda_graph_kv_indptr = [
                self.cuda_graph_kv_indptr,
                self.cuda_graph_kv_indptr.clone(),
            ]
            self.cuda_graph_kv_indices = [
                self.cuda_graph_kv_indices,
                self.cuda_graph_kv_indices.clone(),
            ]

    def init_forward_metadata_capture_cuda_graph(
        self, bs: int, req_pool_indices, seq_lens
    ):
        if self.model_runner.sliding_window_size is None:
            decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                use_cuda_graph=True,
                use_tensor_cores=self.decode_use_tensor_cores,
                paged_kv_indptr_buffer=self.cuda_graph_kv_indptr[: bs + 1],
                paged_kv_indices_buffer=self.cuda_graph_kv_indices,
                paged_kv_last_page_len_buffer=self.cuda_graph_kv_last_page_len[:bs],
            )
        else:
            decode_wrapper = []
            for i in range(2):
                decode_wrapper.append(
                    BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        use_tensor_cores=self.decode_use_tensor_cores,
                        paged_kv_indptr_buffer=self.cuda_graph_kv_indptr[i][: bs + 1],
                        paged_kv_indices_buffer=self.cuda_graph_kv_indices[i],
                        paged_kv_last_page_len_buffer=self.cuda_graph_kv_last_page_len[
                            :bs
                        ],
                    )
                )

        update_flashinfer_indices(
            ForwardMode.DECODE,
            self.model_runner,
            req_pool_indices,
            seq_lens,
            None,
            decode_wrapper,
        )

        self.cuda_graph_metadata[bs] = decode_wrapper

        self.forward_metadata = (False, None, decode_wrapper)

    def init_forward_metadata_replay_cuda_graph(
        self, bs: int, req_pool_indices, seq_lens
    ):
        update_flashinfer_indices(
            ForwardMode.DECODE,
            self.model_runner,
            req_pool_indices[:bs],
            seq_lens[:bs],
            None,
            self.cuda_graph_metadata[bs],
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def forward_extend(self, q, k, v, layer: nn.Module, input_metadata: InputMetadata):
        if not isinstance(self.prefill_wrapper_paged, list):
            prefill_wrapper_paged = self.prefill_wrapper_paged
        else:
            if layer.sliding_window_size != -1:
                prefill_wrapper_paged = self.prefill_wrapper_paged[0]
            else:
                prefill_wrapper_paged = self.prefill_wrapper_paged[1]

        use_ragged, total_num_tokens, decode_wrapper = self.forward_metadata

        if not use_ragged:
            if k is not None:
                assert v is not None
                input_metadata.token_to_kv_pool.set_kv_buffer(
                    layer.layer_id, input_metadata.out_cache_loc, k, v
                )
            o = prefill_wrapper_paged.forward(
                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                input_metadata.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                causal=True,
                sm_scale=layer.scaling,
                window_left=layer.sliding_window_size,
                logits_soft_cap=layer.logit_cap,
            )
        else:
            o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k.contiguous().view(-1, layer.tp_k_head_num, layer.head_dim),
                v.contiguous().view(-1, layer.tp_v_head_num, layer.head_dim),
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
            )

            if input_metadata.extend_no_prefix:
                o = o1
            else:
                o2, s2 = prefill_wrapper_paged.forward_return_lse(
                    q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    input_metadata.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                    causal=False,
                    sm_scale=layer.scaling,
                    logits_soft_cap=layer.logit_cap,
                )

                o, _ = merge_state(o1, s1, o2, s2)

            input_metadata.token_to_kv_pool.set_kv_buffer(
                layer.layer_id, input_metadata.out_cache_loc, k, v
            )

            if total_num_tokens >= global_config.layer_sync_threshold:
                torch.cuda.synchronize()

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(self, q, k, v, layer: nn.Module, input_metadata: InputMetadata):
        use_ragged, total_num_tokens, decode_wrapper = self.forward_metadata

        if isinstance(decode_wrapper, list):
            if layer.sliding_window_size != -1:
                decode_wrapper = decode_wrapper[0]
            else:
                decode_wrapper = decode_wrapper[1]

        if k is not None:
            assert v is not None
            input_metadata.token_to_kv_pool.set_kv_buffer(
                layer.layer_id, input_metadata.out_cache_loc, k, v
            )

        o = decode_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            input_metadata.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)


class TritonAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        # Lazy import to avoid the initialization of cuda context
        from sglang.srt.layers.triton_attention.decode_attention import (
            REDUCE_TORCH_TYPE,
            decode_attention_fwd,
        )
        from sglang.srt.layers.triton_attention.extend_attention import (
            extend_attention_fwd,
        )

        super().__init__()

        self.decode_attention_fwd = decode_attention_fwd
        self.extend_attention_fwd = extend_attention_fwd
        self.REDUCE_TORCH_TYPE = REDUCE_TORCH_TYPE
        self.num_head = model_runner.model_config.num_attention_heads

        self.forward_metadata = None

        self.cuda_graph_max_seq_len = model_runner.model_config.context_len

    def init_forward_metadata(
        self, batch: ScheduleBatch, input_metadata: InputMetadata
    ):
        """Init auxiliary variables for triton attention backend."""

        if input_metadata.forward_mode.is_decode():
            start_loc = torch.zeros_like(input_metadata.seq_lens, dtype=torch.int32)
            start_loc[1:] = torch.cumsum(input_metadata.seq_lens[:-1], dim=0)

            total_num_tokens = torch.sum(input_metadata.seq_lens).item()
            attn_logits = torch.empty(
                (self.num_head, total_num_tokens),
                dtype=self.REDUCE_TORCH_TYPE,
                device="cuda",
            )

            max_seq_len = torch.max(input_metadata.seq_lens).item()
            max_extend_len = None
        else:
            start_loc = attn_logits = max_seq_len = None
            prefix_lens = torch.tensor(batch.prefix_lens_cpu, device="cuda")
            max_extend_len = torch.max(input_metadata.seq_lens - prefix_lens).item()

        self.forward_metadata = start_loc, attn_logits, max_seq_len, max_extend_len

    def init_cuda_graph_state(self, max_bs: int):
        self.cuda_graph_max_total_num_tokens = max_bs * self.cuda_graph_max_seq_len

        self.cuda_graph_start_loc = torch.zeros(
            (max_bs,), dtype=torch.int32, device="cuda"
        )
        self.cuda_graph_attn_logits = torch.empty(
            (self.num_head, self.cuda_graph_max_total_num_tokens),
            dtype=self.REDUCE_TORCH_TYPE,
            device="cuda",
        )

    def init_forward_metadata_capture_cuda_graph(
        self, bs: int, req_pool_indices, seq_lens
    ):
        self.forward_metadata = (
            self.cuda_graph_start_loc,
            self.cuda_graph_attn_logits,
            self.cuda_graph_max_seq_len,
            None,
        )

    def init_forward_metadata_replay_cuda_graph(
        self, bs: int, req_pool_indices, seq_lens
    ):
        self.cuda_graph_start_loc.zero_()
        self.cuda_graph_start_loc[1:bs] = torch.cumsum(seq_lens[: bs - 1], dim=0)

        self.forward_metadata = (
            self.cuda_graph_start_loc,
            self.cuda_graph_attn_logits,
            self.cuda_graph_max_seq_len,
            None,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_extend(self, q, k, v, layer: nn.Module, input_metadata: InputMetadata):
        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        input_metadata.token_to_kv_pool.set_kv_buffer(
            layer.layer_id, input_metadata.out_cache_loc, k, v
        )

        start_loc, attn_logits, max_seq_len, max_extend_len = self.forward_metadata
        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            input_metadata.token_to_kv_pool.get_key_buffer(layer.layer_id),
            input_metadata.token_to_kv_pool.get_value_buffer(layer.layer_id),
            input_metadata.req_to_token_pool.req_to_token,
            input_metadata.req_pool_indices,
            input_metadata.seq_lens,
            input_metadata.extend_seq_lens,
            input_metadata.extend_start_loc,
            max_extend_len,
            layer.scaling,
            layer.logit_cap,
        )
        return o

    def forward_decode(self, q, k, v, layer: nn.Module, input_metadata: InputMetadata):
        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        start_loc, attn_logits, max_seq_len, max_extend_len = self.forward_metadata

        input_metadata.token_to_kv_pool.set_kv_buffer(
            layer.layer_id, input_metadata.out_cache_loc, k, v
        )

        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            input_metadata.token_to_kv_pool.get_key_buffer(layer.layer_id),
            input_metadata.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            input_metadata.req_to_token_pool.req_to_token,
            input_metadata.req_pool_indices,
            start_loc,
            input_metadata.seq_lens,
            attn_logits,
            max_seq_len,
            layer.scaling,
            layer.logit_cap,
        )
        return o
