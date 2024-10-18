from __future__ import annotations

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from sglang.global_config import global_config
from sglang.srt.layers.attention import AttentionBackend
from sglang.srt.layers.attention.flashinfer_utils import (
    WrapperDispatch,
    create_flashinfer_kv_indices_triton,
    update_flashinfer_indices,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_flashinfer_available

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )
    from flashinfer.cascade import merge_state
    from flashinfer.decode import _grouped_size_compiled_for_decode_kernels


class FlashInferAttnBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        # Parse constants
        self.model_runner = model_runner  # TODO: remove this
        if not _grouped_size_compiled_for_decode_kernels(
            model_runner.model_config.num_attention_heads // model_runner.tp_size,
            model_runner.model_config.get_num_kv_heads(model_runner.tp_size),
        ):
            self.decode_use_tensor_cores = True
        else:
            self.decode_use_tensor_cores = False

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.has_cross_attention
        ), "Sliding window and cross attention are not supported together"

        if model_runner.sliding_window_size is not None:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.SLIDING_WINDOW
        elif model_runner.has_cross_attention:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.CROSS_ATTENTION
        else:
            self.num_wrappers = 1
            self.dispatch_reason = None

        # Allocate buffers
        self.workspace_buffer = torch.empty(
            global_config.flashinfer_workspace_size,
            dtype=torch.uint8,
            device=model_runner.device,
        )
        max_bs = model_runner.req_to_token_pool.size
        self.kv_indptr = [
            torch.zeros((max_bs + 1,), dtype=torch.int32, device=model_runner.device)
            for _ in range(self.num_wrappers)
        ]
        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )

        # Create wrappers
        # NOTE: we do not use ragged attention when there are multiple wrappers
        self.prefill_wrapper_ragged = (
            BatchPrefillWithRaggedKVCacheWrapper(self.workspace_buffer, "NHD")
            if self.num_wrappers == 1
            else None
        )

        # Two wrappers: one for sliding window attention and one for full attention.
        # Using two wrappers is unnecessary in the current PR, but are prepared for future PRs
        self.prefill_wrappers_paged = []
        self.decode_wrappers = []
        for _ in range(self.num_wrappers):
            self.prefill_wrappers_paged.append(
                BatchPrefillWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")
            )
            self.decode_wrappers.append(
                BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_tensor_cores=self.decode_use_tensor_cores,
                )
            )

        # Create indices updater
        self.indices_updater_decode = FlashInferIndicesUpdaterDecode(self)

        # Other metadata
        self.forward_metadata = None
        self.cuda_graph_metadata = {}

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode():
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
            )
            self.forward_metadata = (self.decode_wrappers,)
        else:
            prefix_lens = forward_batch.extend_prefix_lens

            # Some heuristics to check whether to use ragged forward
            use_ragged = False
            if (
                torch.sum(forward_batch.seq_lens).item() >= 4096
                and self.num_wrappers == 1
            ):
                use_ragged = True

            extend_no_prefix = not torch.any(forward_batch.extend_prefix_lens).item()

            update_flashinfer_indices(
                forward_batch.forward_mode,
                self.model_runner,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                prefix_lens,
                use_ragged=use_ragged,
            )
            self.forward_metadata = (
                use_ragged,
                extend_no_prefix,
            )

    def init_cuda_graph_state(self, max_bs: int):
        cuda_graph_kv_indices = torch.zeros(
            (max_bs * self.model_runner.model_config.context_len,),
            dtype=torch.int32,
            device="cuda",
        )
        self.cuda_graph_kv_indices = [cuda_graph_kv_indices] + [
            cuda_graph_kv_indices.clone() for _ in range(self.num_wrappers - 1)
        ]

    def init_forward_metadata_capture_cuda_graph(
        self, bs: int, req_pool_indices, seq_lens
    ):
        decode_wrappers = []
        for i in range(self.num_wrappers):
            decode_wrappers.append(
                BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_cuda_graph=True,
                    use_tensor_cores=self.decode_use_tensor_cores,
                    paged_kv_indptr_buffer=self.kv_indptr[i][: bs + 1],
                    paged_kv_indices_buffer=self.cuda_graph_kv_indices[i],
                    paged_kv_last_page_len_buffer=self.kv_last_page_len[:bs],
                )
            )

        self.indices_updater_decode.update(req_pool_indices, seq_lens, decode_wrappers)
        self.cuda_graph_metadata[bs] = decode_wrappers
        self.forward_metadata = (decode_wrappers,)

    def init_forward_metadata_replay_cuda_graph(
        self, bs: int, req_pool_indices, seq_lens
    ):
        self.indices_updater_decode.update(
            req_pool_indices[:bs], seq_lens[:bs], self.cuda_graph_metadata[bs]
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def forward_extend(self, q, k, v, layer: nn.Module, forward_batch: ForwardBatch):
        prefill_wrapper_paged = self.prefill_wrappers_paged[
            self._get_wrapper_idx(layer)
        ]

        use_ragged, extend_no_prefix = self.forward_metadata

        if not use_ragged:
            if k is not None:
                assert v is not None
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer.layer_id, forward_batch.out_cache_loc, k, v
                )
            o = prefill_wrapper_paged.forward(
                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
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

            if extend_no_prefix:
                o = o1
            else:
                o2, s2 = prefill_wrapper_paged.forward_return_lse(
                    q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                    causal=False,
                    sm_scale=layer.scaling,
                    logits_soft_cap=layer.logit_cap,
                )

                o, _ = merge_state(o1, s1, o2, s2)

            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer.layer_id, forward_batch.out_cache_loc, k, v
            )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(self, q, k, v, layer: nn.Module, forward_batch: ForwardBatch):
        decode_wrapper = self.forward_metadata[0][self._get_wrapper_idx(layer)]

        if k is not None:
            assert v is not None
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer.layer_id, forward_batch.out_cache_loc, k, v
            )

        o = decode_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _get_wrapper_idx(self, layer: nn.Module):
        if self.num_wrappers == 1:
            return 0

        if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            return layer.sliding_window_size == -1
        if self.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            return layer.is_cross_attention

        raise ValueError(f"Unknown dispatch reason: {self.dispatch_reason}")


class FlashInferIndicesUpdaterDecode:
    def __init__(self, attn_backend):
        # Constants
        model_runner = attn_backend.model_runner
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            model_runner.tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.max_context_len = model_runner.req_to_token_pool.req_to_token.size(1)

        # Buffers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.decode_wrappers = attn_backend.decode_wrappers

        # Dispatch
        if attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update_sliding_window(self):
        raise NotImplementedError()

    def update_cross_attention(self):
        raise NotImplementedError()

    def update_single_wrapper(self, req_pool_indices, seq_lens, decode_wrappers=None):
        decode_wrappers = decode_wrappers or self.decode_wrappers

        bs = len(seq_lens)
        kv_indptr = self.kv_indptr[0][: bs + 1]
        kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
        # TODO: optimize the blocking call on kv_indptr[-1]
        kv_indices = torch.empty(kv_indptr[-1], dtype=torch.int32, device="cuda")
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            kv_indptr,
            None,
            kv_indices,
            self.max_context_len,
        )
        self.call_begin_forward(
            decode_wrappers[0], kv_indptr, kv_indices, self.kv_last_page_len[:bs]
        )

    def call_begin_forward(self, wrapper, kv_indptr, kv_indices, kv_last_page_len):
        wrapper.end_forward()
        wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            data_type=self.data_type,
            q_data_type=self.q_data_type,
        )
