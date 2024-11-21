from __future__ import annotations

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

import os
from enum import Enum, auto
from typing import TYPE_CHECKING, List

import torch
import triton
import triton.language as tl

from sglang.global_config import global_config
from sglang.srt.layers.attention import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import (
    get_bool_env_var,
    is_flashinfer_available,
    should_use_tensor_core,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

if is_flashinfer_available():
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
    from flashinfer.cascade import MultiLevelCascadeAttentionWrapper


class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


def _update_layer_args(
    layer_args: dict, updated_layer_args: dict, plan_args: dict, wrapper
):
    """Update layer_args and re-plan if necessary"""

    layer_args_changed = layer_args != updated_layer_args
    layer_args.clear()
    layer_args.update(updated_layer_args)
    if layer_args_changed:
        wrapper.plan(**plan_args, **layer_args)


class FlashInferAttnBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        self.decode_use_tensor_cores = should_use_tensor_core(
            kv_cache_dtype=model_runner.kv_cache_dtype,
            num_attention_heads=model_runner.model_config.num_attention_heads
            // model_runner.tp_size,
            num_kv_heads=model_runner.model_config.get_num_kv_heads(
                model_runner.tp_size
            ),
        )

        self.max_context_len = model_runner.model_config.context_len

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        if model_runner.sliding_window_size is not None:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.SLIDING_WINDOW
        elif model_runner.model_config.is_encoder_decoder:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.CROSS_ATTENTION
        else:
            self.num_wrappers = 1
            self.dispatch_reason = None

        self.num_cascade_levels = 1

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
        self.qo_indptr = [
            torch.zeros((max_bs + 1,), dtype=torch.int32, device=model_runner.device)
            for _ in range(self.num_wrappers)
        ]

        # Create wrappers
        # NOTE: we do not use ragged attention when there are multiple wrappers
        self.prefill_wrapper_ragged = (
            BatchPrefillWithRaggedKVCacheWrapper(self.workspace_buffer, "NHD")
            if self.num_wrappers == 1
            else None
        )

        self.multi_level_wrappers = []
        for _ in range(self.num_wrappers):
            self.multi_level_wrappers.append(
                MultiLevelCascadeAttentionWrapper(
                    self.num_cascade_levels,
                    self.workspace_buffer,
                    "NHD",
                )
            )

        # Create indices updater
        self.indices_updater_decode = FlashInferIndicesUpdaterDecode(model_runner, self)
        self.indices_updater_prefill = FlashInferIndicesUpdaterPrefill(
            model_runner, self
        )

        # Other metadata
        self.forward_metadata = None
        self.cuda_graph_metadata = {}

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode():
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                multi_level_wrappers=None,
                encoder_lens=forward_batch.encoder_lens,
            )
            self.forward_metadata = (self.multi_level_wrappers,)
        else:
            prefix_lens = forward_batch.extend_prefix_lens

            # Some heuristics to check whether to use ragged forward
            if forward_batch.extend_num_tokens >= 4096 and self.num_wrappers == 1:
                use_ragged = True
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            else:
                use_ragged = False
                extend_no_prefix = False

            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                prefix_lens,
                use_ragged=use_ragged,
                encoder_lens=forward_batch.encoder_lens,
            )

            self.forward_metadata = (use_ragged, extend_no_prefix)

    def init_cuda_graph_state(self, max_bs: int):
        cuda_graph_kv_indices = torch.zeros(
            (max_bs * self.max_context_len,),
            dtype=torch.int32,
            device="cuda",
        )
        self.cuda_graph_kv_indices = [cuda_graph_kv_indices] + [
            cuda_graph_kv_indices.clone() for _ in range(self.num_wrappers - 1)
        ]

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: torch.Tensor = None,
    ):
        multi_level_wrappers = []
        for i in range(self.num_wrappers):
            multi_level_wrappers.append(
                MultiLevelCascadeAttentionWrapper(
                    self.num_cascade_levels,
                    self.workspace_buffer,
                    "NHD",
                    # Pass below args once flashinfer version is bumped
                    # use_cuda_graph=True,
                    # qo_indptr_buf_arr=[self.qo_indptr[i][: bs + 1]],
                    # paged_kv_indptr_buf_arr=[self.kv_indptr[i][: bs + 1]],
                    # paged_kv_indices_buf_arr=[self.cuda_graph_kv_indices[i]],
                    # paged_kv_last_page_len_buf_arr=[self.kv_last_page_len[:bs]],
                )
            )

        seq_lens_sum = seq_lens.sum().item()
        self.indices_updater_decode.update(
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            multi_level_wrappers=multi_level_wrappers,
            encoder_lens=encoder_lens,
        )
        self.cuda_graph_metadata[bs] = multi_level_wrappers
        self.forward_metadata = (multi_level_wrappers,)

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: torch.Tensor = None,
    ):
        self.indices_updater_decode.update(
            req_pool_indices[:bs],
            seq_lens[:bs],
            seq_lens_sum,
            multi_level_wrappers=self.cuda_graph_metadata[bs],
            encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        prefill_wrapper_paged = self.multi_level_wrappers[self._get_wrapper_idx(layer)]

        use_ragged, extend_no_prefix = self.forward_metadata
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        plan_args = self.indices_updater_prefill.plan_args
        layer_args = self.indices_updater_prefill.layer_args

        if not use_ragged:
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

            _update_layer_args(
                layer_args,
                {
                    "causal": not layer.is_cross_attention,
                    "sm_scale": layer.scaling,
                    "window_left": layer.sliding_window_size,
                    "logits_soft_cap": layer.logit_cap,
                },
                plan_args,
                prefill_wrapper_paged,
            )
            o = prefill_wrapper_paged.run(
                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
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
                _update_layer_args(
                    layer_args,
                    {
                        "causal": False,
                        "sm_scale": layer.scaling,
                        "logits_soft_cap": layer.logit_cap,
                    },
                    plan_args,
                    prefill_wrapper_paged,
                )
                o = prefill_wrapper_paged.run(
                    q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                )

            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        decode_wrapper = self.forward_metadata[0][self._get_wrapper_idx(layer)]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        plan_args = self.indices_updater_decode.plan_args
        layer_args = self.indices_updater_decode.layer_args

        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        _update_layer_args(
            layer_args,
            {
                "sm_scale": layer.scaling,
                "logits_soft_cap": layer.logit_cap,
            },
            plan_args,
            decode_wrapper,
        )
        o = decode_wrapper.run(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _get_wrapper_idx(self, layer: RadixAttention):
        if self.num_wrappers == 1:
            return 0

        if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            return layer.sliding_window_size == -1
        if self.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            return layer.is_cross_attention

        raise ValueError(f"Unknown dispatch reason: {self.dispatch_reason}")


class FlashInferIndicesUpdaterDecode:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            model_runner.tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size

        self.attn_backend = attn_backend

        self.plan_args = {}
        self.layer_args = {}

        # Buffers and wrappers
        self.qo_indptr = attn_backend.qo_indptr
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.multi_level_wrappers = attn_backend.multi_level_wrappers

        # Dispatch
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        multi_level_wrappers: List,
        encoder_lens: torch.Tensor,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        multi_level_wrappers: List,
        encoder_lens: torch.Tensor,
    ):
        multi_level_wrappers = multi_level_wrappers or self.multi_level_wrappers
        self.call_begin_forward(
            multi_level_wrappers[0],
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            self.kv_indptr[0],
            self.qo_indptr[0],
            None,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        multi_level_wrappers: List,
        encoder_lens: torch.Tensor,
    ):
        multi_level_wrappers = multi_level_wrappers or self.multi_level_wrappers

        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Sliding window attention
                paged_kernel_lens_tmp = torch.minimum(  # TODO: replace this with clamp
                    seq_lens,
                    torch.tensor(self.sliding_window_size + 1),
                )
                paged_kernel_lens_sum_tmp = paged_kernel_lens_tmp.sum().item()
                kv_start_idx_tmp = seq_lens - paged_kernel_lens_tmp
            else:
                # Full attention
                paged_kernel_lens_tmp = seq_lens
                paged_kernel_lens_sum_tmp = seq_lens_sum
                kv_start_idx_tmp = None

            self.call_begin_forward(
                multi_level_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens_tmp,
                paged_kernel_lens_sum_tmp,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                kv_start_idx_tmp,
            )

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        multi_level_wrappers: List,
        encoder_lens: torch.Tensor,
    ):
        multi_level_wrappers = multi_level_wrappers or self.multi_level_wrappers

        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Normal attention
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
            else:
                # Cross attention
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                seq_lens_sum = encoder_lens.sum().item()

            self.call_begin_forward(
                multi_level_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                seq_lens_sum,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                kv_start_idx,
            )

    def call_begin_forward(
        self,
        wrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_start_idx: torch.Tensor,
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
            kv_start_idx,
            kv_indices,
            self.req_to_token.shape[1],
        )

        qo_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
        qo_indptr = qo_indptr[: bs + 1]

        self.plan_args = {
            "qo_indptr_arr": [qo_indptr],
            "paged_kv_indptr_arr": [kv_indptr],
            "paged_kv_indices_arr": [kv_indices],
            "paged_kv_last_page_len": [self.kv_last_page_len[:bs]],
            "num_qo_heads": self.num_qo_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "page_size": 1,
            "q_data_type": self.q_data_type,
        }
        wrapper.plan(**self.plan_args, **self.layer_args)


class FlashInferIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            model_runner.tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size

        self.attn_backend = attn_backend

        self.plan_args = {}
        self.layer_args = {}

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.wrapper_ragged = attn_backend.prefill_wrapper_ragged
        self.multi_level_wrappers = attn_backend.multi_level_wrappers

        # Dispatch
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tnesor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        use_ragged: bool,
        encoder_lens: torch.Tensor,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tnesor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        use_ragged: bool,
        encoder_lens: torch.Tensor,
    ):
        if use_ragged:
            paged_kernel_lens = prefix_lens
            paged_kernel_lens_sum = paged_kernel_lens.sum().item()
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

        self.call_begin_forward(
            self.wrapper_ragged,
            self.multi_level_wrappers[0],
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            None,
            self.kv_indptr[0],
            self.qo_indptr[0],
            use_ragged,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        use_ragged: bool,
        encoder_lens: torch.Tensor,
    ):
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # window attention use paged only
                paged_kernel_lens = torch.minimum(
                    seq_lens,
                    torch.tensor(self.sliding_window_size) + seq_lens - prefix_lens,
                )
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()
            else:
                # full attention
                paged_kernel_lens = seq_lens
                paged_kernel_lens_sum = seq_lens_sum

            kv_start_idx = seq_lens - paged_kernel_lens

            self.call_begin_forward(
                self.wrapper_ragged,
                self.multi_level_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
            )

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        use_ragged: bool,
        encoder_lens: torch.Tensor,
    ):
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # normal attention
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
                paged_kernel_lens_sum = seq_lens_sum
            else:
                # cross attention
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()

            self.call_begin_forward(
                self.wrapper_ragged,
                self.multi_level_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
            )

    def call_begin_forward(
        self,
        wrapper_ragged,
        wrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        kv_start_idx: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
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
            kv_start_idx,
            kv_indices,
            self.req_to_token.shape[1],
        )

        qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
        qo_indptr = qo_indptr[: bs + 1]

        # extend part
        if use_ragged:
            wrapper_ragged.end_forward()
            wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

        self.plan_args = {
            "qo_indptr_arr": [qo_indptr],
            "paged_kv_indptr_arr": [kv_indptr],
            "paged_kv_indices_arr": [kv_indices],
            "paged_kv_last_page_len": [self.kv_last_page_len[:bs]],
            "num_qo_heads": self.num_qo_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "page_size": 1,
            "q_data_type": self.q_data_type,
        }
        wrapper.plan(**self.plan_args, **self.layer_args)


@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)
