from __future__ import annotations

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

import os
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch

if os.environ["SGLANG_ENABLE_TORCH_COMPILE"] == "1":
    import logging

    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True

from sglang.global_config import global_config
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.utils import is_sm100_supported
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
from sglang.srt.utils import is_flashinfer_available, next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )
    from flashinfer.cascade import merge_state
    from flashinfer.decode import _get_range_buf, get_seq_lens


class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


@dataclass
class DecodeMetadata:
    decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper]


@dataclass
class PrefillMetadata:
    prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper]
    use_ragged: bool
    extend_no_prefix: bool


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None


class FlashInferAttnBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Parse constants
        self.decode_use_tensor_cores = should_use_tensor_core(
            kv_cache_dtype=model_runner.kv_cache_dtype,
            num_attention_heads=model_runner.model_config.num_attention_heads
            // get_attention_tp_size(),
            num_kv_heads=model_runner.model_config.get_num_kv_heads(
                get_attention_tp_size()
            ),
        )
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill
        self.is_multimodal = model_runner.model_config.is_multimodal

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

        # Qwen2/Qwen3 models require higher flashinfer workspace size
        if (
            "Qwen2ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "Qwen3ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "MiMoForCausalLM" in model_runner.model_config.hf_config.architectures
        ):
            global_config.flashinfer_workspace_size = 512 * 1024 * 1024

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
            self.kv_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
                for _ in range(self.num_wrappers)
            ]
        else:
            assert self.num_wrappers == 1
            self.kv_indptr = [kv_indptr_buf]

        if kv_last_page_len_buf is None:
            self.kv_last_page_len = torch.ones(
                (max_bs,), dtype=torch.int32, device=model_runner.device
            )
        else:
            assert self.num_wrappers == 1
            self.kv_last_page_len = kv_last_page_len_buf

        if not self.skip_prefill:
            self.qo_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
                for _ in range(self.num_wrappers)
            ]

        fmha_backend = "auto"
        if is_sm100_supported():
            fmha_backend = "cutlass"
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend=fmha_backend
        )

        # Two wrappers: one for sliding window attention and one for full attention.
        # Using two wrappers is unnecessary in the current PR, but are prepared for future PRs
        self.prefill_wrappers_paged = []
        self.prefill_wrappers_verify = []
        self.decode_wrappers = []
        for _ in range(self.num_wrappers):
            if not skip_prefill:
                self.prefill_wrappers_paged.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend="fa2",
                    )
                )
                self.prefill_wrappers_verify.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                    )
                )
            self.decode_wrappers.append(
                BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_tensor_cores=self.decode_use_tensor_cores,
                )
            )

        # Create indices updater
        if not skip_prefill:
            self.indices_updater_prefill = FlashInferIndicesUpdaterPrefill(
                model_runner, self
            )  # for verify
        self.indices_updater_decode = FlashInferIndicesUpdaterDecode(model_runner, self)

        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None
        self.decode_cuda_graph_metadata = {}
        self.prefill_cuda_graph_metadata = {}  # For verify
        self.draft_extend_cuda_graph_metadata = {}  # For draft extend

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                decode_wrappers=self.decode_wrappers,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = DecodeMetadata(self.decode_wrappers)
        elif forward_batch.forward_mode.is_draft_extend():
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_wrappers_paged,
                use_ragged=False,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_paged, False, False
            )
        elif forward_batch.forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_wrappers_verify,
                use_ragged=False,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_verify, False, False
            )
        else:
            prefix_lens = forward_batch.extend_prefix_lens

            if self.is_multimodal:
                use_ragged = False
                extend_no_prefix = False
            else:
                use_ragged = True
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)

            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                prefix_lens,
                prefill_wrappers=self.prefill_wrappers_paged,
                use_ragged=use_ragged,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=None,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_paged, use_ragged, extend_no_prefix
            )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        if kv_indices_buf is None:
            cuda_graph_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len,),
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = kv_indices_buf

        self.cuda_graph_kv_indices = [cuda_graph_kv_indices] + [
            cuda_graph_kv_indices.clone() for _ in range(self.num_wrappers - 1)
        ]

        # Ensure tensors are properly allocated
        for i in range(self.num_wrappers):
            # Force allocation by performing a small operation
            if len(self.cuda_graph_kv_indices[i]) > 0:
                self.cuda_graph_kv_indices[i][0] = 0

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device="cuda",
            )
            self.cuda_graph_qk_indptr = [x.clone() for x in self.kv_indptr]
            self.cuda_graph_qo_indptr = [x.clone() for x in self.kv_indptr]

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        if forward_mode.is_decode_or_idle():
            decode_wrappers = []
            for i in range(self.num_wrappers):
                decode_wrappers.append(
                    BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        use_tensor_cores=self.decode_use_tensor_cores,
                        paged_kv_indptr_buffer=self.kv_indptr[i][: num_tokens + 1],
                        paged_kv_indices_buffer=self.cuda_graph_kv_indices[i],
                        paged_kv_last_page_len_buffer=self.kv_last_page_len[
                            :num_tokens
                        ],
                    )
                )
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_decode.update(
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                decode_wrappers=decode_wrappers,
                encoder_lens=encoder_lens,
                spec_info=spec_info,
            )
            self.decode_cuda_graph_metadata[bs] = decode_wrappers
            self.forward_metadata = DecodeMetadata(decode_wrappers)
            for i in range(self.num_wrappers):
                decode_wrappers[i].begin_forward = partial(
                    fast_decode_plan, decode_wrappers[i]
                )
        elif forward_mode.is_target_verify():
            prefill_wrappers = []
            for i in range(self.num_wrappers):
                prefill_wrappers.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                        paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                        paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                        paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                        custom_mask_buf=self.cuda_graph_custom_mask,
                        mask_indptr_buf=self.cuda_graph_qk_indptr[i][: bs + 1],
                    )
                )
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=prefill_wrappers,
                use_ragged=False,
                encoder_lens=encoder_lens,
                spec_info=spec_info,
            )
            self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(prefill_wrappers, False, False)
        elif forward_mode.is_draft_extend():
            prefill_wrappers = []
            for i in range(self.num_wrappers):
                prefill_wrappers.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend="fa2",
                        use_cuda_graph=True,
                        qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                        paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                        paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                        paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                    )
                )

            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=prefill_wrappers,
                use_ragged=False,
                encoder_lens=encoder_lens,
                spec_info=spec_info,
            )
            self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(prefill_wrappers, False, False)
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
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        if forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_sum,
                decode_wrappers=self.decode_cuda_graph_metadata[bs],
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
            )
        elif forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
            )
        elif forward_mode.is_draft_extend():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
            )
        else:
            raise ValueError("Invalid forward mode")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        prefill_wrapper_paged = self.forward_metadata.prefill_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        logits_soft_cap = layer.logit_cap

        q = q.contiguous()
        if not self.forward_metadata.use_ragged:
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )

            o = prefill_wrapper_paged.forward(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                causal=not layer.is_cross_attention,
                sm_scale=layer.scaling,
                window_left=layer.sliding_window_size,
                logits_soft_cap=logits_soft_cap,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
            )
        else:
            if self.forward_metadata.extend_no_prefix:
                o = self.prefill_wrapper_ragged.forward(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=True,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )

            else:
                o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=True,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )
                o2, s2 = prefill_wrapper_paged.forward_return_lse(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                    causal=False,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )

                o, _ = merge_state(o1, s1, o2, s2)

            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        decode_wrapper = self.forward_metadata.decode_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # Call the wrapped function
        o = decode_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            k_scale=layer.k_scale,
            v_scale=layer.v_scale,
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
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        # Dispatch the update function
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
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        decode_wrappers = decode_wrappers or self.decode_wrappers
        self.call_begin_forward(
            decode_wrappers[0],
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            self.kv_indptr[0],
            None,
            spec_info,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
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
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens_tmp,
                paged_kernel_lens_sum_tmp,
                self.kv_indptr[wrapper_id],
                kv_start_idx_tmp,
                spec_info,
            )

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
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
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                seq_lens_sum,
                self.kv_indptr[wrapper_id],
                kv_start_idx,
                spec_info,
            )

    def call_begin_forward(
        self,
        wrapper: BatchDecodeWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        kv_indptr: torch.Tensor,
        kv_start_idx: torch.Tensor,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        if spec_info is None:
            bs = len(req_pool_indices)
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            if wrapper.is_cuda_graph_enabled:
                # Directly write to the cuda graph input buffer
                kv_indices = wrapper._paged_kv_indices_buf
            else:
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
        else:
            kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
            bs = kv_indptr.shape[0] - 1

        wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            data_type=self.data_type,
            q_data_type=self.q_data_type,
            non_blocking=True,
        )


class FlashInferIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged

        # Dispatch the update function
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
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        if use_ragged:
            paged_kernel_lens = prefix_lens
            paged_kernel_lens_sum = paged_kernel_lens.sum().item()
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

        self.call_begin_forward(
            self.prefill_wrapper_ragged,
            prefill_wrappers[0],
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            None,
            self.kv_indptr[0],
            self.qo_indptr[0],
            use_ragged,
            spec_info,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
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
                self.prefill_wrapper_ragged,
                prefill_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
                spec_info,
            )

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
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
                self.prefill_wrapper_ragged,
                prefill_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
                spec_info,
            )

    def call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchPrefillWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        kv_start_idx: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        bs = len(seq_lens)
        if spec_info is None:
            assert len(seq_lens) == len(req_pool_indices)
            # Normal extend
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                paged_kernel_lens_sum + 256,
                dtype=torch.int32,
                device=req_pool_indices.device,
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
            custom_mask = None
        else:
            assert isinstance(spec_info, EagleDraftInput) or isinstance(
                spec_info, EagleVerifyInput
            )
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    paged_kernel_lens,
                    paged_kernel_lens_sum,
                    self.req_to_token,
                )
            )

        # extend part
        if use_ragged:
            wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

        # cached part
        wrapper_paged.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            custom_mask=custom_mask,
            non_blocking=True,
        )


# Use as a fast path to override the indptr in flashinfer's plan function
# This is used to remove some host-to-device copy overhead.
global global_override_indptr_cpu


class FlashInferMultiStepDraftBackend:
    """
    Wrap multiple flashinfer attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        from sglang.srt.speculative.eagle_utils import generate_draft_decode_kv_indices

        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        self.page_size = model_runner.page_size

        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        self.attn_backends = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                FlashInferAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    kv_last_page_len_buf=self.kv_last_page_len,
                )
            )

        self.max_context_len = self.attn_backends[0].max_context_len

        # Cached variables for generate_draft_decode_kv_indices
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: torch.Tensor,
        call_fn: Callable,
    ):
        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        self.generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            forward_batch.req_to_token_pool.req_to_token,
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

        assert forward_batch.spec_info is not None
        assert isinstance(forward_batch.spec_info, EagleDraftInput)

        # Copy the kv_indptr once to avoid multiple device-to-host copies in flashinfer's plan.
        indptr_cpu_whole = self.kv_indptr[:, : bs + 1].cpu()
        global global_override_indptr_cpu

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            global_override_indptr_cpu = indptr_cpu_whole[i]
            call_fn(i, forward_batch)

        global_override_indptr_cpu = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices = torch.empty(
            (
                self.speculative_num_steps,
                forward_batch.batch_size * self.topk * self.max_context_len,
            ),
            dtype=torch.int32,
            device="cuda",
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
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, max_bs * self.max_context_len),
            dtype=torch.int32,
            device="cuda",
        )

        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, kv_indices_buf=self.cuda_graph_kv_indices[i]
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=None,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)


def should_use_tensor_core(
    kv_cache_dtype: torch.dtype,
    num_attention_heads: int,
    num_kv_heads: int,
) -> bool:
    """
    Determine whether to use tensor cores for attention computation.

    Args:
        kv_cache_dtype: Data type of the KV cache
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of key/value heads

    Returns:
        bool: Whether to use tensor cores
    """
    # Try to use environment variable first
    env_override = os.environ.get("SGLANG_FLASHINFER_USE_TENSOR_CORE")
    if env_override is not None:
        return env_override.lower() == "true"

    # Try to use _grouped_size_compiled_for_decode_kernels if available
    # This is for flashinfer <=0.1.6. Otherwise, there is an accuracy bug
    try:
        from flashinfer.decode import _grouped_size_compiled_for_decode_kernels

        if not _grouped_size_compiled_for_decode_kernels(
            num_attention_heads,
            num_kv_heads,
        ):
            return True
        else:
            return False
    except (ImportError, AttributeError):
        pass

    # Calculate GQA group size
    gqa_group_size = num_attention_heads // num_kv_heads

    # Determine based on dtype and GQA group size
    if kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return True
    elif kv_cache_dtype in (torch.float16, torch.half, torch.bfloat16):
        return gqa_group_size > 4
    else:
        return False


# Use as a fast path to override the indptr in flashinfer's plan function
# This is used to remove some host-to-device copy overhead.
global_override_indptr_cpu = None


def fast_decode_plan(
    self,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    pos_encoding_mode: str = "NONE",
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    q_data_type: Optional[Union[str, torch.dtype]] = None,
    kv_data_type: Optional[Union[str, torch.dtype]] = None,
    data_type: Optional[Union[str, torch.dtype]] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    non_blocking: bool = True,
) -> None:
    """
    A faster version of BatchDecodeWithPagedKVCacheWrapper::plan used for FlashInferMultiStepDraftBackend.
    Modifications:
    - Remove unnecessary device-to-device copy for the cuda graph buffers.
    - Remove unnecessary host-to-device copy for the metadata buffers.
    """
    batch_size = len(last_page_len)
    if logits_soft_cap is None:
        logits_soft_cap = 0.0

    # Handle data types consistently
    if data_type is not None:
        if q_data_type is None:
            q_data_type = data_type
        if kv_data_type is None:
            kv_data_type = data_type
    elif q_data_type is None:
        q_data_type = "float16"

    if kv_data_type is None:
        kv_data_type = q_data_type

    if self.use_tensor_cores:
        qo_indptr_host = _get_range_buf(batch_size + 1, "cpu")

    if self.is_cuda_graph_enabled:
        if batch_size != self._fixed_batch_size:
            raise ValueError(
                "The batch size should be fixed in cudagraph mode, the runtime batch size {} "
                " mismatches the batch size set during initialization {}".format(
                    batch_size, self._fixed_batch_size
                )
            )
        if len(indices) > len(self._paged_kv_indices_buf):
            raise ValueError(
                "The size of indices should be less than or equal to the allocated buffer"
            )
    else:
        self._paged_kv_indptr_buf = indptr
        self._paged_kv_indices_buf = indices
        self._paged_kv_last_page_len_buf = last_page_len
        if self.use_tensor_cores:
            self._qo_indptr_buf = qo_indptr_host.to(
                self.device, non_blocking=non_blocking
            )

    # Create empty tensors for dtype info if needed
    empty_q_data = torch.empty(
        0,
        dtype=(
            getattr(torch, q_data_type) if isinstance(q_data_type, str) else q_data_type
        ),
        device=self.device,
    )

    empty_kv_cache = torch.empty(
        0,
        dtype=(
            getattr(torch, kv_data_type)
            if isinstance(kv_data_type, str)
            else kv_data_type
        ),
        device=self.device,
    )

    indptr_host = (
        global_override_indptr_cpu
        if global_override_indptr_cpu is not None
        else indptr.cpu()
    )

    with torch.cuda.device(self.device):

        if self.use_tensor_cores:
            # ALSO convert last_page_len to CPU
            last_page_len_host = last_page_len.cpu()

            kv_lens_arr_host = get_seq_lens(indptr_host, last_page_len_host, page_size)

            try:
                # Make sure we pass exactly 15 arguments for tensor core version
                self._plan_info = self._cached_module.plan(
                    self._float_workspace_buffer,
                    self._int_workspace_buffer,
                    self._pin_memory_int_workspace_buffer,
                    qo_indptr_host,
                    indptr_host,
                    kv_lens_arr_host,
                    batch_size,  # total_num_rows
                    batch_size,
                    num_qo_heads,
                    num_kv_heads,
                    page_size,
                    self.is_cuda_graph_enabled,
                    head_dim,
                    head_dim,
                    False,  # causal
                )
            except Exception as e:
                raise RuntimeError(f"Error in standard plan: {e}")
        else:
            try:
                # Make sure we pass exactly 15 arguments for standard version
                self._plan_info = self._cached_module.plan(
                    self._float_workspace_buffer,
                    self._int_workspace_buffer,
                    self._pin_memory_int_workspace_buffer,
                    indptr_host,
                    batch_size,
                    num_qo_heads,
                    num_kv_heads,
                    page_size,
                    self.is_cuda_graph_enabled,
                    window_left,
                    logits_soft_cap,
                    head_dim,
                    head_dim,
                    empty_q_data,
                    empty_kv_cache,
                )
            except Exception as e:
                raise RuntimeError(f"Error in standard plan: {e}")

    self._pos_encoding_mode = pos_encoding_mode
    self._window_left = window_left
    self._logits_soft_cap = logits_soft_cap
    self._sm_scale = sm_scale
    self._rope_scale = rope_scale
    self._rope_theta = rope_theta
