from __future__ import annotations

"""
Support attention backend for flashinfer MLA.
The flashinfer_mla_disable_ragged flag controls whether to use ragged prefill wrapper and defaults to be false.
When it's set to false, all wrappers are BatchMLAPaged wrapper.
When it's set to true, the backend uses BatchRagged and BatchMLAPaged wrapper for prefilling,
and uses BatchMLAPaged wrapper for decoding.
More details can be found in https://docs.flashinfer.ai/api/mla.html
"""

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import triton

from sglang.global_config import global_config
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashinfer_backend import (
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
from sglang.srt.utils import is_flashinfer_available

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInfo

if is_flashinfer_available():
    from flashinfer import (
        BatchMLAPagedAttentionWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )


@dataclass
class DecodeMetadata:
    decode_wrapper: BatchMLAPagedAttentionWrapper


@dataclass
class PrefillMetadata:
    prefill_wrapper: BatchMLAPagedAttentionWrapper
    use_ragged: bool


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None


class FlashInferMLAAttnBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Parse constants
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.skip_prefill = skip_prefill

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

        if not self.skip_prefill:
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

        if q_indptr_decode_buf is None:
            self.q_indptr_decode = torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=model_runner.device
            )
        else:
            self.q_indptr_decode = q_indptr_decode_buf

        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )

        if not self.skip_prefill:
            self.prefill_wrapper_paged = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                backend="auto",
            )

            # FlashinferMLA backend uses mla wrapper for target verify
            self.prefill_wrapper_verify = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                backend="auto",
            )

        self.decode_wrapper = BatchMLAPagedAttentionWrapper(
            self.workspace_buffer, backend="auto"
        )

        # Create indices updater
        if not skip_prefill:
            self.indices_updater_prefill = FlashInferMLAIndicesUpdaterPrefill(
                model_runner, self
            )

        self.indices_updater_decode = FlashInferMLAIndicesUpdaterDecode(
            model_runner, self
        )

        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None
        self.decode_cuda_graph_metadata = {}
        self.prefill_cuda_graph_metadata = {}  # For verify

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                decode_wrapper=self.decode_wrapper,
                init_metadata_replay=False,
            )
            self.forward_metadata = DecodeMetadata(self.decode_wrapper)
        elif forward_batch.forward_mode.is_draft_extend():
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrapper_paged=self.prefill_wrapper_paged,
                use_ragged=False,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(self.prefill_wrapper_paged, False)
        elif forward_batch.forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrapper_paged=self.prefill_wrapper_verify,
                use_ragged=False,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(self.prefill_wrapper_verify, False)
        else:
            prefix_lens = forward_batch.extend_prefix_lens
            extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            use_ragged = (
                not global_server_args_dict["flashinfer_mla_disable_ragged"]
                and extend_no_prefix
            )

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
        self.cuda_graph_qo_indptr = self.q_indptr_decode.clone()
        self.cuda_graph_kv_indptr = self.kv_indptr.clone()
        self.cuda_graph_kv_lens = torch.ones(
            (max_bs,), dtype=torch.int32, device=self.device
        )

        # For fast decode plan in graph replaying
        self.cuda_graph_qo_indptr_cpu = self.cuda_graph_qo_indptr.to("cpu")
        self.cuda_graph_kv_indptr_cpu = self.cuda_graph_kv_indptr.to("cpu")
        self.fast_decode_kwargs = {
            "qo_indptr_cpu": self.cuda_graph_qo_indptr_cpu,
            "kv_indptr_cpu": self.cuda_graph_kv_indptr_cpu,
            "kv_indices": self.cuda_graph_kv_indices,
        }

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
                qo_indptr=self.cuda_graph_qo_indptr[: num_tokens + 1],
                kv_indptr=self.cuda_graph_kv_indptr[: num_tokens + 1],
                kv_indices=self.cuda_graph_kv_indices,
                kv_len_arr=self.cuda_graph_kv_lens[:num_tokens],
                backend="auto",
            )

            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_decode.update(
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                decode_wrapper=decode_wrapper,
                init_metadata_replay=False,
                spec_info=spec_info,
            )
            self.decode_cuda_graph_metadata[bs] = decode_wrapper
            self.forward_metadata = DecodeMetadata(decode_wrapper)
            decode_wrapper.plan = partial(fast_mla_decode_plan, decode_wrapper)
        elif forward_mode.is_target_verify():
            verify_wrapper = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                use_cuda_graph=True,
                qo_indptr=self.cuda_graph_qo_indptr[: bs + 1],
                kv_indptr=self.cuda_graph_kv_indptr[: bs + 1],
                kv_indices=self.cuda_graph_kv_indices,
                kv_len_arr=self.cuda_graph_kv_lens[:bs],
                backend="auto",
            )
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrapper_paged=verify_wrapper,
                use_ragged=False,
                spec_info=spec_info,
            )
            self.prefill_cuda_graph_metadata[bs] = verify_wrapper
            self.forward_metadata = PrefillMetadata(verify_wrapper, False)
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
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        if forward_mode.is_decode_or_idle():
            assert seq_lens_cpu is not None
            kv_len_arr_cpu = seq_lens_cpu[:bs]
            self.cuda_graph_kv_indptr_cpu[1 : bs + 1] = torch.cumsum(
                kv_len_arr_cpu, dim=0
            )
            self.fast_decode_kwargs.update(
                {
                    "qo_indptr_cpu": self.cuda_graph_qo_indptr_cpu[: bs + 1],
                    "kv_indptr_cpu": self.cuda_graph_kv_indptr_cpu[: bs + 1],
                    "kv_len_arr_cpu": kv_len_arr_cpu,
                }
            )

            self.indices_updater_decode.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_sum,
                decode_wrapper=self.decode_cuda_graph_metadata[bs],
                init_metadata_replay=True,
                spec_info=spec_info,
                **self.fast_decode_kwargs,
            )
        elif forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrapper_paged=self.prefill_cuda_graph_metadata[bs],
                use_ragged=False,
                spec_info=spec_info,
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
        save_kv_cache: bool = True,
    ):

        cache_loc = forward_batch.out_cache_loc
        logits_soft_cap = layer.logit_cap
        prefill_wrapper_paged = self.forward_metadata.prefill_wrapper
        qall = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

        # Save kv cache
        if save_kv_cache and k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        if self.forward_metadata.use_ragged:
            # ragged prefill
            o, _ = self.prefill_wrapper_ragged.forward_return_lse(
                qall,
                k.view(-1, layer.tp_k_head_num, layer.head_dim),
                v.view(-1, layer.tp_k_head_num, layer.v_head_dim),
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )
        else:
            # mla paged prefill
            o = prefill_wrapper_paged.run(
                qall[:, :, : layer.v_head_dim],
                qall[:, :, layer.v_head_dim :],
                k_buf[:, :, : layer.v_head_dim],
                k_buf[:, :, layer.v_head_dim :],
            )

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
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
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.q_indptr = attn_backend.q_indptr_decode

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrapper: BatchMLAPagedAttentionWrapper,
        init_metadata_replay: bool = False,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]] = None,
        **fast_decode_kwargs,
    ):
        decode_wrapper = decode_wrapper or self.decode_wrapper
        self.call_begin_forward(
            decode_wrapper,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            self.q_indptr,
            self.kv_indptr,
            init_metadata_replay,
            spec_info,
            **fast_decode_kwargs,
        )

    def call_begin_forward(
        self,
        wrapper: BatchMLAPagedAttentionWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        q_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        init_metadata_replay: bool = False,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]] = None,
        **fast_decode_kwargs,
    ):
        bs = len(req_pool_indices)
        q_indptr = q_indptr[: bs + 1]
        kv_lens = paged_kernel_lens.to(torch.int32)
        sm_scale = self.scaling
        if spec_info is None:
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = (
                torch.empty(paged_kernel_lens_sum, dtype=torch.int32, device="cuda")
                if not init_metadata_replay
                else fast_decode_kwargs["kv_indices"]
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
        else:
            kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

        if not init_metadata_replay:
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
        else:
            wrapper.plan(
                fast_decode_kwargs["qo_indptr_cpu"],
                fast_decode_kwargs["kv_indptr_cpu"],
                kv_indices,
                fast_decode_kwargs["kv_len_arr_cpu"],
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
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
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
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged

    def update(
        self,
        req_pool_indices: torch.Tnesor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrapper_paged: BatchMLAPagedAttentionWrapper,
        use_ragged: bool,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]] = None,
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
            spec_info,
        )

    def call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchMLAPagedAttentionWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]] = None,
    ):
        bs = len(seq_lens)
        sm_scale = self.scaling

        if spec_info is None:
            assert len(seq_lens) == len(req_pool_indices)
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
            custom_mask = None
        else:
            assert isinstance(spec_info, EagleDraftInput) or isinstance(
                spec_info, EagleVerifyInput
            )
            # TODO: Support topk > 1 with custom mask
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    paged_kernel_lens,
                    paged_kernel_lens_sum,
                    self.req_to_token,
                )
            )

        if use_ragged:
            # ragged prefill
            wrapper_ragged.begin_forward(
                qo_indptr=qo_indptr,
                kv_indptr=qo_indptr,
                num_qo_heads=self.num_local_heads,
                num_kv_heads=self.num_local_heads,
                head_dim_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,
                head_dim_vo=self.v_head_dim,
                q_data_type=self.q_data_type,
            )
        else:
            # mla paged prefill
            kv_len_arr = kv_indptr[1:] - kv_indptr[:-1]
            wrapper_paged.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_len_arr,
                self.num_local_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                1,
                True,
                sm_scale,
                self.q_data_type,
                self.data_type,
            )


class FlashInferMLAMultiStepDraftBackend:
    """
    Wrap multiple flashinfer mla attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        from sglang.srt.speculative.eagle_utils import generate_draft_decode_kv_indices

        if topk > 1:
            raise ValueError(
                f"Currently Flashinfer MLA only supports topk=1 for speculative decoding"
            )
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices

        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.q_indptr_decode = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=model_runner.device
        )

        self.attn_backends = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                FlashInferMLAAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    q_indptr_decode_buf=self.q_indptr_decode,
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
            num_seqs,
            self.topk,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            triton.next_power_of_2(num_seqs),
            triton.next_power_of_2(self.speculative_num_steps),
            triton.next_power_of_2(bs),
        )

        assert forward_batch.spec_info is not None
        assert isinstance(forward_batch.spec_info, EagleDraftInput)

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices = torch.zeros(
            (
                self.speculative_num_steps,
                forward_batch.batch_size * self.topk * self.max_context_len,
            ),
            dtype=torch.int32,
            device="cuda",
        )

        def call_fn(i, forward_batch):
            assert forward_batch.spec_info is not None
            assert isinstance(forward_batch.spec_info, EagleDraftInput)
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int):
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, max_bs * self.max_context_len),
            dtype=torch.int32,
            device="cuda",
        )

        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, kv_indices_buf=self.cuda_graph_kv_indices[i]
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
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)


def fast_mla_decode_plan(
    self,
    qo_indptr_cpu: torch.Tensor,
    kv_indptr_cpu: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_len_arr_cpu: torch.Tensor,
    num_heads: int,
    head_dim_ckv: int,
    head_dim_kpe: int,
    page_size: int,
    causal: bool,
    sm_scale: float,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
) -> None:
    """A faster version of BatchMLAPagedAttentionWrapper::plan,
    for skipping the stream synchronization in original plan function during
    cuda graph replaying.
    """
    self._causal = causal
    self._page_size = page_size
    self._sm_scale = sm_scale

    with self.device as device:
        stream = torch.cuda.current_stream(device).cuda_stream
        self._cached_module.plan(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_cpu,
            kv_indptr_cpu,
            kv_len_arr_cpu,
            num_heads,
            head_dim_ckv,
            causal,
            stream,
        )
