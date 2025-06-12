from __future__ import annotations

"""
Support attention backend for NpuMLA.

#TODO
Enable speculative sampling in NpuMLA
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import torch
import torch_npu

from sglang.global_config import global_config
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashinfer_mla_backend import (
    FlashInferMLAIndicesUpdaterDecode,
)
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.attention.utils import (
    create_flashinfer_kv_indices,
    create_flashmla_kv_indices,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
    from sglang.srt.speculative.spec_info import SpecInfo


PAGE_SIZE = 128
BLOCK_SIZE = 128
MAX_SEQ_LEN = 4096


@dataclass
class NpuMLADecodeMetadata:
    npumla_metadata: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    num_splits: Optional[torch.Tensor] = None
    block_kv_indices: Optional[torch.Tensor] = None

    def __init__(
        self,
        npumla_metadata: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_splits: Optional[torch.Tensor] = None,
        block_kv_indices: Optional[torch.Tensor] = None,
        seq_lens_list=None,
    ):
        self.npumla_metadata = npumla_metadata
        self.num_splits = num_splits
        self.block_kv_indices = block_kv_indices
        self.seq_lens_list = seq_lens_list if seq_lens_list is not None else [1]


class NpuMLAIndicesUpdaterDecode:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        if "depseek" in model_runner.model_config.hf_config.architectures[0].lower():
            self.kv_lora_rank = model_runner.model_config.kv_lora_rank
            self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
            self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
            self.scaling = model_runner.model_config.scaling
        else:
            self.kv_lora_rank = None
        self.data_type = model_runner.dtype
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
        init_metadata_replay: bool = False,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]] = None,
        **fast_decode_kwargs,
    ):
        return self.call_begin_forward(
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
        if spec_info is None:
            kv_indices = (
                torch.zeros(bs * MAX_SEQ_LEN, dtype=torch.int32, device="npu")
                if not init_metadata_replay
                else fast_decode_kwargs["kv_indices"]
            )
            paged_kernel_lens_new = paged_kernel_lens.clone()
            paged_kernel_lens_new.fill_(MAX_SEQ_LEN)
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens_new, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            create_flashinfer_kv_indices(
                bs,
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.shape[1],
            )
            return kv_indices.reshape(bs, -1)


class NpuMLABackend(TorchNativeAttnBackend):
    """npumla attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
    ):
        super().__init__(model_runner)
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.skip_prefill = skip_prefill
        self.forward_metadata: Optional[NpuMLADecodeMetadata] = None

        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        if "depseek" in model_runner.model_config.hf_config.architectures[0].lower():
            self.kv_lora_rank = model_runner.model_config.kv_lora_rank
            self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
            self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
            self.v_head_dim = model_runner.model_config.v_head_dim
            self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
            self.scaling = model_runner.model_config.scaling

        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype

        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens

        self.mask_length = 2048
        self.attn_mask = ~torch.tril(
            torch.ones(
                (self.mask_length, self.mask_length),
                dtype=torch.bool,
                device=model_runner.device,
            )
        )

        max_bs = model_runner.req_to_token_pool.size
        self.kv_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )
        if not self.skip_prefill:
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

        self.q_indptr_decode = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=model_runner.device
        )
        self.indices_updater_decode = NpuMLAIndicesUpdaterDecode(model_runner, self)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        if forward_batch.forward_mode.is_decode_or_idle():
            if False:
                block_kv_indices = self.indices_updater_decode.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    init_metadata_replay=False,
                )
            else:
                max_seqlen_pad = (
                    forward_batch.seq_lens.max().item() + PAGE_SIZE - 1
                ) // PAGE_SIZE
                block_kv_indices = torch.full(
                    (bs, max_seqlen_pad),
                    -1,
                    dtype=torch.int32,
                    device=forward_batch.seq_lens.device,
                )
                create_flashmla_kv_indices(
                    bs,
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    None,
                    block_kv_indices,
                    self.req_to_token.stride(0),
                    max_seqlen_pad,
                )
            self.forward_metadata = NpuMLADecodeMetadata(
                None,
                None,
                block_kv_indices,
                forward_batch.seq_lens.cpu().tolist(),
            )
        else:
            self.forward_metadata = NpuMLADecodeMetadata(
                None,
                None,
                None,
                forward_batch.seq_lens.cpu().tolist(),
            )
            self.forward_metadata.seq_lens_list = forward_batch.seq_lens.cpu().tolist()

    def init_cuda_graph_state(
        self,
        max_bs: int,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        pass

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
        max_seqlen_pad = (num_tokens // bs + PAGE_SIZE - 1) // PAGE_SIZE
        self.forward_metadata = NpuMLADecodeMetadata(
            None,
            None,
            torch.full(
                (bs, max_seqlen_pad),
                0,
                dtype=torch.int32,
                device=seq_lens.device,
            ),
            [1] * bs,
        )

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
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1024

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if k is not None:
            if save_kv_cache:
                if k_rope is not None:
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        forward_batch.out_cache_loc,
                        k,
                        k_rope,
                    )
                else:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer,
                        forward_batch.out_cache_loc,
                        k,
                        v,
                    )
        bs = forward_batch.batch_size
        if q_rope is not None:
            q_nope = q.view(bs, -1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                bs, -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
        else:
            reshape_q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
            q_nope = reshape_q[..., : layer.v_head_dim]
            q_rope = reshape_q[..., layer.v_head_dim :]
            if q_rope.numel() == 0:
                q_rope = None

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
            q.dtype
        )
        if q_rope is None:
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
                layer.layer_id
            ).to(q.dtype)
        else:
            v_cache = k_cache
        o = self._run_npu_forward_decode(
            (q_nope, q_rope), k_cache, v_cache, layer, forward_batch
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = False,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if (
            forward_batch.forward_mode == ForwardMode.EXTEND
            or forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
        ):
            if k_rope is not None:
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    k_rope,
                )
                bs = forward_batch.batch_size
                q_nope = q.view(bs, -1, layer.tp_q_head_num, layer.v_head_dim)
                q_rope = q_rope.view(
                    bs, -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                )
                k_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                    layer.layer_id
                ).to(q.dtype)

                o = self._run_npu_forward_decode(
                    (q_nope, q_rope), k_cache, k_cache, layer, forward_batch
                )
                return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

            else:
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer,
                        forward_batch.out_cache_loc,
                        k,
                        v,
                    )
            use_gqa = layer.tp_q_head_num != layer.tp_k_head_num
            return self._run_npu_forward_extend(q, k, v, layer, forward_batch, use_gqa)

    def _run_npu_forward_extend(self, q, k, v, layer, forward_batch, use_gqa=False):
        """
        q: (b*s, N, q_dim=192)
        k: (b*s, N, k_dim=192)
        v: (b*s, N, v_dim=128)
        """
        if q.ndim == 2:
            q = q.view(q.shape[0], self.num_local_heads, -1)
        bs_qlen, q_heads, q_dim = q.size()
        _, k_heads, k_dim = k.size()
        _, v_heads, v_dim = v.size()

        if False:
            attn_weights = (
                torch.matmul(q, k.transpose(1, 2)) * layer.scaling
            )  # (bs, n, n)
            # assert attention_mask is not None
            # if attention_mask is not None:
            #     attn_weights += attention_mask

            attn_weights = torch.nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(q.dtype)

            # v = v[..., :self.kv_lora_rank]
            attn_ouput = torch.matmul(attn_weights, v)  # (bs, n, v_dim)
            # attn_ouput = attn_ouput.transpose(1,2).contiguous()
        else:
            bs = forward_batch.batch_size
            if use_gqa:
                attn_ouput = torch.empty(
                    bs_qlen, q_heads, v_dim, device=q.device, dtype=q.dtype
                )
                q_len_offset = 0
                for q_len in forward_batch.seq_len:
                    attn_ouput[q_len_offset : q_len_offset + q_len] = (
                        torch.ops.npu.npu_fused_infer_attention_score(
                            q[None, q_len_offset : q_len_offset + q_len],
                            k[None, q_len_offset : q_len_offset + q_len],
                            v[None, q_len_offset : q_len_offset + q_len],
                            num_heads=q_heads,
                            num_key_value_heads=k_heads,
                            input_layout="BSND",  # todo, TND not supports q_heads!=k_heads
                            atten_mask=self.attn_mask.unsqueeze(0),
                            sparse_mode=3,
                            scale=layer.scaling,
                            next_tokens=0,
                        )[0]
                    )
                    q_len_offset += q_len
            else:  # MHA
                if q_dim != v_dim:
                    reshape_q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
                    q_nope, q_rope = reshape_q.split(
                        [self.v_head_dim, self.qk_rope_head_dim], dim=-1
                    )
                    k_nope, k_rope = k.split(
                        [self.v_head_dim, self.qk_rope_head_dim], dim=-1
                    )

                    attn_ouput, _ = torch.ops.npu.npu_fused_infer_attention_score(
                        q_nope,
                        k_nope.unsqueeze(0),
                        v.unsqueeze(0),
                        query_rope=q_rope,
                        key_rope=k_rope.unsqueeze(0),
                        num_heads=q_heads,
                        input_layout="BSND",
                        atten_mask=self.attn_mask,
                        sparse_mode=3,
                        scale=layer.scaling,
                        next_tokens=0,
                    )
                else:
                    attn_ouput, _ = torch.ops.npu.npu_fused_infer_attention_score(
                        q.unsqueeze(0),
                        k.unsqueeze(0),
                        v.unsqueeze(0),
                        num_heads=q_heads,
                        input_layout="BSND",
                        atten_mask=self.attn_mask,
                        sparse_mode=3,
                        scale=layer.scaling,
                        next_tokens=0,
                    )
                attn_ouput = attn_ouput[..., : layer.v_head_dim]

        return attn_ouput.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _run_npu_forward_decode(self, q, k_cache, v_cache, layer, forward_batch):
        """
        q: (b, s, N, q_dim=576)
        k_cache: (tokens_capticy, 1, k_dim=576)
        """
        if not isinstance(q, torch.Tensor):
            q_nope, q_rope = q
        else:
            q_nope, q_rope = q.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        b, s, n, _ = q_nope.size()
        _, k_heads, k_dim = k_cache.size()

        if q_rope is not None:  # MLA
            k_cache = k_cache.view(-1, BLOCK_SIZE, k_dim)
            # k_nope, k_rope = k_cache.split(
            #     [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            # )  # todo, there is a bug
            k_nope = k_cache[..., : self.kv_lora_rank]
            k_rope = k_cache[..., self.kv_lora_rank :]

            attn_ouput, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_nope,
                k_nope,
                k_nope,
                query_rope=q_rope,
                key_rope=k_rope,
                num_heads=n,
                num_key_value_heads=1,
                input_layout="BSND",
                atten_mask=None,
                sparse_mode=0,
                scale=layer.scaling,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=self.forward_metadata.block_kv_indices[
                    : forward_batch.batch_size
                ],
                block_size=BLOCK_SIZE,
                actual_seq_lengths_kv=self.forward_metadata.seq_lens_list,
            )
        else:  # MHA
            if False:
                req_to_token = forward_batch.req_to_token_pool.req_to_token
                attn_output = q_nope.new_zeros((b, s, n, k_dim))
                for seq_idx in range(b):
                    seq_len_kv = forward_batch.seq_len[seq_idx]
                    req_pool_idx = forward_batch.req_pool_indices[seq_idx]
                    per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
                    k = k_cache[per_req_tokens]
                    v = v_cache[per_req_tokens]
                    attn_ouput_idx, _ = torch.ops.npu.npu_fused_infer_attention_score(
                        q_nope[seq_idx].unsqueeze(0),
                        k.unsqueeze(0),
                        v.unsqueeze(0),
                        num_heads=n,
                        num_key_value_heads=k_heads,
                        input_layout="BSND",
                        atten_mask=None,
                        sparse_mode=0,
                        scale=layer.scaling,
                    )
                    attn_ouput[seq_idx] = attn_ouput_idx.squeeze(0)
            else:
                seq_len_kv = forward_batch.seq_lens
                attn_ouput, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q_nope,
                    k_cache.view(-1, PAGE_SIZE, k_heads * k_dim),
                    v_cache.view(-1, PAGE_SIZE, k_heads * k_dim),
                    num_heads=n,
                    num_key_value_heads=k_heads,
                    input_layout="BSND",
                    atten_mask=None,
                    block_size=PAGE_SIZE,
                    block_table=self.forward_metadata.block_kv_indices,
                    actual_seq_lengths_kv=self.forward_metadata.seq_lens_list,
                    scale=layer.scaling,
                )
        attn_ouput = attn_ouput.view(b * s, layer.tp_q_head_num, layer.v_head_dim)
        # attn_ouput = q.new_zeros((q.shape[0], layer.tp_q_head_num, layer.v_head_dim))
        return attn_ouput


# TODO: multi step kv indices optimization
class NpuMLAMultiStepDraftBackend:
    """
    Wrap multiple npumla attention backends as one for multiple consecutive
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
                f"Currently NpuMLA only supports topk=1 for speculative decoding"
            )
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )

        self.attn_backends = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                NpuMLABackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    kv_last_page_len_buf=None,
                )
            )

    def common_template(
        self,
        forward_batch: ForwardBatch,
        call_fn: Callable,
    ):
        assert forward_batch.spec_info is not None

        for i in range(self.speculative_num_steps - 1):
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            assert forward_batch.spec_info is not None
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, call_fn)
