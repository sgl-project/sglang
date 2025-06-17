from __future__ import annotations

"""
Support attention backend for Cutlass MLA.

"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch
import triton

from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
from sglang.srt.layers.attention.utils import create_flashmla_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInfo

_is_cuda = is_cuda()
if _is_cuda:
    from sgl_kernel import cutlass_mla_decode, cutlass_mla_get_workspace_size


# Cutlass MLA only supports pagesize=128
PAGE_SIZE = 128


@dataclass
class CutlassMLADecodeMetadata:
    workspace: Optional[torch.Tensor] = None
    block_kv_indices: Optional[torch.Tensor] = None

    def __init__(
        self,
        workspace: Optional[torch.Tensor] = None,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        self.workspace = workspace
        self.block_kv_indices = block_kv_indices


class CutlassMLABackend(FlashInferMLAAttnBackend):
    """Cutlass attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model_runner, skip_prefill, kv_indptr_buf, kv_last_page_len_buf
        )

        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.forward_metadata: Union[CutlassMLADecodeMetadata] = None
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.scaling = model_runner.model_config.scaling
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim

    def init_forward_metadata(self, forward_batch: ForwardBatch):

        bs = forward_batch.batch_size
        spec_info = forward_batch.spec_info
        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None:
                max_seqlen_pad = triton.cdiv(
                    forward_batch.seq_lens_cpu.max().item(), PAGE_SIZE
                )
                block_kv_indices = torch.full(
                    (bs, max_seqlen_pad),
                    -1,
                    dtype=torch.int32,
                    device=forward_batch.seq_lens.device,
                )
                create_flashmla_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    None,
                    block_kv_indices,
                    self.req_to_token.stride(0),
                    max_seqlen_pad,
                    PAGE_SIZE,
                )
                workspace_size = cutlass_mla_get_workspace_size(
                    max_seqlen_pad * PAGE_SIZE, bs, num_kv_splits=1
                )
                workspace = torch.empty(
                    workspace_size, device="cuda", dtype=torch.uint8
                )
                self.forward_metadata = CutlassMLADecodeMetadata(
                    workspace,
                    block_kv_indices,
                )
            else:
                super().init_forward_metadata(forward_batch)
        else:
            super().init_forward_metadata(forward_batch)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        if block_kv_indices is None:
            cuda_graph_kv_indices = torch.full(
                (max_bs, (self.max_context_len + PAGE_SIZE) // PAGE_SIZE),
                1,
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = block_kv_indices

        workspace_size = cutlass_mla_get_workspace_size(
            cuda_graph_kv_indices.shape[1] * PAGE_SIZE, max_bs, num_kv_splits=1
        )
        self.cuda_graph_mla_workspace = torch.empty(
            workspace_size, device="cuda", dtype=torch.uint8
        )
        self.cuda_graph_kv_indices = cuda_graph_kv_indices

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
            if spec_info is None:
                max_seqlen_pad = self.cuda_graph_kv_indices.shape[1]

                create_flashmla_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    seq_lens,
                    None,
                    self.cuda_graph_kv_indices,
                    self.req_to_token.stride(0),
                    self.cuda_graph_kv_indices.stride(0),
                    PAGE_SIZE,
                )
                self.forward_metadata = CutlassMLADecodeMetadata(
                    self.cuda_graph_mla_workspace,
                    self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
                )
        else:
            super().init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
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

        if forward_mode.is_decode_or_idle():
            assert seq_lens_cpu is not None
            seq_lens = seq_lens[:bs]

            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices[:bs],
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
                PAGE_SIZE,
            )
        else:
            super().init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

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
        cache_loc = forward_batch.out_cache_loc

        if k is not None:
            assert v is not None
            if save_kv_cache:
                if k_rope is not None:
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )
                else:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        v,
                    )

        # Reshape inputs
        if q_rope is not None:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
        else:
            reshaped_q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = reshaped_q[:, :, : layer.v_head_dim]
            q_rope = reshaped_q[:, :, layer.v_head_dim :]

        q_nope = q_nope.to(self.q_data_type)
        q_rope = q_rope.to(self.q_data_type)

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

        o = cutlass_mla_decode(
            q_nope=q_nope,
            q_pe=q_rope,
            kv_c_and_k_pe_cache=k_cache.view(-1, PAGE_SIZE, self.kv_cache_dim),
            seq_lens=forward_batch.seq_lens.to(torch.int32),
            page_table=self.forward_metadata.block_kv_indices,
            workspace=self.forward_metadata.workspace,
            sm_scale=layer.scaling,
            num_kv_splits=1,
        )

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
