from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.lightning_attn.lightning_attn_minimax import (
    LightningAttnPrefillKernel,
    linear_decode_forward_triton,
)
from sglang.srt.layers.attention.lightning_attn.lightning_attn_seg_la import (
    SegLaMeta,
    seg_la_fwd,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
from sglang.srt.utils import get_compiler_backend

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)


def _build_slope_tensor(n_attention_heads: int):
    def get_slopes(n):

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = torch.tensor(get_slopes(n_attention_heads), dtype=torch.float32).reshape(
        n_attention_heads, 1, 1
    )
    return slopes


def is_linear_layer(layer_idx, layer_group_size):
    if layer_idx is None:
        return False
    if layer_group_size > 0:
        return (layer_idx + 1) % layer_group_size != 0
    else:
        return False


def get_num_prefills(forward_batch: ForwardBatch):
    if forward_batch.forward_mode.is_extend():
        return forward_batch.batch_size
    elif forward_batch.forward_mode.is_mixed():
        return (
            len(forward_batch.extend_seq_lens)
            if forward_batch.extend_seq_lens is not None
            else 0
        )
    else:
        return 0


def get_num_prefill_tokens(forward_batch: ForwardBatch):
    if forward_batch.forward_mode.is_extend() or forward_batch.forward_mode.is_mixed():
        if forward_batch.extend_num_tokens is not None:
            return forward_batch.extend_num_tokens
        elif forward_batch.extend_seq_lens is not None:
            return int(forward_batch.extend_seq_lens.sum().item())
        else:
            return 0
    else:
        return 0


def get_num_decode_tokens(forward_batch: ForwardBatch):
    if forward_batch.forward_mode.is_decode():
        return forward_batch.batch_size
    elif forward_batch.forward_mode.is_mixed():
        num_prefills = get_num_prefills(forward_batch)
        return max(0, forward_batch.batch_size - num_prefills)
    else:
        return 0


@dataclass
class LightningAttentionMetadata:
    """Metadata to be init once in the model forward pass,
    each layer's forward pass can reuse the metadata.

    For each init metadata function, we will try set up them in below order
    """

    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor = None
    # Maximum sequence length for query
    max_seq_len_q: int = 1
    # Maximum sequence length for key
    max_seq_len_k: int = 0
    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor = None
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor = None
    # Window size (typically used by Gemma)
    page_table: torch.Tensor = None
    linear_page_table: torch.Tensor = None
    req_pool_indices: torch.Tensor = None
    # sequence lengths for query per request
    seqlens_q: torch.Tensor = None
    batch_size: int = 0
    is_decode_req_tensor: torch.Tensor = None


class LightningAttentionBackend(AttentionBackend):
    """Lightning Attention backend implementation.

    Note about CUDA Graph:
    - We only support CUDA Graph for Decode (Normal Decode and Draft Decode) and Target Verify.
    - We don't support CUDA Graph for Extend and Draft Extend.
    - When server init, init_cuda_graph_state will be called first and then init_cuda_graph_capture will be called.
    - For each forward batch, init_replay_cuda_graph will be called first and then replay the graph.
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        self.forward_metadata: LightningAttentionMetadata = None
        # extra metadata for handling speculative decoding topk > 1, extended draft decode and verify
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.decode_cuda_graph_metadata = {}
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.req_to_constant = model_runner.req_to_token_pool.req_to_constant
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype
        self.page_size = model_runner.page_size
        self.layer_group_size = model_runner.model_config.hf_config.layer_group_size
        self.BLOCK = (
            model_runner.model_config.block
            if hasattr(model_runner.model_config, "block")
            else 256
        )
        total_num_heads = model_runner.model_config.hf_config.num_attention_heads
        tp_heads = total_num_heads // get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        slope_rate = _build_slope_tensor(total_num_heads)
        num_hidden_layers = model_runner.model_config.hf_config.num_hidden_layers

        if num_hidden_layers <= 1:
            slope_rate_list = [slope_rate * (1 + 1e-5)]
        else:
            slope_rate_list = [
                slope_rate * (1 - layer_id / (num_hidden_layers - 1) + 1e-5)
                for layer_id in range(num_hidden_layers)
            ]
        self.tp_slope = [
            slope_rate_list[layer_id][tp_rank * tp_heads : (tp_rank + 1) * tp_heads]
            .contiguous()
            .to(self.device)
            for layer_id in range(num_hidden_layers)
        ]
        self.linear_backend = getattr(
            model_runner.model_config.hf_config, "linear_backend", "minimax"
        )
        logger.info(
            f"linear_backend for linear attention in hybrid_linear_backend: {self.linear_backend}"
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize forward metadata hence all layers in the forward pass can reuse it."""
        metadata = LightningAttentionMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device

        if forward_batch.forward_mode.is_decode_or_idle():
            # Normal Decode
            metadata.req_pool_indices = forward_batch.req_pool_indices
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens.max().item()
            metadata.cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]
            metadata.linear_page_table = (
                forward_batch.req_to_token_pool.req_to_constant[
                    forward_batch.req_pool_indices
                ]
            )
        elif forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            metadata.req_pool_indices = forward_batch.req_pool_indices
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]
            metadata.linear_page_table = (
                forward_batch.req_to_token_pool.req_to_constant[
                    forward_batch.req_pool_indices
                ]
            )
            if (
                any(forward_batch.extend_prefix_lens_cpu)
                or forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
            ):
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                metadata.max_seq_len_q = metadata.max_seq_len_k
                metadata.cu_seqlens_q = metadata.cu_seqlens_k

        metadata.batch_size = forward_batch.batch_size
        batch_seqlen_q = metadata.cu_seqlens_q[1:] - metadata.cu_seqlens_q[:-1]
        batch_seqlen_k = metadata.cu_seqlens_k[1:] - metadata.cu_seqlens_k[:-1]
        is_decode_req_tensor = torch.ones_like(forward_batch.seq_lens)
        is_decode_req_tensor.masked_fill_(batch_seqlen_q == batch_seqlen_k, 0)
        metadata.is_decode_req_tensor = is_decode_req_tensor
        metadata.seqlens_q = metadata.cu_seqlens_q.diff()

        # Convert the page table to a strided format which is needed by FA3 API
        if self.page_size > 1:
            self.strided_indices = torch.arange(
                0, metadata.page_table.shape[1], self.page_size, device=self.device
            )
            metadata.page_table = (
                metadata.page_table[:, self.strided_indices] // self.page_size
            )

        self.forward_metadata = metadata

    def _prefill_and_mix_infer(
        self,
        q,
        k,
        v,
        kv_cache,
        state_indices_tensor,
        forward_batch,
        layer,
        cu_seqlens_q,
        cu_seqlens_k,
    ):
        hidden = []
        for _prefill_idx in range(get_num_prefills(forward_batch)):
            if _prefill_idx >= len(forward_batch.extend_start_loc):
                break
            if _prefill_idx >= len(state_indices_tensor):
                break

            _start = forward_batch.extend_start_loc[_prefill_idx]

            if _prefill_idx + 1 < len(forward_batch.extend_start_loc):
                _end = forward_batch.extend_start_loc[_prefill_idx + 1]
            else:
                if forward_batch.extend_seq_lens is not None and _prefill_idx < len(
                    forward_batch.extend_seq_lens
                ):
                    seq_len = forward_batch.extend_seq_lens[_prefill_idx]
                    _end = _start + seq_len
                else:
                    _end = q.shape[0]

            slot_id = state_indices_tensor[_prefill_idx]
            qs = q[_start:_end].transpose(0, 1).contiguous()
            ks = k[_start:_end].transpose(0, 1).contiguous()
            vs = v[_start:_end].transpose(0, 1).contiguous()
            slice_layer_cache = kv_cache[slot_id, ...]
            batch_seqlen_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            batch_seqlen_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            if batch_seqlen_q[_prefill_idx] == batch_seqlen_k[_prefill_idx]:
                slice_layer_cache.copy_(0).to(torch.float32)
            out_slice = LightningAttnPrefillKernel.jit_linear_forward_prefix(
                qs,
                ks,
                vs,
                slice_layer_cache,
                self.tp_slope[layer.layer_id],
                self.BLOCK,
                layer_idx=layer.layer_id,
            )
            hidden.append(out_slice.contiguous())
        if get_num_decode_tokens(forward_batch) > 0:
            hidden.append(
                self._decode_infer(
                    q, k, v, kv_cache, state_indices_tensor, forward_batch, layer
                )
            )

        if not hidden:
            return torch.empty((0, q.size(-1)), device=q.device, dtype=q.dtype)

        hidden = torch.concat(hidden, dim=0).contiguous()
        return hidden

    def _decode_infer(
        self, q, k, v, kv_cache, state_indices_tensor, forward_batch, layer
    ):
        num_prefill_tokens = get_num_prefill_tokens(forward_batch)
        num_prefills = get_num_prefills(forward_batch)
        q = q[num_prefill_tokens:].unsqueeze(2).contiguous()
        k = k[num_prefill_tokens:].unsqueeze(2).contiguous()
        v = v[num_prefill_tokens:].unsqueeze(2).contiguous()
        slot_id = state_indices_tensor[num_prefills:]

        assert len(slot_id) == q.shape[0], (
            f"slot_id length {len(slot_id)} does not match decode batch size {q.shape[0]}. "
            "This indicates a bug in the upstream logic that should be investigated."
        )
        hidden = linear_decode_forward_triton(
            q, k, v, kv_cache, self.tp_slope[layer.layer_id], slot_id, 32
        )
        return hidden

    def linear_attention_entry(
        self, q, k, v, kv_cache, state_indices_tensor, attn_metadata, layer
    ):
        seg_meta = SegLaMeta(
            batch_size=attn_metadata.batch_size,
            max_q_length=None,
            q_offsets=attn_metadata.cu_seqlens_q,
            s_offsets=state_indices_tensor,
            q_lengths=attn_metadata.seqlens_q,
            s_scales=attn_metadata.is_decode_req_tensor,
            mask=None,
        )
        hidden = seg_la_fwd(
            q=q,
            k=k,
            v=v,
            s=kv_cache,
            decay_scales=self.tp_slope[layer.layer_id],
            meta=seg_meta,
            decouple=True,
        )
        return hidden

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        if self.kv_cache_dtype_str != "auto" and layer.k_scale is not None:
            q = q.to(self.kv_cache_dtype)

        # Get the appropriate page table based on whether we're using local attention
        cu_seqlens_q = metadata.cu_seqlens_q
        cu_seqlens_k = metadata.cu_seqlens_k
        state_indices_tensor = metadata.linear_page_table

        # Do linear attention
        kv_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)[0]
        if self.linear_backend == "minimax":
            o = self._prefill_and_mix_infer(
                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k,
                v,
                kv_cache,
                state_indices_tensor,
                forward_batch,
                layer,
                cu_seqlens_q,
                cu_seqlens_k,
            )
        elif self.linear_backend == "seg_la":
            o = self.linear_attention_entry(
                q, k, v, kv_cache, state_indices_tensor, metadata, layer
            )
        else:
            raise ValueError(
                f"linear backend: {self.linear_backend} is not support for now"
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
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        if self.kv_cache_dtype_str != "auto":
            q = q.to(self.kv_cache_dtype)

        # Do linear attention
        kv_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)[0]
        state_indices_tensor = metadata.linear_page_table
        if self.linear_backend == "minimax":
            o = self._decode_infer(
                q, k, v, kv_cache, state_indices_tensor, forward_batch, layer
            )
        elif self.linear_backend == "seg_la":
            o = self.linear_attention_entry(
                q, k, v, kv_cache, state_indices_tensor, metadata, layer
            )
        else:
            raise ValueError(
                f"linear backend: {self.linear_backend} is not support for now"
            )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Initialize CUDA graph state for the attention backend.

        Args:
            max_bs (int): Maximum batch size to support in CUDA graphs

        This creates fixed-size tensors that will be reused during CUDA graph replay
        to avoid memory allocations.
        """
        # This is being used by normal decode and draft decode when topk == 1
        self.decode_cuda_graph_metadata = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "page_table": torch.zeros(
                max_bs,
                (self.max_context_len + self.page_size - 1) // self.page_size,
                dtype=torch.int32,
                device=self.device,
            ),
            "linear_page_table": torch.full(
                (max_bs,),
                -1,
                dtype=torch.int32,
                device=self.device,
            ),
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            ),
        }

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
        """Initialize forward metadata for capturing CUDA graph."""
        metadata = LightningAttentionMetadata()

        metadata.batch_size = bs
        metadata.is_decode_req_tensor = torch.ones_like(seq_lens)

        if forward_mode.is_decode_or_idle():
            # Normal Decode
            # Get sequence information
            metadata.cache_seqlens_int32 = seq_lens.to(torch.int32)
            batch_size = len(seq_lens)
            device = seq_lens.device
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
            )
            # Precompute maximum sequence length
            metadata.max_seq_len_k = seq_lens.max().item()
            # Precompute page table
            metadata.page_table = self.decode_cuda_graph_metadata["page_table"][
                req_pool_indices, :
            ]
            metadata.linear_page_table = self.decode_cuda_graph_metadata[
                "linear_page_table"
            ][req_pool_indices]
            metadata.req_pool_indices = req_pool_indices
            # Precompute cumulative sequence lengths
            metadata.cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
            metadata.seqlens_q = metadata.cu_seqlens_q.diff()
            self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_metadata = metadata

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
        out_cache_loc: torch.Tensor = None,
    ):
        """Initialize forward metadata for replaying CUDA graph."""
        req_pool_indices_long = req_pool_indices
        # req_pool_indices_long[bs:] = -1
        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs]
        metadata = None

        if forward_mode.is_decode_or_idle():
            # Normal Decode
            metadata = self.decode_cuda_graph_metadata[bs]
            max_len = seq_lens.max().item()
            max_seq_pages = (max_len + self.page_size - 1) // self.page_size
            metadata.max_seq_len_k = max_len
            metadata.req_pool_indices = req_pool_indices_long
            normal_decode_set_metadata(
                metadata,
                self.req_to_token,
                self.req_to_constant,
                req_pool_indices,
                self.decode_cuda_graph_metadata["strided_indices"],
                max_seq_pages,
                seq_lens,
                self.page_size,
            )
            metadata.batch_size = bs
            metadata.is_decode_req_tensor = torch.ones_like(seq_lens)
            if metadata.cu_seqlens_q is not None:
                metadata.seqlens_q = metadata.cu_seqlens_q.diff()
        self.forward_metadata = metadata

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        return 0


@torch.compile(dynamic=True, backend=get_compiler_backend())
def normal_decode_set_metadata(
    metadata,
    req_to_token,
    req_to_constant,
    req_pool_indices,
    strided_indices,
    max_seq_pages,
    seq_lens,
    page_size,
):
    metadata.cache_seqlens_int32 = seq_lens.to(torch.int32)
    metadata.cu_seqlens_k[1:].copy_(torch.cumsum(seq_lens, dim=0, dtype=torch.int32))
    page_indices = req_to_token[
        req_pool_indices[:, None],
        strided_indices[:max_seq_pages][None, :],
    ]
    metadata.page_table[:, :max_seq_pages].copy_(page_indices // page_size)
    metadata.page_table[:, max_seq_pages:].fill_(0)
    metadata.linear_page_table[:].copy_(req_to_constant[req_pool_indices])
