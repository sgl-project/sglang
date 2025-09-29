from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from sglang.srt.custom_op import CustomOp
from sglang.srt.debug_utils.dumper import dumper
from sglang.srt.utils import add_prefix, is_npu

if not is_npu():
    from sglang.srt.layers.attention.nsa.tilelang_kernel import act_quant
    import deep_gemm

from sglang.srt.layers.attention.nsa.utils import NSA_DUAL_STREAM, NSA_USE_REAL_INDEXER
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import add_prefix, align, is_cuda

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

DUAL_STREAM_TOKEN_THRESHOLD = 1024 if is_cuda() else 0


class BaseIndexerMetadata(ABC):
    @abstractmethod
    def get_seqlens_int32(self) -> torch.Tensor:
        """
        Return: (batch_size,) int32 tensor
        """

    @abstractmethod
    def get_page_table_64(self) -> torch.Tensor:
        """
        Return: (batch_size, num_blocks) int32, page table.
                The page size of the table is 64.
        """

    @abstractmethod
    def get_seqlens_expanded(self) -> torch.Tensor:
        """
        Return: (sum_extend_seq_len,) int32 tensor
        """

    @abstractmethod
    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """
        Perform topk selection on the logits and possibly transform the result.

        NOTE that attention backend may override this function to do some
        transformation, which means the result of this topk_transform may not
        be the topk indices of the input logits.

        Return: Anything, since it will be passed to the attention backend
                for further processing on sparse attention computation.
                Don't assume it is the topk indices of the input logits.
        """


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    from fast_hadamard_transform import hadamard_transform

    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
    return hadamard_transform(x, scale=hidden_size**-0.5)


class V32LayerNorm(nn.Module):
    """
    Layer Normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(
            x.float(), (self.dim,), self.weight, self.bias, self.eps
        ).type_as(x)


class Indexer(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_id: int,
        scale_fmt: Optional[str],
        block_size: int = 128,
        rope_scaling: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        if not is_npu():
            self.sm_count = deep_gemm.get_num_sms()
            self.half_device_sm_count = align(self.sm_count // 2, 8)

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.wk = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wk", prefix),
        )
        self.k_norm = V32LayerNorm(self.head_dim)
        # NOTE: weight_proj is not quantized
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.rotary_emb = get_rope_wrapper(
            rope_head_dim,
            rotary_dim=rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,  # type: ignore
            rope_scaling=rope_scaling,
            is_neox_style=False,
            device=global_server_args_dict["device"],
        )
        self.block_size = block_size
        self.scale_fmt = scale_fmt
        self.softmax_scale = self.head_dim**-0.5

    def _forward_fake(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ):
        bs = x.shape[0]
        assert self.index_topk == 2048
        ans = torch.arange(0, self.index_topk, dtype=torch.int32, device=x.device)[
            None, ...
        ].repeat(bs, 1)
        if forward_batch.forward_mode.is_extend():
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.seq_lens_cpu is not None
            )
            which = 0
            for i, (kv_len, qo_len) in enumerate(
                zip(
                    forward_batch.seq_lens_cpu.tolist(),
                    forward_batch.extend_seq_lens_cpu,
                    strict=True,
                )
            ):
                for j in range(kv_len - qo_len, kv_len):
                    ans[which, j + 1 :] = -1
                    which += 1
            assert which == ans.shape[0]
        else:
            assert forward_batch.seq_lens_cpu is not None
            for i, seq_len in enumerate(forward_batch.seq_lens_cpu.tolist()):
                ans[i, seq_len:] = -1

        return ans

    def _get_logits_head_gate(self, x: torch.Tensor, q_scale: torch.Tensor):
        weights, _ = self.weights_proj(x)
        weights = weights * self.n_heads**-0.5
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        return weights

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        positions: torch.Tensor,
        enable_dual_stream: bool,
    ):

        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                self.half_device_sm_count
            ):
                query, _ = self.wq_b(q_lora)
                query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
                q_rope, _ = torch.split(
                    query,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )
            with torch.cuda.stream(self.alt_stream):
                key, _ = self.wk(x)
                key = self.k_norm(key)

                k_rope, _ = torch.split(
                    key,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )

            current_stream.wait_stream(self.alt_stream)
        else:
            query, _ = self.wq_b(q_lora)
            if dumper._enable:
                after_wq_b = query.clone()
            query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)

            q_rope, _ = torch.split(
                query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )

            key, _ = self.wk(x)
            if dumper._enable:
                after_wk = key.clone()
            key = self.k_norm(key)
            if dumper._enable:
                after_k_norm = key.clone()
            k_rope, _ = torch.split(
                key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )

        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

        query[..., : self.rope_head_dim] = q_rope
        key[..., : self.rope_head_dim] = k_rope

        if dumper._enable:
            q_before_hadamard = query.clone()
            k_before_hadamard = key.clone()

        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            query = rotate_activation(query)

            with torch.cuda.stream(self.alt_stream):
                key = rotate_activation(key)
            current_stream.wait_stream(self.alt_stream)
        else:
            query = rotate_activation(query)
            key = rotate_activation(key)

        return query, key

    def _get_topk_paged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        # NOTE(dark): blocksize = 64 is hardcoded in deep_gemm_v32
        assert page_size == 64, "only support page size 64"

        # NOTE(dark): this support extend/decode/decode+graph
        block_tables = metadata.get_page_table_64()

        max_seq_len = block_tables.shape[1] * page_size
        kv_cache_fp8 = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=layer_id
        )
        import deep_gemm_v32

        blocksize = page_size
        seqlens_32 = metadata.get_seqlens_int32()
        # NOTE(dark): 132 is SM count on H200/B200, not magic number
        schedule_metadata = deep_gemm_v32.get_paged_mqa_logits_metadata(
            seqlens_32, blocksize, 132
        )

        assert len(q_fp8.shape) == 3
        q_fp8 = q_fp8.unsqueeze(1)  # the next_n dim is 1 now
        assert len(kv_cache_fp8.shape) == 2
        block_kv = 64
        num_heads_kv = 1
        head_dim_with_sf = 132
        kv_cache_fp8 = kv_cache_fp8.view(
            kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)

        logits = deep_gemm_v32.fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            seqlens_32,
            block_tables,
            schedule_metadata,
            max_seq_len,
            clean_logits=False,
        )

        # NOTE(dark): logits should be cleaned in topk_transform
        topk_result = metadata.topk_transform(logits, self.index_topk)
        return topk_result

    def _get_topk_ragged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)
        k_fp8_list = []
        k_scale_list = []
        ks_list = []
        offset = 0

        block_tables = metadata.get_page_table_64()

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )

        for i in range(forward_batch.batch_size):
            seq_len = forward_batch.seq_lens_cpu[i].item()
            assert isinstance(seq_len, int)
            k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
                layer_id,
                seq_len,
                block_tables[i],
            )
            k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
                layer_id,
                seq_len,
                block_tables[i],
            )
            extend_seq_len = forward_batch.extend_seq_lens_cpu[i]
            ks = torch.full((extend_seq_len,), offset, dtype=torch.int32, device="cuda")
            k_fp8_list.append(k_fp8)
            k_scale_list.append(k_scale)
            ks_list.append(ks)
            offset += extend_seq_len

        k_fp8 = torch.cat(k_fp8_list, dim=0).view(torch.float8_e4m3fn)
        k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
        kv_fp8 = (k_fp8, k_scale)
        ks = torch.cat(ks_list, dim=0)
        seq_lens_expanded = metadata.get_seqlens_expanded()
        ke = ks + seq_lens_expanded

        import deep_gemm_v32

        logits = deep_gemm_v32.fp8_mqa_logits(
            q_fp8,
            kv_fp8,
            weights,
            ks,
            ke,
            clean_logits=False,
        )

        assert logits.shape[0] == len(seq_lens_expanded)
        topk_result = metadata.topk_transform(logits, self.index_topk)

        return topk_result

    def _forward(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        metadata = forward_batch.attn_backend.get_indexer_metadata(
            layer_id, forward_batch
        )

        enable_dual_stream = (
            NSA_DUAL_STREAM
            and self.alt_stream is not None
            and get_is_capture_mode()
            and q_lora.shape[0] > 0
            and q_lora.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
        )

        # skip NSA if attention backend choose to skip this batch
        if metadata is None:
            return None

        if not NSA_USE_REAL_INDEXER:  # temporary
            return self._forward_fake(x, q_lora, positions, forward_batch, layer_id)

        query, key = self._get_q_k_bf16(q_lora, x, positions, enable_dual_stream)

        q_fp8 = query.to(torch.float32)
        k_fp8 = key.to(torch.float32)
        q_scale = torch.ones((query.shape[0], 1), dtype=torch.float32, device="cuda")
        k_scale = torch.ones((key.shape[0], 1), dtype=torch.float32, device="cuda")

        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
            with torch.cuda.stream(self.alt_stream):
                k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)
            current_stream.wait_stream(self.alt_stream)
        else:
            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
            k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)

        # k_fp8: (seq_len, head_dim) fp8_e4m3fn
        # k_buffer: (num_total_tokens + page_size, head_dim) fp8_e4m3fn
        # k_scale: (seq_len, head_dim // block_size = 1) fp8_e4m3fn
        # k_scale_cache: (num_total_tokens + page_size, head_dim // block_size = 1) fp8_e4m3fn
        forward_batch.token_to_kv_pool.set_index_k_and_scale_buffer(
            layer_id=layer_id,
            loc=forward_batch.out_cache_loc,
            index_k=k_fp8,
            index_k_scale=k_scale,
        )

        weights = self._get_logits_head_gate(x, q_scale)

        assert forward_batch.seq_lens_cpu is not None
        if len(forward_batch.seq_lens_cpu) == 0:
            # this seems b/c max-pad, no worries?
            # if x.shape[0] != 0:
            #     print(
            #         "HACK: seq_lens empty but x not empty, hackily return all-invalid topk_result"
            #     )
            return torch.full(
                (x.shape[0], self.index_topk), -1, dtype=torch.int, device="cuda"
            )

        if forward_batch.forward_mode.is_decode_or_idle():
            topk_result = self._get_topk_paged(
                forward_batch, layer_id, q_fp8, weights, metadata
            )
        else:
            topk_result = self._get_topk_ragged(
                forward_batch, layer_id, q_fp8, weights, metadata
            )

        return topk_result

    def forward_cuda(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        return self._forward(x, q_lora, positions, forward_batch, layer_id)

    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> torch.Tensor:
        import custom_ops
        import torch_npu

        from sglang.srt.layers.dp_attention import (
            get_attention_tp_rank,
            get_attention_tp_size,
        )
        from sglang.srt.utils import get_bool_env_var

        if forward_batch.attn_backend.forward_metadata.seq_lens_cpu_int is None:
            actual_seq_lengths_kv = forward_batch.attn_backend.forward_metadata.seq_lens
        else:
            actual_seq_lengths_kv = (
                forward_batch.attn_backend.forward_metadata.seq_lens_cpu_int
            )
        enable_index_cp = (
            get_bool_env_var("SGLANG_USE_AG_AFTER_QLORA") and layer_id >= 4
        )
        is_prefill = forward_batch.forward_mode.is_extend()

        attention_tp_rank = get_attention_tp_rank()
        attention_tp_size = get_attention_tp_size()

        cos_sin = self.rotary_emb.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
        sin = sin.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
        if is_prefill and enable_index_cp:
            slice_length = cos.shape[0] // attention_tp_size
            cos = cos[
                slice_length
                * attention_tp_rank : slice_length
                * (attention_tp_rank + 1)
            ]
            sin = sin[
                slice_length
                * attention_tp_rank : slice_length
                * (attention_tp_rank + 1)
            ]

        slot_mapping = forward_batch.out_cache_loc
        block_table = forward_batch.attn_backend.forward_metadata.block_tables

        bs = x.shape[0]

        q = self.wq_b(q_lora)[0]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
        q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
        q_pe, q_nope = torch.split(
            q,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )  # [bs, 64, 64 + 64]

        q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin).view(
            bs, self.n_heads, self.rope_head_dim
        )  # [bs, n, d]
        q = torch.cat([q_pe, q_nope], dim=-1)

        k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
        k = self.k_norm(k_proj)
        k_pe, k_nope = torch.split(
            k,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )  # [bs, 64 + 64]

        k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
        k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin).view(
            bs, 1, self.rope_head_dim
        )  # [bs, 1, d]
        k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)  # [bs, 1, 128]

        if is_prefill and enable_index_cp:
            k, local_k = (
                torch.empty(
                    (k.shape[0] * attention_tp_size, k.shape[1], k.shape[2]),
                    dtype=k.dtype,
                    device=k.device,
                ),
                k,
            )
            get_attention_tp_group().all_gather_into_tensor(k, local_k)

        forward_batch.token_to_kv_pool.set_index_k_buffer(layer_id, slot_mapping, k)

        indexer_input = {}
        if is_prefill:
            actual_seq_lengths_kv = forward_batch.seq_lens.to(device=q.device)
            actual_seq_lengths_q = forward_batch.seq_lens.cumsum(dim=0).to(
                device=q.device
            )
            if enable_index_cp:
                actual_seq_lengths_q -= bs * attention_tp_rank
                actual_seq_lengths_q = torch.max(
                    actual_seq_lengths_q,
                    torch.zeros_like(actual_seq_lengths_q).to(
                        device=actual_seq_lengths_q.device
                    ),
                )
                actual_seq_lengths_q = torch.min(
                    actual_seq_lengths_q,
                    torch.full(actual_seq_lengths_q.shape, bs).to(
                        device=actual_seq_lengths_q.device
                    ),
                )

        else:
            if forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q is None:
                actual_seq_lengths_q = torch.tensor(
                    [1 + i * 1 for i in range(bs)], dtype=torch.int32, device=k.device
                )
            else:
                actual_seq_lengths_q = (
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q
                )

        past_key_states = forward_batch.token_to_kv_pool.get_index_k_buffer(layer_id)

        x = x.view(-1, self.hidden_size)
        weights = self.weights_proj(x)[0]
        block_table = (
            block_table[: actual_seq_lengths_q.size()[0]] if is_prefill else block_table
        )

        topk_indices = torch.ops.custom.npu_lightning_indexer(
            query=q.view(-1, self.n_heads, self.head_dim),
            key=past_key_states,
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_q.to(torch.int32),
            actual_seq_lengths_key=actual_seq_lengths_kv.to(k.device).to(torch.int32),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )

        if is_prefill and enable_index_cp:
            topk_indices, local_topk_indices = (
                torch.empty(
                    (
                        topk_indices.shape[0] * attention_tp_size,
                        topk_indices.shape[1],
                        topk_indices.shape[2],
                    ),
                    dtype=topk_indices.dtype,
                    device=topk_indices.device,
                ),
                topk_indices,
            )
            get_attention_tp_group().all_gather_into_tensor(
                topk_indices, local_topk_indices
            )

        return topk_indices
