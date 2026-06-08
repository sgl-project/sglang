from __future__ import annotations

from typing import Any, Dict, Optional

import torch

try:
    import sgl_kernel_npu  # noqa: F401
    import torch_npu
except (ImportError, OSError):
    sgl_kernel_npu = None
    torch_npu = None

from transformers import PretrainedConfig

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import get_indexer_weight_stream
from sglang.srt.layers.attention.dsa.dsa_indexer import (
    Indexer,
    _use_ag_after_qlora,
    scattered_to_tp_attn_full,
)
from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.utils import add_prefix


class LongcatProNPUIndexer(Indexer):
    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        index_k_norm_type: str,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_id: int,
        scale_fmt: Optional[str],
        block_size: int = 128,
        rope_scaling: Optional[Dict[str, Any]] = None,
        is_neox_style: bool = True,
        prefix: str = "",
        config: Optional[PretrainedConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__(
            hidden_size=hidden_size,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            index_topk=index_topk,
            q_lora_rank=q_lora_rank,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            layer_id=layer_id,
            scale_fmt=scale_fmt,
            block_size=block_size,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            prefix=prefix,
            quant_config=quant_config,
            alt_stream=alt_stream,
        )
        self.config = config
        self.index_k_norm_type = index_k_norm_type
        self.kv_block_size = getattr(config, "kv_block_size", 1)
        self.q_block_size = getattr(config, "q_block_size", 1)
        self.num_init_tokens = getattr(config, "index_init_tokens", 0)
        self.num_local_tokens = getattr(config, "index_local_tokens", 0)
        self.nsa_enable_prefill_cp = False

        if index_k_norm_type == "rms":
            self.k_norm = RMSNorm(
                self.head_dim, eps=getattr(config, "rms_norm_eps", 1e-6)
            )

        # LongcatPro's reference NPU path keeps indexer routing weights in fp32.
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            params_dtype=torch.float32,
            prefix=add_prefix("weights_proj", prefix),
        )

    def _prepend_zero_to_cu_seqlens(self, cu_seqlens: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=cu_seqlens.device),
                cu_seqlens.to(torch.int64),
            ]
        )

    def _build_cu_seqlens_from_lengths(self, seq_lens: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=seq_lens.device),
                torch.cumsum(seq_lens.to(torch.int64), dim=0),
            ]
        )

    def _can_use_mlp_lightning_indexer(self) -> bool:
        return hasattr(torch.ops, "npu") and hasattr(
            torch.ops.npu, "mlp_lightning_indexer"
        )

    def _get_full_candidate_count(self, actual_seq_lengths_kv: torch.Tensor) -> int:
        # Ascend lightning indexer requires sparse_count <= 2048.
        # For LongCat Pro we keep the candidate pool aligned with the model's
        # configured topk budget instead of expanding it with sequence length.
        return min(self.index_topk, 2048)

    def _run_mlp_lightning_indexer_pa_bsnd(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        past_key_states: torch.Tensor,
        actual_seq_lengths_q: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        block_table: torch.Tensor,
        candidate_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._can_use_mlp_lightning_indexer():
            raise RuntimeError(
                "LongcatProNPUIndexer requires torch.ops.npu.mlp_lightning_indexer. "
                "Please ensure sgl_kernel_npu with mlp_lightning_indexer is "
                "built and imported."
            )

        cur_seq_lengths_query = self._prepend_zero_to_cu_seqlens(
            actual_seq_lengths_q
        )
        cur_seq_lengths_key = self._build_cu_seqlens_from_lengths(
            actual_seq_lengths_kv
        )

        topk_indices_local, topk_values = torch.ops.npu.mlp_lightning_indexer(
            q,
            past_key_states,
            weights.to(torch.float32),
            cur_seq_lengths_query=cur_seq_lengths_query,
            cur_seq_lengths_key=cur_seq_lengths_key,
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=candidate_count,
            kv_block_len=self.kv_block_size,
            q_block_len=self.q_block_size,
            init_num=self.num_init_tokens,
            local_num=self.num_local_tokens,
            sparse_mode=3,
            return_value=True,
        )
        return topk_indices_local, topk_values.to(torch.float32)

    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        layer_id: int,
        layer_scatter_modes=None,
        dynamic_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        if torch_npu is None:
            raise RuntimeError("LongcatProNPUIndexer requires torch_npu")

        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend()
        )
        bs = q_lora.shape[0]

        if self.rotary_emb.is_neox_style:
            if not hasattr(forward_batch, "npu_indexer_sin_cos_cache"):
                cos_sin = self.rotary_emb.cos_sin_cache[positions]
                cos, sin = cos_sin.chunk(2, dim=-1)
                cos = cos.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                sin = sin.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                forward_batch.npu_indexer_sin_cos_cache = (sin, cos)
            else:
                sin, cos = forward_batch.npu_indexer_sin_cos_cache

            if self.alt_stream is not None:
                self.alt_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(self.alt_stream):
                    q_lora = (
                        (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                    )
                    q = self.wq_b(q_lora)[0]
                    q = q.view(bs, self.n_heads, self.head_dim)
                    q_pe, q_nope = torch.split(
                        q,
                        [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                        dim=-1,
                    )
                    q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
                    q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin).view(
                        bs, self.n_heads, self.rope_head_dim
                    )
                    q = torch.cat([q_pe, q_nope], dim=-1)
                    q.record_stream(self.alt_stream)
                    q_rope_event = self.alt_stream.record_event()
            else:
                q_lora = (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                q = self.wq_b(q_lora)[0]
                q = q.view(bs, self.n_heads, self.head_dim)
                q_pe, q_nope = torch.split(
                    q,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )
                q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
                q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin).view(
                    bs, self.n_heads, self.rope_head_dim
                )
                q = torch.cat([q_pe, q_nope], dim=-1)
                q_rope_event = None

            if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    x = x.view(-1, self.hidden_size)
                    weights = self.weights_proj(x.float())[0]
                    weights.record_stream(indexer_weight_stream)
                    weights_event = indexer_weight_stream.record_event()
            else:
                x = x.view(-1, self.hidden_size)
                weights = self.weights_proj(x.float())[0]
                weights_event = None

            k_proj = self.wk(x)[0]
            k = self.k_norm(k_proj)
            if (
                _use_ag_after_qlora
                and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
                and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
            ):
                k = scattered_to_tp_attn_full(k, forward_batch)
            k_pe, k_nope = torch.split(
                k,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )
            k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
            k_pe = torch.ops.npu.npu_rotary_mul(k_pe, cos, sin).view(
                bs, 1, self.rope_head_dim
            )
            k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)
        else:
            if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    x = x.view(-1, self.hidden_size)
                    weights = self.weights_proj(x.float())[0]
                    weights.record_stream(indexer_weight_stream)
                    weights_event = indexer_weight_stream.record_event()
            else:
                x = x.view(-1, self.hidden_size)
                weights = self.weights_proj(x.float())[0]
                weights_event = None

            q_lora = (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
            q = self.wq_b(q_lora)[0]
            q = q.view(bs, self.n_heads, self.head_dim)
            q_pe, q_nope = torch.split(
                q,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )

            k_proj = self.wk(x)[0]
            k = self.k_norm(k_proj)
            k_pe, k_nope = torch.split(
                k,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )
            k_pe = k_pe.unsqueeze(1)

            if layer_id == 0:
                self.rotary_emb.sin_cos_cache = (
                    self.rotary_emb.cos_sin_cache.index_select(0, positions)
                )

            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            k_pe = k_pe.squeeze(1)
            q = torch.cat([q_pe, q_nope], dim=-1)
            k = torch.cat([k_pe, k_nope], dim=-1)
            q_rope_event = None

        get_token_to_kv_pool().set_index_k_buffer(layer_id, forward_batch.out_cache_loc, k)

        if is_prefill:
            actual_seq_lengths_q = forward_batch.extend_seq_lens.cumsum(dim=0).to(
                device=k.device, dtype=torch.int32
            )
        else:
            actual_seq_lengths_q = torch.arange(
                1,
                bs + 1,
                dtype=torch.int32,
                device=k.device,
            )
        forward_metadata = get_attn_backend().forward_metadata
        if forward_metadata.seq_lens_cpu_int is None:
            actual_seq_lengths_kv = forward_metadata.seq_lens
        else:
            actual_seq_lengths_kv = forward_metadata.seq_lens_cpu_int
        actual_seq_lengths_kv = actual_seq_lengths_kv.to(
            device=k.device, dtype=torch.int32
        )

        past_key_states = get_token_to_kv_pool().get_index_k_buffer(layer_id)

        if self.rotary_emb.is_neox_style and q_rope_event is not None:
            torch.npu.current_stream().wait_event(q_rope_event)
        if weights_event is not None:
            torch.npu.current_stream().wait_event(weights_event)
        if (
            _use_ag_after_qlora
            and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
            and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
        ):
            weights = scattered_to_tp_attn_full(weights, forward_batch)

        block_table = get_attn_backend().forward_metadata.block_tables
        if is_prefill:
            block_table = block_table[: actual_seq_lengths_q.size(0)]
        candidate_count = self._get_full_candidate_count(actual_seq_lengths_kv)
        q_indexer = q.view(-1, self.n_heads, self.head_dim)
        topk_indices_local, _ = self._run_mlp_lightning_indexer_pa_bsnd(
            q=q_indexer,
            weights=weights,
            past_key_states=past_key_states,
            actual_seq_lengths_q=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            block_table=block_table,
            candidate_count=candidate_count,
        )
        return topk_indices_local
