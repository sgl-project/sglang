from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Type

import torch

from sglang.kernels.ops.kvcache.dsv4_fp4_index_k_cache import store_fp4_index_k_cache
from sglang.kernels.ops.quantization.fp4_indexer_quant import (
    quantize_fp4_indexer_tensor,
)
from sglang.srt.layers.attention.dsa.dsa_indexer import (
    Indexer,
    _index_k_write_targets,
)
from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
    DSAPagedMQALogitsBackend,
)
from sglang.srt.runtime_context import get_exec

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class Fp4Indexer(Indexer):
    """DSA indexer scoring E2M1 queries against E2M1 index keys, each 32 values
    sharing a UE8M0 exponent (DeepGEMM fp8_fp4 MQA kernels, SM100).

    The query stays BF16 until a logits call quantizes the rows it scores, so
    no query scale reaches the head gates.
    """

    # 64 bytes of packed E2M1 values and four exponent bytes per token.
    _HEAD_DIM_WITH_SF = 68

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The CuTe DSL paged MQA kernel has no FP4 variant.
        self.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.DEEPGEMM

    def _can_use_dsa_indexer_fusion(self, config, is_neox_style: bool) -> bool:
        # The fused Q/K kernels write FP8.
        return False

    def _quantize_query(self, query: torch.Tensor, act_quant):
        return query, None

    @torch.compile(dynamic=True)
    def _get_logits_head_gate_fp32(self, x, q_scale):
        weights = self._project_head_gates(x)
        weights = weights * self.n_heads**-0.5
        return weights.unsqueeze(-1).float() * self.softmax_scale

    @torch.compile(dynamic=True)
    def _apply_q_scale_and_softmax_scale(self, weights: torch.Tensor, q_scale):
        return weights.unsqueeze(-1).float() * self.softmax_scale

    def _quantize_fp4(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_fp4, q_sf = quantize_fp4_indexer_tensor(q, rne=True)
        return q_fp4.view(*q.shape[:-1], self.head_dim // 2), q_sf.view(q.shape[:-1])

    def _paged_mqa_logits(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_metadata: torch.Tensor,
        max_len: int,
        clean_logits: bool = False,
    ) -> torch.Tensor:
        from deep_gemm import fp8_fp4_paged_mqa_logits

        return fp8_fp4_paged_mqa_logits(
            self._quantize_fp4(q),
            kv_cache,
            weights,
            context_lens,
            block_table,
            schedule_metadata,
            max_len,
            clean_logits=clean_logits,
        )

    def _mqa_logits(
        self,
        q: torch.Tensor,
        kv: Tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        ks: torch.Tensor,
        ke: torch.Tensor,
    ) -> torch.Tensor:
        from deep_gemm import fp8_fp4_mqa_logits

        q_fp4, q_sf = self._quantize_fp4(q)
        q_fp4, weights, num_heads = self._pad_heads_for_deep_gemm(q_fp4, weights)
        if q_fp4.shape[1] > num_heads:
            q_sf = torch.nn.functional.pad(q_sf, (0, q_fp4.shape[1] - num_heads))
        k, k_scale = kv
        return fp8_fp4_mqa_logits(
            (q_fp4, q_sf),
            (k.view(torch.int8), k_scale.view(torch.int32)),
            weights,
            ks,
            ke,
            clean_logits=False,
        )

    def _write_index_k_cache(
        self,
        forward_batch: ForwardBatch,
        pool: DSATokenToKVPool,
        layer_id: int,
        key: torch.Tensor,
        act_quant,
        out_cache_loc: torch.Tensor,
    ) -> None:
        for buf, loc, rows in _index_k_write_targets(pool, layer_id, out_cache_loc):
            store_fp4_index_k_cache(
                key if rows is None else key.index_select(0, rows),
                buf,
                loc.contiguous(),
                page_size=pool.index_page_size,
                rne=True,
            )


def dsa_indexer_cls() -> Type[Indexer]:
    return Fp4Indexer if get_exec().kernel.enable_fp4_indexer else Indexer
