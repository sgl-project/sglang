"""Kimi-K3.1 NoPE indexer prologue for gfx950 decode and target verify."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Tuple

import torch

from sglang.kernels.ops.attention.dsa.fp4_index_hip import indexer_q_fp4_head_gate
from sglang.kernels.ops.attention.dsa.head_gate import indexer_q_quant_head_gate
from sglang.srt.layers.cp.utils import is_cp_active


def make_rocm_nope_prologue(indexer) -> Optional[RocmNoPEPrologue]:
    # indexer_q_quant_head_gate quantizes one 128-wide block per head
    if (
        indexer.skip_rope
        and indexer.bf16_head_gate
        and indexer.head_dim == indexer.block_size == 128
    ):
        return RocmNoPEPrologue()
    return None


class RocmNoPEPrologue:
    def __init__(self) -> None:
        # [wk | weights_proj] after merge_k_gate; the two weights become its views
        self.k_gate_weight: Optional[torch.Tensor] = None

    def merge_k_gate(self, indexer) -> None:
        if not (
            indexer.wk.weight.dtype == torch.bfloat16
            and indexer.weights_proj.weight.dtype == torch.bfloat16
        ):
            return
        merged = torch.cat([indexer.wk.weight.data, indexer.weights_proj.weight.data])
        indexer.wk.weight.data = merged[: indexer.head_dim]
        indexer.weights_proj.weight.data = merged[indexer.head_dim :]
        self.k_gate_weight = merged

    @staticmethod
    def applies(x, forward_batch, in_graph_split_op: bool, weights_proj_lora) -> bool:
        mode = forward_batch.forward_mode
        return (
            (mode.is_decode_or_idle() or mode.is_target_verify())
            and not in_graph_split_op
            and not weights_proj_lora
            and (isinstance(x, torch.Tensor) or len(x) == 3)
            and not is_cp_active(forward_batch)
        )

    def forward(
        self,
        indexer,
        x,
        q_lora,
        forward_batch,
        layer_id: int,
        act_quant,
        fuse_k_norm: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """fuse_k_norm: the index-K store applies k_norm (the monokernel's)."""
        # a monokernel plugin that already ran wk, weights_proj and wq_b hands
        # their outputs in (_mono_pre)
        pre = indexer.__dict__.get("_mono_pre")
        if pre is not None:
            key, gate, query = pre["k"], pre["w"], pre["q"]
        else:
            # AITER's fused RMSNorm hands (fp8, scale, bf16); wk and the gate read the bf16
            if isinstance(x, tuple):
                x = x[2]
            if self.k_gate_weight is None:
                key, gate = indexer.wk(x)[0], indexer.weights_proj(x)[0]
            else:
                # wk's unquantized route; the K3.1 bf16 GEMM table tunes N = 160
                layer = SimpleNamespace(weight=self.k_gate_weight)
                key, gate = indexer.wk.quant_method.apply(layer, x).split(
                    [indexer.head_dim, indexer.n_heads], dim=-1
                )
            query = indexer.wq_b(q_lora)[0]
        indexer._store_index_k_cache(
            forward_batch,
            layer_id,
            key if fuse_k_norm else indexer.k_norm(key),
            act_quant=act_quant,
            prenorm=fuse_k_norm,
        )
        if indexer.fp4_index_k:
            return indexer_q_fp4_head_gate(
                query.view(-1, indexer.n_heads, indexer.head_dim),
                gate,
                indexer.softmax_scale,
            )
        return indexer_q_quant_head_gate(
            query.view(-1, indexer.n_heads, indexer.head_dim),
            gate,
            indexer.softmax_scale,
            indexer.scale_fmt,
        )
