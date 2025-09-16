# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
from dataclasses import dataclass
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute

from sglang.srt.layers.attention.cute_ops.seqlen_info import SeqlenInfo


@dataclass(frozen=True)
class BlockInfo:
    m_block_size: cutlass.Constexpr[int]
    n_block_size: cutlass.Constexpr[int]
    is_causal: cutlass.Constexpr[bool]
    is_local: cutlass.Constexpr[bool] = False
    window_size_left: Optional[cutlass.Int32] = None
    window_size_right: Optional[cutlass.Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1

    @cute.jit
    def get_n_block_min_max(
        self, seqlen_info: SeqlenInfo, m_block: cutlass.Int32
    ) -> Tuple[cutlass.Int32, cutlass.Int32]:
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.n_block_size)
        if cutlass.const_expr(
            self.is_causal or (self.is_local and self.window_size_right is not None)
        ):
            m_idx_max = (m_block + 1) * self.m_block_size
            if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_right = (
                n_idx
                if cutlass.const_expr(self.is_causal)
                else n_idx + self.window_size_right
            )
            n_block_max = min(
                n_block_max, cute.ceil_div(n_idx_right, self.n_block_size)
            )
        n_block_min = 0
        if cutlass.const_expr(self.is_local and self.window_size_left is not None):
            m_idx_min = m_block * self.m_block_size
            if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
            n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_left = n_idx - self.window_size_left
            n_block_min = cutlass.max(n_idx_left // self.n_block_size, 0)
        return n_block_min, n_block_max

    @cute.jit
    def get_n_block_min_causal_local_mask(
        self,
        seqlen_info: SeqlenInfo,
        m_block: cutlass.Int32,
        n_block_min: cutlass.Int32,
    ) -> cutlass.Int32:
        """If we have separate iterations with causal or local masking at the start, where do we stop"""
        m_idx_min = m_block * self.m_block_size
        if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
            m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
        n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_idx_right = (
            n_idx
            if cutlass.const_expr(not self.is_local or self.window_size_right is None)
            else n_idx + self.window_size_right
        )
        return cutlass.max(n_block_min, n_idx_right // self.n_block_size)

    @cute.jit
    def get_n_block_min_before_local_mask(
        self,
        seqlen_info: SeqlenInfo,
        m_block: cutlass.Int32,
        n_block_min: cutlass.Int32,
    ) -> cutlass.Int32:
        """If we have separate iterations with local masking at the end, where do we stop the non-masked iterations"""
        if cutlass.const_expr(not self.is_local or self.window_size_left is None):
            return n_block_min
        else:
            m_idx_max = (m_block + 1) * self.m_block_size
            if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_left = n_idx - self.window_size_left
            return cutlass.max(
                n_block_min, cute.ceil_div(n_idx_left, self.n_block_size)
            )
