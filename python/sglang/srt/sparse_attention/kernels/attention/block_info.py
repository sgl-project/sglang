# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
from dataclasses import dataclass
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute

from sglang.srt.sparse_attention.kernels.attention.seqlen_info import SeqlenInfoQK


@dataclass(frozen=True)
class BlockInfo:
    m_block_size: cutlass.Constexpr[int]
    n_block_size: cutlass.Constexpr[int]
    is_causal: cutlass.Constexpr[bool]
    is_local: cutlass.Constexpr[bool] = False
    window_size_left: Optional[cutlass.Int32] = None
    window_size_right: Optional[cutlass.Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1

    sink_size: cutlass.Constexpr[int] = 0
    enable_streaming: cutlass.Constexpr[bool] = False

    @cute.jit
    def get_n_block_min_max(
        self, seqlen_info: SeqlenInfoQK, m_block: cutlass.Int32
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
        seqlen_info: SeqlenInfoQK,
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
        seqlen_info: SeqlenInfoQK,
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

    @cute.jit
    def get_streaming_mask_n_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: cutlass.Int32,
        position_ids: Optional[cute.Tensor] = None,
    ) -> Tuple[cutlass.Int32, cutlass.Int32]:
        """
        Get the start and end of the streaming mask for the given m_block.
        Supports both standard relative attention and absolute position_ids (Chunked Prefill).
        """

        # 1. Get the physical index range of the current Q block in memory.
        # m_idx_min_phys = m_block * self.m_block_size
        # m_idx_max_phys = (m_block + 1) * self.m_block_size

        m_idx_min_phys = 0
        m_idx_max_phys = self.m_block_size

        # 2. Determine the "base position" and "offset" for mask calculation.
        # We need to distinguish between using absolute position_ids and standard relative indexing.

        # NOTE: Using cutlass.const_expr or ensuring position_ids is not None at compile time
        # is crucial to avoid JIT errors.
        if cutlass.const_expr(position_ids is not None):
            # --- Absolute Position Mode (e.g., Chunked Prefill) ---
            # We assume position_ids are loaded or accessible.
            # We also assume position_ids are monotonically increasing within a block.

            m_pos_min = position_ids[m_idx_min_phys]

            # Boundary check: ensure we don't access position_ids out of bounds
            # if the last block is partial.
            valid_idx = min(m_idx_max_phys - 1, seqlen_info.seqlen_q - 1)
            m_pos_max = position_ids[valid_idx]

            # In absolute position mode, the K index is directly limited by the Q position.
            # We assume K_physical_index == K_position_id (K cache is continuous from 0).
            n_idx_max_limit = m_pos_max
            n_idx_min_limit = m_pos_min

            # No offset needed because m_pos is already absolute.
            offset = 0
        else:
            # --- Relative Position Mode (Standard FlashAttn) ---
            # Fallback to physical indices if position_ids are not provided.
            n_idx_max_limit = m_idx_max_phys
            n_idx_min_limit = m_idx_min_phys

            # Add the diagonal offset: (seqlen_k - seqlen_q)
            offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q

        # -------------------------------------------------------
        # 3. Calculate n_block_max (Causal Constraint)
        # -------------------------------------------------------
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.n_block_size)

        if cutlass.const_expr(self.is_causal):
            # The rightmost K position a query can attend to.
            m_idx_max_val = n_idx_max_limit

            # Handle GQA packing if necessary
            if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max_val = cute.ceil_div(
                    m_idx_max_val, self.qhead_per_kvhead_packgqa
                )

            # Apply offset (0 for absolute pos, seqlen_diff for relative pos)
            n_idx_max = m_idx_max_val + offset

            n_block_max = min(n_block_max, cute.ceil_div(n_idx_max, self.n_block_size))

        # -------------------------------------------------------
        # 4. Calculate n_block_min (Local Window Constraint)
        # -------------------------------------------------------
        n_block_min = 0

        if self.enable_streaming and self.window_size_left is not None:
            # The leftmost K position (excluding sink) that needs to be processed.
            m_idx_min_val = n_idx_min_limit

            if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_min_val = m_idx_min_val // self.qhead_per_kvhead_packgqa

            # Apply offset
            n_idx = m_idx_min_val + offset

            # Calculate the left boundary of the local window
            n_idx_left = n_idx - self.window_size_left + 1

            if self.sink_size > 0:
                # If sink is present, start from 0 to include the sink region.
                n_block_min = 0
            else:
                # No sink, start from the window boundary.
                n_block_min = cutlass.max(n_idx_left // self.n_block_size, 0)

        return n_block_min, n_block_max
