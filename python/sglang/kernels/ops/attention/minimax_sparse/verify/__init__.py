# Copyright 2025 XunhaoLai. All rights reserved.
#
# EAGLE3 verify 专用 sparse attention 入口 (Path 1 Strategy B, 固定长度版).
# 与 prefill/ 的区别仅在于: score buffer 第3维用固定上界 max_seqblock_k_upper
# (= cdiv(context_len, block_size_k)), capture/replay 形状恒定 → graph-safe.
# kernel 内部仍按真实 seq_len 做 causal, 与 prefill 完全一致.

from .verify_sparse import minimax_sparse_verify_prefill

__all__ = ["minimax_sparse_verify_prefill"]
