# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""K31HT_IDX_FP8_1PASS (default off): the served FP8 DSA indexer's verify logits at native 5..8 rows per request through the
one-pass kernel (k31_dglogits.dg_logits_1p.paged_mqa_logits_fp8), bitwise == deep_gemm.fp8_paged_mqa_logits. dsa_indexer.py calls
fp8_native_logits_1p in place of deepgemm_paged_mqa_logits_native when the switch is on; unset, nothing here runs.
"""

from __future__ import annotations

import logging


def fp8_native_logits_1p(q_fp8, kv_cache_fp8, weights, ctx_lens_2d, block_tables, schedule_metadata, max_seq_len, *, q_offset, B, next_n):
    """deepgemm_paged_mqa_logits_native's call ([B, next_n] rows, block_tables de-expanded with [::next_n]) through the one-pass FP8 kernel; one
    launch for any batch (the DeepGEMM 0.2.3 paged scheduler partitions the whole batch; each logit's arithmetic does not depend on it)."""
    from sglang.kernels.ops.attention.dsa.paged_mqa_logits import _restore_row_stride
    from sglang.srt.layers.attention.dsa.k31_dglogits import dg_logits_1p

    _engaged("decode_logits_fp8_1p")
    H, D = q_fp8.shape[1], q_fp8.shape[2]
    return _restore_row_stride(
        dg_logits_1p.paged_mqa_logits_fp8(q_fp8[:q_offset].view(B, next_n, H, D), kv_cache_fp8, weights[:q_offset], ctx_lens_2d.contiguous(),
                                          block_tables[::next_n], schedule_metadata, max_seq_len)
    )


logger = logging.getLogger(__name__)
PREC_IDX_FP4_STATS = {"decode_logits_fp8_1p": 0}


def _engaged(site: str) -> None:
    PREC_IDX_FP4_STATS[site] += 1
    if PREC_IDX_FP4_STATS[site] == 1:
        logger.info("PREC_IDX_FP4 ENGAGED site=%s (K31HT_IDX_FP8_1PASS, FP8 index-K)", site)
