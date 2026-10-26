"""DSA indexer prefill top-k through the fused LiteTopK kernel.

With SGLANG_ENABLE_DSA_FUSED_MQA_TOPK the indexer must return the indices of the
dense path it replaces for the same metadata (RAGGED, PAGED, unfused), and keep
chunks below the per-transform bounds on the dense path.
"""

import unittest
from itertools import accumulate, product
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsa import fused_mqa_topk
from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa.dsa_indexer_metadata import DSAIndexerMetadata
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.dsa_fused_mqa_topk import (
    assert_topk_matches_reference,
    make_page_table,
    make_ragged_batch,
    paged_to_relative,
    reference_scores,
    row_requests,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=200, stage="base-b", runner_config="4-gpu-b200")

TOPK = 2048
# Production offsets equal the row starts; shifting them tells the two apart.
_OFFSET_SHIFT = 5
RAGGED, PAGED = TopkTransformMethod.RAGGED, TopkTransformMethod.PAGED


def _indexer(use_fused_mqa_topk):
    indexer = SimpleNamespace(
        index_topk=TOPK,
        num_init_tokens=0,
        num_local_tokens=0,
        logits_with_pp_recv=False,
        use_fused_mqa_topk=use_fused_mqa_topk,
        _pad_heads_for_deep_gemm=Indexer._pad_heads_for_deep_gemm,
    )
    for name in (
        "_with_real_sm_count",
        "_mask_init_and_local_tokens",
        "_fused_mqa_topk_prefill",
    ):
        setattr(indexer, name, MethodType(getattr(Indexer, name), indexer))
    return indexer


def _batch(seq_lens, extend_lens, *, seed):
    batch = make_ragged_batch(seq_lens, extend_lens, seed=seed)
    lens = dict(seq_lens=seq_lens, extend_lens=extend_lens)
    return dict(batch, row_batch=row_requests(extend_lens), **lens)


def _topk_ragged(batch, *, method, use_fused, force_unfused_topk, page_table):
    """Run Indexer._get_topk_ragged on a packed batch whose index-K the pool
    returns already gathered; returns the indices and whether the fused op ran."""
    seq_lens_cpu = torch.tensor(batch["seq_lens"], dtype=torch.int32)
    extend_lens = batch["extend_lens"]
    starts, ends = batch["starts"], batch["ends"]
    attn_metadata = SimpleNamespace(
        real_page_table=torch.zeros(len(extend_lens), 1, dtype=torch.int32),
        indexer_logical_page_table=None,
        indexer_k_start_end=(starts, ends),
        indexer_seq_lens_cpu=seq_lens_cpu,
        indexer_seq_lens=seq_lens_cpu.cuda(),
        dsa_seqlens_expanded=ends - starts,
        token_to_batch_idx=batch["row_batch"],
        topk_indices_offset=(starts + _OFFSET_SHIFT if method == RAGGED else None),
        cu_seqlens_q=torch.tensor(
            [0, *accumulate(extend_lens)], dtype=torch.int32
        ).cuda(),
        page_table_1=page_table,
    )
    metadata = DSAIndexerMetadata(
        attn_metadata=attn_metadata,
        topk_transform_method=method,
        force_unfused_topk=force_unfused_topk,
    )
    keys = batch["k_fp8"].shape[0]
    pool = SimpleNamespace(
        index_page_size=64,
        get_index_k_scale_buffer=lambda *_: (
            batch["k_fp8"].view(torch.uint8),
            batch["k_scale"].view(torch.uint8).view(keys, 4),
        ),
    )
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        seq_lens_cpu=seq_lens_cpu,
        extend_seq_lens_cpu=list(extend_lens),
    )
    with (
        patch.object(dsa_indexer, "get_token_to_kv_pool", return_value=pool),
        patch.object(
            dsa_indexer,
            "fused_mqa_topk_prefill",
            wraps=fused_mqa_topk.fused_mqa_topk_prefill,
        ) as fused_op,
    ):
        out = Indexer._get_topk_ragged(
            _indexer(use_fused),
            False,
            forward_batch,
            0,
            batch["q_fp8"],
            batch["weights"].unsqueeze(-1),
            metadata,
        )
    return out, fused_op.called


def _assert_contract(out, batch, scores, *, method, page_table, transformed=True):
    """Check ``out`` against the oracle in the index form ``method`` produces."""
    offsets = None
    if transformed and method == RAGGED:
        offsets = batch["starts"] + _OFFSET_SHIFT
    if transformed and method == PAGED:
        out = paged_to_relative(out, page_table, batch["seq_lens"], batch["row_batch"])
    assert_topk_matches_reference(out, batch, scores, out_offsets=offsets)


# Scaled-down bounds keep the fp32 oracle small; they differ per transform so
# that a swapped or shared bound shows.
_BOUNDS = {RAGGED: (16384, 256), PAGED: (8192, 128)}


@unittest.skipUnless(
    fused_mqa_topk.is_fused_mqa_topk_supported(), "requires CUDA SM100 or SM103"
)
@patch.object(dsa_indexer, "_FUSED_MQA_TOPK_MIN_KEYS_RAGGED", _BOUNDS[RAGGED][0])
@patch.object(dsa_indexer, "_FUSED_MQA_TOPK_MIN_ROWS_RAGGED", _BOUNDS[RAGGED][1])
@patch.object(dsa_indexer, "_FUSED_MQA_TOPK_MIN_KEYS_PAGED", _BOUNDS[PAGED][0])
@patch.object(dsa_indexer, "_FUSED_MQA_TOPK_MIN_ROWS_PAGED", _BOUNDS[PAGED][1])
class TestFusedMqaTopkDispatch(CustomTestCase):
    def test_fused_branch_keeps_the_dense_index_contract(self):
        # A long request, a cached-prefix one and a request whose rows see fewer
        # keys than top-k. The fused RAGGED transform rebases onto the flattened
        # KV, the fused PAGED one maps through the page table row of the row's
        # request; the unfused top-k leaves both row-relative for the backend.
        seq_lens, extend_lens = [20000, 3000, 1100], [96, 64, 128]
        batch = _batch(seq_lens, extend_lens, seed=7)
        page_table = make_page_table(seq_lens, seed=7)
        scores = reference_scores(batch)
        for method, force_unfused_topk in product((RAGGED, PAGED), (False, True)):
            with self.subTest(
                method=method.name, force_unfused_topk=force_unfused_topk
            ):
                run = dict(
                    method=method,
                    force_unfused_topk=force_unfused_topk,
                    page_table=page_table,
                )
                dense, _ = _topk_ragged(batch, use_fused=False, **run)
                fused, fused_ran = _topk_ragged(batch, use_fused=True, **run)
                self.assertTrue(fused_ran)
                for out in (dense, fused):
                    _assert_contract(
                        out,
                        batch,
                        scores,
                        method=method,
                        page_table=page_table,
                        transformed=not force_unfused_topk,
                    )

    def _assert_dispatch(self, method, seq_lens, extend_lens, *, fused):
        batch = _batch(seq_lens, extend_lens, seed=8)
        page_table = make_page_table(seq_lens, seed=8)
        out, fused_ran = _topk_ragged(
            batch,
            method=method,
            use_fused=True,
            force_unfused_topk=False,
            page_table=page_table,
        )
        self.assertEqual(fused_ran, fused)
        _assert_contract(
            out, batch, reference_scores(batch), method=method, page_table=page_table
        )

    def test_each_transform_keeps_chunks_below_its_bounds_dense(self):
        (ragged_keys, ragged_rows), (paged_keys, paged_rows) = _BOUNDS.values()
        # Above the PAGED bounds, below the RAGGED keys bound.
        between = ([12000, 3000], [96, 96])
        for method, seq_lens, extend_lens, fused in (
            (RAGGED, [ragged_keys - 1, 3000], [192, 128], False),
            (RAGGED, [20000], [ragged_rows - 1], False),
            (PAGED, [paged_keys - 1, 3000], [192, 128], False),
            (PAGED, [20000], [paged_rows - 1], False),
            (RAGGED, *between, False),
            (PAGED, *between, True),
        ):
            with self.subTest(method=method.name, seq_lens=seq_lens, rows=extend_lens):
                self._assert_dispatch(method, seq_lens, extend_lens, fused=fused)


if __name__ == "__main__":
    unittest.main()
