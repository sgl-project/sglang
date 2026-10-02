"""Fused KPool replay and MTP sibling copies preserve captured buffer identity."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.topk import (
    topk_transform_kpool_v2,
    topk_v2_plan_is_written,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.dsa_metadata_kit import (
    BS,
    NEXT_N,
    POOL,
    ROUNDS,
    TOPK,
    addresses,
    apply_metadata,
    assert_metadata_equal,
    inputs,
    make_backend,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _pooled_selection_buffers(metadata, mode):
    if mode.is_decode_or_idle():
        return (
            metadata.pooled_topk_v2_plan,
            metadata.pooled_cache_seqlens_int32,
            metadata.cache_seqlens_int32,
        )
    plan = metadata.kpool_write_plan
    return plan.pool_topk_v2_plan, plan.pool_seqlens_per_q, plan.seqlens_per_q


def _assert_plan_and_selection(test, *, mode, seq, req, width, metadata, out, count):
    plan, pool_lens, token_lens = _pooled_selection_buffers(metadata, mode)
    lengths = seq.cpu().int()
    requests = req.cpu().int()
    if not mode.is_decode_or_idle():
        start = lengths - NEXT_N if mode.is_draft_extend_v2() else lengths
        lengths = (
            start[:, None] + torch.arange(1, NEXT_N + 1, dtype=torch.int32)
        ).flatten()
        requests = requests.repeat_interleave(NEXT_N)
    pools = lengths // POOL
    torch.testing.assert_close(pool_lens.cpu(), pools)
    torch.testing.assert_close(token_lens.cpu(), lengths)
    planned = plan.cpu()
    test.assertEqual(int(planned[0, 1]), count)
    rows = torch.nonzero(pools > planned[0, 0]).flatten()
    expected_items = torch.stack((rows.int(), pools[rows]), dim=1)
    active = planned[1 : count + 1]
    torch.testing.assert_close(active[active[:, 0].argsort()], expected_items)
    test.assertTrue((planned[count + 1 :] == 0).all().item())

    # Ascending scores select the final pools; page tables map tokens to request-local ranges.
    cols = torch.arange(TOPK + POOL - 1)[None, :]
    history = torch.minimum(pools, torch.tensor(TOPK // POOL))[:, None] * POOL
    first = torch.clamp(pools - TOPK // POOL, min=0)[:, None] * POOL
    tail_end = history + (lengths % POOL)[:, None]
    tokens = torch.where(
        cols < history, first + cols, pools[:, None] * POOL + cols - history
    )
    expected = torch.where(
        cols < tail_end, tokens + requests[:, None] * width, -1
    ).int()
    torch.testing.assert_close(
        out.cpu().sort(dim=1).values, expected.sort(dim=1).values
    )


class TestDSAMetadataReplay(CustomTestCase):
    def test_populated_pool_plans_refresh_and_copy_on_graph_replay(self):
        """Populated plans must follow changing lengths and request order in source and sibling replays."""
        bs, width = 48, 280064
        probe = torch.zeros(bs, dtype=torch.int32, device="cuda")
        if not topk_v2_plan_is_written(probe):
            self.skipTest("Device does not write persistent top-k plans for this batch")
        with envs.SGLANG_OPT_USE_TOPK_V2.override(True):
            for mode in (
                ForwardMode.DECODE,
                ForwardMode.TARGET_VERIFY,
                ForwardMode.DRAFT_EXTEND_V2,
            ):
                with self.subTest(mode=mode):
                    self._check_populated_pool_plan_replay(
                        mode=mode, bs=bs, width=width
                    )

    def _check_populated_pool_plan_replay(self, *, mode, bs, width):
        seq, req = inputs(
            lengths=[280000] * 4 + [63] * (bs - 4), requests=list(range(bs))
        )
        source = make_backend(mode=mode, seq=seq, req=req, width=width)
        sibling = make_backend(mode=mode, seq=seq, req=req, width=width)
        src, dst = source.forward_metadata, sibling.forward_metadata
        plan, pool_lens, token_lens = _pooled_selection_buffers(dst, mode)
        pointers = addresses(src), addresses(dst)
        num_rows = pool_lens.shape[0]
        score_source = (
            torch.arange(width // POOL, device="cuda", dtype=torch.float32)
            .div(width // POOL)
            .repeat(num_rows, 1)
        )
        scores = torch.empty_like(score_source)
        out = torch.empty((num_rows, TOPK + POOL - 1), dtype=torch.int32, device="cuda")

        def forward():
            source._update_kpool_metadata_replay(
                metadata=src, seq_lens=seq, req_pool_indices=req, forward_mode=mode
            )
            dst.cache_seqlens_int32.copy_(src.cache_seqlens_int32)
            dst.real_page_table.copy_(src.real_page_table)
            sibling._copy_kpool_metadata_from_sibling(metadata=dst, src_metadata=src)
            # Stand in for the logits producer between plan writes and v2's PDL wait.
            scores.copy_(score_source)
            topk_transform_kpool_v2(
                scores=scores,
                pool_lens=pool_lens,
                out=out,
                pool_size=POOL,
                metadata=plan,
                token_seq_lens=token_lens,
                page_table=dst.real_page_table,
                page_size=64,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            forward()
        torch.cuda.current_stream().wait_stream(stream)
        for phase, long_rows in enumerate(
            (range(4), range(bs - 4, bs), (), (5, 17, 29, 41))
        ):
            lengths = [63 + (row % 4) * 1024 for row in range(bs)]
            for row in long_rows:
                lengths[row] = 280000 - phase * 256 + row % POOL
            requests = torch.arange(bs).roll(phase * 7).tolist()
            seq.copy_(torch.tensor(lengths, device="cuda"))
            req.copy_(torch.tensor(requests, device="cuda"))
            apply_metadata(backend=source, mode=mode, seq=seq, req=req)
            graph.replay()
            count = len(long_rows) * (1 if mode.is_decode_or_idle() else NEXT_N)
            _assert_plan_and_selection(
                self,
                mode=mode,
                seq=seq,
                req=req,
                width=width,
                metadata=dst,
                out=out,
                count=count,
            )
            src_plan, _, _ = _pooled_selection_buffers(src, mode)
            torch.testing.assert_close(plan, src_plan)
            self.assertNotEqual(plan.data_ptr(), src_plan.data_ptr())
            self.assertEqual(pointers, (addresses(src), addresses(dst)))

    def test_fusion_matches_ordinary_metadata(self):
        for mode in (
            ForwardMode.DECODE,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DRAFT_EXTEND_V2,
        ):
            with self.subTest(mode=mode):
                seq, req = inputs(*ROUNDS[0])
                fused = make_backend(mode, seq, req)
                ordinary = make_backend(mode, seq, req, fusion=False)
                pointers = addresses(fused.forward_metadata)
                for lengths, requests in ROUNDS:
                    seq.copy_(torch.tensor(lengths, device="cuda"))
                    req.copy_(torch.tensor(requests, device="cuda"))
                    spec = None
                    if mode.is_draft_extend_v2():
                        spec = SimpleNamespace(
                            num_accept_tokens=torch.tensor(
                                [1, 2, 5, NEXT_N], device="cuda", dtype=torch.int32
                            )
                        )
                    apply_metadata(fused, mode, seq, req, spec)
                    apply_metadata(ordinary, mode, seq, req, spec)
                    assert_metadata_equal(
                        self, fused.forward_metadata, ordinary.forward_metadata
                    )
                    self.assertEqual(pointers, addresses(fused.forward_metadata))

    def test_precomputed_verify_retains_live_tail(self):
        mode = ForwardMode.TARGET_VERIFY
        seq, req = inputs(*ROUNDS[0])
        fused = make_backend(mode, seq, req)
        ordinary = make_backend(mode, seq, req, fusion=False)
        pointers = addresses(fused.forward_metadata)
        for lengths, requests in ROUNDS[1:]:
            seq.copy_(torch.tensor(lengths, device="cuda"))
            req.copy_(torch.tensor(requests, device="cuda"))
            precomputed = fused._precompute_replay_metadata(
                BS, req, seq, seq.cpu(), mode
            )
            fused.init_forward_metadata_replay_cuda_graph_from_precomputed(
                BS, precomputed, mode
            )
            apply_metadata(ordinary, mode, seq, req)
            assert_metadata_equal(
                self, fused.forward_metadata, ordinary.forward_metadata
            )
            self.assertEqual(pointers, addresses(fused.forward_metadata))

    def test_precomputed_and_sibling_copy_refresh_derived_metadata(self):
        mode = ForwardMode.DECODE
        seq, req = inputs(*ROUNDS[0])
        source = make_backend(mode, seq, req)
        sibling = make_backend(mode, seq, req)
        ordinary = make_backend(mode, seq, req, fusion=False)
        pointers = addresses(sibling.forward_metadata)
        for lengths, requests in ROUNDS[1:]:
            seq.copy_(torch.tensor(lengths, device="cuda"))
            req.copy_(torch.tensor(requests, device="cuda"))
            precomputed = source._precompute_replay_metadata(
                BS, req, seq, seq.cpu(), mode
            )
            source.init_forward_metadata_replay_cuda_graph_from_precomputed(
                BS, precomputed, mode
            )
            # An eligible sibling must reuse the derived results, not silently
            # fall through to the full recomputation path.
            with patch.object(
                sibling,
                "init_forward_metadata_replay_cuda_graph_from_precomputed",
                side_effect=AssertionError("unexpected sibling fallback"),
            ):
                sibling._copy_replay_metadata_from_sibling(
                    source, BS, precomputed, mode
                )
            apply_metadata(ordinary, mode, seq, req)
            assert_metadata_equal(
                self, source.forward_metadata, ordinary.forward_metadata
            )
            assert_metadata_equal(
                self, sibling.forward_metadata, ordinary.forward_metadata
            )
            self.assertEqual(pointers, addresses(sibling.forward_metadata))
            self.assertIsNot(
                sibling.forward_metadata.kpool_write_plan,
                source.forward_metadata.kpool_write_plan,
            )


if __name__ == "__main__":
    unittest.main()
