"""Fused KPool replay and shared MTP draft metadata preserve captured buffer identity."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
from sglang.srt.layers.attention.dsa.kpool_fp8_index import build_pooled_page_table_64
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnMultiStepBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.dsa_metadata_kit import (
    BS,
    NEXT_N,
    POOL,
    ROUNDS,
    addresses,
    apply_metadata,
    assert_metadata_equal,
    inputs,
    make_backend,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=50, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestDSAMetadataReplay(CustomTestCase):
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
                    if mode.is_target_verify():
                        metadata = fused.forward_metadata
                        _, _, table, _, _ = IndexerKPool._get_kpool_decode_metadata(
                            SimpleNamespace(index_kpool=POOL),
                            SimpleNamespace(attn_metadata=metadata),
                            metadata.real_page_table,
                            metadata.dsa_seqlens_expanded,
                            64,
                            build_schedule_metadata=False,
                        )
                        self.assertIs(table, metadata.pooled_real_page_table)
                        torch.testing.assert_close(
                            table,
                            build_pooled_page_table_64(metadata.real_page_table, POOL),
                            rtol=0,
                            atol=0,
                        )

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

    def _multi_step(self, *, seq, req, fusion, pool=POOL):
        mode = ForwardMode.DECODE
        multi = object.__new__(DeepseekSparseAttnMultiStepBackend)
        multi.speculative_num_steps = 4
        multi.attn_backends = [
            make_backend(mode, seq, req, fusion=fusion, pool=pool) for _ in range(3)
        ]
        first, *rest = multi.attn_backends
        # Only the first step allocates a full buffer set; the fixture's
        # backends cannot run the full allocation, so it is stubbed.
        with contextlib.ExitStack() as stack:
            full = stack.enter_context(patch.object(first, "init_cuda_graph_state"))
            for backend in rest:
                stack.enter_context(
                    patch.object(
                        backend,
                        "init_cuda_graph_state",
                        side_effect=AssertionError("shared step allocated buffers"),
                    )
                )
            multi.init_cuda_graph_state(max_bs=BS, max_num_tokens=BS)
        full.assert_called_once_with(max_bs=BS, max_num_tokens=BS)
        for backend in rest:
            self.assertEqual(backend.decode_cuda_graph_metadata, {})
        multi.init_forward_metadata_out_graph(
            SimpleNamespace(
                batch_size=BS,
                forward_mode=mode,
                req_pool_indices=req,
                seq_lens=seq,
                seq_lens_sum=int(seq.sum()),
                seq_lens_cpu=seq.cpu(),
                spec_info=None,
            ),
            in_capture=True,
        )
        return multi

    def test_draft_steps_share_first_step_metadata(self):
        mode = ForwardMode.DECODE
        # KPool fusion needs pool > 1; pool 1 is the ordinary DSA path.
        for pool, fusion in ((POOL, True), (POOL, False), (1, False)):
            with self.subTest(pool=pool, fusion=fusion):
                seq, req = inputs(*ROUNDS[0])
                multi = self._multi_step(seq=seq, req=req, fusion=fusion, pool=pool)
                first = multi.attn_backends[0]
                shared = first.forward_metadata
                pointers = addresses(shared)
                for backend in multi.attn_backends:
                    self.assertIs(backend.decode_cuda_graph_metadata[BS], shared)
                ordinary = make_backend(mode, seq, req, fusion=False, pool=pool)
                for lengths, requests in ROUNDS[1:]:
                    seq.copy_(torch.tensor(lengths, device="cuda"))
                    req.copy_(torch.tensor(requests, device="cuda"))
                    # Only the first step refreshes metadata; the others alias it.
                    with patch.object(
                        first,
                        "init_forward_metadata_replay_cuda_graph_from_precomputed",
                        wraps=first.init_forward_metadata_replay_cuda_graph_from_precomputed,
                    ) as refresh:
                        multi.init_forward_metadata_out_graph(
                            SimpleNamespace(
                                batch_size=BS,
                                req_pool_indices=req,
                                seq_lens=seq,
                                seq_lens_cpu=seq.cpu(),
                            )
                        )
                    self.assertEqual(refresh.call_count, 1)
                    apply_metadata(ordinary, mode, seq, req)
                    assert_metadata_equal(self, shared, ordinary.forward_metadata)
                    self.assertEqual(pointers, addresses(shared))
                    for backend in multi.attn_backends:
                        self.assertIs(backend.forward_metadata, shared)
                        self.assertIs(backend.decode_cuda_graph_metadata[BS], shared)


if __name__ == "__main__":
    unittest.main()
