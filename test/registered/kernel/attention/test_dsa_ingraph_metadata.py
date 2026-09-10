"""Captured TARGET_VERIFY metadata follows the graph runner's static inputs."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.dsa_metadata_kit import (
    BS,
    POOL,
    ROUNDS,
    addresses,
    apply_metadata,
    assert_metadata_equal,
    capture_verify_metadata,
    inputs,
    make_backend,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestDSAInGraphMetadata(CustomTestCase):
    def test_captured_verify_refreshes_all_derived_buffers(self):
        self._check_captured_verify_refresh(pool_size=POOL)

    def test_captured_verify_without_kpool(self):
        self._check_captured_verify_refresh(pool_size=1)

    def _check_captured_verify_refresh(self, pool_size):
        mode = ForwardMode.TARGET_VERIFY
        seq, req = inputs(*ROUNDS[0])
        backend = make_backend(
            mode, seq, req, fusion=pool_size > 1, pool_size=pool_size
        )
        ordinary = make_backend(mode, seq, req, fusion=False, pool_size=pool_size)
        if pool_size == 1:
            self.assertFalse(backend.experimental_kpool_metadata_fusion)
            self.assertIsNone(backend.forward_metadata.kpool_write_plan)
        pointers = addresses(backend.forward_metadata)
        graph = capture_verify_metadata(backend, seq, req)
        self.assertIsNotNone(
            getattr(backend.forward_metadata, "_ingraph_verify_metadata", None),
            "the verify hook must install captured replay state",
        )
        for lengths, requests in ROUNDS[1:]:
            seq.copy_(torch.tensor(lengths, device="cuda"))
            req.copy_(torch.tensor(requests, device="cuda"))
            # Eager replay must leave captured metadata refresh to the graph.
            before = backend.forward_metadata.dsa_seqlens_expanded.clone()
            apply_metadata(backend, mode, seq, req)
            torch.testing.assert_close(
                backend.forward_metadata.dsa_seqlens_expanded, before
            )
            graph.replay()
            apply_metadata(ordinary, mode, seq, req)
            assert_metadata_equal(
                self, backend.forward_metadata, ordinary.forward_metadata
            )
            self.assertEqual(pointers, addresses(backend.forward_metadata))

    def test_recapture_replaces_static_input_identities(self):
        mode = ForwardMode.TARGET_VERIFY
        old_seq, old_req = inputs(*ROUNDS[0])
        backend = make_backend(mode, old_seq, old_req)
        old_graph = capture_verify_metadata(backend, old_seq, old_req)
        old_graph.replay()
        old_state = getattr(backend.forward_metadata, "_ingraph_verify_metadata", None)
        self.assertIsNotNone(old_state)

        seq, req = inputs(*ROUNDS[1])
        batch = SimpleNamespace(
            batch_size=BS,
            forward_mode=mode,
            seq_lens=seq,
            req_pool_indices=req,
            spec_info=None,
            out_cache_loc=None,
        )
        # The public capture preparation must retire the previous alias guard
        # before refreshing metadata from a new graph runner's buffers.
        backend.init_forward_metadata_out_graph(batch, in_capture=True)
        self.assertIsNone(
            getattr(backend.forward_metadata, "_ingraph_verify_metadata", None)
        )
        graph = capture_verify_metadata(backend, seq, req)
        state = getattr(backend.forward_metadata, "_ingraph_verify_metadata", None)
        self.assertIsNotNone(state)
        self.assertIsNot(state, old_state)
        pointers = addresses(backend.forward_metadata)
        ordinary = make_backend(mode, seq, req, fusion=False)

        lengths, requests = ROUNDS[2]
        seq.copy_(torch.tensor(lengths, device="cuda"))
        req.copy_(torch.tensor(requests, device="cuda"))
        apply_metadata(backend, mode, seq, req)
        graph.replay()
        apply_metadata(ordinary, mode, seq, req)
        assert_metadata_equal(self, backend.forward_metadata, ordinary.forward_metadata)
        self.assertEqual(pointers, addresses(backend.forward_metadata))
        with self.assertRaisesRegex(RuntimeError, "alias|static buffers"):
            apply_metadata(backend, mode, old_seq, old_req)

    def test_replay_rejects_replaced_input_buffers(self):
        mode = ForwardMode.TARGET_VERIFY
        seq, req = inputs(*ROUNDS[0])
        backend = make_backend(mode, seq, req)
        graph = capture_verify_metadata(backend, seq, req)
        self.assertIsNotNone(
            getattr(backend.forward_metadata, "_ingraph_verify_metadata", None)
        )
        for bad_seq, bad_req in ((seq.clone(), req), (seq, req.clone())):
            with self.subTest(seq_replaced=bad_seq is not seq):
                with self.assertRaisesRegex(RuntimeError, "alias|static buffers"):
                    apply_metadata(backend, mode, bad_seq, bad_req)
        # Views with the same pointer, shape and stride remain valid.
        apply_metadata(backend, mode, seq[:], req[:])
        graph.replay()


if __name__ == "__main__":
    unittest.main()
