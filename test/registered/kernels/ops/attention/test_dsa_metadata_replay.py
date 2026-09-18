"""Fused KPool replay and MTP sibling copies preserve captured buffer identity."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.dsa_metadata_kit import (
    BS,
    NEXT_N,
    ROUNDS,
    addresses,
    apply_metadata,
    assert_metadata_equal,
    inputs,
    make_backend,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


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
