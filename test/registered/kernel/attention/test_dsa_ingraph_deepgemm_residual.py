"""Out-of-graph DeepGEMM schedules must use this round's raw sequence lengths."""

import unittest

import deep_gemm
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.dsa_metadata_kit import (
    NEXT_N,
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


class TestDSAInGraphDeepGEMMResidual(CustomTestCase):
    def test_residual_rebuilds_both_schedules_before_graph_replay(self):
        mode = ForwardMode.TARGET_VERIFY
        seq, req = inputs(*ROUNDS[0])
        backend = make_backend(mode, seq, req)
        ordinary = make_backend(mode, seq, req, fusion=False)
        graph = capture_verify_metadata(backend, seq, req, dg_out_of_graph=True)
        metadata = backend.forward_metadata
        state = getattr(metadata, "_ingraph_verify_metadata", None)
        self.assertIsNotNone(state)
        self.assertIsNotNone(getattr(state, "residual_launch", None))
        self.assertIsNotNone(metadata.kpool_write_plan.pool_schedule_metadata)
        pointers = addresses(metadata)
        offsets = torch.arange(1, NEXT_N + 1, device="cuda", dtype=torch.int32)
        for lengths, requests in ROUNDS[1:]:
            seq.copy_(torch.tensor(lengths, device="cuda"))
            req.copy_(torch.tensor(requests, device="cuda"))
            # These buffers are still from the previous graph round when the
            # residual runs. Poison them to expose an accidental stale read.
            metadata.dsa_seqlens_expanded.fill_(1)
            metadata.paged_mqa_ctx_lens_2d.fill_(1)
            metadata.kpool_write_plan.pool_seqlens_per_q.fill_(1)
            expanded = seq.int()[:, None] + offsets
            if get_platform().is_sm100:
                ctx = (seq.int() + NEXT_N)[:, None].expand(-1, NEXT_N).contiguous()
            else:
                ctx = expanded.reshape(-1, 1).contiguous()
            pool_ctx = (expanded // POOL).reshape(-1, 1).clamp(min=1).contiguous()
            expected_schedule = deep_gemm.get_paged_mqa_logits_metadata(
                ctx, 64, deep_gemm.get_num_sms()
            )
            expected_pool_schedule = deep_gemm.get_paged_mqa_logits_metadata(
                pool_ctx, 64, deep_gemm.get_num_sms()
            )
            apply_metadata(backend, mode, seq, req)
            # Assert before replay: the residual must be sufficient on its own.
            torch.testing.assert_close(
                metadata.paged_mqa_schedule_metadata, expected_schedule
            )
            torch.testing.assert_close(
                metadata.kpool_write_plan.pool_schedule_metadata, expected_pool_schedule
            )
            torch.testing.assert_close(
                metadata.dsa_seqlens_expanded,
                torch.ones_like(metadata.dsa_seqlens_expanded),
            )
            graph.replay()
            apply_metadata(ordinary, mode, seq, req)
            assert_metadata_equal(self, metadata, ordinary.forward_metadata)
            self.assertEqual(pointers, addresses(metadata))


if __name__ == "__main__":
    unittest.main()
