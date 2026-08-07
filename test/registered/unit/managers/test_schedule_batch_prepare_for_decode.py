import types
import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req():
    return types.SimpleNamespace(
        decode_batch_idx=0,
        kv_committed_len=3,
        kv_allocated_len=3,
    )


def _make_decode_batch():
    batch = ScheduleBatch(reqs=[_make_req(), _make_req()])
    batch.device = "cpu"
    batch.model_config = types.SimpleNamespace(is_encoder_decoder=False)
    batch.enable_overlap = False
    batch.spec_algorithm = types.SimpleNamespace(is_none=lambda: True)
    batch.sampling_info = types.SimpleNamespace(
        penalizer_orchestrator=types.SimpleNamespace(is_required=False)
    )
    batch.hisparse_coordinator = None
    batch.seq_lens = torch.tensor([3, 5], dtype=torch.int64)
    batch.seq_lens_cpu = torch.tensor([3, 5], dtype=torch.int64)
    batch.orig_seq_lens = torch.tensor([3, 5], dtype=torch.int32)
    return batch


def _make_mamba_extend_req(prefix_len: int, extend_len: int):
    return types.SimpleNamespace(
        prefix_indices=list(range(prefix_len)),
        extend_range=types.SimpleNamespace(length=extend_len),
        mamba_ping_pong_track_buffer=torch.tensor([17, 18]),
        mamba_next_track_idx=0,
        mamba_branching_seqlen=None,
        mamba_last_track_seqlen=None,
    )


class TestPrepareForDecodeSeqLensOwnership(unittest.TestCase):
    def test_decode_seq_lens_bump_is_out_of_place(self):
        """Each prepare_for_decode call rebinds seq-lens tensors to new +1 objects without mutating the old ones."""
        batch = _make_decode_batch()

        server_args = types.SimpleNamespace(
            enable_mamba_extra_buffer=lambda: False,
        )
        with (
            patch(
                "sglang.srt.managers.schedule_batch.alloc_for_decode",
                return_value=torch.tensor([6, 7], dtype=torch.int64),
            ),
            patch(
                "sglang.srt.managers.schedule_batch.get_server_args",
                return_value=server_args,
            ),
        ):
            for step in range(1, 3):
                prev_seq_lens = batch.seq_lens
                prev_seq_lens_cpu = batch.seq_lens_cpu
                prev_orig_seq_lens = batch.orig_seq_lens
                prev_values = (
                    prev_seq_lens.clone(),
                    prev_seq_lens_cpu.clone(),
                    prev_orig_seq_lens.clone(),
                )

                batch.prepare_for_decode()

                self.assertIsNot(batch.seq_lens, prev_seq_lens)
                self.assertIsNot(batch.seq_lens_cpu, prev_seq_lens_cpu)
                self.assertIsNot(batch.orig_seq_lens, prev_orig_seq_lens)
                expected = torch.tensor([3 + step, 5 + step], dtype=torch.int64)
                self.assertTrue(torch.equal(batch.seq_lens, expected))
                self.assertTrue(torch.equal(batch.seq_lens_cpu, expected))
                self.assertTrue(
                    torch.equal(batch.orig_seq_lens, expected.to(torch.int32))
                )
                self.assertTrue(torch.equal(prev_seq_lens, prev_values[0]))
                self.assertTrue(torch.equal(prev_seq_lens_cpu, prev_values[1]))
                self.assertTrue(torch.equal(prev_orig_seq_lens, prev_values[2]))


class TestMambaCheckpointTracking(unittest.TestCase):
    def _prepare(self, prefix_len: int, extend_len: int):
        batch = object.__new__(ScheduleBatch)
        batch.req_to_token_pool = types.SimpleNamespace(
            get_mamba_ping_pong_other_idx=lambda idx: 1 - idx
        )
        req = _make_mamba_extend_req(prefix_len, extend_len)
        server_args = types.SimpleNamespace(
            mamba_cache_chunk_size=64,
            mamba_radix_checkpoint_interval=512,
            enable_mamba_extra_buffer_lazy=lambda: False,
        )
        with patch(
            "sglang.srt.managers.schedule_batch.get_server_args",
            return_value=server_args,
        ):
            entry = batch._mamba_radix_cache_v2_req_prepare_for_extend(req)
        return req, entry

    def test_tracks_small_chunk_that_crosses_checkpoint(self):
        req, entry = self._prepare(prefix_len=256, extend_len=256)

        self.assertTrue(entry.track_mask)
        self.assertEqual(entry.track_index, 17)
        self.assertEqual(entry.track_seqlen, 512)
        self.assertEqual(req.mamba_last_track_seqlen, 512)
        self.assertEqual(req.mamba_next_track_idx, 1)

    def test_does_not_track_small_chunk_without_crossing(self):
        req, entry = self._prepare(prefix_len=256, extend_len=128)

        self.assertFalse(entry.track_mask)
        self.assertEqual(entry.track_index, 17)
        self.assertEqual(entry.track_seqlen, -1)
        self.assertIsNone(req.mamba_last_track_seqlen)
        self.assertEqual(req.mamba_next_track_idx, 0)

    def test_internal_checkpoint_uses_intermediate_state(self):
        req, entry = self._prepare(prefix_len=400, extend_len=200)

        self.assertTrue(entry.track_mask)
        self.assertEqual(entry.track_seqlen, 513)
        self.assertEqual(req.mamba_last_track_seqlen, 512)


if __name__ == "__main__":
    unittest.main()
