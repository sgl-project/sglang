"""MiniMax-M3 DSpark static-verify metadata contract tests."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.minimax_sparse_backend import (
    MiniMaxSparseAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMiniMaxM3DSparkStaticMetadata(CustomTestCase):
    @staticmethod
    def _make_backend(*, is_dspark: bool = True, is_npu: bool = False):
        backend = object.__new__(MiniMaxSparseAttnBackend)
        backend.is_dspark = is_dspark
        backend.is_npu = is_npu
        backend.speculative_num_draft_tokens = 9
        backend._static_dspark_verify_width_cg = 0
        backend._static_dspark_extend_lens_cg = None
        backend._static_dspark_extend_start_loc_cg = None
        return backend

    @staticmethod
    def _make_forward_batch(
        *,
        forward_mode: ForwardMode = ForwardMode.TARGET_VERIFY,
        ragged_verify_layout=None,
    ):
        return SimpleNamespace(
            forward_mode=forward_mode,
            spec_info=SimpleNamespace(
                draft_token_num=9,
                ragged_verify_layout=ragged_verify_layout,
            ),
            seq_lens=torch.tensor([100, 50], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([109, 59], dtype=torch.int64),
            extend_prefix_lens=None,
            extend_seq_lens=None,
            extend_prefix_lens_cpu=None,
            extend_seq_lens_cpu=None,
            extend_logprob_start_lens_cpu=None,
            extend_num_tokens=None,
            extend_start_loc=None,
            num_padding=0,
        )

    def test_builds_static_verify_metadata_without_changing_seq_lens(self):
        backend = self._make_backend()
        forward_batch = self._make_forward_batch()
        original_seq_lens = forward_batch.seq_lens.clone()

        backend._ensure_static_dspark_verify_metadata(forward_batch)

        torch.testing.assert_close(forward_batch.seq_lens, original_seq_lens)
        torch.testing.assert_close(
            forward_batch.extend_prefix_lens,
            torch.tensor([100, 50], dtype=torch.int32),
        )
        torch.testing.assert_close(
            forward_batch.extend_seq_lens,
            torch.tensor([9, 9], dtype=torch.int32),
        )
        torch.testing.assert_close(
            forward_batch.extend_start_loc,
            torch.tensor([0, 9], dtype=torch.int32),
        )
        self.assertEqual(forward_batch.extend_num_tokens, 18)
        self.assertEqual(forward_batch.extend_prefix_lens_cpu, [100, 50])
        self.assertEqual(forward_batch.extend_seq_lens_cpu, [9, 9])
        self.assertEqual(forward_batch.extend_logprob_start_lens_cpu, [100, 50])

    def test_sparse_kernel_receives_prefix_plus_extend_as_total_k_length(self):
        backend = self._make_backend()
        forward_batch = self._make_forward_batch()
        q = torch.empty((18, 1, 1))

        cu_seqlens, seq_lens, prefix_lens = backend._resolve_extend_meta(
            forward_batch, q
        )

        torch.testing.assert_close(
            cu_seqlens, torch.tensor([0, 9, 18], dtype=torch.int32)
        )
        torch.testing.assert_close(seq_lens, torch.tensor([109, 59], dtype=torch.int32))
        torch.testing.assert_close(
            prefix_lens, torch.tensor([100, 50], dtype=torch.int32)
        )

    def test_capture_treats_dummy_cpu_lengths_as_prefixes(self):
        backend = self._make_backend()
        forward_batch = self._make_forward_batch()
        forward_batch.seq_lens = torch.tensor([1, 1], dtype=torch.int64)
        forward_batch.seq_lens_cpu = torch.tensor([1, 1], dtype=torch.int64)

        backend._ensure_static_dspark_verify_metadata(forward_batch, in_capture=True)

        self.assertEqual(forward_batch.extend_prefix_lens_cpu, [1, 1])
        self.assertEqual(forward_batch.extend_seq_lens_cpu, [9, 9])

    def test_cuda_graph_metadata_addresses_survive_capture_to_replay(self):
        backend = self._make_backend()
        backend.req_to_token = torch.empty((1, 1))
        backend.init_cuda_graph_state(max_bs=4, max_num_tokens=36)

        stable_seq_lens = torch.tensor([1, 1, 1, 1], dtype=torch.int64)
        capture_batch = self._make_forward_batch()
        capture_batch.seq_lens = stable_seq_lens
        capture_batch.seq_lens_cpu = torch.tensor([1, 1, 1, 1], dtype=torch.int64)
        capture_batch.num_padding = 0
        backend._ensure_static_dspark_verify_metadata(capture_batch, in_capture=True)

        captured_prefix_ptr = capture_batch.extend_prefix_lens.data_ptr()
        captured_extend_ptr = capture_batch.extend_seq_lens.data_ptr()
        captured_start_ptr = capture_batch.extend_start_loc.data_ptr()
        self.assertEqual(captured_prefix_ptr, stable_seq_lens.data_ptr())

        stable_seq_lens.copy_(torch.tensor([100, 50, 1, 1]))
        replay_batch = self._make_forward_batch()
        replay_batch.seq_lens = stable_seq_lens
        replay_batch.seq_lens_cpu = torch.tensor([109, 59, 1, 1], dtype=torch.int64)
        replay_batch.num_padding = 2
        backend._ensure_static_dspark_verify_metadata(replay_batch)

        self.assertEqual(
            replay_batch.extend_prefix_lens.data_ptr(), captured_prefix_ptr
        )
        self.assertEqual(replay_batch.extend_seq_lens.data_ptr(), captured_extend_ptr)
        self.assertEqual(replay_batch.extend_start_loc.data_ptr(), captured_start_ptr)
        torch.testing.assert_close(replay_batch.extend_prefix_lens, stable_seq_lens)
        torch.testing.assert_close(
            replay_batch.extend_seq_lens,
            torch.tensor([9, 9, 9, 9], dtype=torch.int32),
        )
        torch.testing.assert_close(
            replay_batch.extend_start_loc,
            torch.tensor([0, 9, 18, 27], dtype=torch.int32),
        )
        self.assertEqual(replay_batch.extend_prefix_lens_cpu, [100, 50, 1, 1])

    def test_replay_keeps_padded_cuda_graph_slots_as_dummy_prefixes(self):
        backend = self._make_backend()
        forward_batch = self._make_forward_batch()
        forward_batch.seq_lens = torch.tensor([100, 50, 1, 1], dtype=torch.int64)
        forward_batch.seq_lens_cpu = torch.tensor([109, 59, 1, 1], dtype=torch.int64)
        forward_batch.num_padding = 2

        backend._ensure_static_dspark_verify_metadata(forward_batch)

        self.assertEqual(forward_batch.extend_prefix_lens_cpu, [100, 50, 1, 1])
        self.assertEqual(forward_batch.extend_seq_lens_cpu, [9, 9, 9, 9])

    def test_other_backends_and_modes_are_unchanged(self):
        cases = (
            (self._make_backend(is_dspark=False), self._make_forward_batch()),
            (self._make_backend(is_npu=True), self._make_forward_batch()),
            (
                self._make_backend(),
                self._make_forward_batch(forward_mode=ForwardMode.DECODE),
            ),
            (
                self._make_backend(),
                self._make_forward_batch(ragged_verify_layout=object()),
            ),
        )
        for backend, forward_batch in cases:
            with self.subTest(
                is_dspark=backend.is_dspark,
                is_npu=backend.is_npu,
                forward_mode=forward_batch.forward_mode,
                ragged=forward_batch.spec_info.ragged_verify_layout is not None,
            ):
                backend._ensure_static_dspark_verify_metadata(forward_batch)
                self.assertIsNone(forward_batch.extend_prefix_lens)
                self.assertIsNone(forward_batch.extend_seq_lens)

    def test_rejects_partial_or_inconsistent_length_metadata(self):
        backend = self._make_backend()
        forward_batch = self._make_forward_batch()
        forward_batch.extend_prefix_lens = torch.tensor([100, 50])
        with self.assertRaisesRegex(RuntimeError, "requires both prefix"):
            backend._ensure_static_dspark_verify_metadata(forward_batch)

        forward_batch = self._make_forward_batch()
        forward_batch.seq_lens_cpu = torch.tensor([8, 59])
        with self.assertRaisesRegex(RuntimeError, "smaller than the verify width"):
            backend._ensure_static_dspark_verify_metadata(forward_batch)


if __name__ == "__main__":
    unittest.main()
