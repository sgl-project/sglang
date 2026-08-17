"""CPU contracts for DSV4 draft CUDA-graph metadata replay."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.attention.deepseek_v4_backend import (  # noqa: E402
    DeepseekV4AttnBackend,
    DeepseekV4MultiStepBackend,
    DSV4RawDecodeMetadata,
    _GraphBucket,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDSV4DraftMetadataReplay(CustomTestCase):
    def _make_batch(self):
        return SimpleNamespace(
            batch_size=2,
            forward_mode=ForwardMode.DECODE,
            input_ids=None,
            positions=torch.arange(2, dtype=torch.int64),
            req_pool_indices=torch.tensor([3, 7], dtype=torch.int64),
            seq_lens=torch.tensor([32, 64], dtype=torch.int64),
            seq_lens_sum=96,
            seq_lens_cpu=None,
            out_cache_loc=torch.arange(6, dtype=torch.int64),
            spec_info=SimpleNamespace(),
        )

    def _make_multi_step_backend(self, batch):
        """3-step draft backend whose per-step raw metadata aliases one shared
        pair of graph buffers (as capture does), each with its own dummy
        out_cache_loc. Returns the shared buffers so a test can mutate them to
        model the runner's per-replay refresh."""
        backend = object.__new__(DeepseekV4MultiStepBackend)
        backend.speculative_num_steps = 3
        graph_req_pool_indices = batch.req_pool_indices.clone()
        graph_seq_lens = batch.seq_lens.clone()
        backend.attn_backends = []
        metadata = []
        for step in range(3):
            step_backend = object.__new__(DeepseekV4AttnBackend)
            step_backend.is_draft_runner = True
            step_backend.needs_cpu_seq_lens = False
            step_backend.is_dspark_draft = False
            step_backend.forward_metadata = None
            step_backend.init_forward_metadata_out_graph = mock.Mock()
            step_backend.replay_cuda_graph_metadata_from = mock.Mock()
            raw = DSV4RawDecodeMetadata(
                req_pool_indices=graph_req_pool_indices[:],
                seq_lens=graph_seq_lens[:],
                out_cache_loc=torch.full(
                    (batch.batch_size,), -(step + 1), dtype=torch.int64
                ),
            )
            step_backend.cuda_graph_metadata_of_bucket_and_bs = {
                bucket: {} for bucket in _GraphBucket
            }
            step_backend.cuda_graph_metadata_of_bucket_and_bs[
                _GraphBucket.DECODE_OR_IDLE
            ][batch.batch_size] = raw
            backend.attn_backends.append(step_backend)
            metadata.append(raw)
        return backend, metadata, graph_req_pool_indices, graph_seq_lens

    def _force_fallback(self, backend, fallback_metadata):
        """Make step 0's re-plan plant ``fallback_metadata``, as the real
        init_forward_metadata_out_graph would."""

        def build_fallback(_forward_batch):
            backend.attn_backends[0].forward_metadata = fallback_metadata

        backend.attn_backends[0].init_forward_metadata_out_graph.side_effect = (
            build_fallback
        )

    def test_capture_aliases_graph_inputs_with_dummy_out_locations(self):
        """Capture yields raw metadata that *views* req/seq (so the runner's
        per-replay buffer refresh is seen) but a fresh dummy out_cache_loc."""
        batch = self._make_batch()
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.is_draft_runner = True
        backend.is_dspark_draft = False
        backend.needs_cpu_seq_lens = False
        backend.MAX_SEQ_LEN_FOR_CAPTURE = 4096
        backend.online_c128_mtp = mock.Mock()
        backend.replay_cuda_graph_metadata_from = mock.Mock()

        def store_first_capture(*, bs, temp_metadata, bucket):
            backend.forward_metadata = temp_metadata

        backend.replay_cuda_graph_metadata_from.side_effect = store_first_capture

        # Exercise the real env-gated producer: SGLANG_PREP_IN_CUDA_GRAPH is the
        # precondition for the fast path, and its raw branch must alias, not clone.
        with envs.SGLANG_PREP_IN_CUDA_GRAPH.override(True):
            backend.init_forward_metadata_out_graph(batch, in_capture=True)

        raw = backend.forward_metadata
        self.assertEqual(
            raw.req_pool_indices.data_ptr(), batch.req_pool_indices.data_ptr()
        )
        self.assertEqual(raw.seq_lens.data_ptr(), batch.seq_lens.data_ptr())
        self.assertIsNot(raw.out_cache_loc, batch.out_cache_loc)
        torch.testing.assert_close(raw.out_cache_loc, torch.zeros_like(batch.seq_lens))
        self.assertIs(backend._current_capture_raw, raw)

    def test_reuse_tracks_refreshed_buffers_without_copying(self):
        """With raw metadata on every step, reuse it in place: no re-plan, no
        copy, reading the runner-refreshed graph buffers, not the runtime batch."""
        batch = self._make_batch()
        backend, metadata, graph_req_pool_indices, graph_seq_lens = (
            self._make_multi_step_backend(batch)
        )
        dummy_ptrs = [m.out_cache_loc.data_ptr() for m in metadata[:2]]

        with (
            mock.patch.object(
                DSV4RawDecodeMetadata,
                "copy_",
                side_effect=AssertionError("raw metadata must not be copied"),
            ),
            mock.patch.object(
                torch.nn.functional,
                "pad",
                side_effect=AssertionError("draft out locations must not be padded"),
            ),
        ):
            backend.init_forward_metadata_out_graph(batch)
            # Runner refreshes the shared graph buffers; the now-divergent runtime
            # batch must be ignored by the no-copy path.
            graph_req_pool_indices.add_(10)
            graph_seq_lens.add_(1)
            batch.req_pool_indices = torch.tensor([90, 91], dtype=torch.int64)
            batch.seq_lens = torch.tensor([900, 901], dtype=torch.int64)
            backend.init_forward_metadata_out_graph(batch)

        for i in range(2):
            step = backend.attn_backends[i]
            self.assertIs(step.forward_metadata, metadata[i])
            step.init_forward_metadata_out_graph.assert_not_called()
            step.replay_cuda_graph_metadata_from.assert_not_called()
            self.assertEqual(metadata[i].out_cache_loc.data_ptr(), dummy_ptrs[i])
        torch.testing.assert_close(
            metadata[0].req_pool_indices, torch.tensor([13, 17], dtype=torch.int64)
        )
        torch.testing.assert_close(
            metadata[0].seq_lens, torch.tensor([33, 65], dtype=torch.int64)
        )
        # The unused final step (num_steps - 1 active) is left untouched.
        self.assertIsNone(backend.attn_backends[2].forward_metadata)

    def test_guards_fall_back_to_replan_copy_path(self):
        """Any condition that breaks the reuse precondition takes the wholesale
        re-plan + copy path: idle mode, cpu/dspark planning, or a step missing
        raw metadata."""

        def idle(backend, batch):
            batch.forward_mode = ForwardMode.IDLE

        def cpu(backend, batch):
            backend.attn_backends[0].needs_cpu_seq_lens = True

        def dspark(backend, batch):
            backend.attn_backends[0].is_dspark_draft = True

        def missing_raw(backend, batch):
            backend.attn_backends[1].cuda_graph_metadata_of_bucket_and_bs[
                _GraphBucket.DECODE_OR_IDLE
            ][batch.batch_size] = object()

        for name, setup in (
            ("idle", idle),
            ("cpu", cpu),
            ("dspark", dspark),
            ("missing_raw", missing_raw),
        ):
            with self.subTest(name=name):
                batch = self._make_batch()
                backend, _, _, _ = self._make_multi_step_backend(batch)
                setup(backend, batch)
                fallback_metadata = object()
                self._force_fallback(backend, fallback_metadata)

                backend.init_forward_metadata_out_graph(batch)

                backend.attn_backends[
                    0
                ].init_forward_metadata_out_graph.assert_called_once()
                backend.attn_backends[
                    1
                ].replay_cuda_graph_metadata_from.assert_called_once_with(
                    bs=batch.batch_size,
                    temp_metadata=fallback_metadata,
                    bucket=_GraphBucket.DECODE_OR_IDLE,
                )
                self.assertIs(
                    backend.attn_backends[0].forward_metadata, fallback_metadata
                )


if __name__ == "__main__":
    unittest.main()
