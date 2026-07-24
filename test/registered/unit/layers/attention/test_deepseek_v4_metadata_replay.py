"""CPU contracts for DSV4 draft CUDA-graph metadata replay."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

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

    def _make_step_backend(
        self,
        batch,
        *,
        graph_req_pool_indices,
        graph_seq_lens,
        dummy_value,
    ):
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.is_draft_runner = True
        backend.needs_cpu_seq_lens = False
        backend.is_dspark_draft = False
        backend.forward_metadata = None
        backend.init_forward_metadata_out_graph = mock.Mock()
        backend.replay_cuda_graph_metadata_from = mock.Mock()

        raw_metadata = DSV4RawDecodeMetadata(
            req_pool_indices=graph_req_pool_indices[:],
            seq_lens=graph_seq_lens[:],
            out_cache_loc=torch.full(
                (batch.batch_size,), dummy_value, dtype=torch.int64
            ),
        )
        backend.cuda_graph_metadata_of_bucket_and_bs = {
            bucket: {} for bucket in _GraphBucket
        }
        backend.cuda_graph_metadata_of_bucket_and_bs[_GraphBucket.DECODE_OR_IDLE][
            batch.batch_size
        ] = raw_metadata
        return backend, raw_metadata

    def _make_multi_step_backend(self, batch):
        backend = object.__new__(DeepseekV4MultiStepBackend)
        backend.speculative_num_steps = 3
        graph_req_pool_indices = batch.req_pool_indices.clone()
        graph_seq_lens = batch.seq_lens.clone()
        step_backends_and_metadata = [
            self._make_step_backend(
                batch,
                graph_req_pool_indices=graph_req_pool_indices,
                graph_seq_lens=graph_seq_lens,
                dummy_value=-(i + 1),
            )
            for i in range(3)
        ]
        backend.attn_backends = [
            step_backend for step_backend, _ in step_backends_and_metadata
        ]
        metadata = [step_metadata for _, step_metadata in step_backends_and_metadata]
        return backend, metadata, graph_req_pool_indices, graph_seq_lens

    def _set_fallback_metadata(self, backend, fallback_metadata):
        def build_fallback_metadata(_forward_batch):
            backend.attn_backends[0].forward_metadata = fallback_metadata

        backend.attn_backends[0].init_forward_metadata_out_graph.side_effect = (
            build_fallback_metadata
        )

    def test_capture_raw_metadata_aliases_graph_inputs_but_uses_dummy_out_locations(
        self,
    ):
        batch = self._make_batch()
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.is_draft_runner = True
        backend.is_dspark_draft = False
        backend.needs_cpu_seq_lens = False
        backend.MAX_SEQ_LEN_FOR_CAPTURE = 4096
        backend.online_c128_mtp = mock.Mock()
        backend.init_forward_metadata_decode = mock.Mock()
        backend.replay_cuda_graph_metadata_from = mock.Mock()

        def make_raw_metadata(**kwargs):
            return DSV4RawDecodeMetadata(
                req_pool_indices=kwargs["req_pool_indices"],
                seq_lens=kwargs["seq_lens"],
                out_cache_loc=kwargs["out_cache_loc"],
            )

        backend.init_forward_metadata_decode.side_effect = make_raw_metadata

        def select_metadata(*, bs, temp_metadata, bucket):
            backend.forward_metadata = temp_metadata

        backend.replay_cuda_graph_metadata_from.side_effect = select_metadata

        backend.init_forward_metadata_out_graph(batch, in_capture=True)

        raw_metadata = backend.forward_metadata
        self.assertEqual(
            raw_metadata.req_pool_indices.data_ptr(),
            batch.req_pool_indices.data_ptr(),
        )
        self.assertEqual(raw_metadata.seq_lens.data_ptr(), batch.seq_lens.data_ptr())
        self.assertEqual(raw_metadata.out_cache_loc.shape, (batch.batch_size,))
        self.assertIsNot(raw_metadata.out_cache_loc, batch.out_cache_loc)
        torch.testing.assert_close(
            raw_metadata.out_cache_loc, torch.zeros_like(batch.seq_lens)
        )
        self.assertIs(backend._current_capture_raw, raw_metadata)

    def test_reuses_captured_raw_metadata_and_ignores_runtime_tensor_identity(self):
        batch = self._make_batch()
        backend, metadata, graph_req_pool_indices, graph_seq_lens = (
            self._make_multi_step_backend(batch)
        )
        dummy_out_locations = [item.out_cache_loc.clone() for item in metadata[:2]]
        dummy_ptrs = [item.out_cache_loc.data_ptr() for item in metadata[:2]]
        self.assertNotEqual(
            metadata[0].req_pool_indices.data_ptr(),
            batch.req_pool_indices.data_ptr(),
        )
        self.assertNotEqual(metadata[0].seq_lens.data_ptr(), batch.seq_lens.data_ptr())

        with (
            mock.patch.object(
                DSV4RawDecodeMetadata,
                "copy_",
                side_effect=AssertionError("raw metadata must not be copied"),
            ),
            mock.patch.object(
                torch.nn.functional,
                "pad",
                side_effect=AssertionError("draft output locations must not be padded"),
            ),
        ):
            backend.init_forward_metadata_out_graph(batch)

            # Model the runner's next grouped refresh of the captured graph
            # inputs. The runtime batch remains separate and is not consulted by
            # the no-copy metadata path.
            graph_req_pool_indices.add_(10)
            graph_seq_lens.add_(1)
            batch.req_pool_indices = torch.tensor([90, 91], dtype=torch.int64)
            batch.seq_lens = torch.tensor([900, 901], dtype=torch.int64)
            backend.init_forward_metadata_out_graph(batch)

        for i in range(2):
            step_backend = backend.attn_backends[i]
            self.assertIs(step_backend.forward_metadata, metadata[i])
            step_backend.init_forward_metadata_out_graph.assert_not_called()
            step_backend.replay_cuda_graph_metadata_from.assert_not_called()
            self.assertEqual(metadata[i].out_cache_loc.data_ptr(), dummy_ptrs[i])
            torch.testing.assert_close(
                metadata[i].out_cache_loc, dummy_out_locations[i]
            )

        torch.testing.assert_close(
            metadata[0].req_pool_indices, torch.tensor([13, 17], dtype=torch.int64)
        )
        torch.testing.assert_close(
            metadata[0].seq_lens, torch.tensor([33, 65], dtype=torch.int64)
        )
        self.assertIsNone(backend.attn_backends[2].forward_metadata)

    def test_later_step_without_raw_metadata_uses_wholesale_fallback(self):
        batch = self._make_batch()
        backend, metadata, _, _ = self._make_multi_step_backend(batch)
        backend.attn_backends[1].cuda_graph_metadata_of_bucket_and_bs[
            _GraphBucket.DECODE_OR_IDLE
        ][batch.batch_size] = object()
        fallback_metadata = object()
        self._set_fallback_metadata(backend, fallback_metadata)

        backend.init_forward_metadata_out_graph(batch)

        backend.attn_backends[0].init_forward_metadata_out_graph.assert_called_once()
        backend.attn_backends[
            1
        ].replay_cuda_graph_metadata_from.assert_called_once_with(
            bs=batch.batch_size,
            temp_metadata=fallback_metadata,
            bucket=_GraphBucket.DECODE_OR_IDLE,
        )
        self.assertIs(backend.attn_backends[0].forward_metadata, fallback_metadata)
        self.assertIsNone(backend.attn_backends[1].forward_metadata)
        self.assertIsNot(metadata[0], fallback_metadata)

    def test_idle_cpu_and_dspark_planning_use_existing_copy_path(self):
        cases = (
            ("idle", "forward_mode", ForwardMode.IDLE),
            ("cpu", "needs_cpu_seq_lens", True),
            ("dspark", "is_dspark_draft", True),
        )
        for name, field, value in cases:
            with self.subTest(name=name):
                batch = self._make_batch()
                backend, _, _, _ = self._make_multi_step_backend(batch)
                if field == "forward_mode":
                    setattr(batch, field, value)
                else:
                    setattr(backend.attn_backends[0], field, value)
                fallback_metadata = object()
                self._set_fallback_metadata(backend, fallback_metadata)

                backend.init_forward_metadata_out_graph(batch)

                backend.attn_backends[
                    0
                ].init_forward_metadata_out_graph.assert_called_once()
                backend.attn_backends[
                    1
                ].replay_cuda_graph_metadata_from.assert_called_once()


if __name__ == "__main__":
    unittest.main()
