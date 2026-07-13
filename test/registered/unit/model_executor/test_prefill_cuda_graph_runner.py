"""CPU coverage for chunked-prefix Full prefill CUDA-graph state."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.model_executor.runner.prefill_cuda_graph_runner as runner_module
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeAttentionBackend:
    supports_full_cuda_graph_chunked_prefix = True

    def __init__(self):
        self.calls = []

    def prepare_full_cuda_graph_chunked_prefix(self, forward_batch, *, in_capture):
        self.calls.append((forward_batch, in_capture))


class _FakeKVIndexKernel:
    def __getitem__(self, grid):
        del grid

        def run(
            req_to_token,
            req_pool_indices,
            starts,
            seq_lens,
            cu_seq_lens,
            output,
            req_to_token_stride,
        ):
            del cu_seq_lens, req_to_token_stride
            cursor = 0
            for row in range(seq_lens.numel()):
                seq_len = int(seq_lens[row])
                start = int(starts[row])
                req = int(req_pool_indices[row])
                output[cursor : cursor + seq_len].copy_(
                    req_to_token[req, start : start + seq_len]
                )
                cursor += seq_len

        return run


class TestPrefillCudaGraphRunnerChunkedPrefix(CustomTestCase):
    def test_backend_contract_and_buffers_are_shared_across_token_buckets(self):
        unsupported = SimpleNamespace(supports_full_cuda_graph_chunked_prefix=False)
        with self.assertRaisesRegex(AssertionError, "does not support"):
            PrefillCudaGraphRunner._assert_chunked_prefix_backend_supported(unsupported)

        backend = _FakeAttentionBackend()
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._capture_req_slots = 3
        runner._prefix_chunk_len = 4
        runner.device = torch.device("cpu")
        runner._prefill_static_buffers = {
            "extend_prefix_lens": torch.zeros(3, dtype=torch.int64)
        }
        runner._prefix_capture_batches = {}
        runner._prefix_capture_buffer_owner = None
        runner.model_runner = SimpleNamespace(
            attn_backend=backend,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(24, dtype=torch.int32).view(3, 8)
            ),
        )

        first = SimpleNamespace(
            req_pool_indices=torch.tensor([2, 0, 1], dtype=torch.int64)
        )
        second = SimpleNamespace(
            req_pool_indices=torch.tensor([2, 0, 1], dtype=torch.int64)
        )
        first_key = ShapeKey(size=8, variant_label="chunked_prefix_1")
        second_key = ShapeKey(size=16, variant_label="chunked_prefix_1")

        with patch.object(
            runner_module,
            "create_chunked_prefix_cache_kv_indices",
            _FakeKVIndexKernel(),
        ):
            runner._prepare_chunked_prefix_capture(first, first_key)
            runner._prepare_chunked_prefix_capture(second, second_key)

            self.assertIs(first.prefix_chunk_starts, second.prefix_chunk_starts)
            self.assertIs(first.prefix_chunk_seq_lens, second.prefix_chunk_seq_lens)
            self.assertIs(
                first.prefix_chunk_cu_seq_lens,
                second.prefix_chunk_cu_seq_lens,
            )
            self.assertIs(
                first.prefix_chunk_kv_indices[0],
                second.prefix_chunk_kv_indices[0],
            )

            runner._prefill_static_buffers["extend_prefix_lens"].copy_(
                torch.tensor([3, 1, 0])
            )
            runner._prepare_chunked_prefix_replay(
                second_key,
                SimpleNamespace(batch_size=2, extend_prefix_lens_cpu=[3, 1]),
            )

        self.assertEqual(second.prefix_chunk_seq_lens.tolist(), [[3, 1, 0]])
        self.assertEqual(
            second.prefix_chunk_kv_indices[0].tolist(),
            [16, 17, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        self.assertEqual(
            backend.calls,
            [(first, True), (second, True), (second, False)],
        )


if __name__ == "__main__":
    unittest.main()
