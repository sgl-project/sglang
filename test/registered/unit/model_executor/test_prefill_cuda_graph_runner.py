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
    def test_prefix_chunk_capacity_is_aggregate_and_can_be_overridden(self):
        model_runner = SimpleNamespace(
            server_args=SimpleNamespace(
                chunked_prefill_size=16,
                cuda_graph_config=SimpleNamespace(
                    prefill=SimpleNamespace(
                        full_prefill_prefix_chunk_tokens=None, max_bs=8
                    )
                ),
            ),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.empty((1, 32), dtype=torch.int32)
            ),
        )

        self.assertEqual(
            PrefillCudaGraphRunner._resolve_prefix_chunk_shape(model_runner, 4),
            (4, 16),
        )

        model_runner.server_args.chunked_prefill_size = -1
        self.assertEqual(
            PrefillCudaGraphRunner._resolve_prefix_chunk_shape(model_runner, 4),
            (2, 8),
        )
        model_runner.server_args.chunked_prefill_size = 16

        model_runner.server_args.cuda_graph_config.prefill.full_prefill_prefix_chunk_tokens = (
            24
        )
        self.assertEqual(
            PrefillCudaGraphRunner._resolve_prefix_chunk_shape(model_runner, 4),
            (6, 24),
        )

        model_runner.server_args.cuda_graph_config.prefill.full_prefill_prefix_chunk_tokens = (
            256
        )
        self.assertEqual(
            PrefillCudaGraphRunner._resolve_prefix_chunk_shape(model_runner, 4),
            (32, 128),
        )

        # At least one token is reserved per request lane even if the requested
        # aggregate capacity is smaller than the fixed request-slot count.
        model_runner.server_args.cuda_graph_config.prefill.full_prefill_prefix_chunk_tokens = (
            2
        )
        self.assertEqual(
            PrefillCudaGraphRunner._resolve_prefix_chunk_shape(model_runner, 4),
            (1, 4),
        )

        model_runner.server_args.cuda_graph_config.prefill.full_prefill_prefix_chunk_tokens = (
            0
        )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            PrefillCudaGraphRunner._resolve_prefix_chunk_shape(model_runner, 4)

    def test_backend_contract_and_buffers_are_shared_across_token_buckets(self):
        unsupported = SimpleNamespace(supports_full_cuda_graph_chunked_prefix=False)
        with self.assertRaisesRegex(AssertionError, "does not support"):
            PrefillCudaGraphRunner._assert_chunked_prefix_backend_supported(unsupported)

        backend = _FakeAttentionBackend()
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._capture_req_slots = 3
        runner._prefix_chunk_len = 2
        runner._prefix_chunk_capacity = 6
        runner._prefix_max_len = 8
        runner._prefix_capture_variants = (1, 2, 4)
        runner.device = torch.device("cpu")
        runner._prefill_static_buffers = {
            "extend_prefix_lens": torch.zeros(3, dtype=torch.int64)
        }
        runner._prefix_capture_batches = {}
        runner._prefix_capture_buffers = None
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
        first_key = ShapeKey(size=8, variant_label="chunked_prefix:4")
        second_key = ShapeKey(size=16, variant_label="chunked_prefix:4")

        with patch.object(
            runner_module,
            "create_chunked_prefix_cache_kv_indices",
            _FakeKVIndexKernel(),
        ):
            runner._prepare_chunked_prefix_capture(first, first_key, 4)
            runner._prepare_chunked_prefix_capture(second, second_key, 4)

            buffers = runner._prefix_capture_buffers
            self.assertIsNotNone(buffers)
            self.assertEqual(first.extend_prefix_lens_cpu, [8, 8, 8])
            self.assertEqual(first.prefix_chunk_num_tokens, [6, 6, 6, 6])
            self.assertIs(first.prefix_chunk_starts, buffers.starts)
            self.assertIs(first.prefix_chunk_seq_lens, buffers.seq_lens)
            self.assertIs(first.prefix_chunk_cu_seq_lens, buffers.cu_seq_lens)
            self.assertIs(first.prefix_chunk_kv_indices[0], buffers.kv_indices[0])
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
            self.assertIs(
                first.prefix_chunk_kv_indices[3],
                second.prefix_chunk_kv_indices[3],
            )

            runner._prefill_static_buffers["extend_prefix_lens"].copy_(
                torch.tensor([5, 1, 0])
            )
            runner._prepare_chunked_prefix_replay(
                second_key,
                SimpleNamespace(batch_size=2, extend_prefix_lens_cpu=[5, 1]),
            )

        self.assertEqual(
            second.prefix_chunk_seq_lens.tolist(),
            [[2, 1, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0]],
        )
        self.assertEqual(
            second.prefix_chunk_kv_indices[0].tolist(),
            [16, 17, 0, 0, 0, 0],
        )
        self.assertEqual(
            second.prefix_chunk_kv_indices[1].tolist(),
            [18, 19, 0, 0, 0, 0],
        )
        self.assertEqual(
            second.prefix_chunk_kv_indices[2].tolist(),
            [20, 0, 0, 0, 0, 0],
        )
        self.assertEqual(second.prefix_chunk_kv_indices[3].tolist(), [0] * 6)
        self.assertEqual(
            backend.calls,
            [(first, True), (second, True), (second, False)],
        )

    def test_prefix_gate_only_applies_to_chunked_prefix_variant(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._capture_req_slots = 4
        runner.capture_hidden_mode = None
        runner.max_num_tokens = 32
        runner.backend = SimpleNamespace()
        runner._prefix_chunk_len = 2
        runner._prefix_capture_variants = (1, 2, 4)

        forward_batch = SimpleNamespace(
            batch_size=1,
            input_ids=torch.zeros(4, dtype=torch.int64),
            input_embeds=None,
            replace_embeds=None,
            forward_mode=SimpleNamespace(is_target_verify=lambda: False),
            capture_hidden_mode=None,
            global_num_tokens_cpu=None,
            return_logprob=False,
            extend_prefix_lens_cpu=[8],
        )

        # Prefix hits in BCG/TC-piecewise and ordinary non-MLA FullCG use the
        # normal graph topology and must retain their existing eligibility.
        runner._capture_chunked_prefix = False
        for is_full_backend in (False, True):
            with self.subTest(is_full_backend=is_full_backend):
                runner._is_full_backend = is_full_backend
                self.assertTrue(runner.can_run_graph(forward_batch))

        # The dedicated chunked-prefix topology has a fixed captured capacity.
        runner._is_full_backend = True
        runner._capture_chunked_prefix = True
        self.assertTrue(runner.can_run_graph(forward_batch))
        self.assertEqual(
            runner._shape_key(4, forward_batch).variant_label,
            "chunked_prefix:4",
        )
        forward_batch.batch_size = 2
        # Capacity is per request, not a sum: three real chunks round up to the
        # four-chunk graph even though the aggregate prefix has eight tokens.
        forward_batch.extend_prefix_lens_cpu = [5, 3]
        self.assertTrue(runner.can_run_graph(forward_batch))
        self.assertEqual(
            runner._shape_key(4, forward_batch).variant_label,
            "chunked_prefix:4",
        )
        forward_batch.extend_prefix_lens_cpu = [9, 1]
        self.assertFalse(runner.can_run_graph(forward_batch))


if __name__ == "__main__":
    unittest.main()
