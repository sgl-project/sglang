"""Regression tests for draft-extend CUDA graph bucket selection.

The runner is built via ``__new__`` to skip ``__init__``'s CUDA allocation and
graph capture; CPU tensors and stubs then let ``can_run_graph()`` and
``execute()`` run on CPU.

Test contract: ``can_run_graph()`` admits and ``execute()`` sizes the replay
bucket from the same raw per-rank request count.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _RecordingBackend:
    """Records the replay-metadata batch size so the test can verify it uses the
    same raw request count admitted by ``can_run_graph()``.
    """

    def __init__(self):
        self.metadata_batch_sizes = []

    def init_forward_metadata_out_graph(self, forward_batch):
        self.metadata_batch_sizes.append(forward_batch.batch_size)


class TestEagleDraftExtendCudaGraphRunner(CustomTestCase):
    def _build_runner(self, *, is_eagle, captured_req_width):
        # Consecutive buckets so the expected replay bs == raw request count.
        max_bs = 8
        max_num_tokens = max_bs * captured_req_width
        backend = _RecordingBackend()
        runner = EAGLEDraftExtendCudaGraphRunner.__new__(
            EAGLEDraftExtendCudaGraphRunner
        )

        runner.require_mlp_tp_gather = True
        runner.require_mlp_sync = False
        runner.require_gathered_buffer = False
        runner.disable_padding = False
        runner.max_bs = max_bs
        runner.capture_bs = list(range(1, max_bs + 1))
        runner.captured_req_width = captured_req_width
        runner.seq_len_fill_value = 1
        runner.forward_mode = object()
        runner.deepep_adapter = SimpleNamespace(replay=lambda: None)
        runner.device_module = SimpleNamespace(
            Event=lambda: SimpleNamespace(record=lambda: None)
        )
        runner.draft_extend_attn_backend = backend
        runner.extend_seq_lens_cpu = [captured_req_width] * max_bs
        runner.buffers = SimpleNamespace(
            input_ids=torch.empty(max_num_tokens, dtype=torch.int64),
            req_pool_indices=torch.empty(max_bs, dtype=torch.int64),
            out_cache_loc=torch.empty(max_num_tokens, dtype=torch.int64),
            positions=torch.empty(max_num_tokens, dtype=torch.int64),
            hidden_states=None,
            seq_lens=torch.empty(max_bs, dtype=torch.int64),
            extend_seq_lens=torch.empty(max_bs, dtype=torch.int32),
            num_correct_drafts=torch.empty(max_bs, dtype=torch.int32),
            num_accept_tokens=torch.empty(max_bs, dtype=torch.int32),
        )
        # spec_algorithm: used by the pre-fix branch only.
        runner.model_runner = SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_eagle=lambda: is_eagle),
            device_timer=None,
        )

        # Record the replay bs selected from capture_bs.
        runner.replayed_bs = None

        def replay_graph(shape_key, forward_batch):
            runner.replayed_bs = shape_key.size
            num_tokens = forward_batch.input_ids.shape[0]
            return SimpleNamespace(
                next_token_logits=torch.empty(num_tokens, 1),
                hidden_states=torch.empty(num_tokens, 1),
            )

        runner._replay_graph = replay_graph
        return runner, backend

    @staticmethod
    def _build_forward_batch(
        *,
        local_bs,
        captured_req_width,
        global_num_tokens_cpu,
        original_global_num_tokens_cpu,
    ):
        num_tokens = local_bs * captured_req_width
        return SimpleNamespace(
            out_cache_loc=torch.arange(num_tokens, dtype=torch.int64),
            batch_size=local_bs,
            input_ids=torch.arange(num_tokens, dtype=torch.int64),
            req_pool_indices=torch.arange(local_bs, dtype=torch.int64),
            seq_lens=torch.ones(local_bs, dtype=torch.int64),
            seq_lens_cpu=None,
            seq_lens_sum=None,
            positions=torch.arange(num_tokens, dtype=torch.int64),
            extend_seq_lens=None,
            extend_seq_lens_cpu=None,
            global_num_tokens_cpu=global_num_tokens_cpu,  # used by the pre-fix branch only
            original_global_num_tokens_cpu=original_global_num_tokens_cpu,
            spec_info=SimpleNamespace(
                num_tokens_per_req=captured_req_width,
                num_correct_drafts=None,
            ),
        )

    def test_replay_bucket_uses_raw_request_count(self):
        cases = (
            # EAGLE: the global max is 5 requests. After width scaling and
            # attn_tp alignment, there are 16 tokens per rank.
            # Pre-fix execute() derives 16 // 2, selecting bucket 8.
            # Fixed execute() derives max([5, 3]), selecting bucket 5.
            ("eagle_ceil_aligned", True, 2, 3, [16, 16], [5, 3]),
            # Standalone: the global max is 7 requests. No ceil alignment.
            # Pre-fix execute() derives max([35, 20]), exceeding max bucket 8.
            # Fixed execute() derives max([7, 4]), selecting bucket 7.
            ("standalone_scaled", False, 5, 4, [35, 20], [7, 4]),
        )
        for (
            name,
            is_eagle,
            captured_req_width,
            local_bs,
            global_num_tokens,
            original_global_num_tokens,
        ) in cases:
            with self.subTest(case=name):
                runner, backend = self._build_runner(
                    is_eagle=is_eagle, captured_req_width=captured_req_width
                )
                forward_batch = self._build_forward_batch(
                    local_bs=local_bs,
                    captured_req_width=captured_req_width,
                    global_num_tokens_cpu=global_num_tokens,
                    original_global_num_tokens_cpu=original_global_num_tokens,
                )
                expected_replay_bs = max(original_global_num_tokens)

                # Ensure graph-replay admission uses the raw request count.
                self.assertTrue(runner.can_run_graph(forward_batch))

                # Ensure execution sizes the replay bucket from the same raw request count.
                runner.execute(forward_batch)
                self.assertEqual(runner.replayed_bs, expected_replay_bs)
                self.assertEqual(backend.metadata_batch_sizes, [expected_replay_bs])


if __name__ == "__main__":
    unittest.main()
