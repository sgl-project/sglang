"""Regression test for the EAGLE draft-extend shared-read fence."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative import eagle_draft_extend_cuda_graph_runner as runner_module
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingEvent:
    def __init__(self, trace):
        self.trace = trace

    def record(self):
        self.trace.append("read_done")


class _RecordingDevice:
    def __init__(self, trace):
        self.trace = trace

    def Event(self):
        return _RecordingEvent(self.trace)


class _RecordingAttentionBackend:
    def __init__(self, trace):
        self.trace = trace

    def init_forward_metadata_out_graph(self, forward_batch):
        self.trace.append("metadata")


class TestEagleDraftExtendSharedReadEvent(CustomTestCase):
    def test_event_is_published_after_graph_replay(self):
        trace = []
        runner = EAGLEDraftExtendCudaGraphRunner.__new__(
            EAGLEDraftExtendCudaGraphRunner
        )
        runner.device_module = _RecordingDevice(trace)
        runner.deepep_adapter = SimpleNamespace(replay=lambda: None)
        runner.buffers = SimpleNamespace(
            input_ids=torch.empty(1, dtype=torch.int64),
            seq_lens=torch.empty(1, dtype=torch.int64),
            out_cache_loc=torch.empty(1, dtype=torch.int64),
            positions=torch.empty(1, dtype=torch.int64),
            req_pool_indices=torch.empty(1, dtype=torch.int64),
            extend_seq_lens=torch.empty(1, dtype=torch.int32),
            num_correct_drafts=torch.empty(1, dtype=torch.int32),
            num_accept_tokens=torch.empty(1, dtype=torch.int32),
            select_index=torch.empty(1, dtype=torch.int64),
            hidden_states=None,
            global_num_tokens_gpu=None,
            global_num_tokens_for_logprob_gpu=None,
            seq_lens_cpu=torch.empty(1, dtype=torch.int64),
        )
        runner.require_mlp_tp_gather = False
        runner.require_gathered_buffer = False
        runner.captured_req_width = 1
        runner.capture_bs = [1]
        runner.seq_len_fill_value = 1
        runner.extend_seq_lens_cpu = [1]
        runner.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        runner.draft_extend_attn_backend = _RecordingAttentionBackend(trace)
        runner.model_runner = SimpleNamespace(
            device_timer=None,
            shared_read_done_event=None,
            spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
        )
        runner._make_graph_key = lambda bs: bs

        def replay_graph(shape_key, forward_batch):
            trace.append("replay")
            return LogitsProcessorOutput(
                next_token_logits=torch.zeros(1, 4),
                hidden_states=torch.zeros(1, 2),
            )

        runner._replay_graph = replay_graph
        forward_batch = SimpleNamespace(
            batch_size=1,
            input_ids=torch.tensor([1], dtype=torch.int64),
            seq_lens=torch.tensor([2], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([2], dtype=torch.int64),
            seq_lens_sum=2,
            out_cache_loc=torch.tensor([0], dtype=torch.int64),
            positions=torch.tensor([2], dtype=torch.int64),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            extend_seq_lens=None,
            extend_seq_lens_cpu=None,
            out_cache_loc_dsv4=None,
            spec_info=SimpleNamespace(
                num_correct_drafts=torch.tensor([0], dtype=torch.int32),
                num_accept_tokens=torch.tensor([1], dtype=torch.int32),
                hidden_states=None,
            ),
        )

        def copy_group(dst, src):
            for dst_tensor, src_tensor in zip(dst, src):
                dst_tensor.copy_(src_tensor)

        with patch.object(runner_module, "_grouped_foreach_copy_", copy_group):
            runner.execute(forward_batch, torch.tensor([0], dtype=torch.int64))

        self.assertEqual(trace, ["metadata", "replay", "read_done"])
        self.assertIsNotNone(runner.model_runner.shared_read_done_event)


if __name__ == "__main__":
    unittest.main()
