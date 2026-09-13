from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
import torch

from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "declared, expected_order",
    [
        (SharedReadEnds.PRE_REPLAY, ["metadata", "record", "graph"]),
        (SharedReadEnds.IN_REPLAY, ["metadata", "graph", "record"]),
        (SharedReadEnds.POST_REPLAY, ["metadata", "graph", "record"]),
        (SharedReadEnds.UNKNOWN, ["metadata", "graph"]),
    ],
)
def test_draft_extend_fence_covers_declared_reads(declared, expected_order):
    runner = EAGLEDraftExtendCudaGraphRunner.__new__(EAGLEDraftExtendCudaGraphRunner)
    order = []
    event = SimpleNamespace(record=lambda: order.append("record"))
    # A preceding draft phase may have published an earlier event. UNKNOWN
    # must clear it so the scheduler takes its whole-forward fallback.
    stale_event = object()
    runner.model_runner = SimpleNamespace(
        device_timer=None, shared_read_done_event=stale_event
    )
    runner.device_module = SimpleNamespace(Event=lambda: event)
    runner.deepep_adapter = SimpleNamespace(replay=lambda: None)
    runner.forward_mode = ForwardMode.DRAFT_EXTEND_V2
    runner.require_mlp_tp_gather = False
    runner.require_gathered_buffer = False
    runner.captured_req_width = 6
    runner.capture_bs = [1]
    runner.seq_len_fill_value = 1
    runner.extend_seq_lens_cpu = [6]
    runner.buffers = SimpleNamespace(
        input_ids=torch.zeros(6, dtype=torch.int64),
        seq_lens=torch.zeros(1, dtype=torch.int64),
        out_cache_loc=torch.zeros(6, dtype=torch.int64),
        positions=torch.zeros(6, dtype=torch.int64),
        req_pool_indices=torch.zeros(1, dtype=torch.int64),
        extend_seq_lens=torch.zeros(1, dtype=torch.int32),
        num_correct_drafts=torch.zeros(1, dtype=torch.int32),
        num_accept_tokens=torch.zeros(1, dtype=torch.int32),
        select_index=torch.zeros(1, dtype=torch.int64),
        hidden_states=None,
        seq_lens_cpu=torch.zeros(1, dtype=torch.int64),
    )
    backend = create_autospec(AttentionBackend, instance=True)
    backend.shared_read_ends.return_value = declared
    backend.init_forward_metadata_out_graph.side_effect = lambda _: order.append(
        "metadata"
    )
    runner.draft_extend_attn_backend = backend

    def replay(shape_key, forward_batch):
        order.append("graph")
        return LogitsProcessorOutput(
            next_token_logits=torch.zeros(1, 2),
            hidden_states=torch.ones(1, 2),
        )

    runner._replay_graph = replay
    batch = SimpleNamespace(
        batch_size=1,
        input_ids=torch.arange(6),
        seq_lens=torch.tensor([12]),
        out_cache_loc=torch.arange(6),
        positions=torch.arange(6),
        req_pool_indices=torch.tensor([1]),
        extend_seq_lens=torch.tensor([6], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([12]),
        extend_seq_lens_cpu=[6],
        seq_lens_sum=12,
        spec_info=SimpleNamespace(
            num_correct_drafts=torch.tensor([5], dtype=torch.int32),
            num_accept_tokens=torch.tensor([6], dtype=torch.int32),
            hidden_states=None,
        ),
    )

    runner.execute(batch, torch.tensor([5]))

    assert order == expected_order
    backend.shared_read_ends.assert_called_once_with(ForwardMode.DRAFT_EXTEND_V2)
    assert runner.model_runner.shared_read_done_event is (
        None if declared is SharedReadEnds.UNKNOWN else event
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
