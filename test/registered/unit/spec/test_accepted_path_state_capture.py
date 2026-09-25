"""Per-token state captures must follow the KV through the accepted-path move.

A capture tape (routed experts, indexer topk) is keyed by KV slot:
``TopkCaptureOutput.finalize`` scatters row ``i`` into ``out_cache_loc[i]``, and
readback goes through ``req_to_token``. ``move_accept_tokens_to_target_kvcache``
moves each accepted tree node's KV to the front of its per-request block, so the
capture rows have to be compacted the same way or the committed slots keep the
routing of whichever tree node was originally allocated there.

These tests drive ``_compact_state_captures_to_front`` against the slot layout
``eagle_prepare_for_verify`` / ``NGRAMWorker`` build, so they need no GPU, no
model and no KV pool.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.spec_utils import _compact_state_captures_to_front
from sglang.srt.state_capturer.base import BaseTopkCapturer, TopkCaptureOutput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BS = 2
NUM_DRAFT_TOKENS = 8
NUM_LAYERS = 3
TOPK_SIZE = 4
NUM_SLOTS = 64
# The verify block starts partway into the pool so slot != row index.
SLOT_BASE = 16
SEQ_LEN = 5


def _node_capture(node: int) -> torch.Tensor:
    """A per-node capture row, distinct from the zero-initialised host cache."""
    return torch.full((NUM_LAYERS, TOPK_SIZE), node + 1, dtype=torch.int32)


class TestAcceptedPathStateCapture(CustomTestCase):
    def _verify_layout(self):
        """out_cache_loc and req_to_token as the verify prologue builds them.

        ``out_cache_loc`` is ``[bs * num_draft_tokens]`` with block ``b`` equal to
        ``req_to_token[b][seq_len : seq_len + num_draft_tokens]``, so committed
        position ``seq_len + k`` and block position ``k`` name the same slot.
        """
        out_cache_loc = torch.arange(
            SLOT_BASE, SLOT_BASE + BS * NUM_DRAFT_TOKENS, dtype=torch.int64
        )
        req_to_token = torch.zeros((BS, SEQ_LEN + NUM_DRAFT_TOKENS), dtype=torch.int64)
        req_to_token[:, SEQ_LEN:] = out_cache_loc.view(BS, NUM_DRAFT_TOKENS)
        return out_cache_loc, req_to_token

    def _finalized_tape(self, accept_index, *, compact: bool):
        out_cache_loc, req_to_token = self._verify_layout()
        capture = TopkCaptureOutput(
            out_cache_loc=out_cache_loc,
            topk=torch.stack(
                [_node_capture(node) for node in range(BS * NUM_DRAFT_TOKENS)]
            ),
            host_cache=SimpleNamespace(
                buffer=torch.zeros(
                    (NUM_SLOTS, NUM_LAYERS, TOPK_SIZE), dtype=torch.int32
                )
            ),
        )
        if compact:
            _compact_state_captures_to_front((capture, None), accept_index, BS)
        capture.finalize()
        return capture.host_cache.buffer, req_to_token

    def _committed_rows(self, tape, req_to_token, req, accept_len):
        """Read back exactly what ``BaseTopkCapturer.get_topk`` would read."""
        return tape[req_to_token[req][SEQ_LEN : SEQ_LEN + accept_len]]

    def test_committed_slots_hold_the_accepted_path(self):
        # Neither request accepts the front chain: req 0 takes nodes 0/3/5,
        # req 1 takes 0/1/6/7 (-1 pads an unaccepted tail).
        accept_index = torch.tensor([[0, 3, 5, -1], [8, 9, 14, 15]], dtype=torch.int32)
        accept_lens = [3, 4]
        tape, req_to_token = self._finalized_tape(accept_index, compact=True)

        for req, accept_len in enumerate(accept_lens):
            got = self._committed_rows(tape, req_to_token, req, accept_len)
            want = torch.stack(
                [_node_capture(int(node)) for node in accept_index[req, :accept_len]]
            )
            self.assertTrue(
                torch.equal(got, want),
                f"req {req}: committed slots hold {got[:, 0, 0].tolist()}, "
                f"expected the accepted path {want[:, 0, 0].tolist()}",
            )

    def test_without_compaction_committed_slots_hold_the_front_chain(self):
        """Pins the failure the compaction removes, so the test above cannot
        pass vacuously: the slots the KV move targets are the front of each
        block, whose capture rows belong to nodes 0..accept_len-1."""
        accept_index = torch.tensor([[0, 3, 5, -1], [8, 9, 14, 15]], dtype=torch.int32)
        accept_lens = [3, 4]
        tape, req_to_token = self._finalized_tape(accept_index, compact=False)

        for req, accept_len in enumerate(accept_lens):
            got = self._committed_rows(tape, req_to_token, req, accept_len)
            front_chain = torch.stack(
                [_node_capture(req * NUM_DRAFT_TOKENS + k) for k in range(accept_len)]
            )
            self.assertTrue(torch.equal(got, front_chain))
            accepted = torch.stack(
                [_node_capture(int(node)) for node in accept_index[req, :accept_len]]
            )
            self.assertFalse(torch.equal(got, accepted))

    def test_front_chain_acceptance_is_an_identity(self):
        """topk == 1 never runs the KV move because the accepted path already is
        the front chain; the compaction must agree and change nothing."""
        accept_index = torch.tensor([[0, 1, 2, 3], [8, 9, 10, 11]], dtype=torch.int32)
        compacted, _ = self._finalized_tape(accept_index, compact=True)
        untouched, _ = self._finalized_tape(accept_index, compact=False)
        self.assertTrue(torch.equal(compacted, untouched))

    def test_pipeline_stage_without_capture_compacts_accepted_kv(self):
        out_cache_loc, _ = self._verify_layout()
        accept_index = torch.tensor([[0, 3, 5, -1], [8, 9, 14, 15]])
        accept_lens = torch.tensor([3, 4])
        fwd_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            seq_lens_cpu=torch.tensor([SEQ_LEN, SEQ_LEN]),
        )
        scheduler = SimpleNamespace(token_to_kv_pool_allocator=object())
        with patch(
            "sglang.srt.speculative.spec_utils.move_accept_tokens_to_target_kvcache",
            autospec=True,
        ) as move:
            SchedulerPPMixin._pp_spec_compact_accept_kv(
                scheduler,
                SimpleNamespace(),
                fwd_batch,
                ["first", "second"],
                ["second"],
                out_cache_loc,
                PPProxyTensors(
                    {
                        "spec_accept_index": accept_index,
                        "spec_accept_lens": accept_lens,
                    }
                ),
            )
        move.assert_called_once()
        args, kwargs = move.call_args
        self.assertIs(args[0], fwd_batch)
        self.assertTrue(torch.equal(args[1], accept_index))
        self.assertTrue(torch.equal(args[2], accept_lens - 1))
        self.assertEqual(kwargs["state_captures"], ())

    def test_non_overlap_verify_defers_capture_until_accepted_path_move(self):
        out_cache_loc, req_to_token = self._verify_layout()
        capturer = object.__new__(BaseTopkCapturer)
        capturer.host_cache = SimpleNamespace(
            buffer=torch.zeros((NUM_SLOTS, NUM_LAYERS, TOPK_SIZE), dtype=torch.int32)
        )
        capturer._get_local_slice = Mock(
            return_value=torch.stack(
                [_node_capture(node) for node in range(BS * NUM_DRAFT_TOKENS)]
            )
        )
        runner = SimpleNamespace(
            forward_pass_id=0,
            msprobe_debugger=None,
            is_draft_worker=False,
            canary_manager=None,
            enable_elastic_ep=False,
            decode_cuda_graph_runner=None,
            eplb_manager=None,
            _forward_raw=Mock(return_value=SimpleNamespace(can_run_graph=False)),
        )
        forward_batch = SimpleNamespace(
            apply_deprecated_skip_attn_backend_init=Mock(),
            forward_mode=ForwardMode.TARGET_VERIFY,
            out_cache_loc=out_cache_loc,
            out_cache_loc_virtual=None,
        )
        with patch.multiple(
            "sglang.srt.model_executor.model_runner",
            get_schedule=Mock(
                return_value=SimpleNamespace(disable_overlap_schedule=True)
            ),
            get_parallel=Mock(return_value=SimpleNamespace(pp_size=1)),
            get_exec=Mock(
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(elastic_ep_backend=None)
                )
            ),
            get_global_experts_capturer=Mock(return_value=capturer),
            get_global_indexer_capturer=Mock(return_value=None),
            get_global_expert_distribution_recorder=Mock(
                return_value=SimpleNamespace(
                    with_forward_pass=Mock(side_effect=lambda *args: nullcontext({}))
                )
            ),
            build_step_span_name=Mock(return_value="verify"),
            profile_range=Mock(side_effect=lambda *args: nullcontext()),
            dumper=SimpleNamespace(may_enable=False),
        ):
            output = ModelRunner.forward(runner, forward_batch)
        self.assertIsNotNone(output.routed_experts_output)
        self.assertFalse(capturer.host_cache.buffer.any())
        accept_index = torch.tensor([[0, 3, 5, -1], [8, 9, 14, 15]])
        _compact_state_captures_to_front(
            (output.routed_experts_output,), accept_index, BS
        )
        output.routed_experts_output.finalize()
        for req, accept_len in enumerate([3, 4]):
            got = self._committed_rows(
                capturer.host_cache.buffer, req_to_token, req, accept_len
            )
            want = torch.stack(
                [_node_capture(int(node)) for node in accept_index[req, :accept_len]]
            )
            self.assertTrue(torch.equal(got, want))


if __name__ == "__main__":
    unittest.main()
