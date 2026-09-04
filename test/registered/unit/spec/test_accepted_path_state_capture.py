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
from types import SimpleNamespace

import torch

from sglang.srt.speculative.spec_utils import _compact_state_captures_to_front
from sglang.srt.state_capturer.base import TopkCaptureOutput
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


if __name__ == "__main__":
    unittest.main()
