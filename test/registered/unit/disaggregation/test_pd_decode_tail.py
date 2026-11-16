"""Decode-tail PD mode (RFC #50547): the transfer truncation and the tail
extend admission surface.

With `--disaggregation-decode-tail-tokens W`, the prefill transfers KV for
`[0, N - W)` of an N-token prompt and drops the handoff token at commit; the
decode worker admits the request through a real extend over the last W prompt
tokens (`prepare_for_tail_extend`) and samples the first token itself, so
grammar, penalties, logprobs and a speculative drafter live entirely on
decode. These tests pin the pure pieces of that contract; the end-to-end
paired-worker behavior is covered by the PD disaggregation E2E suite.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation import decode as decode_module
from sglang.srt.disaggregation import prefill as prefill_module
import sglang.srt.disaggregation.decode_schedule_batch_mixin as mixin_module
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, pre_alloc_fill_len


class TailTransferInputLenTest(unittest.TestCase):
    def test_cap_stops_at_n_minus_w(self):
        self.assertEqual(prefill_module.tail_transfer_input_len(1000, 10), 990)

    def test_short_prompt_falls_back_to_full_transfer(self):
        # N <= W + 1 cannot leave both a non-empty transfer and a non-empty
        # tail: the request falls back to the full transfer and the decode
        # takes the prebuilt path (the handoff token stays authoritative).
        self.assertEqual(prefill_module.tail_transfer_input_len(5, 10), 5)
        self.assertEqual(prefill_module.tail_transfer_input_len(1, 10), 1)
        self.assertEqual(prefill_module.tail_transfer_input_len(2, 10), 2)
        self.assertEqual(prefill_module.tail_transfer_input_len(11, 10), 11)

    def test_cap_is_identity_without_tail(self):
        self.assertEqual(prefill_module.tail_transfer_input_len(1000, 0), 1000)


class IsTailTransferTest(unittest.TestCase):
    """The commit gate: tail mode on, and the transfer is not fake warmup."""

    def _assert_gate(self, tail_tokens, req, expected):
        gate = decode_module._is_tail_transfer
        with patch.object(
            decode_module,
            "get_disagg",
            return_value=SimpleNamespace(disaggregation_decode_tail_tokens=tail_tokens),
        ):
            # The patch must stay active across the call: the gate reads the
            # runtime config through get_disagg on every invocation.
            self.assertEqual(gate(req), expected)

    def _req(self, fake=False, with_handoff=False):
        req = SimpleNamespace(
            rid="chat-1",
            output_ids=[42] if with_handoff else [],
        )
        if fake:
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        else:
            req.bootstrap_host = "10.0.0.5"
        return req

    def test_real_transfer_in_tail_mode(self):
        self._assert_gate(8, self._req(), True)

    def test_fake_warmup_transfers_keep_their_handoff_id(self):
        self._assert_gate(8, self._req(fake=True), False)

    def test_gate_closed_without_tail_mode(self):
        self._assert_gate(0, self._req(), False)

    def test_handoff_token_marks_the_prebuilt_fallback(self):
        # N <= W + 1: the prefill handoff token survives -> prebuilt path.
        self._assert_gate(8, self._req(with_handoff=True), False)


class PrepareForTailExtendTest(unittest.TestCase):
    """The batch construction: the transferred rows are the attention prefix,
    the tail W tokens are the extend, and nothing is re-allocated."""

    PROMPT_LEN = 64
    COMMITTED = 56

    def _make_req(self, index):
        return SimpleNamespace(
            rid=f"r{index}",
            kv=SimpleNamespace(req_pool_idx=index, kv_committed_len=self.COMMITTED),
            extend_end=self.PROMPT_LEN,
            fill_ids=list(range(self.PROMPT_LEN)),
            origin_input_ids=list(range(self.PROMPT_LEN)),
            output_ids=[],
            is_retracted=False,
            multimodal_inputs=None,
        )

    def _prepare(self, reqs):
        batch = SimpleNamespace(
            reqs=reqs,
            device="cpu",
            return_logprob=False,
            model_config=SimpleNamespace(vocab_size=128),
        )
        req_to_token = torch.zeros(len(reqs), 256, dtype=torch.int64)
        for i, req in enumerate(reqs):
            # The receiver wrote the transferred KV's row indices for
            # [0, committed) and preallocated the tail with the prompt grid.
            req_to_token[i, : req.extend_end] = torch.arange(req.extend_end) + 100 * (
                i + 1
            )
        batch.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)

        with patch.object(
            mixin_module.SamplingBatchInfo,
            "from_schedule_batch",
            staticmethod(lambda batch, vocab_size: SimpleNamespace()),
        ):
            ScheduleBatchDisaggregationDecodeMixin.prepare_for_tail_extend(batch)
        return batch

    def test_extend_covers_only_the_tail(self):
        req = self._make_req(0)
        # pre_alloc_fill_len with empty output_ids == the full prompt length.
        self.assertEqual(pre_alloc_fill_len(req), self.PROMPT_LEN)

        batch = self._prepare([req])
        self.assertEqual(batch.extend_num_tokens, 8)
        # The attention prefix is the transferred range, not the tree view.
        self.assertEqual(batch.prefix_lens, [self.COMMITTED])
        self.assertEqual(batch.extend_lens, [8])
        # input_ids are the tail prompt tokens, not the handoff token.
        self.assertEqual(batch.input_ids.tolist(), list(range(56, 64)))
        # The rows come from the preallocated grid — no fresh allocation.
        self.assertEqual(
            batch.out_cache_loc.tolist(),
            (torch.arange(56, 64) + 100).tolist(),
        )
        self.assertEqual(batch.seq_lens_cpu.tolist(), [self.PROMPT_LEN])

    def test_two_requests_flatten_in_order(self):
        reqs = [self._make_req(0), self._make_req(1)]
        batch = self._prepare(reqs)
        self.assertEqual(batch.extend_num_tokens, 16)
        self.assertEqual(batch.prefix_lens, [self.COMMITTED, self.COMMITTED])
        self.assertEqual(
            batch.input_ids.tolist(), list(range(56, 64)) + list(range(56, 64))
        )
        self.assertEqual(
            batch.out_cache_loc.tolist(),
            (torch.arange(56, 64) + 100).tolist()
            + (torch.arange(56, 64) + 200).tolist(),
        )
        self.assertEqual(batch.seq_lens_cpu.tolist(), [64, 64])


if __name__ == "__main__":
    unittest.main()
