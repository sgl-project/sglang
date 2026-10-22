"""Greedy next-token ids from a vocab-parallel argmax - CPU only.

A forward whose spec input sets ``greedy_tokens_only`` gets
``LogitsProcessorOutput.next_token_ids`` (the argmax of the logits it would
otherwise return) and ``next_token_logits=None``: each tensor-parallel rank
reduces its part of the vocabulary to (max, index) and the ranks gather only
those pairs.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers import logits_processor as logits_mod
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _VerifyInput(SpecInput):
    def __init__(self, greedy: bool):
        super().__init__(SpecInputType.EAGLE_VERIFY)
        self.greedy_tokens_only = greedy


def _shard(start: int, valid: int):
    return SimpleNamespace(org_vocab_start_index=start, num_org_elements=valid)


def _processor(shard_logits, shard, *, tp_gather, gatherer=None, **flags):
    """A LogitsProcessor stand-in whose LM head returns ``shard_logits``."""
    processor = SimpleNamespace(
        do_tensor_parallel_all_gather=tp_gather,
        do_tensor_parallel_all_gather_dp_attn=flags.get("dp_attn", False),
        use_attn_tp_group=flags.get("attn_tp", False),
        final_logit_softcapping=flags.get("softcap", None),
        logit_scale=flags.get("logit_scale", None),
        _greedy_tokens_gatherer=gatherer,
        _compute_lm_head=lambda hidden_states, lm_head: shard_logits.clone(),
    )
    processor._get_greedy_tokens = lambda hs, head: LogitsProcessor._get_greedy_tokens(
        processor, hs, head
    )
    return processor


@contextlib.contextmanager
def _no_forward_context():
    forward = mock.MagicMock()
    forward.scoped.side_effect = lambda **kw: contextlib.nullcontext()
    with mock.patch.object(logits_mod, "get_forward", return_value=forward):
        yield


def _tp_greedy_ids(full_logits, tp, padding=3):
    """Every rank's _get_greedy_tokens over the shards of ``full_logits``,
    with the gather done by hand (padded shard columns hold large values)."""
    rows, vocab = full_logits.shape
    per = -(-vocab // tp)
    shards = []
    for rank in range(tp):
        start = rank * per
        valid = max(0, min(vocab, start + per) - start)
        cols = torch.full((rows, per + padding), 1e9)
        cols[:, :valid] = full_logits[:, start : start + valid]
        shards.append((cols, _shard(start, valid)))
    sent = []

    def record(x):
        sent.append(x.clone())
        return x

    for cols, shard in shards:  # what each rank contributes to the gather
        _processor(cols, shard, tp_gather=True, gatherer=record)._get_greedy_tokens(
            None, SimpleNamespace(shard_indices=shard)
        )
    gathered = torch.cat(sent, dim=-1)
    outs = []
    for cols, shard in shards:
        processor = _processor(cols, shard, tp_gather=True, gatherer=lambda x: gathered)
        outs.append(
            processor._get_greedy_tokens(None, SimpleNamespace(shard_indices=shard))
        )
    return outs


class TestGreedyTokenIds(CustomTestCase):
    def test_matches_argmax_of_gathered_logits(self):
        torch.manual_seed(0)
        with _no_forward_context():
            for tp in (1, 2, 4):
                logits = torch.randn(5, 37)
                for ids in _tp_greedy_ids(logits, tp):
                    self.assertEqual(ids.dtype, torch.int64)
                    torch.testing.assert_close(ids, torch.argmax(logits, dim=-1))

    def test_ties_across_shards_pick_the_lowest_id(self):
        logits = torch.zeros(2, 12)
        logits[0, [2, 9]] = 5.0  # shard 0 and shard 1 tie
        logits[1, [7, 11]] = 5.0  # two ties inside the last shard
        with _no_forward_context():
            for ids in _tp_greedy_ids(logits, tp=2):
                self.assertEqual(ids.tolist(), [2, 7])

    def test_logit_scale_is_applied(self):
        logits = torch.tensor([[1.0, 3.0, 2.0]])
        processor = _processor(logits, _shard(0, 3), tp_gather=False, logit_scale=-1.0)
        with _no_forward_context():
            ids = processor._get_greedy_tokens(
                None, SimpleNamespace(shard_indices=_shard(0, 3))
            )
        self.assertEqual(ids.tolist(), [0])

    def test_declines_logits_that_are_not_plain_shards(self):
        head = SimpleNamespace(shard_indices=_shard(0, 3))
        logits = torch.zeros(1, 3)
        for flags in ({"dp_attn": True}, {"attn_tp": True}, {"softcap": 30.0}):
            processor = _processor(logits, _shard(0, 3), tp_gather=True, **flags)
            self.assertIsNone(processor._get_greedy_tokens(None, head))
        processor = _processor(logits, _shard(0, 3), tp_gather=False)
        self.assertIsNone(processor._get_greedy_tokens(None, SimpleNamespace()))


def _forward(processor, metadata, hidden_states, lm_head):
    with mock.patch.object(
        logits_mod.layernorm_sp,
        "maybe_exit_gather",
        side_effect=lambda hidden_states, hidden_states_before_norm, **kw: (
            hidden_states,
            hidden_states_before_norm,
        ),
    ):
        return LogitsProcessor.forward(
            processor, None, hidden_states, lm_head, metadata
        )


class TestForwardReturnsTokenIds(CustomTestCase):
    def _processor(self, ids):
        processor = SimpleNamespace()
        processor._get_pruned_states = lambda hs, before_norm, aux, metadata: (
            hs,
            None,
            None,
            None,
            None,
            [],
        )
        processor._get_hidden_states_to_store = lambda *args: None
        processor._get_logits = mock.MagicMock(return_value=torch.zeros(3, 5))
        processor._get_greedy_tokens = mock.MagicMock(return_value=ids)
        return processor

    def test_greedy_input_gets_ids_and_no_logits(self):
        ids = torch.tensor([4, 0, 2])
        processor = self._processor(ids)
        metadata = LogitsMetadata(
            forward_mode=ForwardMode.TARGET_VERIFY, greedy_tokens_only=True
        )
        out = _forward(processor, metadata, torch.randn(3, 4), object())
        processor._get_logits.assert_not_called()
        self.assertIs(out.next_token_ids, ids)
        self.assertIsNone(out.next_token_logits)

    def test_other_forwards_and_declined_ids_get_logits(self):
        for greedy, ids in ((False, torch.tensor([1, 1, 1])), (True, None)):
            processor = self._processor(ids)
            metadata = LogitsMetadata(
                forward_mode=ForwardMode.TARGET_VERIFY, greedy_tokens_only=greedy
            )
            out = _forward(processor, metadata, torch.randn(3, 4), object())
            processor._get_logits.assert_called_once()
            self.assertIsNone(out.next_token_ids)
            self.assertIs(out.next_token_logits, processor._get_logits.return_value)
            if not greedy:
                processor._get_greedy_tokens.assert_not_called()

    def test_metadata_reads_the_spec_input_flag(self):
        base = dict(
            _original_forward_mode=None,
            forward_mode=ForwardMode.TARGET_VERIFY,
            return_logprob=False,
            capture_hidden_mode=None,
            next_token_logits_buffer=None,
            extend_seq_lens=None,
            extend_seq_lens_cpu=None,
            extend_logprob_start_lens_cpu=None,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_input_logprob_token_ids_gpu=None,
            is_prefill_only=False,
            attn_tp_sequence_sharded=False,
            global_num_tokens_gpu=None,
            dp_local_start_pos=None,
            dp_local_num_tokens=None,
            global_dp_buffer_len=None,
            global_num_tokens_for_logprob_cpu=None,
            global_num_tokens_for_logprob_gpu=None,
            mm_input_embeds=None,
        )
        for spec_info, expected in (
            (_VerifyInput(True), True),
            (_VerifyInput(False), False),
            (None, False),
        ):
            batch = SimpleNamespace(**base, spec_info=spec_info)
            metadata = LogitsMetadata.from_forward_batch(batch)
            self.assertIs(metadata.greedy_tokens_only, expected)


if __name__ == "__main__":
    unittest.main()
