"""Unit tests for the session context forward primitives in srt/managers."""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import _resolve_query_attention
from sglang.srt.managers.scheduler import (
    _tail_coverage_intact,
    _token_positions_dims,
)
from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(**fields):
    req = MagicMock()
    for name, value in fields.items():
        setattr(req, name, value)
    return req


class TestResolveQueryAttention(CustomTestCase):
    def test_single_mode_and_empty_batch(self):
        self.assertEqual(
            _resolve_query_attention([_req(query_attention="bidirectional")]),
            "bidirectional",
        )
        self.assertIsNone(_resolve_query_attention([_req(query_attention=None)]))
        # DP idle batches build ScheduleBatch.init_new([]) routinely
        self.assertIsNone(_resolve_query_attention([]))

    def test_mixed_modes_raise(self):
        with self.assertRaises(ValueError):
            _resolve_query_attention(
                [_req(query_attention=None), _req(query_attention="bidirectional")]
            )


class TestBatchPurityHelpers(CustomTestCase):
    def test_token_positions_dims(self):
        self.assertIsNone(_token_positions_dims(_req(token_positions=None)))
        self.assertEqual(_token_positions_dims(_req(token_positions=[1, 2, 3])), 1)
        self.assertEqual(
            _token_positions_dims(_req(token_positions=[[9], [0], [0]])), 3
        )

    def test_tail_coverage(self):
        plain = _req(input_embeds=None, query_attention=None)
        self.assertTrue(_tail_coverage_intact(plain))

        covered = _req(
            input_embeds=[[0.0]] * 4,
            query_attention="bidirectional",
            origin_input_ids=list(range(10)),
            prefix_indices=list(range(6)),
        )
        self.assertTrue(_tail_coverage_intact(covered))

        evicted = _req(
            input_embeds=[[0.0]] * 4,
            query_attention="bidirectional",
            origin_input_ids=list(range(10)),
            prefix_indices=list(range(3)),
        )
        self.assertFalse(_tail_coverage_intact(evicted))


class TestSkipInsertChunkFrontier(CustomTestCase):
    def test_chunked_skip_insert_still_advances_prefix(self):
        req = _req(skip_radix_cache_insert=True, req_pool_idx=0)
        req.extend_range.end = 5
        tree_cache = MagicMock()
        tree_cache.req_to_token_pool.req_to_token = torch.arange(32).reshape(1, 32)

        maybe_cache_unfinished_req(req, tree_cache, chunked=True)
        tree_cache.cache_unfinished_req.assert_not_called()
        self.assertEqual(req.prefix_indices.tolist(), [0, 1, 2, 3, 4])

    def test_unchunked_skip_insert_is_a_no_op(self):
        req = _req(skip_radix_cache_insert=True)
        tree_cache = MagicMock()
        maybe_cache_unfinished_req(req, tree_cache)
        tree_cache.cache_unfinished_req.assert_not_called()

    def test_normal_requests_reach_the_tree_cache(self):
        req = _req(skip_radix_cache_insert=False)
        tree_cache = MagicMock()
        maybe_cache_unfinished_req(req, tree_cache, chunked=True)
        tree_cache.cache_unfinished_req.assert_called_once_with(req, chunked=True)


class TestGenerateReqInputEmbeds(CustomTestCase):
    def test_ids_win_over_embeds_without_query_attention(self):
        obj = GenerateReqInput(
            input_ids=[1, 2, 3],
            input_embeds=[[0.0], [0.0], [0.0]],
        )
        obj.normalize_batch_and_arguments()
        self.assertIsNone(obj.input_embeds)

    def test_context_forwards_keep_ids_and_embeds(self):
        # regression: the batch-size normalization used to clear input_embeds
        # whenever input_ids was present, silently degrading context forwards
        # to plain embedding lookups
        obj = GenerateReqInput(
            input_ids=[1, 2, 3],
            input_embeds=[[0.0], [0.0]],
            query_attention="bidirectional",
            token_positions=[1, 2],
        )
        obj.normalize_batch_and_arguments()
        self.assertEqual(obj.input_embeds, [[0.0], [0.0]])
        self.assertTrue(obj.is_single)


if __name__ == "__main__":
    unittest.main()


class TestPrefillHiddenStateSlicing(CustomTestCase):
    """Guards the batched-context-forward fix: rows are per-request extend
    widths, and the offset advances for every processed request."""

    def _req(self, *, returning, origin_len, transport=None):
        req = MagicMock()
        req.return_hidden_states = returning
        req.hidden_states = []
        req.hidden_states_transport = transport
        req.extend_batch_idx = 1
        req.origin_input_ids = list(range(origin_len))
        return req

    def _append(self, req, logits_output, offset, extend_len):
        from sglang.srt.managers.scheduler_components.batch_result_processor import (
            SchedulerBatchResultProcessor,
        )
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        return SchedulerBatchResultProcessor._append_prefill_hidden_states(
            None,
            req=req,
            logits_output=logits_output,
            hidden_state_offset=offset,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            extend_input_len=extend_len,
            store=True,
        )

    def test_two_requests_get_their_own_rows(self):
        from types import SimpleNamespace

        hidden = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        logits_output = SimpleNamespace(hidden_states=hidden)
        # origin lengths deliberately differ from extend widths (radix hits)
        req0 = self._req(returning=True, origin_len=9)
        req1 = self._req(returning=True, origin_len=7)

        offset = self._append(req0, logits_output, 0, 2)
        offset = self._append(req1, logits_output, offset, 3)
        self.assertEqual(offset, 5)
        self.assertEqual(req0.hidden_states, [hidden[0:2].tolist()])
        self.assertEqual(req1.hidden_states, [hidden[2:5].tolist()])

    def test_non_returning_request_still_advances_the_offset(self):
        from types import SimpleNamespace

        hidden = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        logits_output = SimpleNamespace(hidden_states=hidden)
        skipped = self._req(returning=False, origin_len=5)
        returning = self._req(returning=True, origin_len=5)

        offset = self._append(skipped, logits_output, 0, 3)
        self._append(returning, logits_output, offset, 1)
        self.assertEqual(skipped.hidden_states, [])
        self.assertEqual(returning.hidden_states, [hidden[3:4].tolist()])

    def test_chunked_prefill_merges_into_one_entry(self):
        from types import SimpleNamespace

        hidden = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        logits_output = SimpleNamespace(hidden_states=hidden)
        req = self._req(returning=True, origin_len=4)
        self._append(req, logits_output, 0, 2)
        req.extend_batch_idx = 2
        self._append(req, logits_output, 2, 2)
        self.assertEqual(len(req.hidden_states), 1)
        self.assertEqual(req.hidden_states[0], hidden.tolist())


class TestContextForwardEnvelope(CustomTestCase):
    """Each v1 envelope pin must reject on its own."""

    def _stub(self, **overrides):
        from types import SimpleNamespace

        server_args = SimpleNamespace(
            attention_backend="fa3",
            prefill_attention_backend=None,
            decode_attention_backend=None,
            disaggregation_mode="null",
            speculative_algorithm=None,
            page_size=1,
            chunked_prefill_size=8192,
            disable_radix_cache=False,
            tp_size=1,
            pp_size=1,
        )
        server_args.get_attention_backends = lambda: (
            server_args.prefill_attention_backend or server_args.attention_backend,
            server_args.decode_attention_backend or server_args.attention_backend,
        )
        for name, value in overrides.items():
            setattr(server_args, name, value)
        return SimpleNamespace(
            server_args=server_args,
            model_config=SimpleNamespace(hidden_size=4),
        )

    def _obj(self, **overrides):
        obj = GenerateReqInput(
            input_ids=[1, 2, 3],
            input_embeds=[[0.0] * 4],
            query_attention="bidirectional",
            token_positions=[2],
            sampling_params={"max_new_tokens": 0},
            return_hidden_states=True,
        )
        for name, value in overrides.items():
            setattr(obj, name, value)
        return obj

    def _validate(self, stub, obj):
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        TokenizerManager._validate_context_forward_fields(stub, obj)

    def test_base_envelope_passes(self):
        self._validate(self._stub(), self._obj())

    def test_each_pin_rejects(self):
        cases = [
            (self._stub(), self._obj(query_attention="diagonal")),
            (self._stub(), self._obj(input_embeds=None)),
            (self._stub(), self._obj(sampling_params={"max_new_tokens": 1})),
            (self._stub(attention_backend="triton"), self._obj()),
            (self._stub(prefill_attention_backend="triton"), self._obj()),
            (self._stub(disaggregation_mode="prefill"), self._obj()),
            (self._stub(speculative_algorithm="EAGLE"), self._obj()),
            (self._stub(page_size=16), self._obj()),
            (self._stub(tp_size=2), self._obj()),
            (self._stub(pp_size=2), self._obj()),
            (
                self._stub(chunked_prefill_size=2),
                self._obj(
                    input_ids=[1, 2, 3],
                    input_embeds=[[0.0] * 4] * 3,
                    token_positions=[0, 1, 2],
                ),
            ),
        ]
        for stub, obj in cases:
            with self.assertRaises(ValueError):
                self._validate(stub, obj)

    def test_integer_and_empty_embed_rows_report_a_width(self):
        """JSON clients emit `0`, not `0.0`. A float-only probe returned None
        for those rows, and the caller treats None as "skip the hidden-size
        check", which re-opened the in-kernel scheduler crash the pin exists
        to prevent."""
        from sglang.srt.managers.tokenizer_manager import _input_embeds_width

        self.assertEqual(_input_embeds_width([[0, 0, 0]]), 3)
        self.assertEqual(_input_embeds_width([[0] * 4]), 4)
        self.assertEqual(_input_embeds_width([[0, 1.0, 2.0, 3.0]]), 4)
        self.assertEqual(_input_embeds_width([[]]), 0)
        self.assertEqual(_input_embeds_width([[0.0] * 4]), 4)
        self.assertEqual(
            _input_embeds_width(
                {"transport": "shm", "shape": [4, 8], "dtype": "float32", "name": "x"}
            ),
            8,
        )
        self.assertIsNone(_input_embeds_width([["a", "b"]]))

    def test_split_prefill_backend_is_honored(self):
        # base backend non-fa is fine when the resolved prefill backend is fa3
        self._validate(
            self._stub(attention_backend="triton", prefill_attention_backend="fa3"),
            self._obj(),
        )

    def test_shm_transport_rejects_parallel_sampling(self):
        obj = self._obj(
            hidden_states_transport="shm",
            sampling_params={"max_new_tokens": 0, "n": 2},
        )
        with self.assertRaises(ValueError):
            self._validate(self._stub(), obj)
