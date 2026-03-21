"""
Unit tests for sglang.srt.constrained.grammar_manager.

Test Coverage:
- GrammarManager initialization, queue management, len, clear
- process_req_with_grammar: dispatch by constraint type (json, regex, ebnf,
  structural_tag), no-constraint requests, no-backend error, cache hits,
  cached invalid grammar abort
- abort_requests: single abort, abort all, future cancellation
- get_ready_grammar_requests: future completion, invalid grammar handling,
  timeout with max poll iterations, aborted request handling, queue cleanup

Usage:
    python -m pytest test_grammar_manager.py -v
"""

import unittest
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
    InvalidGrammarObject,
)
from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(2.0, "stage-a-cpu-only")


def _make_scheduler(grammar_backend_name="none", skip_tokenizer=False):
    """Create a mock scheduler with necessary attributes."""
    scheduler = MagicMock()
    scheduler.server_args.grammar_backend = grammar_backend_name
    scheduler.server_args.skip_tokenizer_init = skip_tokenizer
    scheduler.server_args.reasoning_parser = None
    scheduler.server_args.constrained_json_whitespace_pattern = None
    scheduler.server_args.constrained_json_disable_any_whitespace = False

    # Distributed group mocks
    scheduler.dp_tp_cpu_group = MagicMock()
    scheduler.dp_tp_group.world_size = 1
    scheduler.dp_tp_group.first_rank = 0
    scheduler.dp_tp_group.is_first_rank = True

    return scheduler


def _make_req(
    json_schema=None, regex=None, ebnf=None, structural_tag=None, rid="req-1"
):
    """Create a mock request with sampling params."""
    req = MagicMock()
    req.rid = rid
    req.sampling_params.json_schema = json_schema
    req.sampling_params.regex = regex
    req.sampling_params.ebnf = ebnf
    req.sampling_params.structural_tag = structural_tag
    req.require_reasoning = False
    req.grammar = None
    req.grammar_key = None
    req.grammar_wait_ct = 0
    req.finished.return_value = False
    return req


class TestGrammarManagerInit(unittest.TestCase):
    """Test GrammarManager initialization."""

    @patch("sglang.srt.constrained.grammar_manager.create_grammar_backend")
    def test_init_with_backend(self, mock_create):
        mock_create.return_value = MagicMock(spec=BaseGrammarBackend)
        scheduler = _make_scheduler("xgrammar")
        scheduler.server_args.skip_tokenizer_init = False

        mgr = GrammarManager(scheduler)
        self.assertIsNotNone(mgr.grammar_backend)
        self.assertEqual(len(mgr), 0)

    def test_init_skip_tokenizer(self):
        scheduler = _make_scheduler(skip_tokenizer=True)
        mgr = GrammarManager(scheduler)
        self.assertIsNone(mgr.grammar_backend)

    @patch("sglang.srt.constrained.grammar_manager.create_grammar_backend")
    def test_len_and_has_waiting(self, mock_create):
        mock_create.return_value = None
        scheduler = _make_scheduler()
        mgr = GrammarManager(scheduler)
        self.assertEqual(len(mgr), 0)
        self.assertFalse(mgr.has_waiting_grammars())

    @patch("sglang.srt.constrained.grammar_manager.create_grammar_backend")
    def test_clear_resets_backend(self, mock_create):
        mock_backend = MagicMock(spec=BaseGrammarBackend)
        mock_create.return_value = mock_backend
        scheduler = _make_scheduler()
        scheduler.server_args.skip_tokenizer_init = False

        mgr = GrammarManager(scheduler)
        mgr.clear()
        mock_backend.reset.assert_called_once()

    @patch("sglang.srt.constrained.grammar_manager.create_grammar_backend")
    def test_clear_no_backend(self, mock_create):
        mock_create.return_value = None
        scheduler = _make_scheduler()
        mgr = GrammarManager(scheduler)
        mgr.clear()  # Should not raise


class TestProcessReqWithGrammar(unittest.TestCase):
    """Test process_req_with_grammar dispatch and caching."""

    def _make_mgr(self):
        scheduler = _make_scheduler()
        scheduler.server_args.skip_tokenizer_init = True
        mgr = GrammarManager(scheduler)
        mgr.grammar_backend = MagicMock(spec=BaseGrammarBackend)
        return mgr

    def test_no_constraint_returns_false(self):
        mgr = self._make_mgr()
        req = _make_req()  # No constraints
        result = mgr.process_req_with_grammar(req)
        self.assertFalse(result)
        self.assertEqual(len(mgr.grammar_queue), 0)

    def test_json_schema_cache_miss(self):
        mgr = self._make_mgr()
        future = Future()
        mgr.grammar_backend.get_cached_or_future_value.return_value = (future, False)

        req = _make_req(json_schema='{"type": "object"}')
        result = mgr.process_req_with_grammar(req)

        self.assertTrue(result)
        self.assertEqual(len(mgr.grammar_queue), 1)
        self.assertEqual(req.grammar_key, ("json", '{"type": "object"}'))

    def test_regex_cache_miss(self):
        mgr = self._make_mgr()
        future = Future()
        mgr.grammar_backend.get_cached_or_future_value.return_value = (future, False)

        req = _make_req(regex="[a-z]+")
        result = mgr.process_req_with_grammar(req)

        self.assertTrue(result)
        self.assertEqual(req.grammar_key, ("regex", "[a-z]+"))

    def test_ebnf_cache_miss(self):
        mgr = self._make_mgr()
        future = Future()
        mgr.grammar_backend.get_cached_or_future_value.return_value = (future, False)

        req = _make_req(ebnf="root ::= 'hello'")
        result = mgr.process_req_with_grammar(req)

        self.assertTrue(result)
        self.assertEqual(req.grammar_key, ("ebnf", "root ::= 'hello'"))

    def test_structural_tag_cache_miss(self):
        mgr = self._make_mgr()
        future = Future()
        mgr.grammar_backend.get_cached_or_future_value.return_value = (future, False)

        req = _make_req(structural_tag='{"structures": [], "triggers": []}')
        result = mgr.process_req_with_grammar(req)

        self.assertTrue(result)
        self.assertEqual(
            req.grammar_key,
            ("structural_tag", '{"structures": [], "triggers": []}'),
        )

    def test_cache_hit_returns_false(self):
        """Cache hit should NOT add to grammar queue."""
        mgr = self._make_mgr()
        grammar_obj = MagicMock(spec=BaseGrammarObject)
        mgr.grammar_backend.get_cached_or_future_value.return_value = (
            grammar_obj,
            True,
        )

        req = _make_req(json_schema='{"type": "object"}')
        result = mgr.process_req_with_grammar(req)

        self.assertFalse(result)
        self.assertEqual(len(mgr.grammar_queue), 0)
        self.assertIs(req.grammar, grammar_obj)

    def test_cache_hit_invalid_grammar_aborts(self):
        """Cache hit with InvalidGrammarObject should abort the request."""
        mgr = self._make_mgr()
        invalid = InvalidGrammarObject("bad schema")
        mgr.grammar_backend.get_cached_or_future_value.return_value = (invalid, True)

        req = _make_req(json_schema="bad")
        result = mgr.process_req_with_grammar(req)

        self.assertFalse(result)
        req.set_finish_with_abort.assert_called_once()
        self.assertIn("bad schema", req.set_finish_with_abort.call_args[0][0])

    def test_no_backend_aborts(self):
        """No grammar backend should abort request."""
        scheduler = _make_scheduler()
        scheduler.server_args.skip_tokenizer_init = True
        mgr = GrammarManager(scheduler)
        mgr.grammar_backend = None

        req = _make_req(json_schema='{"type": "object"}')
        result = mgr.process_req_with_grammar(req)

        self.assertFalse(result)
        req.set_finish_with_abort.assert_called_once()
        self.assertIn("not supported", req.set_finish_with_abort.call_args[0][0])

    def test_json_takes_priority_over_other_constraints(self):
        """When json_schema is set, it should be used regardless of other fields."""
        mgr = self._make_mgr()
        future = Future()
        mgr.grammar_backend.get_cached_or_future_value.return_value = (future, False)

        req = _make_req(json_schema='{"type": "object"}', regex="[a-z]+")
        mgr.process_req_with_grammar(req)
        self.assertEqual(req.grammar_key, ("json", '{"type": "object"}'))

    def test_require_reasoning_forwarded_to_backend(self):
        """require_reasoning from the request should be passed to the backend."""
        mgr = self._make_mgr()
        grammar_obj = MagicMock(spec=BaseGrammarObject)
        mgr.grammar_backend.get_cached_or_future_value.return_value = (
            grammar_obj,
            True,
        )

        req = _make_req(json_schema="schema")
        req.require_reasoning = True
        mgr.process_req_with_grammar(req)

        mgr.grammar_backend.get_cached_or_future_value.assert_called_once_with(
            ("json", "schema"), True
        )

    def test_has_waiting_grammars_after_enqueue(self):
        mgr = self._make_mgr()
        future = Future()
        mgr.grammar_backend.get_cached_or_future_value.return_value = (future, False)

        self.assertFalse(mgr.has_waiting_grammars())
        req = _make_req(json_schema="schema")
        mgr.process_req_with_grammar(req)
        self.assertTrue(mgr.has_waiting_grammars())
        self.assertEqual(len(mgr), 1)


class TestAbortRequests(unittest.TestCase):
    """Test abort_requests handling."""

    def _make_mgr_with_queue(self):
        scheduler = _make_scheduler()
        scheduler.server_args.skip_tokenizer_init = True
        mgr = GrammarManager(scheduler)
        mgr.grammar_backend = MagicMock(spec=BaseGrammarBackend)
        return mgr

    def test_abort_by_rid_prefix(self):
        mgr = self._make_mgr_with_queue()
        req = _make_req(rid="req-123")
        future = MagicMock(spec=Future)
        req.grammar = future
        mgr.grammar_queue.append(req)

        abort_req = MagicMock()
        abort_req.abort_all = False
        abort_req.rid = "req-123"

        mgr.abort_requests(abort_req)
        future.cancel.assert_called_once()
        req.set_finish_with_abort.assert_called_once()

    def test_abort_non_matching_rid(self):
        mgr = self._make_mgr_with_queue()
        req = _make_req(rid="req-999")
        req.grammar = MagicMock(spec=Future)
        mgr.grammar_queue.append(req)

        abort_req = MagicMock()
        abort_req.abort_all = False
        abort_req.rid = "req-123"

        mgr.abort_requests(abort_req)
        req.set_finish_with_abort.assert_not_called()

    def test_abort_all(self):
        mgr = self._make_mgr_with_queue()
        reqs = []
        for i in range(3):
            req = _make_req(rid=f"req-{i}")
            req.grammar = MagicMock(spec=Future)
            mgr.grammar_queue.append(req)
            reqs.append(req)

        abort_req = MagicMock()
        abort_req.abort_all = True
        abort_req.rid = ""

        mgr.abort_requests(abort_req)
        for req in reqs:
            req.set_finish_with_abort.assert_called_once()

    def test_abort_empty_queue(self):
        """Aborting on an empty queue should not raise."""
        mgr = self._make_mgr_with_queue()
        abort_req = MagicMock()
        abort_req.abort_all = True
        abort_req.rid = ""
        mgr.abort_requests(abort_req)  # Should not raise

    def test_abort_prefix_match(self):
        """rid.startswith means prefix matching, not exact matching."""
        mgr = self._make_mgr_with_queue()
        req = _make_req(rid="req-123-suffix")
        req.grammar = MagicMock(spec=Future)
        mgr.grammar_queue.append(req)

        abort_req = MagicMock()
        abort_req.abort_all = False
        abort_req.rid = "req-123"

        mgr.abort_requests(abort_req)
        req.set_finish_with_abort.assert_called_once()


class TestGetReadyGrammarRequests(unittest.TestCase):
    """Test get_ready_grammar_requests polling and result handling."""

    def _make_mgr(self):
        scheduler = _make_scheduler()
        scheduler.server_args.skip_tokenizer_init = True
        mgr = GrammarManager(scheduler)
        mgr.grammar_backend = MagicMock(spec=BaseGrammarBackend)
        # Use very short poll interval for tests
        mgr.SGLANG_GRAMMAR_POLL_INTERVAL = 0.01
        mgr.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = 3
        return mgr

    def test_ready_future_returns_req(self):
        mgr = self._make_mgr()

        grammar_obj = MagicMock(spec=BaseGrammarObject)
        grammar_obj.copy.return_value = grammar_obj
        future = Future()
        future.set_result(grammar_obj)

        req = _make_req(json_schema="schema")
        req.grammar = future
        req.grammar_key = ("json", "schema")
        mgr.grammar_queue.append(req)

        result = mgr.get_ready_grammar_requests()
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], req)
        self.assertIs(req.grammar, grammar_obj)
        # Cache should be set
        mgr.grammar_backend.set_cache.assert_called_once()
        # Queue should be empty
        self.assertEqual(len(mgr.grammar_queue), 0)

    def test_invalid_grammar_aborts_req(self):
        mgr = self._make_mgr()

        invalid = InvalidGrammarObject("compile error")
        invalid_copy = InvalidGrammarObject("compile error")
        invalid.copy = MagicMock(return_value=invalid_copy)
        future = Future()
        future.set_result(invalid)

        req = _make_req(json_schema="bad")
        req.grammar = future
        req.grammar_key = ("json", "bad")
        mgr.grammar_queue.append(req)

        result = mgr.get_ready_grammar_requests()
        self.assertEqual(len(result), 1)
        req.set_finish_with_abort.assert_called_once()
        self.assertIn("compile error", req.set_finish_with_abort.call_args[0][0])

    def test_aborted_req_removed_from_queue(self):
        mgr = self._make_mgr()

        req = _make_req(json_schema="schema")
        req.finished.return_value = True  # Already aborted
        req.grammar = None
        mgr.grammar_queue.append(req)

        result = mgr.get_ready_grammar_requests()
        self.assertEqual(len(result), 1)
        self.assertEqual(len(mgr.grammar_queue), 0)

    def test_timeout_aborts_req(self):
        mgr = self._make_mgr()
        mgr.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = 1

        future = Future()  # Never completes
        req = _make_req(json_schema="slow")
        req.grammar = future
        req.grammar_key = ("json", "slow")
        req.grammar_wait_ct = 0
        mgr.grammar_queue.append(req)

        # First call: not ready, increments wait_ct to 1 (== max_poll)
        result = mgr.get_ready_grammar_requests()
        # Should timeout and abort
        self.assertEqual(len(result), 1)
        req.set_finish_with_abort.assert_called_once()
        self.assertIn("timed out", req.set_finish_with_abort.call_args[0][0])
        # Cache should store InvalidGrammarObject for timeout
        mgr.grammar_backend.set_cache.assert_called_once()
        cached_key, cached_val = mgr.grammar_backend.set_cache.call_args[0]
        self.assertEqual(cached_key, ("json", "slow"))
        self.assertIsInstance(cached_val, InvalidGrammarObject)

    def test_pending_future_stays_in_queue(self):
        """Futures that aren't done stay in the queue."""
        mgr = self._make_mgr()
        mgr.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = 100  # High to avoid timeout

        future = Future()  # Never completes
        req = _make_req(json_schema="pending")
        req.grammar = future
        req.grammar_key = ("json", "pending")
        req.grammar_wait_ct = 0
        mgr.grammar_queue.append(req)

        result = mgr.get_ready_grammar_requests()
        self.assertEqual(len(result), 0)
        self.assertEqual(len(mgr.grammar_queue), 1)
        self.assertEqual(req.grammar_wait_ct, 1)

    def test_mixed_ready_and_pending(self):
        mgr = self._make_mgr()
        mgr.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = 100

        # Ready request
        grammar_obj = MagicMock(spec=BaseGrammarObject)
        grammar_obj.copy.return_value = grammar_obj
        done_future = Future()
        done_future.set_result(grammar_obj)
        ready_req = _make_req(json_schema="ready", rid="r1")
        ready_req.grammar = done_future
        ready_req.grammar_key = ("json", "ready")

        # Pending request
        pending_future = Future()
        pending_req = _make_req(json_schema="pending", rid="r2")
        pending_req.grammar = pending_future
        pending_req.grammar_key = ("json", "pending")
        pending_req.grammar_wait_ct = 0

        mgr.grammar_queue = [ready_req, pending_req]

        result = mgr.get_ready_grammar_requests()
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], ready_req)
        self.assertEqual(len(mgr.grammar_queue), 1)
        self.assertIs(mgr.grammar_queue[0], pending_req)

    def test_empty_queue(self):
        """get_ready_grammar_requests on empty queue should return empty list."""
        mgr = self._make_mgr()
        result = mgr.get_ready_grammar_requests()
        self.assertEqual(len(result), 0)
        self.assertEqual(len(mgr.grammar_queue), 0)

    def test_progressive_timeout(self):
        """Request with partial wait_ct should timeout after remaining iterations."""
        mgr = self._make_mgr()
        mgr.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = 3

        future = Future()  # Never completes
        req = _make_req(json_schema="slow")
        req.grammar = future
        req.grammar_key = ("json", "slow")
        req.grammar_wait_ct = 2  # Already waited 2 iterations
        mgr.grammar_queue.append(req)

        # wait_ct increments to 3 (== max), should timeout
        result = mgr.get_ready_grammar_requests()
        self.assertEqual(len(result), 1)
        req.set_finish_with_abort.assert_called_once()
        self.assertIn("timed out", req.set_finish_with_abort.call_args[0][0])

    def test_future_exception_propagates(self):
        """A future that raised an exception should propagate on .result()."""
        mgr = self._make_mgr()

        future = Future()
        future.set_exception(RuntimeError("compilation crashed"))

        req = _make_req(json_schema="crash")
        req.grammar = future
        req.grammar_key = ("json", "crash")
        mgr.grammar_queue.append(req)

        with self.assertRaises(RuntimeError):
            mgr.get_ready_grammar_requests()

    @patch("sglang.srt.constrained.grammar_manager.torch.distributed.all_gather_object")
    def test_multi_rank_sync_intersects_ready_unions_failed(self, mock_all_gather):
        """With multiple ranks, ready = intersection, failed = union."""
        mgr = self._make_mgr()
        mgr.grammar_sync_size = 2  # Enable multi-rank path

        # Two requests: idx 0 ready on both ranks, idx 1 ready only on rank 0
        grammar_obj = MagicMock(spec=BaseGrammarObject)
        grammar_obj.copy.return_value = grammar_obj
        done_future = Future()
        done_future.set_result(grammar_obj)

        req0 = _make_req(json_schema="s0", rid="r0")
        req0.grammar = done_future
        req0.grammar_key = ("json", "s0")

        pending_future = Future()
        req1 = _make_req(json_schema="s1", rid="r1")
        req1.grammar = pending_future
        req1.grammar_key = ("json", "s1")
        req1.grammar_wait_ct = 0

        mgr.grammar_queue = [req0, req1]
        mgr.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = 100

        # Simulate all_gather: rank 0 has {0} ready, rank 1 has {0,1} ready
        def fake_all_gather(output_list, _obj, group=None):  # noqa: ARG001
            output_list[0] = ({0}, set())  # rank 0: only idx 0 ready
            output_list[1] = ({0, 1}, set())  # rank 1: both ready

        mock_all_gather.side_effect = fake_all_gather

        result = mgr.get_ready_grammar_requests()
        # Intersection of ready: {0} ∩ {0,1} = {0}
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], req0)
        # req1 stays in queue
        self.assertEqual(len(mgr.grammar_queue), 1)
        self.assertIs(mgr.grammar_queue[0], req1)

    @patch("sglang.srt.constrained.grammar_manager.torch.distributed.all_gather_object")
    def test_multi_rank_sync_unions_failed(self, mock_all_gather):
        """Failed requests from any rank should be unioned."""
        mgr = self._make_mgr()
        mgr.grammar_sync_size = 2
        mgr.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = 1

        pending_future = Future()  # Never completes
        req = _make_req(json_schema="slow", rid="r0")
        req.grammar = pending_future
        req.grammar_key = ("json", "slow")
        req.grammar_wait_ct = 0

        mgr.grammar_queue = [req]

        # Simulate: rank 0 has no ready and idx 0 failed, rank 1 has no ready/failed
        def fake_all_gather(output_list, _obj, group=None):  # noqa: ARG001
            output_list[0] = (set(), {0})  # rank 0: idx 0 timed out
            output_list[1] = (set(), set())  # rank 1: nothing

        mock_all_gather.side_effect = fake_all_gather

        result = mgr.get_ready_grammar_requests()
        # Union of failed: {} ∪ {0} = {0}
        self.assertEqual(len(result), 1)
        req.set_finish_with_abort.assert_called_once()
        self.assertIn("timed out", req.set_finish_with_abort.call_args[0][0])
        self.assertEqual(len(mgr.grammar_queue), 0)


if __name__ == "__main__":
    unittest.main()
