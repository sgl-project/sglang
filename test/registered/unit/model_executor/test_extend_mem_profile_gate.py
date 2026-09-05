"""Which eager forwards the extend memory profiler records.

Covers the mode/token predicate (``extend_mem_profile_tokens``) against
ForwardBatch-like objects shaped the way ``prepare_mlp_sync_batch`` leaves
them, and the ModelRunner eager call site: ``record`` must be entered only for
a genuine prefill extend with enough real tokens, and nothing profiler-related
may run when the profiler is disabled.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import (
    ModelRunner,
    extend_mem_profile_tokens,
)
from sglang.srt.utils import extend_mem_profile
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _batch(
    mode,
    *,
    original_mode=None,
    original_num_tokens=None,
    non_padded=None,
    input_ids=0,
):
    """ForwardBatch-like object with the fields the predicate reads."""
    return SimpleNamespace(
        forward_mode=mode,
        _original_forward_mode=original_mode,
        _original_num_tokens=original_num_tokens,
        num_token_non_padded_cpu=non_padded,
        input_ids=torch.zeros(input_ids, dtype=torch.int64),
    )


def plain_extend(tokens=4096):
    # No DP padding: the count is the batch's own token count.
    return _batch(ForwardMode.EXTEND, non_padded=tokens, input_ids=tokens)


def padded_extend(real=3000, padded=4096):
    # A real extend rank under DP MAX_LEN: input_ids padded to the peer's
    # length, mode unchanged, pre-padding count recorded.
    return _batch(
        ForwardMode.EXTEND,
        original_num_tokens=real,
        non_padded=real,
        input_ids=padded,
    )


def idle_hybrid_converted(peer_tokens=4096):
    # Idle hybrid-SSM rank converted to EXTEND with a fabricated request:
    # forward_batch_info overwrites num_token_non_padded_cpu with the peer's
    # padded length and pads input_ids to it; positions had 0 rows.
    return _batch(
        ForwardMode.EXTEND,
        original_mode=ForwardMode.IDLE,
        original_num_tokens=0,
        non_padded=peer_tokens,
        input_ids=peer_tokens,
    )


def decode_converted(bs=8, peer_tokens=4096):
    # Decode rows padded to 1-token extends next to a prefill peer.
    return _batch(
        ForwardMode.EXTEND,
        original_mode=ForwardMode.DECODE,
        original_num_tokens=bs,
        non_padded=bs,
        input_ids=peer_tokens,
    )


def mixed(tokens=4096):
    return _batch(ForwardMode.MIXED, non_padded=tokens, input_ids=tokens)


def target_verify(reqs=256, draft=4):
    return _batch(
        ForwardMode.TARGET_VERIFY, non_padded=reqs * draft, input_ids=reqs * draft
    )


def decode(bs=256):
    return _batch(ForwardMode.DECODE, non_padded=bs, input_ids=bs)


def idle():
    return _batch(ForwardMode.IDLE, non_padded=0, input_ids=0)


class ExtendMemProfileTokensTest(unittest.TestCase):
    def test_plain_extend_reports_its_token_count(self):
        self.assertEqual(extend_mem_profile_tokens(plain_extend(4096)), 4096)

    def test_padded_extend_reports_the_pre_padding_count(self):
        self.assertEqual(
            extend_mem_profile_tokens(padded_extend(real=3000, padded=4096)), 3000
        )

    def test_idle_hybrid_rank_converted_to_extend_is_excluded(self):
        # The review's case: idle Mamba/KDA rank paired with a 4096-token
        # prefill must not be profiled as a 4096-token extend.
        self.assertEqual(extend_mem_profile_tokens(idle_hybrid_converted(4096)), 0)

    def test_decode_rows_converted_to_extend_are_excluded(self):
        self.assertEqual(extend_mem_profile_tokens(decode_converted()), 0)

    def test_mixed_is_excluded(self):
        self.assertEqual(extend_mem_profile_tokens(mixed(4096)), 0)

    def test_target_verify_is_excluded(self):
        self.assertEqual(extend_mem_profile_tokens(target_verify(256, 4)), 0)

    def test_decode_and_idle_are_excluded(self):
        self.assertEqual(extend_mem_profile_tokens(decode()), 0)
        self.assertEqual(extend_mem_profile_tokens(idle()), 0)

    def test_falls_back_to_input_ids_without_any_count(self):
        batch = _batch(ForwardMode.EXTEND, input_ids=2048)
        self.assertEqual(extend_mem_profile_tokens(batch), 2048)


class _RecordSpy:
    """Stands in for extend_mem_profile.record: remembers each call and
    whether the returned scope was entered."""

    def __init__(self):
        self.calls = []
        self.entered = []

    def __call__(self, num_tokens, min_tokens=extend_mem_profile.DEFAULT_MIN_TOKENS):
        self.calls.append(num_tokens)
        if num_tokens < min_tokens:
            return extend_mem_profile._NOOP_SCOPE
        spy = self

        class _Scope:
            def __enter__(self_inner):
                spy.entered.append(num_tokens)

            def __exit__(self_inner, *exc):
                return False

        return _Scope()


def _runner():
    runner = ModelRunner.__new__(ModelRunner)
    runner.eager_runner = SimpleNamespace(
        execute=mock.Mock(side_effect=lambda fb, pp_proxy_tensors=None: ("out", fb))
    )
    return runner


class ModelRunnerEagerGateTest(unittest.TestCase):
    def setUp(self):
        self._was_enabled = extend_mem_profile.ENABLED

    def tearDown(self):
        extend_mem_profile._bind(self._was_enabled)

    def test_record_is_entered_only_for_the_qualifying_extend(self):
        extend_mem_profile._bind(True)
        runner = _runner()
        spy = _RecordSpy()
        batches = [
            plain_extend(4096),
            idle_hybrid_converted(4096),
            decode_converted(),
            mixed(4096),
            target_verify(256, 4),
            decode(256),
            idle(),
            plain_extend(512),  # genuine extend, below min_tokens
        ]
        with mock.patch.object(extend_mem_profile, "record", spy):
            for batch in batches:
                out = runner._execute_eager(batch, None)
                self.assertEqual(out, ("out", batch))
        self.assertEqual(runner.eager_runner.execute.call_count, len(batches))
        self.assertEqual(spy.calls, [4096, 0, 0, 0, 0, 0, 0, 512])
        self.assertEqual(spy.entered, [4096])

    def test_disabled_runner_never_touches_the_profiler(self):
        extend_mem_profile._bind(False)
        runner = _runner()
        batch = plain_extend(4096)
        with mock.patch.object(extend_mem_profile, "record") as record, mock.patch(
            "sglang.srt.model_executor.model_runner.extend_mem_profile_tokens"
        ) as tokens:
            out = runner._execute_eager(batch, None)
        self.assertEqual(out, ("out", batch))
        runner.eager_runner.execute.assert_called_once_with(
            batch, pp_proxy_tensors=None
        )
        self.assertFalse(record.called)
        self.assertFalse(tokens.called)

    def test_exception_from_execute_propagates_through_the_scope(self):
        extend_mem_profile._bind(True)
        runner = _runner()
        runner.eager_runner.execute.side_effect = RuntimeError("CUDA out of memory")
        spy = _RecordSpy()
        with mock.patch.object(extend_mem_profile, "record", spy):
            with self.assertRaises(RuntimeError):
                runner._execute_eager(plain_extend(4096), None)
        self.assertEqual(spy.entered, [4096])


if __name__ == "__main__":
    unittest.main()
