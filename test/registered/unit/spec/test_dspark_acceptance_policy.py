"""Unit tests for the DSpark pluggable acceptance-policy seam.

Covers the DSparkAcceptancePolicy base-class contract, the worker hook
defaults and lifecycle forwarding, and TargetVerifyExecutor's policy
routing (native fallback, folded-accept rejection, argument plumbing, and
the (correct_len, bonus, cap_trim_lens) contract passthrough).
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.speculative.dspark_components.acceptance_policy import (
    DSparkAcceptancePolicy,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    TargetVerifyExecutor,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _SpyPolicy(DSparkAcceptancePolicy):
    """Records lifecycle calls; returns a fixed or native-delegated result."""

    def __init__(self, result=None, delegate_native=False):
        self.accept_calls = []
        self.bound_batches = []
        self.finished = []
        self._result = result
        self._delegate_native = delegate_native

    def bind_batch(self, batch):
        self.bound_batches.append(batch)

    def note_request_finished(self, *, rid, natural_stop):
        self.finished.append((rid, natural_stop))

    def accept(self, **kwargs):
        self.accept_calls.append(kwargs)
        if self._delegate_native:
            return kwargs["native_accept"]()
        return self._result


class _NoopTpSync:
    def sync(self, site, tensor):
        pass


class TestPolicyBaseClass(CustomTestCase):
    def test_lifecycle_hooks_are_default_noops(self):
        policy = DSparkAcceptancePolicy()
        policy.bind_batch(object())
        policy.note_request_finished(rid="r0", natural_stop=True)

    def test_accept_is_abstract(self):
        policy = DSparkAcceptancePolicy()
        with self.assertRaises(NotImplementedError):
            policy.accept(
                candidates=torch.zeros(1, 2, dtype=torch.int64),
                target_logits=None,
                cutoff_verify_lens=None,
                req_pool_indices=torch.zeros(1, dtype=torch.int64),
                all_greedy=True,
                native_accept=lambda: None,
            )


class TestWorkerHook(CustomTestCase):
    def test_default_hook_returns_none(self):
        # Unbound call: the default hook must not touch worker state, so a
        # bare sentinel works as `self` without constructing the worker.
        self.assertIsNone(DSparkWorkerV2._build_acceptance_policy(object()))

    def test_note_request_finished_forwards_to_policy(self):
        # __new__ skips the heavy __init__; the method only needs the two
        # attributes it touches.
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        policy = _SpyPolicy()

        class _Observers:
            def note_request_finished(self, *, rid, natural_stop):
                pass

        worker._observers = _Observers()
        worker._acceptance_policy = policy
        worker.note_request_finished(rid="r1", natural_stop=False)
        self.assertEqual(policy.finished, [("r1", False)])


class TestExecutorSeam(CustomTestCase):
    """TargetVerifyExecutor routes acceptance through the policy or native."""

    def _executor(self, policy):
        return TargetVerifyExecutor(
            target_worker=None,
            gamma=2,
            verify_num_draft_tokens=3,
            model_runner=None,
            kv_injector=None,
            tp_sync=_NoopTpSync(),
            verify_epilogue=None,
            simulate_acc_len=0.0,
            acceptance_policy=policy,
        )

    def _kwargs(self, **overrides):
        kwargs = dict(
            folded_accept=False,
            bs=2,
            verify_ids_2d=torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.int64),
            target_logits=torch.randn(6, 8),
            draft_block=None,
            sampling_info=None,
            draft_input=None,
            layout=None,
            prefix_lens=torch.tensor([10, 20], dtype=torch.int64),
            draft_tokens=torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
            req_pool_indices=torch.tensor([5, 6], dtype=torch.int64),
        )
        kwargs.update(overrides)
        return kwargs

    def test_folded_accept_rejects_policy(self):
        executor = self._executor(_SpyPolicy())
        with self.assertRaisesRegex(RuntimeError, "folded"):
            executor.accept_and_finalize(**self._kwargs(folded_accept=True))

    def test_folded_accept_without_policy_reads_epilogue(self):
        class _Epilogue:
            def read_accept(self, bs):
                return ("accept", bs)

        executor = self._executor(None)
        executor.verify_epilogue = _Epilogue()
        out = executor.accept_and_finalize(**self._kwargs(folded_accept=True))
        self.assertEqual(out, ("accept", 2))

    def test_policy_receives_native_contract_arguments(self):
        policy = _SpyPolicy(
            result=(
                torch.tensor([1, 2], dtype=torch.int32),
                torch.tensor([100, 200], dtype=torch.int64),
                torch.tensor([0, 1], dtype=torch.int32),
            )
        )
        kwargs = self._kwargs()
        executor = self._executor(policy)
        out = executor.accept_and_finalize(**kwargs)

        self.assertEqual(len(policy.accept_calls), 1)
        call = policy.accept_calls[0]
        self.assertIs(call["candidates"], kwargs["verify_ids_2d"])
        self.assertIs(call["target_logits"], kwargs["target_logits"])
        self.assertIsNone(call["cutoff_verify_lens"])  # layout is None
        self.assertIs(call["req_pool_indices"], kwargs["req_pool_indices"])
        self.assertTrue(call["all_greedy"])  # sampling_info is None

        # Passthrough: the policy's tuple is authoritative downstream.
        self.assertTrue(torch.equal(out.correct_len, torch.tensor([1, 2])))
        self.assertTrue(torch.equal(out.bonus, torch.tensor([100, 200])))
        # FinalizeAcceptLens: commit = correct_len + 1; new = prefix + commit.
        self.assertTrue(torch.equal(out.commit_lens, torch.tensor([2, 3])))
        self.assertTrue(torch.equal(out.new_seq_lens, torch.tensor([12, 23])))
        self.assertTrue(torch.equal(out.cap_trim_lens, torch.tensor([0, 1])))
        # BuildOutTokens: draft prefix kept, bonus scattered at correct_len.
        self.assertTrue(
            torch.equal(
                out.out_tokens,
                torch.tensor([[1, 100, 0], [3, 4, 200]], dtype=torch.int64),
            )
        )

    def test_policy_without_req_pool_indices_raises(self):
        policy = _SpyPolicy(
            result=(
                torch.tensor([1], dtype=torch.int32),
                torch.tensor([1], dtype=torch.int64),
                torch.tensor([0], dtype=torch.int32),
            )
        )
        executor = self._executor(policy)
        with self.assertRaisesRegex(RuntimeError, "request-pool indices"):
            executor.accept_and_finalize(**self._kwargs(req_pool_indices=None))

    def test_no_policy_calls_native_accept(self):
        executor = self._executor(None)
        sentinel = (
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([100, 200], dtype=torch.int64),
            torch.tensor([0, 1], dtype=torch.int32),
        )
        with patch(
            "sglang.srt.speculative.dspark_components.dspark_verify.accept_draft_tokens",
            return_value=sentinel,
        ) as native:
            out = executor.accept_and_finalize(**self._kwargs())
        native.assert_called_once()
        self.assertTrue(torch.equal(out.correct_len, sentinel[0]))

    def test_policy_can_delegate_to_native(self):
        executor = self._executor(_SpyPolicy(delegate_native=True))
        sentinel = (
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([100, 200], dtype=torch.int64),
            torch.tensor([0, 1], dtype=torch.int32),
        )
        with patch(
            "sglang.srt.speculative.dspark_components.dspark_verify.accept_draft_tokens",
            return_value=sentinel,
        ):
            out = executor.accept_and_finalize(**self._kwargs())
        self.assertTrue(torch.equal(out.correct_len, sentinel[0]))
        self.assertTrue(torch.equal(out.bonus, sentinel[1]))


if __name__ == "__main__":
    unittest.main(verbosity=3)
