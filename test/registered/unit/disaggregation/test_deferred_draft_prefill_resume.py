"""Unsupported draft-prefix restores must terminate one request, not the worker."""

import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation import decode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDeferredDraftPrefillResume(unittest.TestCase):
    def queue(self, enabled=True):
        queue = object.__new__(decode.DecodePreallocQueue)
        self.pool = Mock()
        queue.scheduler = SimpleNamespace(
            deferred_draft_prefill_pool=self.pool if enabled else None,
            retire_unadmitted_request=Mock(),
        )
        queue.tree_cache = object()
        queue.queue = []
        queue.retracted_queue = []
        queue.held_rebootstrap_reqs = []
        return queue

    def setUp(self):
        self.discard = patch.object(decode, "discard_kv_cache_backup").start()
        self.prepare_abort = patch.object(decode, "prepare_abort").start()
        patch.object(
            decode,
            "get_disagg",
            return_value=SimpleNamespace(
                disaggregation_decode_retraction_backup="cpu_tensor"
            ),
        ).start()
        self.addCleanup(patch.stopall)

    def test_fresh_request_does_not_enter_resume_cleanup(self):
        queue = self.queue()
        self.assertFalse(queue.reject_deferred_draft_resume(SimpleNamespace()))
        self.discard.assert_not_called()
        self.prepare_abort.assert_not_called()
        queue.scheduler.retire_unadmitted_request.assert_not_called()

    def test_every_resume_marker_rejects_before_allocating_or_transferring(self):
        for fields, arguments in (
            ({}, {"is_retracted": True}),
            ({}, {"is_rebootstrap": True}),
            ({"is_retracted": True}, {}),
            ({"pd_rebootstrap_in_progress": True}, {}),
        ):
            with self.subTest(fields=fields, arguments=arguments):
                queue = self.queue()
                req = SimpleNamespace(**fields)
                self.assertTrue(queue.reject_deferred_draft_resume(req, **arguments))
                self.discard.assert_called_with(req, queue.tree_cache, "cpu_tensor")
                self.pool.free.assert_called_once_with(req)
                self.prepare_abort.assert_called_with(
                    req,
                    "Deferred draft prefill cannot resume retracted or rebootstrapped "
                    "requests: the complete draft prefix state is unavailable",
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                queue.scheduler.retire_unadmitted_request.assert_called_once_with(req)
                self.assertFalse(req.pd_deferred_draft_prefill)
                self.assertEqual(queue.queue + queue.retracted_queue, [])

    def test_retracted_add_never_reenters_the_draft_initialization_queue(self):
        queue = self.queue()
        queue._check_if_req_exceed_kv_capacity = Mock(
            side_effect=AssertionError("Must reject before allocation checks")
        )
        req = SimpleNamespace(output_ids=[8, 9], is_retracted=True)
        queue.add(req, is_retracted=True)
        self.assertEqual(req.output_ids, [8, 9])
        queue._check_if_req_exceed_kv_capacity.assert_not_called()
        self.assertEqual(queue.queue + queue.retracted_queue, [])

    def test_true_rebootstrap_is_not_held_until_resume(self):
        queue = self.queue()
        req = SimpleNamespace(output_ids=[8, 9], pd_rebootstrap_in_progress=True)
        queue.hold_rebootstrap(req)
        self.assertEqual(queue.held_rebootstrap_reqs, [])
        self.assertEqual(req.output_ids, [8, 9])
        queue.scheduler.retire_unadmitted_request.assert_called_once_with(req)

    def test_disabled_feature_preserves_native_rebootstrap_holding(self):
        queue = self.queue(enabled=False)
        req = SimpleNamespace(pd_rebootstrap_in_progress=True)
        queue.hold_rebootstrap(req)
        self.assertEqual(queue.held_rebootstrap_reqs, [req])
        self.discard.assert_not_called()
        queue.scheduler.retire_unadmitted_request.assert_not_called()

    def test_old_scheduler_fixture_without_feature_attribute_still_resumes(self):
        queue = self.queue(enabled=False)
        del queue.scheduler.deferred_draft_prefill_pool
        self.assertFalse(
            queue.reject_deferred_draft_resume(SimpleNamespace(), is_retracted=True)
        )
        self.discard.assert_not_called()


if __name__ == "__main__":
    unittest.main()
