"""Under prefill_complete a second prefill sender on a held room fails alone."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.bootstrap import DeferredBootstrap
from sglang.srt.disaggregation.common.conn import CommonKVSender, KVTransferError
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ROOM = 1000


def _prefill_manager():
    # prefill_complete runs only with Mooncake, so the manager is Mooncake's.
    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager.is_dummy_cp_rank = False
    manager.request_status = {}
    manager.transfer_infos = {}
    manager.req_to_decode_prefix_len = {}
    manager.room_owners = {}
    manager._room_owner_lock = threading.Lock()
    manager._deferred_ack_targets = {}
    manager._deferred_ack_poisoned_rooms = set()
    manager._staging_outstanding = {}
    manager.failure_records = {}
    manager.failure_lock = threading.Lock()
    manager.bootstrap_timeout = 300
    manager.send_kv_status_message = MagicMock()
    manager.server_args = SimpleNamespace(enable_pd_role_switch=False)
    manager.deferred_bootstrap = DeferredBootstrap(timeout=10)
    return manager


def _sender(manager, fingerprint):
    with get_context().override_server_args(dp_size=1, enable_trace=False):
        return CommonKVSender(manager, "unused", ROOM, prompt_fingerprint=fingerprint)


class TestDeferredDuplicatePrefill(CustomTestCase):
    def test_second_sender_fails_without_touching_owner(self):
        for name, other_fingerprint in (("other", "b" * 16), ("same", "a" * 16)):
            with self.subTest(other_request=name):
                manager = _prefill_manager()
                owner = _sender(manager, "a" * 16)
                self.assertEqual(owner.poll(), KVPoll.Bootstrapping)
                other = _sender(manager, other_fingerprint)
                self.assertEqual(other.poll(), KVPoll.Failed)
                self.assertEqual(owner.poll(), KVPoll.Bootstrapping)
                self.assertNotIn(ROOM, manager.failure_records)
                with self.assertRaisesRegex(KVTransferError, "in use by another"):
                    other.failure_exception()
                self.assertEqual(owner.poll(), KVPoll.Bootstrapping)
                self.assertFalse(manager.deferred_bootstrap.rooms[ROOM].failed)


if __name__ == "__main__":
    unittest.main()
