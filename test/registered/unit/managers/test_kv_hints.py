import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.kv_hints import (  # noqa: E402
    DerefHint,
    KvHintManager,
    KvHints,
)
from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKvHints(unittest.TestCase):
    def test_request_dict_converts_to_typed_envelope(self):
        request = GenerateReqInput(
            text="hello",
            kv_hints={"deref": {"action_id": "compact-1"}},
        )

        self.assertIsNotNone(request.kv_hints.deref)
        self.assertEqual(request.kv_hints.deref.action_id, "compact-1")

    def test_capability_requires_an_installed_handler(self):
        self.assertEqual(KvHintManager().capabilities(), [])
        self.assertEqual(
            KvHintManager(session_refs=object()).capabilities(),
            ["kv_hint.deref.v1"],
        )

    def test_deref_without_session_is_ignored(self):
        request = type(
            "Request",
            (),
            {"session_id": None, "session_generation": None, "kv_hints": None},
        )()
        manager = KvHintManager(session_refs=object())

        manager.on_request(
            request,
            KvHints(deref=DerefHint()),
        )

        self.assertIsNone(request.kv_hints)

    def test_deref_action_is_applied_once_per_session(self):
        session_refs = Mock()
        session_refs.evict_radix_session.return_value = SimpleNamespace(
            status="evicted", generation=2, indexed_component_leaves=1
        )
        request = SimpleNamespace(
            session_id="session-1",
            session_generation=1,
            kv_hints=KvHints(deref=DerefHint(action_id="compact-turn-1")),
        )
        manager = KvHintManager(session_refs=session_refs)

        manager.on_request_success(request, has_reusable_leaf=False)
        manager.on_request_success(request, has_reusable_leaf=False)

        session_refs.evict_radix_session.assert_called_once_with("session-1", 1)


if __name__ == "__main__":
    unittest.main()
