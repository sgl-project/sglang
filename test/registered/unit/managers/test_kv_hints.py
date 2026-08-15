import unittest

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
            kv_hints={"deref": {}},
        )

        self.assertIsNotNone(request.kv_hints.deref)

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


if __name__ == "__main__":
    unittest.main()
