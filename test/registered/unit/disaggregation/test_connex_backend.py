"""CPU unit tests for the connex disaggregation transfer backend wiring."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.connex import ConnexKVManager
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVBootstrapServer,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    TransferBackend,
    get_kv_class,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestConnexBackend(CustomTestCase):
    def test_backend_classes_follow_mooncake_protocol(self):
        backend = TransferBackend("connex")
        expected = {
            KVClassType.MANAGER: MooncakeKVManager,
            KVClassType.SENDER: MooncakeKVSender,
            KVClassType.RECEIVER: MooncakeKVReceiver,
            KVClassType.BOOTSTRAP_SERVER: MooncakeKVBootstrapServer,
        }
        for class_type, base in expected.items():
            cls = get_kv_class(backend, class_type)
            self.assertTrue(issubclass(cls, base))
            self.assertIsNot(cls, base)

    def test_init_engine_uses_connex_engine_for_role(self):
        calls = []
        fake = types.ModuleType("connex.adapters.sglang.pd_backend")
        fake.get_engine = lambda ip, role, gpu: calls.append((ip, role, gpu)) or "E"
        mgr = ConnexKVManager.__new__(ConnexKVManager)
        mgr.local_ip = "10.0.0.1"
        mgr.kv_args = SimpleNamespace(gpu_id=3)
        mgr.disaggregation_mode = DisaggregationMode.DECODE
        with patch.dict(sys.modules, {"connex.adapters.sglang.pd_backend": fake}):
            mgr.init_engine()
        self.assertEqual(mgr.engine, "E")
        self.assertEqual(calls, [("10.0.0.1", "decode", 3)])

    def test_missing_connex_package_has_install_hint(self):
        mgr = ConnexKVManager.__new__(ConnexKVManager)
        mgr.disaggregation_mode = DisaggregationMode.PREFILL
        with patch.dict(sys.modules, {"connex.adapters.sglang.pd_backend": None}):
            with self.assertRaisesRegex(ImportError, "connex-ai"):
                mgr.init_engine()


if __name__ == "__main__":
    unittest.main()
