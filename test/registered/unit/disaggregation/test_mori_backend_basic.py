"""Basic CPU unit tests for Mori disaggregation control paths."""

import importlib.util
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch


def _fake_mori_modules():
    if importlib.util.find_spec("mori") is not None:
        return {}

    mori_module = types.ModuleType("mori")
    mori_module.__path__ = []
    cpp_module = types.ModuleType("mori.cpp")
    io_module = types.ModuleType("mori.io")

    cpp_module.TransferStatus = object
    for name in (
        "BackendType",
        "EngineDesc",
        "IOEngine",
        "IOEngineConfig",
        "MemoryDesc",
        "MemoryLocationType",
        "PollCqMode",
        "RdmaBackendConfig",
        "StatusCode",
    ):
        setattr(io_module, name, type(name, (), {}))

    mori_module.cpp = cpp_module
    mori_module.io = io_module
    return {
        "mori": mori_module,
        "mori.cpp": cpp_module,
        "mori.io": io_module,
    }


with patch.dict(sys.modules, _fake_mori_modules()):
    from sglang.srt.disaggregation.common.conn import CommonKVManager
    from sglang.srt.disaggregation.mori.conn import MoriKVManager
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMoriKVManager(unittest.TestCase):
    def test_decode_initialization_starts_heartbeat_checker(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.disaggregation_mode = DisaggregationMode.DECODE
        engine = MagicMock()
        lifecycle = MagicMock()

        with (
            patch.object(CommonKVManager, "__init__", return_value=None),
            patch.object(MoriKVManager, "_init_engine", return_value=engine),
            patch.object(MoriKVManager, "_register_local_buffers"),
            patch.object(MoriKVManager, "_start_decode_thread") as start_decode,
            patch.object(
                MoriKVManager, "_start_heartbeat_checker_thread"
            ) as start_heartbeat,
            patch("sglang.srt.disaggregation.mori.conn.zmq.Context"),
        ):
            lifecycle.attach_mock(start_decode, "decode")
            lifecycle.attach_mock(start_heartbeat, "heartbeat")

            MoriKVManager.__init__(
                manager,
                SimpleNamespace(),
                DisaggregationMode.DECODE,
                SimpleNamespace(),
            )

        self.assertEqual(lifecycle.mock_calls, [call.decode(), call.heartbeat()])


if __name__ == "__main__":
    unittest.main()
