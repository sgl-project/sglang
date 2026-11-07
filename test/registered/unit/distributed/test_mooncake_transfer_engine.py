"""Ascend endpoint compatibility at the worker-thread/transport boundary."""

import ipaddress
import threading
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from sglang.srt.distributed.device_communicators import mooncake_transfer_engine
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _EndpointTransport:
    """Model the external host:port parser and P2P descriptor publication.

    The affected Ascend Direct wheel parses the last colon as the RPC port
    separator. Passing host:port:npu_id therefore publishes host:port:rpc_port,
    which cannot be resolved when a retry fetches the descriptor by its name.
    """

    def __init__(self, *, accepts_device_suffix=False):
        self.accepts_device_suffix = accepts_device_suffix
        self.endpoint = None

    def initialize(self, endpoint, metadata, protocol, device_name):
        self.thread = threading.get_ident()
        assert metadata == "P2PHANDSHAKE"
        self.endpoint = endpoint
        if self.accepts_device_suffix:
            endpoint, suffix = endpoint.rsplit(":npu_", 1)
            self.physical_device = int(suffix)
        host, _ = endpoint.rsplit(":", 1)
        self.descriptor_name = f"{host}:16162"
        return 0

    def refresh_descriptor(self):
        host, _ = self.descriptor_name.rsplit(":", 1)
        return ipaddress.ip_address(host)


class TestMooncakeAscendEndpoint(CustomTestCase):
    def _engine(self, gpu_id, *, accepts_device_suffix=False):
        transport = _EndpointTransport(accepts_device_suffix=accepts_device_suffix)
        engine = mooncake_transfer_engine.MooncakeTransferEngine.__new__(
            mooncake_transfer_engine.MooncakeTransferEngine
        )
        engine.gpu_id = gpu_id
        engine.engine = transport
        return engine, transport

    def _config(self, *, suffix=True, physical_id=-1):
        stack = ExitStack()
        for field, value in (
            (envs.ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE, True),
            (envs.SGLANG_MOONCAKE_ASCEND_USE_DEVICE_SUFFIX, suffix),
            (envs.ASCEND_NPU_PHY_ID, physical_id),
        ):
            stack.enter_context(field.override(value))
        stack.enter_context(
            patch.object(mooncake_transfer_engine, "get_free_port", return_value=30557)
        )
        # Device binding belongs to the platform boundary, covered by #47277.
        stack.enter_context(
            patch("sglang.srt.disaggregation.common.utils.bind_thread_to_device")
        )
        return stack

    def test_plain_endpoint_remains_resolvable_after_metadata_refresh(self):
        """The legacy suffix must not turn a retry hostname into IP:port."""
        for gpu_id in (0, 3):
            with self.subTest(gpu_id=gpu_id):
                engine, transport = self._engine(gpu_id)
                with self._config(suffix=False):
                    engine.initialize("192.0.2.10", "")
                self.assertEqual(str(transport.refresh_descriptor()), "192.0.2.10")
                self.assertNotEqual(transport.thread, threading.get_ident())

    def test_device_suffix_mode_preserves_explicit_physical_device(self):
        engine, transport = self._engine(1, accepts_device_suffix=True)
        with self._config(physical_id=5):
            engine.initialize("192.0.2.10", "")
        self.assertEqual(transport.physical_device, 5)
        self.assertEqual(str(transport.refresh_descriptor()), "192.0.2.10")

    def test_plain_endpoint_rejects_an_unrepresentable_physical_override(self):
        engine, transport = self._engine(1)
        with self._config(suffix=False, physical_id=5):
            with self.assertRaisesRegex(ValueError, "ASCEND_NPU_PHY_ID"):
                engine.initialize("192.0.2.10", "")
        self.assertIsNone(transport.endpoint)


if __name__ == "__main__":
    unittest.main()
