"""Unit tests for srt/checkpoint_engine/checkpoint_engine_worker.py — no server, no model loading.

Focus: device resolution so the ZMQ handshake key matches checkpoint-engine's
ParameterServer (ps.py::_get_physical_gpu_id) on every backend -- ``GPU-<uuid>``
for CUDA/XPU and ``NPU-<uuid>`` for NPU. These paths are pure namespace routing
(``get_device`` / ``get_device_module`` / ``is_npu``) and are fully mockable on CPU.

Skipped entirely unless the ``checkpoint-engine`` extra is installed, since the
worker module refuses to import without it.
"""

from sglang.test.ci.ci_register import register_cpu_ci, register_xpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

import importlib.util
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.utils import is_xpu
from sglang.test.test_utils import CustomTestCase

# checkpoint-engine is an optional extra (sglang[checkpoint-engine]) that CI does
# not install, and the worker module raises ImportError at import time without it.
# Probe first so this file stays importable: an unguarded import would fail
# collection and take the whole file down rather than skipping.
_HAS_CHECKPOINT_ENGINE = importlib.util.find_spec("checkpoint_engine") is not None

if _HAS_CHECKPOINT_ENGINE:
    from sglang.srt.checkpoint_engine.checkpoint_engine_worker import (
        SGLangCheckpointEngineWorkerExtensionImpl,
    )

_WORKER_MOD = "sglang.srt.checkpoint_engine.checkpoint_engine_worker"
_NO_CKPT_ENGINE = "requires the checkpoint-engine optional dependency"


@unittest.skipUnless(_HAS_CHECKPOINT_ENGINE, _NO_CKPT_ENGINE)
class TestWorkerDeviceResolution(CustomTestCase):
    """get_device_uuid / get_device_id must route through the active accelerator
    namespace and emit the key the ParameterServer expects."""

    def _make_worker(self):
        # model_runner is unused by the device-resolution methods under test.
        return SGLangCheckpointEngineWorkerExtensionImpl(model_runner=MagicMock())

    def _fake_device_module(self, *, current=3, uuid="abcd-1234"):
        mod = MagicMock()
        mod.current_device.return_value = current
        props = MagicMock()
        props.uuid = uuid
        mod.get_device_properties.return_value = props
        return mod

    def test_device_uuid_cuda(self):
        worker = self._make_worker()
        fake = self._fake_device_module(current=0, uuid="cuda-uuid")
        with (
            patch(f"{_WORKER_MOD}.is_npu", return_value=False),
            patch(f"{_WORKER_MOD}.get_device_module", return_value=fake),
        ):
            self.assertEqual(worker.get_device_uuid(), "GPU-cuda-uuid")
            self.assertEqual(worker.get_device_id(), 0)

    def test_device_uuid_xpu(self):
        worker = self._make_worker()
        fake = self._fake_device_module(current=2, uuid="xpu-uuid")
        with (
            patch(f"{_WORKER_MOD}.is_npu", return_value=False),
            patch(f"{_WORKER_MOD}.get_device_module", return_value=fake),
        ):
            # XPU shares CUDA's GPU-<uuid> format; only the namespace differs.
            self.assertEqual(worker.get_device_uuid(), "GPU-xpu-uuid")
            self.assertEqual(worker.get_device_id(), 2)

    def test_device_uuid_npu_uses_npu_prefix(self):
        # NPU must NOT be treated as CUDA: the ParameterServer keys it as
        # NPU-<npu_generate_uuid()>, so a GPU-<uuid> key would never resolve.
        worker = self._make_worker()
        with (
            patch(f"{_WORKER_MOD}.is_npu", return_value=True),
            patch(
                "checkpoint_engine.device_utils.npu_generate_uuid",
                return_value="1.2.3.4-0",
            ),
        ):
            self.assertEqual(worker.get_device_uuid(), "NPU-1.2.3.4-0")

    def test_device_uuid_wraps_assertion_error(self):
        worker = self._make_worker()
        fake = MagicMock()
        fake.current_device.return_value = 1
        fake.get_device_properties.side_effect = AssertionError("no uuid")
        with (
            patch(f"{_WORKER_MOD}.is_npu", return_value=False),
            patch(f"{_WORKER_MOD}.get_device_module", return_value=fake),
            self.assertRaises(ValueError),
        ):
            worker.get_device_uuid()


@unittest.skipUnless(_HAS_CHECKPOINT_ENGINE, _NO_CKPT_ENGINE)
@unittest.skipUnless(is_xpu(), "requires an Intel XPU")
class TestWorkerDeviceUuidOnXpu(CustomTestCase):
    """Hardware-gated: the real XPU key must match what checkpoint-engine's
    ParameterServer derives, or the ZMQ handshake silently fails on XPU."""

    def test_real_uuid_matches_parameter_server(self):
        from checkpoint_engine.device_utils import DeviceManager
        from checkpoint_engine.ps import _get_physical_gpu_id

        worker = SGLangCheckpointEngineWorkerExtensionImpl(model_runner=MagicMock())
        key = worker.get_device_uuid()

        self.assertTrue(key.startswith("GPU-"), key)
        self.assertEqual(worker.get_device_id(), torch.xpu.current_device())

        # Independently derived by the ParameterServer side; the two must agree.
        dm = DeviceManager()
        self.assertEqual(dm.device_type, "xpu")
        self.assertEqual(key, _get_physical_gpu_id(dm, torch.xpu.current_device()))


if __name__ == "__main__":
    unittest.main(verbosity=3)
