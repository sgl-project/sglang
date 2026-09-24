"""Unit tests for srt/checkpoint_engine/checkpoint_engine_worker.py — no server, no model loading.

Focus: device resolution so the ZMQ handshake key matches checkpoint-engine's
ParameterServer (ps.py::_get_physical_gpu_id) on every backend -- ``GPU-<uuid>``
for CUDA/XPU and ``NPU-<uuid>`` for NPU. These paths are pure namespace routing
(``get_device`` / ``get_device_module`` / ``is_npu``) and are fully mockable on CPU.

Skipped entirely unless the ``checkpoint-engine`` extra is installed, since the
worker module refuses to import without it.
"""

from sglang.test.ci.ci_register import register_cpu_ci

# Unit tests may register CPU suites only (scripts/lint/check_registered_tests.py).
# TestWorkerDeviceUuidOnXpu below still self-skips off XPU, so it stays runnable
# by hand on an Intel GPU host.
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import importlib.util
import sys
import unittest
from types import ModuleType
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

    def _fake_device_module(self, *, current=3):
        mod = MagicMock()
        mod.current_device.return_value = current
        return mod

    def _fake_platform(self, *, uuid="abcd-1234"):
        # The uuid now comes from the platform layer (current_platform), which
        # already returns str(get_device_properties(id).uuid) for cuda/xpu.
        plat = MagicMock()
        plat.get_device_uuid.return_value = uuid
        return plat

    def test_device_uuid_cuda(self):
        worker = self._make_worker()
        fake = self._fake_device_module(current=0)
        with (
            patch(f"{_WORKER_MOD}.is_npu", return_value=False),
            patch(f"{_WORKER_MOD}.get_device_module", return_value=fake),
            patch(
                f"{_WORKER_MOD}.current_platform", self._fake_platform(uuid="cuda-uuid")
            ),
        ):
            self.assertEqual(worker.get_device_uuid(), "GPU-cuda-uuid")
            self.assertEqual(worker.get_device_id(), 0)

    def test_device_uuid_xpu(self):
        worker = self._make_worker()
        fake = self._fake_device_module(current=2)
        with (
            patch(f"{_WORKER_MOD}.is_npu", return_value=False),
            patch(f"{_WORKER_MOD}.get_device_module", return_value=fake),
            patch(
                f"{_WORKER_MOD}.current_platform", self._fake_platform(uuid="xpu-uuid")
            ),
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
        fake = self._fake_device_module(current=1)
        plat = MagicMock()
        plat.get_device_uuid.side_effect = AssertionError("no uuid")
        with (
            patch(f"{_WORKER_MOD}.is_npu", return_value=False),
            patch(f"{_WORKER_MOD}.get_device_module", return_value=fake),
            patch(f"{_WORKER_MOD}.current_platform", plat),
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


def _checkpoint_engine_stub():
    """get_model_loader never calls into checkpoint-engine, so the worker can be
    imported against a stand-in where the extra is not installed (as on CPU CI)."""
    worker = ModuleType("checkpoint_engine.worker")
    worker.update_weights_from_ipc = MagicMock()
    package = ModuleType("checkpoint_engine")
    package.worker = worker
    return {"checkpoint_engine": package, "checkpoint_engine.worker": worker}


class _DoubledAtRuntime:
    """The runtime form is 2x the checkpoint value, as with FNUZ block weights or
    doubled KV scales; writing a checkpoint tensor onto it corrupts the layer."""

    def process_weights_after_loading(self, layer):
        layer.weight.data.mul_(2)
        layer.runtime_form = True

    def restore_weights_before_loading(self, layer):
        if layer.runtime_form:
            layer.weight.data.div_(2)
            layer.runtime_form = False


class _ConvertedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Module()
        self.layer.weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        self.layer.quant_method = _DoubledAtRuntime()
        self.layer.quant_method.process_weights_after_loading(self.layer)
        self.written_onto = []

    def load_weights(self, weights):
        self.written_onto.append(self.layer.weight.tolist())
        for _, value in weights:
            self.layer.weight.data.copy_(value)


class TestWorkerModelLoaderRestores(CustomTestCase):
    """An IPC reload opens no weight-update session, so the loader itself must put the
    converted layers back in checkpoint form before the first write."""

    def setUp(self):
        stubs = {} if _HAS_CHECKPOINT_ENGINE else _checkpoint_engine_stub()
        with patch.dict(sys.modules, stubs):
            self.worker = importlib.import_module(_WORKER_MOD)

    def test_every_update_writes_onto_checkpoint_form(self):
        model = _ConvertedModel()
        worker = self.worker.SGLangCheckpointEngineWorkerExtensionImpl(
            model_runner=MagicMock(model=model)
        )
        with (
            patch.object(self.worker, "get_device", return_value="cpu"),
            patch.object(self.worker, "get_device_module") as device_module,
        ):
            device_module.return_value.current_device.return_value = 0
            load, post_hook = worker.get_model_loader(), worker.get_post_hook()
            for value in (3.0, 5.0):
                load(iter([("layer.weight", torch.full((2,), value))]))
                load(iter([]))  # a later bucket of the same update
                post_hook()

        # Without the restore, the second update would land on 6.0 (3.0 converted).
        self.assertEqual(
            model.written_onto, [[1.0, 1.0], [3.0, 3.0], [3.0, 3.0], [5.0, 5.0]]
        )
        self.assertEqual(model.layer.weight.tolist(), [10.0, 10.0])


if __name__ == "__main__":
    unittest.main(verbosity=3)
