# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import subprocess
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock


def _install_fake_checkpoint_engine():
    """Install a fake ``checkpoint_engine`` package.

    The real package is an optional dependency
    (``pip install sglang[checkpoint-engine]``) and is absent from the CPU CI
    environment, while the module under test imports
    ``checkpoint_engine.worker`` unconditionally. The fake modules are also
    used to verify the NPU/GPU UUID key conventions of the IPC handshake.
    """
    if "checkpoint_engine" in sys.modules:
        return
    pkg = types.ModuleType("checkpoint_engine")
    worker = types.ModuleType("checkpoint_engine.worker")
    worker.update_weights_from_ipc = mock.MagicMock(name="update_weights_from_ipc")
    device_utils = types.ModuleType("checkpoint_engine.device_utils")
    device_utils.npu_generate_uuid = lambda: "fake-npu-uuid"
    pkg.worker = worker
    pkg.device_utils = device_utils
    sys.modules["checkpoint_engine"] = pkg
    sys.modules["checkpoint_engine.worker"] = worker
    sys.modules["checkpoint_engine.device_utils"] = device_utils


_install_fake_checkpoint_engine()

import sglang.srt.checkpoint_engine.checkpoint_engine_worker as ce_worker
from sglang.srt.checkpoint_engine.checkpoint_engine_worker import (
    SGLangCheckpointEngineWorkerExtension,
    SGLangCheckpointEngineWorkerExtensionImpl,
)


class TestImportGuard(unittest.TestCase):
    def test_import_error_without_package(self):
        # A fresh interpreter has no fake package installed, so importing the
        # module must fail with the pip-install hint.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sglang.srt.checkpoint_engine.checkpoint_engine_worker",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pip install sglang[checkpoint-engine]", result.stderr)


class TestBaseExtension(unittest.TestCase):
    def setUp(self):
        sys.modules["checkpoint_engine.worker"].update_weights_from_ipc.reset_mock()

    def test_abstract_accessors_raise(self):
        ext = SGLangCheckpointEngineWorkerExtension()
        for accessor in (ext.get_device_uuid, ext.get_device_id, ext.get_model_loader):
            with self.assertRaises(NotImplementedError):
                accessor()

    def test_get_post_hook_default_none(self):
        self.assertIsNone(SGLangCheckpointEngineWorkerExtension().get_post_hook())

    def test_update_weights_from_ipc_unknown_uuid_raises(self):
        class _Stub(SGLangCheckpointEngineWorkerExtension):
            def get_device_uuid(self):
                return "GPU-known"

            def get_device_id(self):
                return 0

            def get_model_loader(self):
                return lambda *args, **kwargs: None

        stub = _Stub()
        with self.assertRaises(ValueError) as ctx:
            stub.update_weights_from_ipc({"GPU-other": "sock"})
        self.assertIn("GPU-known", str(ctx.exception))
        self.assertIn("GPU-other", str(ctx.exception))

    def test_update_weights_from_ipc_dispatches_to_worker(self):
        class _Stub(SGLangCheckpointEngineWorkerExtension):
            def get_device_uuid(self):
                return "GPU-known"

            def get_device_id(self):
                return 3

            def get_model_loader(self):
                return "fake-loader"

        stub = _Stub()
        stub.update_weights_from_ipc({"GPU-known": "sock-path"})
        fake = sys.modules["checkpoint_engine.worker"].update_weights_from_ipc
        fake.assert_called_once()
        args, kwargs = fake.call_args
        self.assertEqual(args[1], "sock-path")
        self.assertEqual(kwargs["device_id"], 3)
        self.assertEqual(kwargs["run"], "fake-loader")
        self.assertIsNone(kwargs["post_hook"])


class TestExtensionImpl(unittest.TestCase):
    def _make_impl(self, model):
        return SGLangCheckpointEngineWorkerExtensionImpl(SimpleNamespace(model=model))

    def test_get_model_loader_returns_load_weights(self):
        def load_weights(weights):
            return None

        impl = self._make_impl(SimpleNamespace(load_weights=load_weights))
        self.assertIs(impl.get_model_loader(), load_weights)

    def test_get_device_uuid_gpu_path(self):
        impl = self._make_impl(SimpleNamespace())
        with mock.patch.object(ce_worker, "is_npu", return_value=False):
            with mock.patch.object(
                ce_worker,
                "get_device_module",
                return_value=SimpleNamespace(current_device=3),
            ):
                with mock.patch.object(
                    ce_worker,
                    "current_platform",
                    SimpleNamespace(
                        get_device_uuid=lambda device_id: f"uuid-{device_id}"
                    ),
                ):
                    self.assertEqual(impl.get_device_uuid(), "GPU-uuid-3")

    def test_get_device_uuid_gpu_path_wraps_assertion_error(self):
        def _boom(device_id):
            raise AssertionError("no uuid")

        impl = self._make_impl(SimpleNamespace())
        with mock.patch.object(ce_worker, "is_npu", return_value=False):
            with mock.patch.object(
                ce_worker,
                "get_device_module",
                return_value=SimpleNamespace(current_device=0),
            ):
                with mock.patch.object(
                    ce_worker,
                    "current_platform",
                    SimpleNamespace(get_device_uuid=_boom),
                ):
                    with self.assertRaises(ValueError):
                        impl.get_device_uuid()

    def test_get_device_uuid_npu_path(self):
        # NPU devices key the ZMQ handshake as "NPU-<uuid>" (see the
        # docstring on the impl), generated via checkpoint_engine helpers.
        impl = self._make_impl(SimpleNamespace())
        with mock.patch.object(ce_worker, "is_npu", return_value=True):
            self.assertEqual(impl.get_device_uuid(), "NPU-fake-npu-uuid")


class TestPostHook(unittest.TestCase):
    def _run_post_hook(self, model):
        impl = SGLangCheckpointEngineWorkerExtensionImpl(SimpleNamespace(model=model))
        with mock.patch.object(ce_worker, "get_device", return_value="cpu"):
            with mock.patch.object(
                ce_worker,
                "get_device_module",
                return_value=SimpleNamespace(current_device=0),
            ):
                hook = impl.get_post_hook()
                hook()

    def test_quant_modules_processed(self):
        quant_method = mock.Mock()
        quant_module = SimpleNamespace(quant_method=quant_method)
        plain_module = SimpleNamespace()
        model = SimpleNamespace(
            named_modules=lambda: [("q", quant_module), ("p", plain_module)]
        )
        self._run_post_hook(model)
        quant_method.process_weights_after_loading.assert_called_once_with(quant_module)

    def test_post_load_weights_called_when_present(self):
        post_load_weights = mock.Mock()
        model = SimpleNamespace(
            named_modules=lambda: [], post_load_weights=post_load_weights
        )
        self._run_post_hook(model)
        post_load_weights.assert_called_once()

    def test_exceptions_are_swallowed(self):
        # A failing quant post-process must not propagate out of the hook;
        # it is logged as a warning instead.
        quant_method = mock.Mock()
        quant_method.process_weights_after_loading.side_effect = RuntimeError("boom")
        model = SimpleNamespace(
            named_modules=lambda: [("q", SimpleNamespace(quant_method=quant_method))]
        )
        with self.assertLogs(
            "sglang.srt.checkpoint_engine.checkpoint_engine_worker", level="WARNING"
        ):
            self._run_post_hook(model)


if __name__ == "__main__":
    unittest.main()
