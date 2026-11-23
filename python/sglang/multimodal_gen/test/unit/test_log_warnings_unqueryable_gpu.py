# SPDX-License-Identifier: Apache-2.0
"""log_warnings walks every physical GPU at import time.

A GPU that NVML (or MTML) cannot query, for example one that has fallen off
the bus, must be skipped instead of failing the platform module import.
"""

import importlib
import sys
import types
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.runtime import platforms
from sglang.multimodal_gen.runtime.platforms import cuda as cuda_platform


def _fake_device_queries(lib, names, bad_index, fail_at, error):
    """Patch the device count, handle and name queries of `lib`."""

    def get_handle(index):
        if index == bad_index and fail_at == "handle":
            raise error
        return index

    def get_name(handle):
        if handle == bad_index and fail_at == "name":
            raise error
        return names[handle]

    return [
        patch.object(lib, "nvmlInit", return_value=None, create=True),
        patch.object(lib, "nvmlShutdown", return_value=None, create=True),
        patch.object(lib, "nvmlDeviceGetCount", return_value=len(names), create=True),
        patch.object(
            lib, "nvmlDeviceGetHandleByIndex", side_effect=get_handle, create=True
        ),
        patch.object(lib, "nvmlDeviceGetName", side_effect=get_name, create=True),
    ]


class _LogWarningsCases:
    """Shared cases; subclasses set platform_cls, lib, logger and env_var."""

    platform_cls = None
    lib = None
    logger = None
    env_var = None

    def _run(self, names, bad_index, fail_at, error):
        with ExitStack() as stack:
            for p in _fake_device_queries(self.lib, names, bad_index, fail_at, error):
                stack.enter_context(p)
            stack.enter_context(patch.dict("os.environ", {self.env_var: ""}))
            warning = stack.enter_context(patch.object(self.logger, "warning"))
            self.platform_cls.log_warnings()
            return [call.args for call in warning.call_args_list]

    def test_unqueryable_gpu_is_skipped(self):
        names = ["GPU A", "GPU A", "GPU B"]
        for fail_at in ("handle", "name"):
            with self.subTest(fail_at=fail_at):
                warnings = self._run(names, 1, fail_at, self.lost_error())

                skipped = [w for w in warnings if w[0].startswith("Skipping GPU")]
                self.assertEqual(len(skipped), 1)
                self.assertEqual(skipped[0][1], 1)
                # The check still runs over the GPUs that answered.
                mixed = [w for w in warnings if w[0].startswith("Detected different")]
                self.assertEqual(len(mixed), 1)
                self.assertEqual(mixed[0][1], "GPU A, GPU B")

    def test_other_errors_still_raise(self):
        with self.assertRaises(RuntimeError):
            self._run(["GPU A", "GPU A"], 1, "name", RuntimeError("unexpected"))


class TestCudaLogWarnings(_LogWarningsCases, unittest.TestCase):
    platform_cls = cuda_platform.NvmlCudaPlatform
    lib = cuda_platform.pynvml
    logger = cuda_platform.logger
    env_var = "CUDA_DEVICE_ORDER"

    def lost_error(self):
        return self.lib.NVMLError(self.lib.NVML_ERROR_GPU_IS_LOST)


class TestCudaPlatformImport(unittest.TestCase):
    def test_import_survives_a_lost_gpu(self):
        # With NVML available the module calls log_warnings at import time.
        name = "sglang.multimodal_gen.runtime.platforms.cuda"
        lib = cuda_platform.pynvml
        lost = lib.NVMLError(lib.NVML_ERROR_GPU_IS_LOST)
        original = sys.modules[name]
        with ExitStack() as stack:
            for p in _fake_device_queries(lib, ["GPU A", "GPU A"], 1, "handle", lost):
                stack.enter_context(p)
            # Restore the module and the parent package attribute afterwards.
            stack.enter_context(patch.dict(sys.modules))
            stack.enter_context(patch.object(platforms, "cuda", original))
            sys.modules.pop(name)
            reloaded = importlib.import_module(name)
            self.assertIs(reloaded.CudaPlatform, reloaded.NvmlCudaPlatform)


class _FakeMTMLError(Exception):
    pass


class TestMusaLogWarnings(_LogWarningsCases, unittest.TestCase):
    env_var = "MUSA_DEVICE_ORDER"

    @classmethod
    def setUpClass(cls):
        # pymtml and torchada are only installed on MUSA machines. With MTML
        # unavailable, importing the module picks the non-MTML platform and its
        # import-time log_warnings is a no-op, so MtmlMusaPlatform can be
        # exercised directly.
        pymtml = types.ModuleType("pymtml")
        pymtml.NVMLError = _FakeMTMLError
        pymtml.nvmlInit = MagicMock(side_effect=_FakeMTMLError("no MTML"))
        pymtml.nvmlShutdown = MagicMock()
        cls._modules = patch.dict(
            sys.modules, {"pymtml": pymtml, "torchada": types.ModuleType("torchada")}
        )
        cls._modules.start()
        cls._had_musa_attr = hasattr(platforms, "musa")
        cls._musa_attr = getattr(platforms, "musa", None)
        sys.modules.pop("sglang.multimodal_gen.runtime.platforms.musa", None)
        musa = importlib.import_module("sglang.multimodal_gen.runtime.platforms.musa")
        cls.platform_cls = musa.MtmlMusaPlatform
        cls.lib = pymtml
        cls.logger = musa.logger

    @classmethod
    def tearDownClass(cls):
        # Drop the stubbed module from sys.modules and the parent package so a
        # later import does not pick it up.
        sys.modules.pop("sglang.multimodal_gen.runtime.platforms.musa", None)
        cls._modules.stop()
        if cls._had_musa_attr:
            platforms.musa = cls._musa_attr
        elif hasattr(platforms, "musa"):
            del platforms.musa

    def lost_error(self):
        return _FakeMTMLError("GPU is lost")


if __name__ == "__main__":
    unittest.main()
