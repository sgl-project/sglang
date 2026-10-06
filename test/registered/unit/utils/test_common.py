import contextlib
import importlib
import io
import os
import sys
import tempfile
import unittest
from array import array
from unittest import mock

import torch
from PIL import Image

from sglang.srt.runtime_context import override_platform
from sglang.srt.utils.common import (
    _flashinfer_has_fused_dcp_reduce,
    _get_device_sm_via_nvml,
    _load_image,
    fi_a2a_platform_blocker,
    flatten_arrays_to_int64_tensor,
    get_device_sm_nvidia_smi,
    get_nvidia_driver_version_str,
)
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small-amd")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFlattenArraysToInt64Tensor(CustomTestCase):
    """`flatten_arrays_to_int64_tensor` is invoked by `prepare_for_extend`
    to build the per-batch input_ids tensor (pinned, async H2D) from a
    list of array.array('q') per-req get_fill_ids() slices. Tests the
    full matrix of (device, pin) the production code paths through.
    """

    DEVICES = ("cpu", "cuda")
    PIN_OPTIONS = (False, True)

    def _check(self, parts: list, expected: list[int]) -> None:
        for device in self.DEVICES:
            for pin in self.PIN_OPTIONS:
                with self.subTest(device=device, pin=pin):
                    out = flatten_arrays_to_int64_tensor(parts, device, pin)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    self.assertEqual(out.dtype, torch.int64)
                    self.assertEqual(out.device.type, device)
                    self.assertEqual(out.shape, (len(expected),))
                    self.assertEqual(out.cpu().tolist(), expected)

    def test_single_part(self):
        parts = [array("q", [1, 2, 3, 4, 5])]
        self._check(parts, [1, 2, 3, 4, 5])

    def test_multiple_parts(self):
        parts = [
            array("q", [10, 20, 30]),
            array("q", [100, 200]),
            array("q", [1000]),
        ]
        self._check(parts, [10, 20, 30, 100, 200, 1000])


class TestNvidiaDriverVersionStr(CustomTestCase):
    """`get_nvidia_driver_version_str` is typed as `str | None`: it returns
    `None` when nvidia-smi is missing, fails, or emits an empty string. These
    tests exercise both the success and the None-return paths by monkey-
    patching `subprocess.run`, so they don't require a GPU. The function is
    `@lru_cache`d, so the cache is cleared around each test to make the patch
    observable.
    """

    def setUp(self):
        get_nvidia_driver_version_str.cache_clear()

    def tearDown(self):
        get_nvidia_driver_version_str.cache_clear()

    def test_returns_version_string(self):
        import subprocess

        class _R:
            stdout = "595.58.03\n"

        original = subprocess.run
        subprocess.run = lambda *a, **k: _R()
        try:
            self.assertEqual(get_nvidia_driver_version_str(), "595.58.03")
        finally:
            subprocess.run = original

    def test_returns_none_on_empty_output(self):
        import subprocess

        class _R:
            stdout = "\n"

        original = subprocess.run
        subprocess.run = lambda *a, **k: _R()
        try:
            self.assertIsNone(get_nvidia_driver_version_str())
        finally:
            subprocess.run = original

    def test_returns_none_on_called_process_error(self):
        import subprocess

        original = subprocess.run

        def boom(*a, **k):
            raise subprocess.CalledProcessError(1, "nvidia-smi")

        subprocess.run = boom
        try:
            self.assertIsNone(get_nvidia_driver_version_str())
        finally:
            subprocess.run = original

    def test_returns_none_on_file_not_found(self):
        import subprocess

        original = subprocess.run

        def boom(*a, **k):
            raise FileNotFoundError("nvidia-smi")

        subprocess.run = boom
        try:
            self.assertIsNone(get_nvidia_driver_version_str())
        finally:
            subprocess.run = original


class TestGetDeviceSmNvidiaSmi(CustomTestCase):
    """`get_device_sm_nvidia_smi` parses nvidia-smi output into a (major,
    minor) tuple and falls back to (0, 0) -- logging via `logger.error` --
    when nvidia-smi fails. The success path needs a GPU; the fallback path is
    covered here by forcing a failure and asserting the (0, 0) return. The
    fallback path needs no GPU, so this test runs on CPU.
    """

    def test_fallback_on_failure_returns_zero_zero(self):
        import subprocess

        original = subprocess.run

        def boom(*a, **k):
            raise subprocess.CalledProcessError(1, "nvidia-smi")

        subprocess.run = boom
        try:
            self.assertEqual(get_device_sm_nvidia_smi(), (0, 0))
        finally:
            subprocess.run = original


class TestLoadImage(CustomTestCase):
    def test_corrupt_image_bytes_raise_value_error(self):
        buf = io.BytesIO()
        Image.new("RGB", (16, 16), (10, 120, 200)).save(buf, format="PNG")
        valid_png = buf.getvalue()

        broken_chunk_png = bytearray(valid_png)
        idat_pos = broken_chunk_png.index(b"IDAT")
        broken_chunk_png[idat_pos - 1] -= 6

        img = _load_image(image_bytes=valid_png, gpu_image_decode=False)
        self.assertEqual(img.size, (16, 16))

        cases = [
            b"not an image",
            b"",
            valid_png[: len(valid_png) // 2],
            bytes(broken_chunk_png),
        ]
        for image_bytes in cases:
            with self.subTest(length=len(image_bytes)):
                with self.assertRaisesRegex(ValueError, "Could not decode image"):
                    _load_image(
                        image_bytes=image_bytes,
                        gpu_image_decode=False,
                    )


class _FakePynvml:
    """Records the NVML index it was asked for, so a test can tell which
    physical GPU the helper would have reported."""

    def __init__(self, capability=(9, 0)):
        self.capability = capability
        self.requested_index = None
        self.initialized = False

    def nvmlInit(self):
        self.initialized = True

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, index):
        self.requested_index = index
        return f"handle-{index}"

    def nvmlDeviceGetCudaComputeCapability(self, handle):
        return self.capability


class TestGetDeviceSmViaNvml(CustomTestCase):
    """The torch ordinal and the NVML index differ under CUDA_VISIBLE_DEVICES
    and MIG; without that mapping the helper must return None, not GPU 0."""

    def test_torch_exposes_the_mapping_api(self):
        # The cases below install the private attribute themselves, so they stay
        # green on a torch that dropped it while the helper silently falls back.
        self.assertTrue(hasattr(torch.cuda, "_get_nvml_device_index"))

    def test_maps_the_torch_ordinal_to_the_nvml_index(self):
        fake = _FakePynvml(capability=(9, 0))
        with (
            mock.patch.dict(sys.modules, {"pynvml": fake}),
            mock.patch.object(
                torch.cuda, "_get_nvml_device_index", lambda index: 3, create=True
            ),
        ):
            self.assertEqual(_get_device_sm_via_nvml(), 90)
        self.assertEqual(fake.requested_index, 3)

    def test_returns_none_when_the_mapping_api_is_absent(self):
        fake = _FakePynvml()
        saved = torch.cuda.__dict__.pop("_get_nvml_device_index", None)
        try:
            with mock.patch.dict(sys.modules, {"pynvml": fake}):
                self.assertIsNone(_get_device_sm_via_nvml())
        finally:
            if saved is not None:
                torch.cuda._get_nvml_device_index = saved
        self.assertFalse(fake.initialized, "must not query NVML without the mapping")

    def test_returns_none_when_the_mapping_api_raises(self):
        fake = _FakePynvml()

        def boom(index):
            raise RuntimeError("no such device")

        with (
            mock.patch.dict(sys.modules, {"pynvml": fake}),
            mock.patch.object(torch.cuda, "_get_nvml_device_index", boom, create=True),
        ):
            self.assertIsNone(_get_device_sm_via_nvml())
        self.assertFalse(fake.initialized, "must not query NVML without the mapping")


class TestFiA2aPlatformBlocker(CustomTestCase):
    @contextlib.contextmanager
    def _flashinfer_package(self, *, with_fused_op: bool):
        """Install a flashinfer package that fails when imported."""
        with tempfile.TemporaryDirectory() as root:
            package = os.path.join(root, "flashinfer")
            os.makedirs(os.path.join(package, "comm"))
            with open(os.path.join(package, "__init__.py"), "w") as f:
                f.write("raise ImportError('flashinfer was imported')\n")
            if with_fused_op:
                open(os.path.join(package, "comm", "dcp_lse_reduce.py"), "w").close()
            with (
                mock.patch.dict(sys.modules),
                mock.patch.object(sys, "path", [root, *sys.path]),
            ):
                for name in [n for n in sys.modules if n.split(".")[0] == "flashinfer"]:
                    del sys.modules[name]
                importlib.invalidate_caches()
                _flashinfer_has_fused_dcp_reduce.cache_clear()
                try:
                    yield
                finally:
                    _flashinfer_has_fused_dcp_reduce.cache_clear()

    @override_platform(is_sm100=True)
    def test_finds_the_fused_op_without_importing_flashinfer(self):
        """The blocker runs while the launcher resolves arguments, so it must
        find FlashInfer's fused op on disk without importing flashinfer."""
        for with_fused_op in (True, False):
            with (
                self.subTest(with_fused_op=with_fused_op),
                self._flashinfer_package(with_fused_op=with_fused_op),
            ):
                reason = fi_a2a_platform_blocker(
                    dcp_size=8, tp_size=8, pp_size=1, nnodes=1
                )
                self.assertNotIn("flashinfer", sys.modules)
                if with_fused_op:
                    self.assertIsNone(reason)
                else:
                    self.assertIn("decode_cp_a2a_lse_reduce", reason)

    @override_platform(is_sm100=True)
    def test_dcp_group_must_share_one_nvlink_domain(self):
        # (MNNVL fabric, dcp_size, tp_size, pp_size, nnodes, blocked)
        cases = (
            (False, 8, 16, 1, 2, False),
            (False, 16, 16, 1, 2, True),
            (False, 8, 8, 2, 2, False),
            (True, 16, 16, 1, 2, False),
        )
        for fabric, dcp_size, tp_size, pp_size, nnodes, blocked in cases:
            with (
                self.subTest(
                    fabric=fabric, dcp_size=dcp_size, pp_size=pp_size, nnodes=nnodes
                ),
                mock.patch(
                    "sglang.srt.utils.common.is_mnnvl_fabric_device",
                    return_value=fabric,
                ),
                mock.patch(
                    "sglang.srt.utils.common._flashinfer_has_fused_dcp_reduce",
                    return_value=True,
                ),
            ):
                reason = fi_a2a_platform_blocker(
                    dcp_size=dcp_size, tp_size=tp_size, pp_size=pp_size, nnodes=nnodes
                )
                self.assertEqual(reason is not None, blocked, reason)


if __name__ == "__main__":
    unittest.main()
