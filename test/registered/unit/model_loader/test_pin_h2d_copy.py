"""Unit tests for model-load H2D copy pinning."""

import contextlib
import os
import threading
import unittest
from unittest.mock import mock_open, patch

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import torch

from sglang.srt.model_loader import loader as loader_module
from sglang.srt.model_loader import pin_h2d_copy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeDevice:
    def __init__(self, device_type, index=None):
        self.type = device_type
        self.index = index


class _FakeStream:
    def __init__(self, device_idx):
        self.device_idx = device_idx
        self.synchronized = False

    def synchronize(self):
        self.synchronized = True


class _FakeCuda:
    def __init__(self):
        self.current_device_idx = 0
        self.created_streams = []
        self.device_contexts = []
        self.stream_contexts = []

    def is_available(self):
        return True

    def current_device(self):
        return self.current_device_idx

    @contextlib.contextmanager
    def device(self, device_idx):
        previous = self.current_device_idx
        self.current_device_idx = device_idx
        self.device_contexts.append(device_idx)
        try:
            yield
        finally:
            self.current_device_idx = previous

    def Stream(self):
        stream = _FakeStream(self.current_device_idx)
        self.created_streams.append(stream)
        return stream

    @contextlib.contextmanager
    def stream(self, stream):
        self.stream_contexts.append(stream.device_idx)
        yield


class _FakeTensor:
    def __init__(self, *, is_cuda=False, device=None, pinned=False, pin_error=False):
        self.is_cuda = is_cuda
        self.device = device or _FakeDevice("cuda" if is_cuda else "cpu", 0)
        self._pinned = pinned
        self._pin_error = pin_error
        self.copy_calls = []
        self.pinned_tensor = None

    def is_pinned(self):
        return self._pinned

    def pin_memory(self):
        if self._pin_error:
            raise RuntimeError("pin failed")
        self.pinned_tensor = _FakeTensor(
            is_cuda=False, device=_FakeDevice("cpu"), pinned=True
        )
        return self.pinned_tensor

    def copy_(self, src, *args, **kwargs):
        self.copy_calls.append((src, args, kwargs))
        return self


class _FakeTorch:
    Tensor = _FakeTensor

    def __init__(self):
        self.cuda = _FakeCuda()


class TestPinH2DCopy(CustomTestCase):
    def setUp(self):
        pin_h2d_copy._is_grace_cpu_platform.cache_clear()

    def tearDown(self):
        pin_h2d_copy._is_grace_cpu_platform.cache_clear()

    @patch("sglang.srt.model_loader.pin_h2d_copy.platform.machine")
    @patch("sglang.srt.model_loader.pin_h2d_copy.subprocess.check_output")
    @patch("builtins.open", new_callable=mock_open, read_data="")
    def test_grace_cpu_platform_detected_from_lscpu(
        self, _mock_file, mock_check_output, mock_machine
    ):
        mock_machine.return_value = "aarch64"
        mock_check_output.return_value = (
            "BIOS Vendor ID: NVIDIA\n"
            "BIOS Model name: Grace A02P 900-2G548-0001-000 CPU @ 3.3GHz\n"
        )

        self.assertTrue(pin_h2d_copy._is_grace_cpu_platform())

    @patch("sglang.srt.model_loader.pin_h2d_copy.platform.machine")
    @patch("sglang.srt.model_loader.pin_h2d_copy.subprocess.check_output")
    @patch("builtins.open", side_effect=OSError)
    def test_grace_cpu_platform_rejects_non_grace_arm(
        self, _mock_file, mock_check_output, mock_machine
    ):
        mock_machine.return_value = "aarch64"
        mock_check_output.return_value = "Vendor ID: ARM\nModel name: Neoverse-V2\n"

        self.assertFalse(pin_h2d_copy._is_grace_cpu_platform())

    @patch("sglang.srt.model_loader.pin_h2d_copy.platform.machine")
    @patch("sglang.srt.model_loader.pin_h2d_copy.subprocess.check_output")
    def test_grace_cpu_platform_rejects_non_arm(self, mock_check_output, mock_machine):
        mock_machine.return_value = "x86_64"

        self.assertFalse(pin_h2d_copy._is_grace_cpu_platform())
        mock_check_output.assert_not_called()

    @patch("sglang.srt.model_loader.pin_h2d_copy._is_grace_cpu_platform")
    def test_context_does_not_patch_when_cuda_unavailable(self, mock_is_grace):
        fake_torch = _FakeTorch()
        original_copy = fake_torch.Tensor.copy_
        fake_torch.cuda.is_available = lambda: False
        mock_is_grace.return_value = True

        with patch.object(pin_h2d_copy, "torch", fake_torch):
            with pin_h2d_copy.pin_h2d_copy_during_load():
                self.assertIs(fake_torch.Tensor.copy_, original_copy)
            self.assertIs(fake_torch.Tensor.copy_, original_copy)

    @patch("sglang.srt.model_loader.pin_h2d_copy._is_grace_cpu_platform")
    def test_context_does_not_patch_when_not_grace(self, mock_is_grace):
        fake_torch = _FakeTorch()
        original_copy = fake_torch.Tensor.copy_
        mock_is_grace.return_value = False

        with patch.object(pin_h2d_copy, "torch", fake_torch):
            with pin_h2d_copy.pin_h2d_copy_during_load():
                self.assertIs(fake_torch.Tensor.copy_, original_copy)
            self.assertIs(fake_torch.Tensor.copy_, original_copy)

    @patch("sglang.srt.model_loader.pin_h2d_copy._is_grace_cpu_platform")
    def test_cpu_to_cuda_copy_uses_destination_device_stream_and_restores(
        self, mock_is_grace
    ):
        fake_torch = _FakeTorch()
        original_copy = fake_torch.Tensor.copy_
        mock_is_grace.return_value = True
        dst = _FakeTensor(is_cuda=True, device=_FakeDevice("cuda", 2))
        src = _FakeTensor(is_cuda=False, device=_FakeDevice("cpu"), pinned=False)

        with patch.object(pin_h2d_copy, "torch", fake_torch), patch.object(
            pin_h2d_copy, "_wload_pin_tls", threading.local()
        ):
            with pin_h2d_copy.pin_h2d_copy_during_load():
                self.assertIsNot(fake_torch.Tensor.copy_, original_copy)
                self.assertIs(dst.copy_(src), dst)
            self.assertIs(fake_torch.Tensor.copy_, original_copy)

        self.assertEqual(fake_torch.cuda.device_contexts, [2])
        self.assertEqual(fake_torch.cuda.stream_contexts, [2])
        self.assertEqual([stream.device_idx for stream in fake_torch.cuda.created_streams], [2])
        self.assertTrue(fake_torch.cuda.created_streams[0].synchronized)
        self.assertEqual(dst.copy_calls, [(src.pinned_tensor, (), {"non_blocking": True})])

    @patch("sglang.srt.model_loader.pin_h2d_copy._is_grace_cpu_platform")
    def test_pin_failure_falls_back_to_original_copy(self, mock_is_grace):
        fake_torch = _FakeTorch()
        mock_is_grace.return_value = True
        dst = _FakeTensor(is_cuda=True, device=_FakeDevice("cuda", 1))
        src = _FakeTensor(
            is_cuda=False, device=_FakeDevice("cpu"), pinned=False, pin_error=True
        )

        with patch.object(pin_h2d_copy, "torch", fake_torch), patch.object(
            pin_h2d_copy, "_wload_pin_tls", threading.local()
        ):
            with pin_h2d_copy.pin_h2d_copy_during_load():
                self.assertIs(dst.copy_(src), dst)

        self.assertEqual(dst.copy_calls, [(src, (), {})])

    def test_default_model_loader_wraps_load_weights_with_pin_context(self):
        events = []

        class DummyModel:
            def load_weights(self, weights):
                events.append(("load", list(weights)))

            def named_modules(self):
                return []

        @contextlib.contextmanager
        def fake_pin_context():
            events.append(("enter", None))
            try:
                yield
            finally:
                events.append(("exit", None))

        weights = [("weight", torch.empty(0))]
        with patch.object(loader_module, "is_cuda_alike", return_value=False), patch.object(
            loader_module, "pin_h2d_copy_during_load", fake_pin_context
        ):
            loader_module.DefaultModelLoader.load_weights_and_postprocess(
                DummyModel(), weights, torch.device("cpu")
            )

        self.assertEqual(events, [("enter", None), ("load", weights), ("exit", None)])


if __name__ == "__main__":
    unittest.main()
