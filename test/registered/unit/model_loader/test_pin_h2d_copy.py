"""Unit tests for model-load H2D copy pinning."""

import contextlib
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import torch

from sglang.srt.model_loader import loader as loader_module
from sglang.srt.model_loader import weight_utils as pin_h2d_copy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeDevice:
    def __init__(self, device_type, index=None):
        self.type = device_type
        self.index = index


class _FakeCuda:
    def __init__(self, available=True):
        self.available = available

    def is_available(self):
        return self.available


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

    def __init__(self, *, cuda_available=True):
        self.cuda = _FakeCuda(available=cuda_available)


class TestPinH2DCopy(CustomTestCase):
    def _run_with_fake_torch(self, fake_torch):
        return patch.object(pin_h2d_copy, "torch", fake_torch)

    def test_context_does_not_patch_when_cuda_unavailable(self):
        fake_torch = _FakeTorch(cuda_available=False)
        original_copy = fake_torch.Tensor.copy_

        with self._run_with_fake_torch(fake_torch):
            with pin_h2d_copy.pin_h2d_copy_during_load():
                self.assertIs(fake_torch.Tensor.copy_, original_copy)
            self.assertIs(fake_torch.Tensor.copy_, original_copy)

    def test_cpu_to_cuda_copy_pins_source_and_restores(self):
        fake_torch = _FakeTorch()
        original_copy = fake_torch.Tensor.copy_
        dst = _FakeTensor(is_cuda=True, device=_FakeDevice("cuda", 2))
        src = _FakeTensor(is_cuda=False, device=_FakeDevice("cpu"), pinned=False)

        with self._run_with_fake_torch(fake_torch):
            with pin_h2d_copy.pin_h2d_copy_during_load():
                self.assertIsNot(fake_torch.Tensor.copy_, original_copy)
                self.assertIs(dst.copy_(src), dst)
            self.assertIs(fake_torch.Tensor.copy_, original_copy)

        self.assertEqual(
            dst.copy_calls, [(src.pinned_tensor, (), {"non_blocking": True})]
        )

    def test_pin_failure_falls_back_to_original_copy(self):
        fake_torch = _FakeTorch()
        dst = _FakeTensor(is_cuda=True, device=_FakeDevice("cuda", 1))
        src = _FakeTensor(
            is_cuda=False, device=_FakeDevice("cpu"), pinned=False, pin_error=True
        )

        with self._run_with_fake_torch(fake_torch):
            with pin_h2d_copy.pin_h2d_copy_during_load():
                self.assertIs(dst.copy_(src), dst)

        self.assertEqual(dst.copy_calls, [(src, (), {})])

    def test_non_intercepted_copies_use_original_copy(self):
        fake_torch = _FakeTorch()
        cases = [
            (
                _FakeTensor(is_cuda=False, device=_FakeDevice("cpu")),
                _FakeTensor(is_cuda=False, device=_FakeDevice("cpu")),
            ),
            (
                _FakeTensor(is_cuda=True, device=_FakeDevice("cuda", 0)),
                _FakeTensor(is_cuda=True, device=_FakeDevice("cuda", 0)),
            ),
            (
                _FakeTensor(is_cuda=True, device=_FakeDevice("cuda", 0)),
                _FakeTensor(is_cuda=False, device=_FakeDevice("cpu"), pinned=True),
            ),
            (_FakeTensor(is_cuda=True, device=_FakeDevice("cuda", 0)), object()),
        ]

        with self._run_with_fake_torch(fake_torch):
            with pin_h2d_copy.pin_h2d_copy_during_load():
                for dst, src in cases:
                    self.assertIs(dst.copy_(src), dst)

        for dst, src in cases:
            self.assertEqual(dst.copy_calls, [(src, (), {})])

    def test_copy_restored_after_exception_in_context_body(self):
        fake_torch = _FakeTorch()
        original_copy = fake_torch.Tensor.copy_

        with self._run_with_fake_torch(fake_torch):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with pin_h2d_copy.pin_h2d_copy_during_load():
                    self.assertIsNot(fake_torch.Tensor.copy_, original_copy)
                    raise RuntimeError("boom")
            self.assertIs(fake_torch.Tensor.copy_, original_copy)

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
        with patch.object(
            loader_module, "is_cuda_alike", return_value=False
        ), patch.object(loader_module, "pin_h2d_copy_during_load", fake_pin_context):
            loader_module.DefaultModelLoader.load_weights_and_postprocess(
                DummyModel(), weights, torch.device("cpu")
            )

        self.assertEqual(events, [("enter", None), ("load", weights), ("exit", None)])


if __name__ == "__main__":
    unittest.main()
