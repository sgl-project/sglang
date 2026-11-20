# SPDX-License-Identifier: Apache-2.0
"""CPU guards and ownership tests for the optional FlashInfer adapter."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.runtime.distributed.device_communicators import (
    flashinfer_ulysses as fi,
)


class TestFlashInferUlysses(unittest.TestCase):
    def tearDown(self):
        fi._CONTEXTS.clear()

    def test_disabled_does_not_initialize_flashinfer(self):
        with (
            patch.object(fi, "enabled", return_value=False),
            patch.object(fi, "_Context") as context,
        ):
            self.assertIsNone(fi.exchange(torch.empty(1, 2, 4, 8), None, scatter=True))
            self.assertIsNone(
                fi.exchange_qkv(*(torch.empty(2, 4, 8) for _ in range(3)), None)
            )
            context.assert_not_called()

    def test_cpu_falls_back_even_when_enabled(self):
        with (
            patch.object(fi, "enabled", return_value=True),
            patch.object(fi, "_Context") as context,
        ):
            self.assertIsNone(fi.exchange(torch.empty(1, 2, 4, 8), None, scatter=True))
            context.assert_not_called()

    def test_compile_autograd_and_capture_do_not_initialize(self):
        fake = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16, ndim=4)
        for grad, compiling, capturing in (
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ):
            with (
                patch.object(fi, "enabled", return_value=True),
                patch.object(torch.version, "hip", None),
                patch.object(torch, "is_grad_enabled", return_value=grad),
                patch.object(torch.compiler, "is_compiling", return_value=compiling),
                patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=capturing
                ),
                patch.object(fi, "_Context") as context,
            ):
                self.assertIsNone(fi.exchange(fake, None, scatter=True))
                context.assert_not_called()

    def test_pcie_results_survive_subsequent_exchange(self):
        context = fi._Context.__new__(fi._Context)
        registered = torch.empty(1, 4, 2, 8)

        def scatter(x, out):
            return out.fill_(x.flatten()[0])

        context.comm = SimpleNamespace(
            backend="pcie",
            allocate_output=Mock(return_value=registered),
            scatter_heads=scatter,
        )
        context.outputs = {}
        first = context.exchange(torch.ones(1, 2, 4, 8), "scatter_heads")
        second = context.exchange(torch.full((1, 2, 4, 8), 2.0), "scatter_heads")
        self.assertTrue(torch.equal(first, torch.ones_like(first)))
        self.assertTrue(torch.equal(second, torch.full_like(second, 2)))
        context.comm.allocate_output.assert_called_once()

    def test_current_geometry_is_reused_per_operation(self):
        group, x = object(), torch.empty(1, 2, 4, 8)
        with patch.object(fi, "_Context", side_effect=lambda *args: Mock()) as create:
            qkv = fi._context(group, x, 3 * x.numel(), "qkv")
            output = fi._context(group, x, x.numel(), "gather_heads")
            self.assertIs(fi._context(group, x, 3 * x.numel(), "qkv"), qkv)
            self.assertIs(fi._context(group, x, x.numel(), "gather_heads"), output)
            self.assertEqual(create.call_count, 2)
            qkv.close.assert_not_called()
            output.close.assert_not_called()

    def test_more_than_nine_lengths_retire_before_collective_replacement(self):
        group, events, created = object(), [], []

        def create(group, x, numel):
            events.append("init")
            context = Mock()
            context.close.side_effect = lambda: events.append("close")
            created.append(context)
            return context

        with patch.object(fi, "_Context", side_effect=create):
            for length in range(1, 25):
                x = torch.empty(1, length, 4, 8)
                fi._context(group, x, 3 * x.numel(), "qkv")
                fi._context(group, x, x.numel(), "gather_heads")
                self.assertEqual(len(fi._CONTEXTS), 2)
        self.assertEqual(events[:2], ["init", "init"])
        self.assertEqual(events[2:], ["close", "init"] * 46)
        for context in created[:-2]:
            context.close.assert_called_once_with()

    def test_equal_payload_size_with_different_shape_retires_registration(self):
        group = object()
        with patch.object(fi, "_Context", side_effect=lambda *args: Mock()):
            first = fi._context(group, torch.empty(1, 2, 4, 8), 64, "scatter_heads")
            second = fi._context(group, torch.empty(1, 4, 2, 8), 64, "scatter_heads")
        self.assertIsNot(first, second)
        first.close.assert_called_once_with()
        self.assertEqual(len(fi._CONTEXTS), 1)

    def test_failed_replacement_is_not_cached_and_can_be_retried(self):
        group, first, replacement = object(), Mock(), Mock()
        with patch.object(
            fi,
            "_Context",
            side_effect=[first, RuntimeError("init failed"), replacement],
        ):
            fi._context(group, torch.empty(1), 1, "qkv")
            with self.assertRaisesRegex(RuntimeError, "init failed"):
                fi._context(group, torch.empty(2), 2, "qkv")
            self.assertFalse(fi._CONTEXTS)
            self.assertIs(fi._context(group, torch.empty(2), 2, "qkv"), replacement)
        first.close.assert_called_once_with()

    def test_pcie_results_survive_geometry_retirement(self):
        context_type = fi._Context

        def create(group, x, numel):
            context = context_type.__new__(context_type)
            context.comm = SimpleNamespace(
                backend="pcie",
                allocate_output=lambda x, op: torch.empty_like(x),
                scatter_heads=lambda x, out: out.copy_(x),
                close=Mock(),
            )
            context.outputs, context.workspace = {}, object()
            return context

        group = object()
        with patch.object(fi, "_Context", side_effect=create):
            x = torch.ones(1, 2, 4, 8)
            old = fi._context(group, x, x.numel(), "scatter_heads")
            retained = old.exchange(x, "scatter_heads")
            y = torch.full((1, 3, 4, 8), 2.0)
            current = fi._context(group, y, y.numel(), "scatter_heads")
            current.exchange(y, "scatter_heads")
        torch.testing.assert_close(retained, x, rtol=0, atol=0)
        old.comm.close.assert_called_once_with()
        self.assertFalse(old.outputs)
        self.assertIsNone(old.workspace)

    def test_failed_collective_close_remains_available_for_retry(self):
        group, first, replacement = object(), Mock(), Mock()
        first.close.side_effect = [RuntimeError("close failed"), None]
        with patch.object(fi, "_Context", side_effect=[first, replacement]):
            fi._context(group, torch.empty(1), 1, "qkv")
            with self.assertRaisesRegex(RuntimeError, "close failed"):
                fi._context(group, torch.empty(2), 2, "qkv")
            self.assertIn(first, fi._CONTEXTS.values())
            self.assertIs(fi._context(group, torch.empty(2), 2, "qkv"), replacement)
        self.assertEqual(first.close.call_count, 2)

    def test_close_releases_every_communicator(self):
        contexts = [Mock() for _ in range(2)]
        fi._CONTEXTS.update(enumerate(contexts))
        fi.close_all()
        for context in contexts:
            context.close.assert_called_once_with()
        self.assertFalse(fi._CONTEXTS)


if __name__ == "__main__":
    unittest.main()
