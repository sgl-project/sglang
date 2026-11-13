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

    def test_cache_limit_fails_before_constructing_collective(self):
        with (
            patch.object(fi, "_MAX_CONTEXTS", 0),
            patch.object(fi, "_Context") as context,
        ):
            with self.assertRaisesRegex(RuntimeError, "geometry cache is full"):
                fi._context(object(), torch.empty(1), 1)
            context.assert_not_called()

    def test_close_releases_every_communicator(self):
        contexts = [SimpleNamespace(comm=Mock()) for _ in range(2)]
        fi._CONTEXTS.update(enumerate(contexts))
        fi.close_all()
        for context in contexts:
            context.comm.close.assert_called_once_with()
        self.assertFalse(fi._CONTEXTS)


if __name__ == "__main__":
    unittest.main()
