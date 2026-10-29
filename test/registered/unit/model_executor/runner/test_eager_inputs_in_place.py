"""The eager runner's input load copies into its buffers unless its caller holds the
inputs at fixed addresses (eager_inputs_in_place), as EAGLE's draft graph steps do.
A capture alone does not skip the copy. Pure Python, so it runs on CPU CI."""

import unittest
from dataclasses import dataclass
from unittest import mock

import torch

from sglang.srt.model_executor.runner.eager_runner import (
    EagerRunner,
    eager_inputs_in_place,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@dataclass
class _Batch:
    batch_size: int
    input_ids: torch.Tensor
    input_embeds: torch.Tensor = None


class TestEagerInputsInPlace(CustomTestCase):
    def _runner(self):
        runner = EagerRunner.__new__(EagerRunner)
        runner._eager_registry = mock.Mock(name="registry")
        return runner

    def test_the_load_passes_inputs_through_only_inside_the_context(self):
        batch = _Batch(batch_size=2, input_ids=torch.arange(8))
        runner = self._runner()
        with eager_inputs_in_place():
            loaded = runner.load_batch(batch)
        runner._eager_registry.fill_from.assert_not_called()
        self.assertIsNot(loaded, batch)
        self.assertIs(loaded.input_ids, batch.input_ids)

        # outside it, and inside a capture alone, the inputs are copied as before
        for capturing in (False, True):
            with self.subTest(capturing=capturing):
                runner = self._runner()
                with mock.patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=capturing
                ):
                    runner.load_batch(batch)
                runner._eager_registry.fill_from.assert_called_once()

    def test_the_context_nests_and_restores(self):
        batch = _Batch(batch_size=1, input_ids=torch.arange(4))
        runner = self._runner()
        with eager_inputs_in_place():
            with eager_inputs_in_place():
                pass
            runner.load_batch(batch)
        runner._eager_registry.fill_from.assert_not_called()
        runner.load_batch(batch)
        runner._eager_registry.fill_from.assert_called_once()


if __name__ == "__main__":
    unittest.main()
