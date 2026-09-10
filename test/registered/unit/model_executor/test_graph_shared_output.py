import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor import graph_shared_output
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGraphSharedOutput(CustomTestCase):
    def setUp(self) -> None:
        GraphSharedOutput._process_shared = None

    def tearDown(self) -> None:
        GraphSharedOutput._process_shared = None

    @staticmethod
    def _model_runner(*, memory_pool, max_rows: int = 8):
        return SimpleNamespace(
            device="cuda",
            cuda_graph_persistent_pool=memory_pool,
            server_args=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(backend=Backend.FULL, bs=[1])
                )
            ),
            max_decode_logits_rows=mock.Mock(return_value=max_rows),
        )

    def test_logits_buffer_allocates_in_selected_pool(self) -> None:
        pool = object()
        allocation_context = mock.MagicMock()
        backing = torch.empty((8, 16))

        def allocate(*_args, **_kwargs):
            allocation_context.__enter__.assert_called_once_with()
            allocation_context.__exit__.assert_not_called()
            return backing

        output = GraphSharedOutput(
            device=torch.device("cuda"),
            max_rows=8,
            memory_pool=pool,
        )
        with (
            mock.patch.object(
                torch.cuda, "use_mem_pool", return_value=allocation_context
            ) as use_mem_pool,
            mock.patch.object(torch, "zeros", side_effect=allocate) as zeros,
        ):
            result = output.get_logits_buffer(16, rows=3)

        use_mem_pool.assert_called_once_with(pool)
        zeros.assert_called_once_with(
            (8, 16), dtype=torch.float, device=torch.device("cuda")
        )
        allocation_context.__exit__.assert_called_once()
        self.assertEqual(result.shape, (3, 16))
        self.assertEqual(result.data_ptr(), backing.data_ptr())

    def test_logits_buffer_uses_nullcontext_without_pool(self) -> None:
        allocation_context = contextlib.nullcontext()
        backing = torch.empty((4, 12))
        output = GraphSharedOutput(device=torch.device("cpu"), max_rows=4)

        with (
            mock.patch.object(
                graph_shared_output.contextlib,
                "nullcontext",
                return_value=allocation_context,
            ) as nullcontext,
            mock.patch.object(torch.cuda, "use_mem_pool") as use_mem_pool,
            mock.patch.object(torch, "zeros", return_value=backing),
        ):
            result = output.get_logits_buffer(12, rows=2)

        nullcontext.assert_called_once_with()
        use_mem_pool.assert_not_called()
        self.assertEqual(result.shape, (2, 12))
        self.assertEqual(result.data_ptr(), backing.data_ptr())

    def test_process_shared_reuse_requires_same_pool(self) -> None:
        config = SimpleNamespace(decode=SimpleNamespace(backend=Backend.FULL, bs=[1]))
        with mock.patch.object(
            graph_shared_output,
            "get_exec",
            return_value=SimpleNamespace(
                graph=SimpleNamespace(cuda_graph_config=config)
            ),
        ):
            first_pool = object()
            first = GraphSharedOutput.create_for_model_runner(
                self._model_runner(memory_pool=first_pool)
            )
            reused = GraphSharedOutput.create_for_model_runner(
                self._model_runner(memory_pool=first_pool, max_rows=4)
            )
            replacement_pool = object()
            replaced = GraphSharedOutput.create_for_model_runner(
                self._model_runner(memory_pool=replacement_pool, max_rows=4)
            )

        self.assertIs(reused, first)
        self.assertIsNot(replaced, first)
        self.assertIs(replaced.memory_pool, replacement_pool)


if __name__ == "__main__":
    unittest.main()
