import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.runner.base_runner import BaseRunner
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPersistentCudaGraphPool(CustomTestCase):
    def test_model_runner_uses_nullcontext_without_pool(self):
        model_runner = ModelRunner.__new__(ModelRunner)
        model_runner.cuda_graph_persistent_pool = None

        context = model_runner.cuda_graph_persistent_pool_context()

        self.assertIsInstance(context, contextlib.nullcontext)

    def test_attention_state_initializes_in_selected_pool(self):
        events = []

        class RecordingContext:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, *_args):
                events.append("exit")

        pool = object()
        model_runner = ModelRunner.__new__(ModelRunner)
        model_runner.cuda_graph_persistent_pool = pool
        backend = mock.Mock()
        backend.init_cuda_graph_state.side_effect = lambda *_args: events.append(
            "initialize"
        )
        runner = SimpleNamespace(
            model_runner=model_runner,
            max_bs=8,
            max_num_token=128,
        )

        with mock.patch.object(
            torch.cuda, "use_mem_pool", return_value=RecordingContext()
        ) as use_mem_pool:
            DecodeCudaGraphRunner._init_attention_cuda_graph_state(runner, backend)

        use_mem_pool.assert_called_once_with(pool)
        backend.init_cuda_graph_state.assert_called_once_with(8, 128)
        self.assertEqual(events, ["enter", "initialize", "exit"])

    def test_shared_inputs_use_pool_except_externally_owned_storage(self):
        pool = object()
        buffers = mock.Mock()
        runner = SimpleNamespace(
            model_runner=SimpleNamespace(cuda_graph_persistent_pool=pool),
            buffers=buffers,
        )

        DecodeCudaGraphRunner._share_input_buffers(runner)

        buffers.share_buffers.assert_called_once_with(
            memory_pool=pool,
            memory_pool_exclusions=frozenset(
                {
                    "next_token_logits_buffer",
                    "ngram_embedding_info.token_table",
                }
            ),
        )

    def test_base_runner_constructs_tbo_plugin_in_selected_pool(self):
        events = []
        in_context = False

        class RecordingContext:
            def __enter__(self):
                nonlocal in_context
                in_context = True
                events.append("enter")

            def __exit__(self, *_args):
                nonlocal in_context
                in_context = False
                events.append("exit")

        def construct_plugin():
            self.assertTrue(in_context)
            events.append("construct")
            return object()

        model_runner = SimpleNamespace(
            device="cpu",
            server_args=SimpleNamespace(tp_size=1, pp_size=1, enable_pdmux=False),
            is_draft_worker=False,
            cuda_graph_persistent_pool_context=RecordingContext,
        )
        parallel = SimpleNamespace(
            tp_size=1,
            dp_size=1,
            pp_size=1,
            attn_tp_size=1,
            attn_tp_rank=0,
        )
        return_hidden_states_mode = SimpleNamespace(need_capture=lambda: False)
        runner = SimpleNamespace()

        with (
            mock.patch(
                "sglang.srt.model_executor.runner.base_runner.get_parallel",
                return_value=parallel,
            ),
            mock.patch(
                "sglang.srt.model_executor.runner.base_runner.get_disagg",
                return_value=SimpleNamespace(enable_pdmux=False),
            ),
            mock.patch(
                "sglang.srt.model_executor.runner.base_runner.get_server_return_hidden_states_mode",
                return_value=return_hidden_states_mode,
            ),
            mock.patch(
                "sglang.srt.model_executor.runner.base_runner.TboCudaGraphRunnerPlugin",
                side_effect=construct_plugin,
            ),
        ):
            BaseRunner.__init__(runner, model_runner)

        self.assertEqual(events, ["enter", "construct", "exit"])


if __name__ == "__main__":
    unittest.main()
