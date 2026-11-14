import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.dllm.algorithm.nemotron_fast_diffuser import NemotronFastDiffuser
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_mode import ForwardMode

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNemotronFastDiffuser(CustomTestCase):
    def test_launch_keeps_graph_choices_and_disables_prefix_caching(self):
        for decode, prefill in (
            (Backend.FULL, Backend.BREAKABLE),
            (Backend.DISABLED, Backend.FULL),
            (Backend.DISABLED, Backend.DISABLED),
        ):
            with self.subTest(decode=decode, prefill=prefill):
                args = SimpleNamespace(
                    device="cuda",
                    dllm_fdfo=True,
                    pp_size=1,
                    dcp_size=1,
                    attn_cp_size=1,
                    enable_dp_attention=False,
                    enable_lora=False,
                    _model_config=SimpleNamespace(quantization=None),
                    speculative_algorithm=None,
                    enable_two_batch_overlap=False,
                    cuda_graph_config=CudaGraphConfig(
                        decode=PhaseConfig(backend=decode),
                        prefill=PhaseConfig(backend=prefill),
                    ),
                    disable_radix_cache=False,
                    chunked_prefill_size=128,
                )
                NemotronFastDiffuser.configure_server_args(args)
                resolved = resolving_view(args)
                self.assertEqual(resolved.cuda_graph_config.decode.backend, decode)
                self.assertEqual(resolved.cuda_graph_config.prefill.backend, prefill)
                self.assertTrue(resolved.disable_radix_cache)
                self.assertEqual(resolved.chunked_prefill_size, -1)

    def test_context_seeds_survive_scheduler_handoff_and_batched_denoising(self):
        algorithm = NemotronFastDiffuser(
            DllmConfig(
                algorithm="NemotronFastDiffuser",
                algorithm_config={"threshold": 0.9},
                block_size=4,
                mask_id=3,
                max_running_requests=2,
                first_done_first_out_mode=True,
                requires_separate_context_encoding=True,
            )
        )
        context_logits = torch.tensor([[0.0, 0.0, 5.0, -5.0], [5.0, 0.0, 0.0, -5.0]])
        block_logits = torch.tensor(
            [
                [
                    [0.0, 5.0, 0.0, -10.0],
                    [4.0, 0.0, 0.0, -10.0],
                    [0.0, 4.0, 0.0, -10.0],
                    [0.0, 0.0, 1.0, -10.0],
                ],
                [
                    [0.0, 0.0, 4.0, -10.0],
                    [0.0, 1.0, 0.0, -10.0],
                    [0.0, 0.0, 2.0, -10.0],
                    [4.0, 0.0, 0.0, -10.0],
                ],
            ]
        )

        def forward(batch, pp_proxy_tensors=None):
            output = (
                LogitsProcessorOutput(
                    full_logits=block_logits[-batch.batch_size :].flatten(0, 1),
                    next_token_logits=None,
                )
                if batch.forward_mode.is_dllm_extend()
                else LogitsProcessorOutput(next_token_logits=context_logits)
            )
            return SimpleNamespace(logits_output=output, can_run_graph=False)

        embedding = torch.nn.Embedding(7, 2)
        runner = SimpleNamespace(
            forward=forward,
            model=SimpleNamespace(get_input_embeddings=lambda: embedding),
        )
        context_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND, input_ids=torch.tensor([5, 6])
        )
        result = algorithm.run(runner, context_batch)
        requests = [SimpleNamespace(dllm_algo_state=None) for _ in range(2)]
        scheduler = SimpleNamespace(
            dllm_config=SimpleNamespace(requires_separate_context_encoding=True),
            metrics_reporter=SimpleNamespace(report_prefill_stats=Mock()),
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            reqs=requests,
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        SchedulerDllmMixin.process_batch_result_dllm(
            scheduler,
            batch,
            SimpleNamespace(
                copy_done=None,
                dllm_algo_state=result[3],
                can_run_cuda_graph=False,
            ),
        )
        states = [request.dllm_algo_state for request in requests]
        inputs = torch.zeros(8, dtype=torch.long)
        denoise_batch = SimpleNamespace(
            forward_mode=ForwardMode.DLLM_EXTEND, input_ids=inputs, batch_size=2
        )
        for expected in (
            [[2, 0, 1, 3], [0, 3, 3, 0]],
            [[2, 0, 1, 2], [0, 3, 2, 0]],
        ):
            output = algorithm.run(runner, denoise_batch, states)
            self.assertEqual(inputs.view(2, 4).tolist(), expected)
            states = output[3]
        self.assertEqual(output[2], [4, 0])
        self.assertIsNone(states[0])
        remaining = SimpleNamespace(
            forward_mode=ForwardMode.DLLM_EXTEND,
            input_ids=inputs.view(2, 4)[1].clone(),
            batch_size=1,
        )
        final = algorithm.run(runner, remaining, states[1:])
        self.assertEqual(final[2], [4])
        self.assertEqual(remaining.input_ids.tolist(), [0, 1, 2, 0])


if __name__ == "__main__":
    unittest.main()
