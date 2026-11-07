import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.dllm.algorithm.nemotron_linear_spec import NemotronLinearSpec
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_mode import ForwardMode

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNemotronLinearSpec(CustomTestCase):
    def test_verification_commits_only_matching_prefix_and_carries_correction(self):
        algorithm = NemotronLinearSpec(
            DllmConfig(
                algorithm="NemotronLinearSpec",
                algorithm_config={},
                block_size=4,
                mask_id=0,
                max_running_requests=3,
                first_done_first_out_mode=True,
                requires_separate_context_encoding=True,
                retains_causal_prefix=True,
            )
        )
        embedding = torch.nn.Embedding(8, 2)
        context_tokens = torch.tensor([4, 5, 6])
        draft_tokens = torch.tensor([[7, 1, 2, 3]] * 3)
        verify_tokens = torch.tensor([[6, 2, 3, 7], [1, 6, 3, 7], [1, 2, 3, 7]])
        phases = []

        def logits(tokens):
            return torch.nn.functional.one_hot(tokens, num_classes=8).float() * 20

        def forward(batch, pp_proxy_tensors=None):
            if not batch.forward_mode.is_dllm_extend():
                output = LogitsProcessorOutput(next_token_logits=logits(context_tokens))
            else:
                phases.append(batch.dllm_is_causal)
                torch.testing.assert_close(
                    batch.input_embeds, embedding(batch.input_ids)
                )
                if batch.dllm_is_causal:
                    self.assertEqual(
                        batch.input_ids.view(3, 4).tolist(),
                        [[4, 1, 2, 3], [5, 1, 2, 3], [6, 1, 2, 3]],
                    )
                    tokens = verify_tokens
                else:
                    self.assertEqual(
                        batch.input_ids.view(3, 4).tolist(),
                        [[4, 0, 0, 0], [5, 0, 0, 0], [6, 0, 0, 0]],
                    )
                    tokens = draft_tokens
                output = LogitsProcessorOutput(
                    next_token_logits=None, full_logits=logits(tokens).flatten(0, 1)
                )
            return SimpleNamespace(logits_output=output, can_run_graph=False)

        runner = SimpleNamespace(
            forward=forward,
            model=SimpleNamespace(get_input_embeddings=lambda: embedding),
        )
        context = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        states = algorithm.run(runner, context)[3]
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DLLM_EXTEND,
            input_ids=torch.zeros(12, dtype=torch.int64),
            batch_size=3,
            dllm_is_causal=False,
        )
        result = algorithm.run(runner, batch, states)
        self.assertEqual(phases, [False, True])
        self.assertEqual(result[2], [1, 2, 4])
        self.assertEqual([state["seed"].item() for state in result[3]], [6, 6, 7])
        self.assertFalse(batch.dllm_is_causal)

    def test_verification_failure_restores_draft_attention_phase(self):
        algorithm = NemotronLinearSpec(
            DllmConfig(
                algorithm="NemotronLinearSpec",
                algorithm_config={},
                block_size=2,
                mask_id=0,
                max_running_requests=1,
                first_done_first_out_mode=True,
            )
        )
        embedding = torch.nn.Embedding(3, 2)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DLLM_EXTEND,
            input_ids=torch.zeros(2, dtype=torch.int64),
            batch_size=1,
            dllm_is_causal=False,
        )

        def forward(batch, pp_proxy_tensors=None):
            if batch.dllm_is_causal:
                raise RuntimeError("verification failed")
            return SimpleNamespace(
                logits_output=LogitsProcessorOutput(
                    next_token_logits=None,
                    full_logits=torch.tensor([[0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]),
                ),
                can_run_graph=False,
            )

        runner = SimpleNamespace(
            forward=forward,
            model=SimpleNamespace(get_input_embeddings=lambda: embedding),
        )
        with self.assertRaisesRegex(RuntimeError, "verification failed"):
            algorithm.run(
                runner, batch, [{"seed": torch.tensor(1), "started": False, "steps": 0}]
            )
        self.assertFalse(batch.dllm_is_causal)


if __name__ == "__main__":
    unittest.main()
