"""Unit tests for the pi0-style continuous-action POC model."""

from types import SimpleNamespace

import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models.pi0_poc import Pi0ForActionPrediction
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestPi0Poc(CustomTestCase):
    def _model(self):
        config = SimpleNamespace(
            vocab_size=16,
            hidden_size=8,
            pi0_action_dim=4,
            pi0_action_horizon=3,
            pi0_num_inference_steps=5,
            pi0_dummy_token_id=2,
        )
        return Pi0ForActionPrediction(config)

    def test_forward_emits_dummy_logits_and_actions(self):
        model = self._model()
        forward_batch = SimpleNamespace(
            batch_size=2,
            seq_lens=[3, 2],
            extend_start_loc=torch.tensor([0, 3]),
            extend_seq_lens=torch.tensor([3, 2]),
            sampling_info=SimpleNamespace(
                custom_params=[
                    {"state": [0.0, 0.1], "num_inference_steps": 4, "seed": 11},
                    {"proprio_state": [1.0, 1.1, 1.2, 1.3], "seed": 13},
                ]
            ),
        )

        output = model(
            torch.tensor([1, 2, 3, 4, 5]),
            torch.arange(5),
            forward_batch,
        )

        self.assertEqual(output.next_token_logits.shape, (2, 16))
        self.assertEqual(output.next_token_logits[0].argmax().item(), 2)
        self.assertEqual(output.customized_info["action_horizon"], [3, 3])
        self.assertEqual(output.customized_info["action_dim"], [4, 4])
        self.assertEqual(output.customized_info["num_inference_steps"], [4, 5])
        self.assertEqual(len(output.customized_info["actions"]), 2)
        self.assertEqual(len(output.customized_info["actions"][0]), 3)
        self.assertEqual(len(output.customized_info["actions"][0][0]), 4)

    def test_forward_is_deterministic_for_same_request(self):
        model = self._model()
        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens=[3],
            sampling_info=SimpleNamespace(
                custom_params=[
                    {"state": [0.0, 0.1, 0.2, 0.3], "seed": 17},
                ]
            ),
        )
        input_ids = torch.tensor([1, 2, 3])

        output_a = model(input_ids, torch.arange(3), forward_batch)
        output_b = model(input_ids, torch.arange(3), forward_batch)

        self.assertEqual(
            output_a.customized_info["actions"],
            output_b.customized_info["actions"],
        )
