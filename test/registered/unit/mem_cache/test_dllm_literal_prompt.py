"""Literal mask IDs inside prompts must never be denoised or emitted."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.dllm.algorithm.joint_threshold import JointThreshold
from sglang.srt.dllm.algorithm.low_confidence import LowConfidence
from sglang.srt.dllm.config import DllmConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Runner:
    def forward(self, batch, **kwargs):
        logits = torch.zeros(4, 16)
        logits[:, 7] = 20
        return SimpleNamespace(
            logits_output=SimpleNamespace(full_logits=logits), can_run_graph=False
        )


class TestDllmLiteralPrompt(unittest.TestCase):
    def test_prompt_positions_are_immutable_in_both_algorithms(self):
        for cls, vectorized in (
            (LowConfidence, False),
            (JointThreshold, False),
            (JointThreshold, True),
        ):
            for fdfo in (False, True):
                for prompt in ([], [15, 2], [15, 2, 15, 3]):
                    with self.subTest(
                        algorithm=cls.__name__,
                        vectorized=vectorized,
                        fdfo=fdfo,
                        prompt=prompt,
                    ):
                        algorithm = cls(
                            DllmConfig(
                                cls.__name__,
                                {"vectorized_decoding": vectorized},
                                4,
                                15,
                                1,
                                fdfo,
                            )
                        )
                        ids = prompt + [15] * (4 - len(prompt))
                        batch = SimpleNamespace(
                            batch_size=1,
                            input_ids=torch.tensor(ids),
                            dllm_prompt_mask=(torch.arange(4) < len(prompt)).unsqueeze(
                                0
                            ),
                        )
                        self.assertEqual(
                            algorithm._block_start_list(batch), [len(prompt)]
                        )
                        states = None
                        for _ in range(32):
                            _, output, accepted, states, _ = algorithm.run(
                                _Runner(), batch, states
                            )
                            self.assertEqual(
                                batch.input_ids[: len(prompt)].tolist(), prompt
                            )
                            if not fdfo or accepted == [4]:
                                break
                        else:
                            self.fail("denoising did not finish")
                        self.assertEqual(
                            batch.input_ids.tolist(), prompt + [7] * (4 - len(prompt))
                        )
                        if not fdfo:
                            if len(prompt) == 4:
                                self.assertEqual(output, [])
                            else:
                                self.assertEqual(
                                    output[0].tolist(), [7] * (4 - len(prompt))
                                )


if __name__ == "__main__":
    unittest.main()
