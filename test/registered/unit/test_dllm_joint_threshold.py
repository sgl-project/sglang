import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.dllm.algorithm.joint_threshold as joint_threshold_module
from sglang.srt.dllm.algorithm.joint_threshold import JointThreshold
from sglang.srt.dllm.config import DllmConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestJointThresholdVectorization(unittest.TestCase):
    @staticmethod
    def _config(
        *,
        block_size: int = 4,
        fdfo: bool = True,
        vectorized_decoding: bool | None = None,
        threshold: float = 0.5,
        edit_threshold: float = 0,
        max_post_edit_steps: int = 2,
        penalty_lambda: float = 0.25,
    ) -> DllmConfig:
        algorithm_config = {
            "threshold": threshold,
            "edit_threshold": edit_threshold,
            "max_post_edit_steps": max_post_edit_steps,
            "penalty_lambda": penalty_lambda,
        }
        if vectorized_decoding is not None:
            algorithm_config["vectorized_decoding"] = vectorized_decoding
        return DllmConfig(
            algorithm="JointThreshold",
            algorithm_config=algorithm_config,
            block_size=block_size,
            mask_id=66,
            max_running_requests=4,
            first_done_first_out_mode=fdfo,
        )

    @staticmethod
    def _input_ids(block_size: int) -> torch.Tensor:
        return torch.tensor(
            [66] * block_size
            + list(range(block_size // 2))
            + [66] * (block_size - block_size // 2)
            + list(range(block_size))
            + [66] * block_size,
            dtype=torch.int64,
        )

    @staticmethod
    def _logits(
        block_size: int, step: int, batch_size: int = 4, seed: int = 1000
    ) -> torch.Tensor:
        vocab_size = 67
        generator = torch.Generator().manual_seed(seed + block_size + step)
        logits = torch.randn(
            batch_size * block_size,
            vocab_size,
            generator=generator,
            dtype=torch.float32,
        )
        logits[:, 66] = -100

        positions = torch.arange(batch_size * block_size)
        winners = (positions + step + 1) % (vocab_size - 1)
        logits[positions, winners] += torch.where(
            positions < block_size,
            torch.tensor(8.0),
            torch.tensor(2.0),
        )
        return logits

    @staticmethod
    def _random_input_ids(
        *, block_size: int, batch_size: int, seed: int
    ) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        input_ids = torch.randint(
            low=0,
            high=66,
            size=(batch_size, block_size),
            generator=generator,
            dtype=torch.int64,
        )
        mask_positions = (
            torch.rand(
                batch_size,
                block_size,
                generator=generator,
            )
            < 0.6
        )
        input_ids[mask_positions] = 66
        input_ids[0] = 66
        if batch_size > 1:
            input_ids[-1] = torch.arange(block_size) % 66
        return input_ids.flatten()

    @staticmethod
    def _state_values(states, *, vectorized: bool, fdfo: bool):
        if vectorized and not fdfo:
            shared = states[0]
            return (
                shared["finished"].tolist(),
                shared["post_edit_steps"].tolist(),
            )
        return (
            [state["finished"] for state in states],
            [state["post_edit_steps"] for state in states],
        )

    @staticmethod
    def _mark_last_row_finished(states, *, vectorized: bool, fdfo: bool):
        if vectorized and not fdfo:
            states[0]["finished"][-1] = True
        else:
            states[-1]["finished"] = True

    @staticmethod
    def _mark_rows_finished(states, rows: list[int], *, vectorized: bool, fdfo: bool):
        if vectorized and not fdfo:
            states[0]["finished"][rows] = True
        else:
            for row in rows:
                states[row]["finished"] = True

    def test_cuda_and_npu_default_to_vectorized_decoding(self):
        with (
            patch.object(joint_threshold_module, "_is_cuda", True),
            patch.object(joint_threshold_module, "_is_npu", False),
        ):
            self.assertTrue(JointThreshold(self._config()).vectorized_decoding)

        with (
            patch.object(joint_threshold_module, "_is_cuda", False),
            patch.object(joint_threshold_module, "_is_npu", False),
        ):
            self.assertFalse(JointThreshold(self._config()).vectorized_decoding)

        with (
            patch.object(joint_threshold_module, "_is_cuda", False),
            patch.object(joint_threshold_module, "_is_npu", True),
        ):
            self.assertTrue(JointThreshold(self._config()).vectorized_decoding)

    def test_explicit_vectorization_override_is_preserved(self):
        with (
            patch.object(joint_threshold_module, "_is_cuda", True),
            patch.object(joint_threshold_module, "_is_npu", False),
        ):
            self.assertFalse(
                JointThreshold(
                    self._config(vectorized_decoding=False)
                ).vectorized_decoding
            )

        with (
            patch.object(joint_threshold_module, "_is_cuda", False),
            patch.object(joint_threshold_module, "_is_npu", False),
        ):
            self.assertTrue(
                JointThreshold(
                    self._config(vectorized_decoding=True)
                ).vectorized_decoding
            )

    def test_vectorized_steps_match_per_row_steps(self):
        for fdfo in (False, True):
            for block_size in (4, 32):
                with self.subTest(fdfo=fdfo, block_size=block_size):
                    per_row = JointThreshold(
                        self._config(
                            block_size=block_size,
                            fdfo=fdfo,
                            vectorized_decoding=False,
                        )
                    )
                    vectorized = JointThreshold(
                        self._config(
                            block_size=block_size,
                            fdfo=fdfo,
                            vectorized_decoding=True,
                        )
                    )

                    input_ids = self._input_ids(block_size)
                    per_row_batch = SimpleNamespace(
                        batch_size=4, input_ids=input_ids.clone()
                    )
                    vectorized_batch = SimpleNamespace(
                        batch_size=4, input_ids=input_ids.clone()
                    )
                    per_row_states = per_row.init_step_state(per_row_batch)
                    vectorized_states = vectorized.init_step_state(vectorized_batch)
                    self._mark_last_row_finished(
                        per_row_states, vectorized=False, fdfo=fdfo
                    )
                    self._mark_last_row_finished(
                        vectorized_states, vectorized=True, fdfo=fdfo
                    )

                    for step in range(4):
                        logits = self._logits(block_size, step)
                        per_row_done = per_row.step(
                            per_row_batch, logits.clone(), per_row_states
                        )
                        vectorized_done = vectorized.step(
                            vectorized_batch, logits.clone(), vectorized_states
                        )

                        self.assertEqual(vectorized_done, per_row_done)
                        self.assertTrue(
                            torch.equal(
                                vectorized_batch.input_ids,
                                per_row_batch.input_ids,
                            )
                        )
                        self.assertEqual(
                            self._state_values(
                                vectorized_states,
                                vectorized=True,
                                fdfo=fdfo,
                            ),
                            self._state_values(
                                per_row_states,
                                vectorized=False,
                                fdfo=fdfo,
                            ),
                        )

    def test_vectorized_steps_match_per_row_across_stress_matrix(self):
        cases = (
            # batch, block, FDFO, threshold, edit threshold, penalty, seed
            (1, 4, False, 0.0, 0.0, 0.0, 11),
            (1, 32, True, 0.95, 0.95, 3.0, 12),
            (2, 4, True, 0.5, 0.0, 0.25, 13),
            (2, 32, False, 0.95, 0.5, 3.0, 14),
            (7, 4, False, 0.0, 0.95, 0.25, 15),
            (7, 32, True, 0.5, 0.5, 0.0, 16),
        )
        for (
            batch_size,
            block_size,
            fdfo,
            threshold,
            edit_threshold,
            penalty_lambda,
            seed,
        ) in cases:
            with self.subTest(
                batch_size=batch_size,
                block_size=block_size,
                fdfo=fdfo,
                threshold=threshold,
                edit_threshold=edit_threshold,
                penalty_lambda=penalty_lambda,
            ):
                config_args = {
                    "block_size": block_size,
                    "fdfo": fdfo,
                    "threshold": threshold,
                    "edit_threshold": edit_threshold,
                    "max_post_edit_steps": 3,
                    "penalty_lambda": penalty_lambda,
                }
                per_row = JointThreshold(
                    self._config(
                        **config_args,
                        vectorized_decoding=False,
                    )
                )
                vectorized = JointThreshold(
                    self._config(
                        **config_args,
                        vectorized_decoding=True,
                    )
                )

                input_ids = self._random_input_ids(
                    block_size=block_size,
                    batch_size=batch_size,
                    seed=seed,
                )
                per_row_batch = SimpleNamespace(
                    batch_size=batch_size, input_ids=input_ids.clone()
                )
                vectorized_batch = SimpleNamespace(
                    batch_size=batch_size, input_ids=input_ids.clone()
                )
                per_row_states = per_row.init_step_state(per_row_batch)
                vectorized_states = vectorized.init_step_state(vectorized_batch)

                finished_rows = [
                    row for row in range(batch_size) if (row + seed) % 5 == 0
                ]
                self._mark_rows_finished(
                    per_row_states,
                    finished_rows,
                    vectorized=False,
                    fdfo=fdfo,
                )
                self._mark_rows_finished(
                    vectorized_states,
                    finished_rows,
                    vectorized=True,
                    fdfo=fdfo,
                )

                for step in range(block_size + 5):
                    logits = self._logits(
                        block_size,
                        step,
                        batch_size=batch_size,
                        seed=seed,
                    )
                    if step == 0 and seed % 2:
                        logits.zero_()
                        logits[:, 66] = -100

                    per_row_done = per_row.step(
                        per_row_batch, logits.clone(), per_row_states
                    )
                    vectorized_done = vectorized.step(
                        vectorized_batch, logits.clone(), vectorized_states
                    )

                    self.assertEqual(vectorized_done, per_row_done)
                    self.assertTrue(
                        torch.equal(
                            vectorized_batch.input_ids,
                            per_row_batch.input_ids,
                        )
                    )
                    self.assertEqual(
                        self._state_values(
                            vectorized_states,
                            vectorized=True,
                            fdfo=fdfo,
                        ),
                        self._state_values(
                            per_row_states,
                            vectorized=False,
                            fdfo=fdfo,
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
