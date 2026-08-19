import unittest

import torch

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
    _select_output_rollout_trajectory,
)
from sglang.multimodal_gen.runtime.managers.gpu_worker import (
    _concat_rollout_trajectory_data,
)
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDitTrajectory,
    RolloutTrajectoryData,
)

_STEPS = 4


def _per_output_trajectory(output_index: int) -> RolloutTrajectoryData:
    """One per-output forward's trajectory: batch dim 1, values tagged by index."""
    tag = float(output_index + 1)
    return RolloutTrajectoryData(
        rollout_log_probs=torch.full((1, _STEPS), tag),
        dit_trajectory=RolloutDitTrajectory(
            latents=torch.full((1, _STEPS + 1, 2), tag),
            timesteps=torch.arange(_STEPS).float(),
            sigmas=torch.linspace(1.0, 0.0, _STEPS + 1),
        ),
    )


class TestMultiOutputRolloutTrajectory(unittest.TestCase):
    """Guards per-sample rollout trajectories across the multi-output merge.

    A ``num_outputs_per_prompt=K`` request runs as K per-output forwards, each
    producing its own trajectory. The merge used to keep only the first
    (``if merged.rollout_trajectory_data is None``) and the per-output result
    assembly never narrowed it, so all K samples reported output 0's log-probs.
    For GRPO that makes the group's advantages cancel and the gradient vanish.
    """

    def test_merge_concatenates_per_output_trajectories(self):
        merged = _concat_rollout_trajectory_data(
            [_per_output_trajectory(i) for i in range(3)]
        )

        self.assertEqual(tuple(merged.rollout_log_probs.shape), (3, _STEPS))
        self.assertEqual(
            [row[0].item() for row in merged.rollout_log_probs], [1.0, 2.0, 3.0]
        )
        self.assertEqual(tuple(merged.dit_trajectory.latents.shape), (3, _STEPS + 1, 2))

    def test_merge_keeps_group_shared_schedule_from_first_output(self):
        merged = _concat_rollout_trajectory_data(
            [_per_output_trajectory(i) for i in range(3)]
        )

        # timesteps/sigmas describe the shared schedule, so they must not gain a
        # batch dim when K outputs are merged.
        self.assertEqual(tuple(merged.dit_trajectory.timesteps.shape), (_STEPS,))
        self.assertEqual(tuple(merged.dit_trajectory.sigmas.shape), (_STEPS + 1,))

    def test_merge_drops_trajectory_when_only_some_outputs_have_one(self):
        """A partial group cannot be aligned to ``[K, ...]``, so drop it.

        Returning the one present row would silently label it as the whole
        group's trajectory -- the same broadcast bug in a different disguise.
        """
        partial = [_per_output_trajectory(0), None, _per_output_trajectory(2)]

        self.assertIsNone(_concat_rollout_trajectory_data(partial))

    def test_single_output_group_is_passed_through(self):
        only = _per_output_trajectory(0)

        merged = _concat_rollout_trajectory_data([only])

        self.assertIs(merged, only)

    def test_each_output_index_selects_its_own_row(self):
        merged = _concat_rollout_trajectory_data(
            [_per_output_trajectory(i) for i in range(3)]
        )

        selected = [
            _select_output_rollout_trajectory(merged, index) for index in range(3)
        ]

        values = [entry.rollout_log_probs.flatten()[0].item() for entry in selected]
        self.assertEqual(values, [1.0, 2.0, 3.0])
        for entry in selected:
            # Keep-dim so a grouped result matches an unexpanded one's shape.
            self.assertEqual(tuple(entry.rollout_log_probs.shape), (1, _STEPS))
            self.assertEqual(
                tuple(entry.dit_trajectory.latents.shape), (1, _STEPS + 1, 2)
            )

    def test_select_without_output_index_returns_group_trajectory(self):
        merged = _concat_rollout_trajectory_data(
            [_per_output_trajectory(i) for i in range(3)]
        )

        self.assertIs(_select_output_rollout_trajectory(merged, None), merged)

    def test_select_tolerates_missing_trajectory(self):
        self.assertIsNone(_select_output_rollout_trajectory(None, 0))


if __name__ == "__main__":
    unittest.main()
