import unittest

import torch

from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDenoisingEnv,
    RolloutDitTrajectory,
    RolloutTrajectoryData,
    concat_rollout_trajectory_data,
    select_output_rollout_trajectory,
)

_STEPS = 4


def _per_output_trajectory(
    output_index: int, with_env: bool = False
) -> RolloutTrajectoryData:
    """One per-output forward's trajectory: batch dim 1, values tagged by index."""
    tag = float(output_index + 1)
    env = None
    if with_env:
        env = RolloutDenoisingEnv(
            pos_cond_kwargs={
                "encoder_hidden_states": torch.full((1, 3), tag),
                # Qwen-Image layout: shared [seq, ch] RoPE caches + per-row lists.
                "freqs_cis": (torch.full((5, 4), tag), torch.full((6, 4), tag)),
                "freqs_complex": (torch.full((5, 2), tag), torch.full((6, 2), tag)),
                "txt_seq_lens": [3],
                "img_shapes": [[(1, 2, 2)]],
            },
            guidance=torch.full((1,), tag),
        )
    return RolloutTrajectoryData(
        rollout_log_probs=torch.full((1, _STEPS), tag),
        dit_trajectory=RolloutDitTrajectory(
            latents=torch.full((1, _STEPS + 1, 2), tag),
            timesteps=torch.arange(_STEPS).float(),
            sigmas=torch.linspace(1.0, 0.0, _STEPS + 1),
        ),
        denoising_env=env,
    )


class TestMultiOutputRolloutTrajectory(unittest.TestCase):
    """Guard per-sample rollout trajectories across the multi-output merge."""

    def test_merge_concatenates_per_output_trajectories(self):
        merged = concat_rollout_trajectory_data(
            [_per_output_trajectory(i) for i in range(3)]
        )

        self.assertEqual(tuple(merged.rollout_log_probs.shape), (3, _STEPS))
        self.assertEqual(
            [row[0].item() for row in merged.rollout_log_probs], [1.0, 2.0, 3.0]
        )
        self.assertEqual(tuple(merged.dit_trajectory.latents.shape), (3, _STEPS + 1, 2))

    def test_merge_keeps_group_shared_schedule_from_first_output(self):
        merged = concat_rollout_trajectory_data(
            [_per_output_trajectory(i) for i in range(3)]
        )

        # timesteps/sigmas describe the shared schedule, so they must not gain a
        # batch dim when K outputs are merged.
        self.assertEqual(tuple(merged.dit_trajectory.timesteps.shape), (_STEPS,))
        self.assertEqual(tuple(merged.dit_trajectory.sigmas.shape), (_STEPS + 1,))

    def test_merge_drops_trajectory_when_only_some_outputs_have_one(self):
        """Drop a partial trajectory group that cannot stay output-aligned."""
        partial = [_per_output_trajectory(0), None, _per_output_trajectory(2)]

        self.assertIsNone(concat_rollout_trajectory_data(partial))

    def test_single_output_group_is_passed_through(self):
        only = _per_output_trajectory(0)

        merged = concat_rollout_trajectory_data([only])

        self.assertIs(merged, only)

    def test_each_output_index_selects_its_own_row(self):
        merged = concat_rollout_trajectory_data(
            [_per_output_trajectory(i) for i in range(3)]
        )

        selected = [
            select_output_rollout_trajectory(merged, index) for index in range(3)
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
        merged = concat_rollout_trajectory_data(
            [_per_output_trajectory(i) for i in range(3)]
        )

        self.assertIs(select_output_rollout_trajectory(merged, None), merged)

    def test_select_tolerates_missing_trajectory(self):
        self.assertIsNone(select_output_rollout_trajectory(None, 0))

    def test_merge_concatenates_per_output_conditioning_env(self):
        """Dynamic batching must not collapse every output's conditioning onto
        output 0's (regression: concat used to keep only ``first.denoising_env``,
        so a trainer replaying request k's DiT got request 0's prompt)."""
        merged = concat_rollout_trajectory_data(
            [_per_output_trajectory(i, with_env=True) for i in range(3)]
        )

        env = merged.denoising_env
        self.assertEqual(
            env.pos_cond_kwargs["encoder_hidden_states"][:, 0].tolist(),
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(env.guidance.tolist(), [1.0, 2.0, 3.0])

    def test_select_output_index_slices_its_own_conditioning_env(self):
        merged = concat_rollout_trajectory_data(
            [_per_output_trajectory(i, with_env=True) for i in range(3)]
        )

        selected = [
            select_output_rollout_trajectory(merged, index) for index in range(3)
        ]

        values = [
            entry.denoising_env.pos_cond_kwargs["encoder_hidden_states"]
            .flatten()[0]
            .item()
            for entry in selected
        ]
        self.assertEqual(values, [1.0, 2.0, 3.0])
        self.assertEqual(
            [entry.denoising_env.guidance.item() for entry in selected],
            [1.0, 2.0, 3.0],
        )


class TestQwenImageSharedConditioning(unittest.TestCase):
    def _merged(self):
        return concat_rollout_trajectory_data(
            [_per_output_trajectory(i, with_env=True) for i in range(3)]
        )

    def test_concat_keeps_rope_caches_unbatched(self):
        kw = self._merged().denoising_env.pos_cond_kwargs
        self.assertEqual(tuple(kw["freqs_cis"][0].shape), (5, 4))
        self.assertEqual(tuple(kw["freqs_cis"][1].shape), (6, 4))
        self.assertEqual(tuple(kw["freqs_complex"][0].shape), (5, 2))
        self.assertEqual(kw["txt_seq_lens"], [3, 3, 3])
        self.assertEqual(len(kw["img_shapes"]), 3)

    def test_select_keeps_rope_caches_whole(self):
        for index in range(3):
            kw = select_output_rollout_trajectory(
                self._merged(), index
            ).denoising_env.pos_cond_kwargs
            self.assertEqual(tuple(kw["freqs_cis"][0].shape), (5, 4))
            self.assertEqual(tuple(kw["freqs_cis"][1].shape), (6, 4))
            self.assertEqual(tuple(kw["freqs_complex"][1].shape), (6, 2))
            self.assertEqual(kw["txt_seq_lens"], [3])
            self.assertEqual(len(kw["img_shapes"]), 1)
            self.assertEqual(tuple(kw["encoder_hidden_states"].shape), (1, 3))

    def test_single_output_with_env_keeps_rope_caches(self):
        only = _per_output_trajectory(0, with_env=True)
        kw = select_output_rollout_trajectory(only, 0).denoising_env.pos_cond_kwargs
        self.assertEqual(tuple(kw["freqs_cis"][0].shape), (5, 4))


if __name__ == "__main__":
    unittest.main()
