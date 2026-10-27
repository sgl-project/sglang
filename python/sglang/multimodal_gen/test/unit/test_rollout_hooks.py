"""Unit tests for init_rollout_env_if_enabled's conditioning snapshot.

finalize_rollout_if_enabled always overwrites env.pos_cond_kwargs /
neg_cond_kwargs with the SP-gathered CPU copy before a trainer reads them, so
init must not pay for a CPU copy of its own that will just be discarded.
"""

import unittest
from types import SimpleNamespace

import torch

# Importing schedule_batch (pipelines_core) before rollout_hooks avoids a
# circular import: rollout_hooks imports schedule_batch, which pulls in the
# denoising stage, which imports rollout_hooks back -- only an issue when
# rollout_hooks is the first of the two to start initializing.
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (  # noqa: F401
    Req,
)
from sglang.multimodal_gen.runtime.post_training.rollout_hooks import (
    init_rollout_env_if_enabled,
)


def _rollout_batch(*, collect_env: bool, collect_traj: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        rollout=True,
        rollout_return_denoising_env=collect_env,
        rollout_return_dit_trajectory=collect_traj,
    )


class TestInitRolloutEnv(unittest.TestCase):
    def test_pos_neg_cond_kwargs_are_kept_as_is_not_copied_to_cpu(self):
        """finalize_rollout_if_enabled always replaces these with its own CPU
        copy; init copying them too would be a wasted host sync with no
        effect on the value a trainer ever reads."""
        batch = _rollout_batch(collect_env=True)
        pos = {"prompt_embeds": torch.zeros(2, 2)}
        neg = {"prompt_embeds": torch.ones(2, 2)}

        init_rollout_env_if_enabled(
            batch=batch,
            image_kwargs={},
            pos_cond_kwargs=pos,
            neg_cond_kwargs=neg,
            guidance=None,
        )

        env = batch._rollout_denoising_env_state["env"]
        self.assertIs(env.pos_cond_kwargs, pos)
        self.assertIs(env.neg_cond_kwargs, neg)
        self.assertIs(batch._rollout_denoising_env_state["pos_cond_kwargs_src"], pos)
        self.assertIs(batch._rollout_denoising_env_state["neg_cond_kwargs_src"], neg)

    def test_empty_neg_cond_kwargs_stay_none(self):
        batch = _rollout_batch(collect_env=True)
        init_rollout_env_if_enabled(
            batch=batch,
            image_kwargs={},
            pos_cond_kwargs={"a": torch.zeros(1)},
            neg_cond_kwargs={},
            guidance=None,
        )
        self.assertIsNone(batch._rollout_denoising_env_state["env"].neg_cond_kwargs)

    def test_neither_collect_flag_set_skips_state_entirely(self):
        batch = _rollout_batch(collect_env=False, collect_traj=False)
        init_rollout_env_if_enabled(
            batch=batch,
            image_kwargs={},
            pos_cond_kwargs={"a": torch.zeros(1)},
            neg_cond_kwargs={},
            guidance=None,
        )
        self.assertIsNone(batch._rollout_denoising_env_state)


if __name__ == "__main__":
    unittest.main()
