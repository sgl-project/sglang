import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTargetHiddenKvInjector(CustomTestCase):
    def _make_injector(self) -> TargetHiddenKvInjector:
        draft_model = SimpleNamespace(write_target_hidden_kv=Mock())
        draft_model_runner = SimpleNamespace(token_to_kv_pool=SimpleNamespace())
        return TargetHiddenKvInjector(
            draft_model=draft_model,
            draft_model_runner=draft_model_runner,
            model_runner=SimpleNamespace(device="cpu"),
            device="cpu",
            verify_num_draft_tokens=2,
            block_pos_offsets=torch.arange(2),
        )

    def test_rejects_cp_local_hidden_with_global_indices(self):
        """A CP-local hidden shard must not be written with global cache indices."""
        injector = self._make_injector()

        with self.assertRaisesRegex(
            ValueError, "one hidden row and cache location per position"
        ):
            injector.inject_target_hidden(
                target_hidden=torch.zeros((2, 4)),
                cache_loc=torch.arange(4),
                positions=torch.arange(4),
            )


if __name__ == "__main__":
    unittest.main()
