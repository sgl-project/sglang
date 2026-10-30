import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.speculative.dspark_components.dspark_verify import (
    TargetVerifyExecutor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSparkHiSparseHooks(CustomTestCase):
    def test_static_verify_binds_hisparse_slots_before_the_target_forward(self):
        events = []
        verify_cache_loc = torch.arange(100, 108)
        coordinator = MagicMock()
        coordinator.prepare_spec_verify.side_effect = lambda batch: events.append(
            ("prepare", batch.out_cache_loc.clone())
        )
        batch = SimpleNamespace(
            out_cache_loc=torch.zeros(8, dtype=torch.int64),
            seq_lens_cpu=None,
            seq_lens_sum=None,
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            hisparse_coordinator=coordinator,
        )
        executor = object.__new__(TargetVerifyExecutor)
        executor.verify_num_draft_tokens = 4
        executor._verify_backend_self_adds_seq_lens = lambda: True

        def forward(**kwargs):
            events.append(("verify", None))
            return SimpleNamespace(logits_output=None)

        executor._forward_prepared_verify = forward
        executor.run_non_compact(
            batch=batch,
            draft_input=SimpleNamespace(nxt_kv_lens_cpu=None),
            verify_ids_2d=torch.zeros((2, 4), dtype=torch.int64),
            verify_window=SimpleNamespace(
                positions_2d=torch.zeros((2, 4), dtype=torch.int64),
                verify_cache_loc=verify_cache_loc,
                kv_loc_plan=None,
                iteration_ids=None,
            ),
            sampling_info=None,
        )

        self.assertEqual([name for name, _ in events], ["prepare", "verify"])
        torch.testing.assert_close(events[0][1], verify_cache_loc)

    def test_prefix_accept_lens_map_to_verify_slot_indices(self):
        coordinator = object.__new__(HiSparseCoordinator)
        coordinator.spec_swap = MagicMock(num_draft_tokens=4)
        batch = object()

        coordinator.commit_spec_accept_lens(batch, torch.tensor([3, 1, 4]))

        call = coordinator.spec_swap.commit_accept_tokens.call_args
        self.assertIs(call.args[0], batch)
        torch.testing.assert_close(
            call.args[1],
            torch.tensor([[0, 1, 2, -1], [4, -1, -1, -1], [8, 9, 10, 11]]),
        )


if __name__ == "__main__":
    unittest.main()
