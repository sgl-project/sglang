"""finalize_symm_workspaces() allocates pending workspaces in registration order."""

import unittest
from unittest.mock import patch

from sglang.srt.distributed.device_communicators import symm_workspace
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")


class _PendingWorkspace:
    def __init__(self, label: str, finalized: list):
        self.label = label
        self.finalized = finalized

    def finalize(self) -> None:
        self.finalized.append(self.label)


class TestFinalizeOrder(CustomTestCase):
    def test_groups_finalize_in_registration_order_not_name_order(self):
        """Every allocation is a collective rendezvous, so a rank must reach a group
        it shares with a peer after the same allocations as that peer. Registration
        order gives that; the order of the group names does not."""
        finalized: list = []
        # Registered as "2", "10", "1": sorting the names would give "1", "10", "2".
        registered = {
            "2": [_PendingWorkspace("2#0", finalized)],
            "10": [
                _PendingWorkspace("10#0", finalized),
                _PendingWorkspace("10#1", finalized),
            ],
            "1": [_PendingWorkspace("1#0", finalized)],
        }
        with patch.object(symm_workspace, "_WORKSPACES", registered):
            symm_workspace.finalize_symm_workspaces()
        self.assertEqual(finalized, ["2#0", "10#0", "10#1", "1#0"])


if __name__ == "__main__":
    unittest.main()
