"""MoE dynamic guests turn on only when the whole TP group has peer access."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.muse_spark import muse_spark_dynamic_guests as guests
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Group:
    """A one-host TP8 group (hashable, as `tp_group_on_one_host` caches on it)."""

    world_size = 8

    def __init__(self, rank: int):
        self.rank_in_group, self.cpu_group = rank, object()


class TestMuseSparkDynamicGuests(unittest.TestCase):
    def _prepare(self, *, rank: int, no_peer_access, other_ranks_agree=True):
        """Run the setup on ``rank`` of a one-host TP8 group whose GPU pairs in
        ``no_peer_access`` lack P2P; ``other_ranks_agree`` is the other ranks' vote."""
        layer = SimpleNamespace(
            use_fused_w4a16=True, global_num_experts=256, num_local_experts=32
        )
        group = _Group(rank)

        def gather(out, obj, group):
            out[:] = list(range(8))

        def all_reduce(t, op, group):
            t.fill_(min(int(t.item()), int(other_ranks_agree)))

        with (
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(tp_group=group, nnodes=1),
            ),
            patch.object(
                torch._C,
                "_accelerator_getAllocatorSettings",
                return_value="",
                create=True,
            ),
            patch.object(guests.torch.cuda, "current_device", return_value=rank),
            patch.object(guests.torch.cuda, "device_count", return_value=8),
            patch.object(
                guests.torch.cuda,
                "can_device_access_peer",
                side_effect=lambda a, b: {a, b} not in no_peer_access,
            ),
            patch.object(guests.dist, "all_gather_object", side_effect=gather),
            patch.object(guests.dist, "all_reduce", side_effect=all_reduce),
        ):
            guests.maybe_prepare_dynamic_guests([layer])
        return hasattr(layer, "dynamic_guest_setup")

    def test_guests_need_peer_access_across_the_whole_group(self):
        """A GPU pair without P2P (PCIe, VMs) cannot open CUDA IPC handles: guests
        stay off on every rank, including ranks whose own peers are all reachable."""
        self.assertTrue(self._prepare(rank=0, no_peer_access=[]))
        self.assertFalse(self._prepare(rank=3, no_peer_access=[{3, 5}]))
        self.assertFalse(
            self._prepare(rank=0, no_peer_access=[], other_ranks_agree=False)
        )


if __name__ == "__main__":
    unittest.main()
