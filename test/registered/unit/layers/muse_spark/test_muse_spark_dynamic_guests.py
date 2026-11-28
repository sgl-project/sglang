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
    def _prepare(
        self,
        *,
        rank: int,
        no_peer_access=(),
        enable_p2p_check=False,
        one_visible_device=False,
        other_ranks_agree=True,
        collectives=None,
    ):
        """Run the setup on ``rank`` of a one-host TP8 group on GPUs ``GPU-0..7``:
        pairs in ``no_peer_access`` lack P2P and ``other_ranks_agree`` is the other
        ranks' vote. ``one_visible_device`` shows each rank only its own GPU, as
        cuda:0. The collectives the rank calls are appended to ``collectives``."""
        collectives = [] if collectives is None else collectives
        layer = SimpleNamespace(
            use_fused_w4a16=True, global_num_experts=256, num_local_experts=32
        )
        group = _Group(rank)
        # local device index -> physical GPU
        physical = [rank] if one_visible_device else list(range(8))

        def gather(out, obj, group):
            collectives.append("all_gather_object")
            out[:] = [f"GPU-{i}" for i in range(8)]

        def all_reduce(t, op, group):
            collectives.append("all_reduce")
            t.fill_(min(int(t.item()), int(other_ranks_agree)))

        def pair(pairs):
            return lambda a, b: {physical[a], physical[b]} not in pairs

        with (
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(
                    tp_group=group, nnodes=1, enable_p2p_check=enable_p2p_check
                ),
            ),
            patch.object(
                torch._C,
                "_accelerator_getAllocatorSettings",
                return_value="",
                create=True,
            ),
            patch.object(
                guests.torch.cuda,
                "current_device",
                return_value=physical.index(rank),
            ),
            patch.object(guests.torch.cuda, "device_count", return_value=len(physical)),
            patch.object(
                guests.torch.cuda,
                "get_device_properties",
                side_effect=lambda i: SimpleNamespace(uuid=f"GPU-{physical[i]}"),
            ),
            patch.object(
                guests.torch.cuda,
                "can_device_access_peer",
                side_effect=pair(no_peer_access),
            ),
            patch.object(guests.dist, "all_gather_object", side_effect=gather),
            patch.object(guests.dist, "all_reduce", side_effect=all_reduce),
        ):
            guests.maybe_prepare_dynamic_guests([layer])
        return hasattr(layer, "dynamic_guest_setup")

    def test_guests_need_peer_access_across_the_whole_group(self):
        """A GPU pair without P2P (PCIe, VMs) cannot open CUDA IPC handles: guests
        stay off on every rank, including ranks whose own peers are all reachable."""
        self.assertTrue(self._prepare(rank=0))
        self.assertFalse(self._prepare(rank=3, no_peer_access=[{3, 5}]))
        self.assertFalse(self._prepare(rank=0, other_ranks_agree=False))

    def test_every_rank_makes_the_same_collectives(self):
        """Ranks that find different peers still make the same collective calls in
        the same order, or the vote hangs; --enable-p2p-check keeps guests off on
        every rank before any collective."""
        for options in ({}, {"enable_p2p_check": True}):
            calls = []
            for rank in range(8):
                calls.append([])
                self._prepare(
                    rank=rank,
                    no_peer_access=[{3, 5}],
                    collectives=calls[-1],
                    **options,
                )
            with self.subTest(**options):
                expected = [] if options else ["all_gather_object", "all_reduce"]
                self.assertEqual(calls, [expected] * 8)
        # The probed check is not available here: guests stay off even with full P2P.
        self.assertFalse(self._prepare(rank=0, enable_p2p_check=True))

    def test_unseen_peer_gpus_keep_guests_off(self):
        """With one visible device per process every rank is cuda:0 and cannot check
        its peers' GPUs, so guests stay off instead of trusting the device index."""
        self.assertFalse(self._prepare(rank=2, one_visible_device=True))


if __name__ == "__main__":
    unittest.main()
