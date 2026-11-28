"""MoE dynamic guests turn on only when the whole TP group has peer access."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed.device_communicators import custom_all_reduce_utils
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
        probe_fails=(),
        one_visible_device=False,
        other_ranks_agree=True,
    ):
        """Run the setup on ``rank`` of a one-host TP8 group on GPUs ``GPU-0..7``:
        pairs in ``no_peer_access`` lack P2P, pairs in ``probe_fails`` fail the
        --enable-p2p-check probe, and ``other_ranks_agree`` is the other ranks' vote.
        ``one_visible_device`` shows each rank only its own GPU, as cuda:0."""
        layer = SimpleNamespace(
            use_fused_w4a16=True, global_num_experts=256, num_local_experts=32
        )
        group = _Group(rank)
        # local device index -> physical GPU
        physical = [rank] if one_visible_device else list(range(8))

        def gather(out, obj, group):
            out[:] = [f"GPU-{i}" for i in range(8)]

        def all_reduce(t, op, group):
            t.fill_(min(int(t.item()), int(other_ranks_agree)))

        def pair(pairs):
            return lambda a, b: {physical[a], physical[b]} not in pairs

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
            patch.object(
                custom_all_reduce_utils,
                "gpu_p2p_access_check",
                side_effect=pair(probe_fails),
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
        self.assertFalse(self._prepare(rank=0, probe_fails=[{0, 6}]))

    def test_unseen_peer_gpus_keep_guests_off(self):
        """With one visible device per process every rank is cuda:0 and cannot check
        its peers' GPUs, so guests stay off instead of trusting the device index."""
        self.assertFalse(self._prepare(rank=2, one_visible_device=True))


if __name__ == "__main__":
    unittest.main()
