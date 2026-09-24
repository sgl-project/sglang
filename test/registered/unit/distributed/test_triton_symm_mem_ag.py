"""CPU-only unit tests for the ``MultimemAllGatherer`` enable guard.

These tests exercise **only** ``MultimemAllGatherer.__init__`` with the parallel
state and the CUDA capability queries faked out. No GPU, no NCCL and no
symmetric-memory rendezvous is involved, so they run on any machine.

This file mirrors ``python/sglang/srt/distributed/device_communicators/triton_symm_mem_ag.py``
and is meant to live at ``test/registered/unit/distributed/test_triton_symm_mem_ag.py``.

    pytest test/registered/unit/distributed/test_triton_symm_mem_ag.py -q
"""

import importlib
import os
import types
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MOD = "sglang.srt.distributed.device_communicators.triton_symm_mem_ag"
PARALLEL_STATE = "sglang.srt.distributed.parallel_state"
DIST = "sglang.srt.distributed"


def _load_module():
    # If this tree carries the `SGLANG_DISABLE_MULTIMEM_AG` opt-out (added by
    # #36110, and not present on current main), make sure it is off: it would
    # bypass the guard entirely and these tests must exercise upstream
    # behaviour rather than the escape hatch.
    os.environ.pop("SGLANG_DISABLE_MULTIMEM_AG", None)
    try:
        return importlib.import_module(MOD)
    except Exception as exc:  # pragma: no cover - environment problem
        raise RuntimeError("cannot import %s: %r" % (MOD, exc))


def _build(
    *,
    world_size,
    nnodes,
    same_node=True,
    peer_access=True,
    enabled=True,
    device_count=2,
):
    """Construct the gatherer with every device-related query faked."""
    mod = _load_module()
    tp_group = types.SimpleNamespace(world_size=world_size, cpu_group=object())

    with (
        mock.patch(DIST + ".get_tp_group", return_value=tp_group),
        mock.patch(
            MOD + ".get_parallel",
            return_value=types.SimpleNamespace(nnodes=nnodes),
        ),
        mock.patch(
            PARALLEL_STATE + ".in_the_same_node_as",
            return_value=[same_node] * max(world_size, 1),
        ),
        mock.patch("torch.cuda.device_count", return_value=device_count),
        mock.patch(
            "torch.cuda.can_device_access_peer", side_effect=lambda i, j: peer_access
        ),
    ):
        return mod.MultimemAllGatherer(8, enabled=enabled), mod


class TestMultimemAllGathererGuard(CustomTestCase):
    def test_disabled_flag_always_uses_nccl(self):
        """``enabled=False`` short-circuits before any topology query."""
        gatherer, _ = _build(world_size=2, nnodes=1, enabled=False)
        self.assertIs(gatherer._state, None)

    def test_single_rank_is_left_alone(self):
        """TP=1 has nothing to exchange, so the gatherer stays lazy."""
        gatherer, mod = _build(world_size=1, nnodes=1)
        self.assertIs(gatherer._state, mod.MultimemAllGatherer._UNINIT)

    def test_cross_node_is_disabled(self):
        """Existing contract: a TP group spanning nodes falls back to NCCL."""
        gatherer, _ = _build(world_size=2, nnodes=2, same_node=False)
        self.assertIs(gatherer._state, None)

    def test_single_node_with_peer_access_stays_enabled(self):
        """Regression guard: multimem must not be disabled where P2P exists."""
        gatherer, mod = _build(world_size=2, nnodes=1, peer_access=True)
        self.assertIs(gatherer._state, mod.MultimemAllGatherer._UNINIT)

    def test_single_node_without_peer_access_is_disabled(self):
        """A single node is not necessarily multicast-capable.

        Two consumer GPUs on PCIe with no NVLink/P2P are co-located (``nnodes == 1``)
        but have no multicast fabric. Without a capability condition the multimem
        path stays enabled and the process dies inside ``symm_mem.rendezvous``
        called from ``create_state`` -- i.e. *before* the ``multicast_ptr == 0``
        fallback in ``_build`` gets a chance to run.
        """
        gatherer, _ = _build(world_size=2, nnodes=1, peer_access=False)
        self.assertIs(gatherer._state, None)


if __name__ == "__main__":
    unittest.main()
