"""DWDP small-param all_gather layout handling tests.

Covers python/sglang/srt/layers/moe/dwdp/dwdp_manager.py:
  - DwdpManager._memory_major_permutation: dim-order recovery for
    non-contiguous (mn-major) expert tensors, contiguous fast path, and the
    error paths (expert dim not outermost / unrecoverable layout).
  - DwdpManager._allgather_small_params: gather happens on the
    memory-contiguous view and the logical (mn-major) dim order is restored
    before replace_expert_tensor.
dist.all_gather is mocked; dwdp_size=2 rank-0 view.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from sglang.srt.layers.moe.dwdp.dwdp_manager import DwdpManager
from sglang.test.test_utils import CustomTestCase


def _mn_major(E, n, sk, seed=0):
    """DeepGEMM-style mn-major expert scale: storage (E, sk, n) contiguous,
    logical view (E, n, sk) non-contiguous."""
    storage = torch.arange(E * n * sk, dtype=torch.float32).reshape(E, sk, n)
    storage += seed * E * n * sk
    return storage.transpose(1, 2)


class _FakeExperts:
    def __init__(self, tensors):
        self.tensors = dict(tensors)
        self.replaced = {}

    def named_per_expert_tensors(self, local_experts):
        return list(self.tensors.items())

    def replace_expert_tensor(self, pname, full):
        self.replaced[pname] = full


def _make_manager(dwdp_size, num_routed, num_per_worker):
    manager = object.__new__(DwdpManager)
    manager.dwdp_size = dwdp_size
    manager.layout = SimpleNamespace(
        num_routed_experts=num_routed,
        num_experts_per_worker=num_per_worker,
    )
    return manager


class TestMemoryMajorPermutation(CustomTestCase):
    def test_contiguous_returns_none(self):
        t = torch.randn(4, 8)
        self.assertIsNone(DwdpManager._memory_major_permutation(t))

    def test_mn_major_recovers_storage_order(self):
        t = _mn_major(E=4, n=3, sk=5)
        self.assertFalse(t.is_contiguous())
        perm = DwdpManager._memory_major_permutation(t)
        self.assertEqual(perm, [0, 2, 1])
        view = t.permute(perm)
        self.assertTrue(view.is_contiguous())
        # The permuted view is just a relabeling of the same memory.
        self.assertEqual(view.data_ptr(), t.data_ptr())

    def test_expert_dim_not_outermost_raises(self):
        # dim 0 has the smallest stride, so no permutation keeps experts
        # outermost -> unrecoverable for expert-concatenating all_gather.
        t = torch.randn(3, 5, 4).permute(1, 2, 0)  # shape (5,4,3), stride (4,1,20)
        with self.assertRaises(ValueError):
            DwdpManager._memory_major_permutation(t)

    def test_sliced_layout_raises(self):
        # Stride order keeps dim 0 first but the view is not contiguous in
        # any dim order (strided slice).
        t = torch.randn(2, 8)[:, ::2]  # shape (2,4), stride (8,2)
        with self.assertRaises(ValueError):
            DwdpManager._memory_major_permutation(t)


class TestAllgatherSmallParams(CustomTestCase):
    def _run(self, tensors):
        """Gather flow with dwdp_size=2, 2 local experts per rank, rank 0 view.

        The mocked all_gather fills shard i with rank i's memory-major data;
        rank 0's contribution is the tensor the manager actually passed.
        """
        E_local, dwdp_size, num_total = 2, 2, 4
        manager = _make_manager(dwdp_size, num_total, E_local)

        experts = _FakeExperts(tensors)
        moe_layers = [(0, experts)]
        group = SimpleNamespace(device_group=object())

        sends = {}

        def fake_all_gather(shards, send, group=None):
            sends[id(shards)] = send
            # This rank's own data goes to shard 0; peer data to shard 1.
            shards[0].copy_(send)
            shards[1].copy_(send + 100.0)

        with patch("torch.distributed.all_gather", side_effect=fake_all_gather):
            manager._allgather_small_params(moe_layers, group)

        return experts, sends

    def test_mn_major_scale_gathered_and_restored(self):
        # (E=2, n=3, sk=4) mn-major per rank -> (E=4, n=3, sk=4) mn-major full.
        data = _mn_major(E=2, n=3, sk=4, seed=0)
        experts, sends = self._run({"w13_scale": data})

        # The tensor handed to all_gather must be contiguous (NCCL requirement)
        # and in storage order (E, sk, n).
        send = next(iter(sends.values()))
        self.assertTrue(send.is_contiguous())
        self.assertEqual(tuple(send.shape), (2, 4, 3))

        full = experts.replaced["w13_scale"]
        self.assertEqual(tuple(full.shape), (4, 3, 4))
        # Logical content: shard 0 = local data, shard 1 = peer (send + 100).
        expected = torch.cat([data, data + 100.0], dim=0)
        self.assertTrue(torch.equal(full, expected))
        # Layout restored: full is again an mn-major (non-contiguous) view whose
        # memory order is the natural (E, sk, n) storage order.
        self.assertFalse(full.is_contiguous())
        self.assertTrue(
            torch.equal(
                full.permute(0, 2, 1).flatten(), expected.permute(0, 2, 1).flatten()
            )
        )

    def test_contiguous_weight_gathered_unchanged(self):
        data = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        experts, _ = self._run({"w2_weight": data})

        full = experts.replaced["w2_weight"]
        self.assertEqual(tuple(full.shape), (4, 8))
        self.assertTrue(torch.equal(full, torch.cat([data, data + 100.0], dim=0)))
        self.assertTrue(full.is_contiguous())

    def test_multiple_tensors_and_layers(self):
        t0 = _mn_major(E=2, n=3, sk=4, seed=1)
        t1 = torch.randn(2, 6)
        layer0 = _FakeExperts({"w13_scale": t0})
        layer1 = _FakeExperts({"w2_weight": t1})
        manager = _make_manager(2, 4, 2)
        group = SimpleNamespace(device_group=object())

        def fake_all_gather(shards, send, group=None):
            shards[0].copy_(send)
            shards[1].copy_(send * 2.0)

        with patch("torch.distributed.all_gather", side_effect=fake_all_gather):
            manager._allgather_small_params([(7, layer0), (8, layer1)], group)

        self.assertEqual(tuple(layer0.replaced["w13_scale"].shape), (4, 3, 4))
        self.assertTrue(
            torch.equal(layer0.replaced["w13_scale"], torch.cat([t0, t0 * 2.0], dim=0))
        )
        self.assertTrue(
            torch.equal(layer1.replaced["w2_weight"], torch.cat([t1, t1 * 2.0], dim=0))
        )


if __name__ == "__main__":
    unittest.main()
