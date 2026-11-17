"""Device-IPC K/V gather must match the NCCL gather it replaces.

The transport swaps two all-gathers for peer writes into an IPC mapping plus a
one-element ordering collective. The result it hands attention has to stay
byte-identical to the collective's: K and V concatenated along the sequence in
rank order. These run the exchange on CPU with the two ranks' staging buffers
cross-wired, which is what the IPC mapping is, so the slot rotation and the
rank-ordered placement are covered without a GPU.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.distributed.device_communicators import (
    ipc_kv_gather as mod,
)
from sglang.test.test_utils import CustomTestCase

SEQ, HEADS, DIM = 4, 2, 8


def _pair():
    """Two initialized states whose staging buffers map each other, as IPC does."""
    states = []
    for rank in range(2):
        st = mod._State()
        st.rank = rank
        st.inited = True
        st.token = torch.zeros(1, dtype=torch.int32)
        st.token_out = torch.zeros(2, dtype=torch.int32)
        states.append(st)
    key = ((1, SEQ, HEADS, DIM), torch.float32)
    buffers = [torch.zeros(2, 2, 1, SEQ, HEADS, DIM) for _ in range(2)]
    for rank, st in enumerate(states):
        st.staging[key] = (buffers[rank], buffers[1 - rank])
    return states


def _kv(rank, step):
    base = 100 * step + 10 * rank
    k = torch.full((1, SEQ, HEADS, DIM), float(base + 1))
    v = torch.full((1, SEQ, HEADS, DIM), float(base + 2))
    return k, v


class TestIpcKvGather(CustomTestCase):
    def test_matches_rank_ordered_concat_over_several_calls(self):
        states = _pair()
        for step in range(4):
            kv = [_kv(rank, step) for rank in range(2)]
            # The token collective guarantees both staging writes land before
            # either read, so stage both ranks, then read both.
            slots = []
            for rank, st in enumerate(states):
                k, v = kv[rank]
                local, _ = st.staging[((1, SEQ, HEADS, DIM), torch.float32)]
                slot = st.calls % 2
                local[slot][0].copy_(k)
                local[slot][1].copy_(v)
                slots.append(slot)
            outs = []
            for rank, st in enumerate(states):
                k, v = kv[rank]
                with patch.object(
                    mod.dist, "all_gather_into_tensor", lambda *a, **k_: None
                ):
                    outs.append(st.gather(k, v))
            want_k = torch.cat([kv[0][0], kv[1][0]], dim=1)
            want_v = torch.cat([kv[0][1], kv[1][1]], dim=1)
            for rank, out in enumerate(outs):
                self.assertEqual(tuple(out.shape), (2, 1, 2 * SEQ, HEADS, DIM))
                self.assertTrue(
                    torch.equal(out[0], want_k), f"K rank {rank} step {step}"
                )
                self.assertTrue(
                    torch.equal(out[1], want_v), f"V rank {rank} step {step}"
                )
            # consecutive calls must alternate slots, so a slot is only
            # rewritten after an intervening collective
            self.assertEqual(slots, [step % 2, step % 2])

    def test_nccl_fallback_has_the_same_layout(self):
        """A rank that never completed the IPC setup must still gather."""
        st = mod._State()
        st.rank = 0
        st.inited = True
        st.failed = True
        k, v = _kv(0, 0)
        peer_k, peer_v = _kv(1, 0)

        def fake_all_gather(output, input_, group=None):
            peer = next(
                s for t, s in ((k, peer_k), (v, peer_v)) if torch.equal(t[0], input_)
            )
            output.copy_(torch.cat([input_.unsqueeze(0), peer[0].unsqueeze(0)], dim=0))

        with patch.object(mod.dist, "all_gather_into_tensor", fake_all_gather):
            out = st.gather(k, v)
        self.assertTrue(torch.equal(out[0], torch.cat([k, peer_k], dim=1)))
        self.assertTrue(torch.equal(out[1], torch.cat([v, peer_v], dim=1)))

    def test_unsupported_shapes_fall_back(self):
        """The wrapper must decline anything the exchange cannot represent."""
        k = torch.zeros(1, SEQ, HEADS, DIM)
        with patch.object(mod, "ENABLED", True):
            self.assertIsNone(mod.ipc_kv_gather(k, k, 1), "non-2-rank group")
            self.assertIsNone(mod.ipc_kv_gather(k, k.to(torch.float16), 2), "dtype")
            self.assertIsNone(mod.ipc_kv_gather(k[0], k[0], 2), "rank-3 tensor")
            batched = torch.zeros(2, SEQ, HEADS, DIM)
            self.assertIsNone(mod.ipc_kv_gather(batched, batched, 2), "batch > 1")
            short = torch.zeros(1, SEQ // 2, HEADS, DIM)
            self.assertIsNone(mod.ipc_kv_gather(k, short, 2), "K/V shape mismatch")
        with patch.object(mod, "ENABLED", False):
            self.assertIsNone(mod.ipc_kv_gather(k, k, 2), "disabled by default")

    def test_pair_without_peer_access_keeps_the_collective(self):
        """A mapping needs peer access; without it the transport must stand down."""
        st = mod._State()
        k, v = _kv(0, 0)
        peer_k, peer_v = _kv(1, 0)

        def fake_all_gather(output, input_, group=None):
            peer = next(
                s for t, s in ((k, peer_k), (v, peer_v)) if torch.equal(t[0], input_)
            )
            output.copy_(torch.cat([input_.unsqueeze(0), peer[0].unsqueeze(0)], dim=0))

        reached = {}

        def fake_can_access(a, b):
            reached["checked"] = (a, b)
            return False

        sp_group = SimpleNamespace(device_group=object(), rank_in_group=0)
        with (
            patch(
                "sglang.multimodal_gen.runtime.distributed.parallel_state.get_sp_group",
                lambda: sp_group,
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform.is_rocm",
                lambda: True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.distributed.device_communicators"
                ".ipc_a2a._peer_cuda_device",
                lambda group, rank, device: 1,
            ),
            patch.object(mod.dist, "get_world_size", lambda group=None: 2),
            patch.object(mod.dist, "all_gather_into_tensor", fake_all_gather),
            patch.object(torch.cuda, "can_device_access_peer", fake_can_access),
        ):
            out = st.gather(k, v)

        # Without this the test would also pass on an unrelated init failure,
        # which is what an earlier version of it actually did.
        self.assertIn("checked", reached, "must reach the peer-access check")
        self.assertTrue(st.failed, "must latch the decline instead of retrying")
        self.assertEqual(st.staging, {}, "must not have exchanged any handle")
        self.assertTrue(torch.equal(out[0], torch.cat([k, peer_k], dim=1)))
        self.assertTrue(torch.equal(out[1], torch.cat([v, peer_v], dim=1)))

    def test_staging_is_capped(self):
        """Staging is never freed, so distinct shapes must evict in order."""
        st = mod._State()
        st.rank = 0
        st.inited = True
        with patch.object(mod.envs, "SGLANG_DIFFUSION_IPC_KV_GATHER_MAX_BUFFERS", 2):
            with patch.object(mod._State, "_share", lambda self, t: t):
                keys = []
                for seq in (2, 4, 6):
                    k = torch.zeros(1, seq, HEADS, DIM)
                    st._get_staging(k)
                    keys.append(((1, seq, HEADS, DIM), torch.float32))
        self.assertEqual(len(st.staging), 2)
        self.assertNotIn(keys[0], st.staging, "oldest key must be the one evicted")
        self.assertEqual(list(st.staging), keys[1:])


if __name__ == "__main__":
    unittest.main()
