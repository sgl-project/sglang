"""GDN ReplaySSM circular commit: fold and track the compact history.

The folded and tracked checkpoints must match a float64 reference of the same
closed form for several value/key head ratios, both checkpoint dtypes, a
partial V tile, slot-strided and layer-inner state pools, null rows, rings of
16 to 128 steps (histories longer than one fold chunk), with and without track
rows, and with bf16 rings plus residuals or fp32 rings without them. Ring
entries past the committed history are NaN and must not be read, and every
slot the commit does not own stays bit-identical. Every case runs through both
commit kernels (per value head and per head group), and the ring length picks
the kernel.
"""

import itertools
import unittest
from unittest import mock

import torch

from sglang.kernels.ops.attention.linear import gdn_replayssm_spec_decode
from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_circular,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")

DEVICE = "cuda"
LAYERS, SLOTS = 2, 12
# active slot, replay row, history length (ring 16), fold, track slot, track step
ROWS = [
    (3, 0, 9, 1, -1, -1),
    (-1, 1, 6, 1, 8, 2),  # null row: never touched
    (5, 2, 4, 0, 9, 1),  # track only
    (7, 3, 12, 1, 10, 3),  # fold and track
    (0, 4, 5, 0, -1, -1),  # nothing to do
]
ACCEPTED = [4, 3, 2, 5, 1]
CACHE_BASES = [3, 11, 15, 0, 7]


def _reference(old, keys, updates, gates, n):
    """float64 S' = exp(G_n) S + sum_{t<n} exp(G_n - G_t) k_t^T u_t."""
    gates = gates[:n].double()
    prefix = torch.cumsum(gates, 0)
    total = prefix[-1] if n > 0 else gates.new_zeros(())
    decay = torch.exp(total - prefix)
    delta = keys[:n].double().T @ (updates[:n].double() * decay[:, None])
    return old.double() * torch.exp(total) + delta.T


class TestGdnReplayssmCircularCommit(CustomTestCase):
    def _run(
        self,
        h,
        hv,
        K,
        V,
        state_dtype,
        layout="contiguous",
        ring=16,
        residual=True,
        track=True,
        head_group=False,
    ):
        gen = torch.Generator(device=DEVICE).manual_seed(h * 100 + hv + ring)

        def rand(*shape, scale, dtype=torch.bfloat16):
            out = torch.randn(*shape, generator=gen, device=DEVICE) * scale
            return out.to(dtype)

        if layout == "slot_strided":
            pool = rand(LAYERS, 2 * SLOTS, hv, V, K, scale=0.25, dtype=state_dtype)
            state = pool[:, 1::2]
        elif layout == "layer_inner":
            pool = rand(SLOTS, LAYERS, hv, V, K, scale=0.25, dtype=state_dtype)
            state = pool.transpose(0, 1)
        else:
            pool = state = rand(LAYERS, SLOTS, hv, V, K, scale=0.25, dtype=state_dtype)
        initial_pool = pool.clone()
        initial = state.clone()
        # Histories scale with the ring so long rings fold several chunks.
        histories = [r[2] * ring // 16 for r in ROWS]
        bases = [b * ring // 16 for b in CACHE_BASES]
        nreq = len(ROWS)
        # The pool keeps residuals only for low-precision rings.
        ring_dtype = torch.bfloat16 if residual else torch.float32
        d = rand(LAYERS, nreq, hv, ring, V, scale=0.25, dtype=ring_dtype)
        k = rand(LAYERS, nreq, h, ring, K, scale=0.08, dtype=ring_dtype)
        if residual:
            d_lo = rand(LAYERS, nreq, hv, ring, V, scale=1e-3)
            k_lo = rand(LAYERS, nreq, h, ring, K, scale=1e-4)
        else:
            d_lo = torch.zeros_like(d)
            k_lo = torch.zeros_like(k)
        g = -0.2 * torch.rand(LAYERS, nreq, hv, ring, generator=gen, device=DEVICE)
        g *= 16 / ring  # keep the total decay over the history comparable
        # Ring entries past the committed history are stale; they must not be read.
        for (_, req, *_), n_hist in zip(ROWS, histories):
            stale = (bases[req] + torch.arange(n_hist, ring)) % ring
            for buf in (d, k, d_lo, k_lo, g):
                buf[:, req, :, stale] = float("nan")

        def ints(values):
            return torch.tensor(values, device=DEVICE, dtype=torch.int32)

        state_indices = ints([r[0] for r in ROWS])
        replay_indices = ints([r[1] for r in ROWS])
        write_pos = ints(histories)
        cache_base = ints(bases)
        is_flush = ints([r[3] for r in ROWS])
        accepted = ints(ACCEPTED)
        with mock.patch.object(
            gdn_replayssm_spec_decode,
            "_commit_by_head_group",
            return_value=head_group,
        ):
            commit_gdn_replayssm_circular(
                checkpoint_state=state,
                d_cache=d,
                k_cache=k,
                g_cache=g,
                d_residual_cache=d_lo if residual else None,
                k_residual_cache=k_lo if residual else None,
                state_batch_indices=state_indices,
                replay_indices=replay_indices,
                write_pos=write_pos,
                cache_base=cache_base,
                is_flush=is_flush,
                accept_lens=accepted,
                mamba_track_indices=ints([r[4] for r in ROWS]) if track else None,
                mamba_steps_to_track=ints([r[5] for r in ROWS]) if track else None,
                null_block_id=-1,
            )

        if state_dtype == torch.bfloat16:
            tol = dict(rtol=8e-3, atol=1e-3)
        elif residual:
            tol = dict(rtol=1e-5, atol=1e-5)
        else:
            # fp32 rings fold with fp32 tensor-core operands (TF32 by default).
            tol = dict(rtol=1e-3, atol=1e-3)
        written = set()
        for (slot, req, _, fold, track_slot, track_step), n_hist, acc in zip(
            ROWS, histories, ACCEPTED
        ):
            if slot < 0:
                continue
            phys = (bases[req] + torch.arange(ring)) % ring
            targets = []
            if fold:
                targets.append((slot, n_hist))
            if track and track_slot >= 0 and track_step >= 0:
                targets.append((track_slot, n_hist - acc + track_step + 1))
            for target, n in targets:
                written.add(target)
                for layer in range(LAYERS):
                    for i_hv in range(hv):
                        i_h = i_hv // (hv // h)
                        ref = _reference(
                            initial[layer, slot, i_hv].float(),
                            (k[layer, req, i_h].float() + k_lo[layer, req, i_h])[phys],
                            (d[layer, req, i_hv].float() + d_lo[layer, req, i_hv])[
                                phys
                            ],
                            g[layer, req, i_hv][phys],
                            n,
                        )
                        torch.testing.assert_close(
                            state[layer, target, i_hv].double(),
                            ref.to(state_dtype).double(),
                            **tol,
                        )
        for slot in set(range(SLOTS)) - written:
            self.assertTrue(torch.equal(state[:, slot], initial[:, slot]), slot)
        if layout == "slot_strided":
            self.assertTrue(torch.equal(pool[:, 0::2], initial_pool[:, 0::2]))
        folded = [r[3] == 1 and r[0] >= 0 for r in ROWS]
        self.assertEqual(
            write_pos.tolist(), [0 if f else n for f, n in zip(folded, histories)]
        )
        self.assertEqual(
            cache_base.tolist(),
            [(b + n) % ring if f else b for f, b, n in zip(folded, bases, histories)],
        )
        self.assertEqual(is_flush.tolist(), [0, 1, 0, 0, 0])

    def test_head_ratios_dtypes_and_pool_layouts(self):
        # (key heads, value heads, K, V); V = 80 leaves a partial V tile.
        for h, hv, k, v in [
            (2, 2, 64, 80),
            (2, 4, 64, 80),
            (2, 6, 64, 80),
            (1, 4, 64, 80),
            (4, 12, 128, 128),
        ]:
            for state_dtype in (torch.float32, torch.bfloat16):
                for layout, head_group in itertools.product(
                    ("contiguous", "slot_strided", "layer_inner"), (False, True)
                ):
                    with self.subTest(
                        h=h,
                        hv=hv,
                        k=k,
                        v=v,
                        state_dtype=state_dtype,
                        layout=layout,
                        head_group=head_group,
                    ):
                        self._run(
                            h, hv, k, v, state_dtype, layout, head_group=head_group
                        )

    def test_ring_lengths_track_and_residual(self):
        for ring in (16, 32, 64, 128):
            for residual in (True, False):
                for track in (True, False):
                    for state_dtype, head_group in itertools.product(
                        (torch.float32, torch.bfloat16), (False, True)
                    ):
                        with self.subTest(
                            ring=ring,
                            residual=residual,
                            track=track,
                            state_dtype=state_dtype,
                            head_group=head_group,
                        ):
                            self._run(
                                2,
                                6,
                                128,
                                128,
                                state_dtype,
                                ring=ring,
                                residual=residual,
                                track=track,
                                head_group=head_group,
                            )

    def test_ring_length_picks_the_kernel(self):
        """Rings of one 16-step fold chunk commit per head group, longer rings
        per value head."""

        def ints(values):
            return torch.tensor(values, device=DEVICE, dtype=torch.int32)

        for ring, head_group in ((16, True), (32, False), (64, False), (128, False)):
            with (
                self.subTest(ring=ring),
                mock.patch.object(
                    gdn_replayssm_spec_decode, "gdn_replayssm_head_group_commit_kernel"
                ) as by_group,
                mock.patch.object(
                    gdn_replayssm_spec_decode, "gdn_replayssm_compact_commit_kernel"
                ) as by_head,
            ):
                commit_gdn_replayssm_circular(
                    checkpoint_state=torch.zeros(1, 2, 6, 128, 128, device=DEVICE),
                    d_cache=torch.zeros(1, 1, 6, ring, 128, device=DEVICE),
                    k_cache=torch.zeros(1, 1, 2, ring, 128, device=DEVICE),
                    g_cache=torch.zeros(1, 1, 6, ring, device=DEVICE),
                    d_residual_cache=None,
                    k_residual_cache=None,
                    state_batch_indices=ints([0]),
                    replay_indices=ints([0]),
                    write_pos=ints([4]),
                    cache_base=ints([0]),
                    is_flush=ints([1]),
                    accept_lens=ints([4]),
                )
                launched, skipped = (
                    (by_group, by_head) if head_group else (by_head, by_group)
                )
                launched.__getitem__.assert_called_once()
                skipped.__getitem__.assert_not_called()


if __name__ == "__main__":
    unittest.main()
