"""GDN ReplaySSM spec verify and commit: the batched gfx950 launches are bitwise
equal to the launches they replace.

Past ``_VERIFY_RESIDENT_PROGRAMS`` verify programs the verify takes the
normalized q/k tiles and the ``k @ k^T`` / ``k @ q^T`` products from
``gdn_replayssm_qk_prep_kernel`` (one program per request and key head) instead
of recomputing them in every (V tile, value head) program. From
``_LAYER_LOOP_MIN_PROGRAMS`` (request, value head) programs the commit folds
every layer in one program. The steps below have ring history, rows with and
without a fold, track rows and a padding row.
"""

import math
import unittest
from unittest import mock

import torch

import sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode as spec
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, stage="base-b", runner_config="1-gpu-small-amd-mi35x")

# The batched launches need K = V = 128, a ring of 16 and 4 draft tokens. At 48
# value heads, 11 rows (10 requests and a padding row) are the fewest whose
# verify preps beside main's 64-wide tile, and 33 rows prep beside its 128-wide
# tile and take the layer-loop commit.
H, HV, K, V, L, D = 16, 48, 128, 128, 16, 4
LAYERS, REQUESTS = 2, 32
RINGS = ("d", "k", "g", "rawv", "rawk")
NEVER = 1 << 40


class _Grids:
    """Stands in for a Triton kernel and records the grid of every launch."""

    def __init__(self, kernel):
        self.kernel, self.grids = kernel, []

    def __getitem__(self, grid):
        self.grids.append(grid)
        return self.kernel[grid]


@unittest.skipUnless(is_gfx95_supported(), "ROCm gfx95 only")
class TestGdnReplaySSMBatchedLaunch(CustomTestCase):
    def test_batched_launches_match_per_program_launches(self):
        gen = torch.Generator(device="cuda").manual_seed(0)

        def rand(*shape, scale=1.0, dtype=torch.bfloat16):
            x = torch.randn(*shape, device="cuda", generator=gen) * scale
            return x.to(dtype)

        def randint(lo, hi, size, dtype=torch.int32):
            return torch.randint(lo, hi, size, device="cuda", generator=gen).to(dtype)

        # A server's checkpoint pool exceeds 2 GiB, so on AMD Triton compiles
        # its pointer without 32-bit offsets; this one does too. Slots n.. hold
        # the track snapshots.
        pool = torch.empty(1 << 30, device="cuda", dtype=torch.bfloat16)
        shape = (LAYERS, 2 * REQUESTS, HV, V, K)
        numel = math.prod(shape)
        state = pool[:numel].view(shape).normal_(0, 0.1, generator=gen)
        folded = pool[numel : 2 * numel].view(shape)

        def step(n):
            """n requests and a padding row after some steps of ring history."""
            B, T = n + 1, (n + 1) * D
            rings = {
                "d": rand(LAYERS, B, HV, L, V, scale=0.1),
                "k": rand(LAYERS, B, H, L, K, scale=0.1),
                "g": -rand(LAYERS, B, HV, L, scale=0.1, dtype=torch.float32).abs(),
                "rawv": rand(LAYERS, B, HV, L, V, scale=1e-3),
                "rawk": rand(LAYERS, B, H, L, K, scale=1e-3),
            }
            slots = torch.cat(
                [
                    torch.randperm(n, device="cuda", generator=gen),
                    torch.tensor([-1], device="cuda"),
                ]
            ).to(torch.int32)
            cursors = {
                "write_pos": randint(0, L - D + 1, (B,)),
                "cache_base": randint(0, L, (B,)),
                "is_flush": torch.zeros(B, device="cuda", dtype=torch.int8),
            }
            inputs = dict(
                q=rand(T, H, K),
                k=rand(T, H, K),
                v=rand(T, HV, V),
                a=rand(T, HV),
                b=rand(T, HV),
                A_log=rand(HV, dtype=torch.float32),
                dt_bias=rand(HV, dtype=torch.float32),
                query_start_loc=torch.arange(
                    0, T + 1, D, device="cuda", dtype=torch.int32
                ),
                ssm_state_indices=slots,
                replay_indices=torch.randperm(B, device="cuda", generator=gen).to(
                    torch.int32
                ),
                max_cache_len=L,
                max_spec_len=D,
                null_block_id=-1,
                launch_mode="verify",
            )
            return rings, cursors, inputs

        def verify(rings, cursors, inputs, resident):
            ring = {name: t[0].clone() for name, t in rings.items()}
            out = torch.zeros_like(inputs["v"])
            with (
                mock.patch.object(spec, "_VERIFY_RESIDENT_PROGRAMS", resident),
                mock.patch.object(
                    spec,
                    "gdn_replayssm_qk_prep_kernel",
                    _Grids(spec.gdn_replayssm_qk_prep_kernel),
                ) as prep,
            ):
                spec.gdn_replayssm_spec_decode(
                    **inputs,
                    checkpoint_state=state[0],
                    d_cache=ring["d"],
                    k_cache=ring["k"],
                    g_cache=ring["g"],
                    rawv_cache=ring["rawv"],
                    rawk_cache=ring["rawk"],
                    beta_cache=None,
                    out=out,
                    **cursors,
                )
            return out, ring, prep.grids

        for n in (10, REQUESTS):
            rings, cursors, inputs = step(n)
            out, ring, grids = verify(
                rings, cursors, inputs, spec._VERIFY_RESIDENT_PROGRAMS
            )
            self.assertEqual(grids, [(n + 1, H)])
            ref_out, ref_ring, ref_grids = verify(rings, cursors, inputs, NEVER)
            self.assertEqual(ref_grids, [])
            self.assertTrue(out[: n * D].flatten(1).ne(0).any(1).all())
            self.assertTrue(torch.equal(out, ref_out), n)
            for name in RINGS:
                self.assertTrue(torch.equal(ring[name], ref_ring[name]), (n, name))

        # Commit: rings as written by the last verify, every layer; the
        # accepted prefix is already in write_pos. Requests 0..2 skip the fold,
        # odd requests snapshot a track step.
        n, B = REQUESTS, REQUESTS + 1
        rows = inputs["replay_indices"]
        for name in RINGS:
            rings[name][0] = ring[name]
        accept = randint(1, D + 1, (B,))
        cursors["write_pos"] += accept
        cursors["is_flush"][rows[3:].long()] = 1
        odd = torch.arange(B, device="cuda") % 2 == 1
        track = torch.arange(n, 2 * n + 1, device="cuda")
        track_step = torch.where(odd, randint(0, D, (B,), torch.int64) % accept, -1)

        def commit(min_programs):
            folded.copy_(state)
            cur = {name: t.clone() for name, t in cursors.items()}
            with (
                mock.patch.object(spec, "_LAYER_LOOP_MIN_PROGRAMS", min_programs),
                mock.patch.object(
                    spec,
                    "gdn_replayssm_compact_commit_kernel",
                    _Grids(spec.gdn_replayssm_compact_commit_kernel),
                ) as kernel,
            ):
                spec.commit_gdn_replayssm_circular(
                    checkpoint_state=folded,
                    d_cache=rings["d"],
                    k_cache=rings["k"],
                    g_cache=rings["g"],
                    d_residual_cache=rings["rawv"],
                    k_residual_cache=rings["rawk"],
                    state_batch_indices=inputs["ssm_state_indices"],
                    replay_indices=rows,
                    accept_lens=accept,
                    mamba_track_indices=track,
                    mamba_steps_to_track=track_step,
                    null_block_id=-1,
                    **cur,
                )
            return folded.clone(), cur, kernel.grids

        out, cur, grids = commit(spec._LAYER_LOOP_MIN_PROGRAMS)
        self.assertEqual(grids, [(1, B, HV)])
        ref_out, ref_cur, ref_grids = commit(NEVER)
        self.assertEqual(ref_grids, [(2, B, HV * LAYERS)])
        self.assertFalse(torch.equal(out, state))
        self.assertTrue(torch.equal(out, ref_out))
        for name in cursors:
            self.assertTrue(torch.equal(cur[name], ref_cur[name]), name)


if __name__ == "__main__":
    unittest.main()
