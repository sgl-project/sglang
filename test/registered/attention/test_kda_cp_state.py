"""Correctness tests for KDA context-parallel state pre-processing.

Single-GPU simulation of W CP ranks: torch.distributed.all_gather_into_tensor
is mocked with a record/replay double so the REAL end-to-end path
(chunk_kda -> chunk_gated_delta_rule_fwd_h_cp_pre_process -> merge -> main
kernel) runs per simulated rank without a process group. The pre-scan is
deterministic and depends only on rank-local inputs, so pass 1 records each
rank's hm buffer, pass 2 replays the stacked gather result. Each simulated
rank gets its own clone of the state pool, matching serving semantics where
every CP rank holds a pool replica and the merge writes the identical global
final state into each replica.

References (established empirically, see test/manual/kda_cp_diag2.py):
* sequential shard chain — run the shards one by one through the pool slot,
  i.e. exactly chunked-prefill semantics with the same per-shard chunk grid
  as CP. CP must reproduce this tightly (only the h0 delivery differs:
  affine merge chain vs sequential carry). Tolerance 2e-3; observed 0 /
  1.15e-06 / 4.44e-04.
* monolithic single run — differs from ANY re-chunked run by bf16 rounding
  (~4e-3 with well-scaled inputs; NOT a CP artifact). Loose cross-check 1e-2.
Inputs use the well-scaled distribution of the existing KDA CI test:
unscaled randn gates/values are numerically adversarial for the output path
(a monolithic run is already 0.17 off the fp32 naive recurrence).

Guarded failure modes:
1. cp4 cases — the merge chain delivering a wrong per-rank h0 or wrong pool
   writeback (affine pre-scan/merge math, pool seeding, scratch handoff).
2. cp8_empty_shards — regression for the empty-sequence trap: the base chunk
   pipeline (prepare_chunk_indices) mis-attributes chunks when cu_seqlens
   contains zero-length sequences, leaving kg/w/u uninitialized; the CP shard
   layout must therefore compact empty shards out and route hm rows via
   local_seq_ids (identity affine for absent sequences).
3. cp1_passthrough — an inactive context must be a bit-exact no-op.
4. untouched-slot checks — the merge must never write pool slots outside the
   batch.

Keep in sync with test/manual/kda_cp_standalone_check.py (a dependency-light
mirror for boxes where the sglang.test import chain is broken).
"""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.fla.chunk_delta_h_cp import (
    LinearAttnCPContext,
    build_cp_shard_layout,
)
from sglang.kernels.ops.attention.fla.kda import chunk_kda
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-large")

CHAIN_TOL = 2e-3
MONO_TOL = 1e-2


def _norm_ratio(actual: torch.Tensor, ref: torch.Tensor) -> float:
    ref = ref.float()
    return ((actual.float() - ref).norm() / ref.norm().clamp(min=1e-12)).item()


class _RecordReplayGather:
    """Fake all_gather_into_tensor for single-process CP simulation."""

    def __init__(self, world_size: int):
        self.world_size = world_size
        self.recorded = {}
        self.replay = None
        self.current_rank = None

    def __call__(self, out: torch.Tensor, inp: torch.Tensor, group=None):
        if self.replay is None:
            self.recorded[self.current_rank] = inp.clone()
            out.zero_()
        else:
            out.copy_(self.replay)

    def build_replay(self):
        assert len(self.recorded) == self.world_size
        self.replay = torch.stack([self.recorded[r] for r in range(self.world_size)])


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestKDAContextParallelState(CustomTestCase):
    H = 4
    D = 128
    NUM_SLOTS = 8

    def setUp(self):
        torch.manual_seed(42)
        self.device = "cuda"

    def _make_inputs(self, total_tokens: int):
        h, d = self.H, self.D
        shape = (1, total_tokens, h, d)
        return {
            "q": torch.randn(shape, dtype=torch.bfloat16, device=self.device),
            "k": torch.randn(shape, dtype=torch.bfloat16, device=self.device),
            "v": torch.randn(shape, dtype=torch.bfloat16, device=self.device) * 0.1,
            "g": (
                torch.randn(shape, dtype=torch.float32, device=self.device) * 0.5 - 2.0
            ).to(torch.bfloat16),
            "beta": torch.rand(
                1, total_tokens, h, dtype=torch.bfloat16, device=self.device
            )
            .float()
            .sigmoid(),
            "A_log": torch.randn(1, 1, h, 1, dtype=torch.float32, device=self.device)
            * 0.1,
            "dt_bias": torch.randn(h * d, dtype=torch.float32, device=self.device)
            * 0.1,
        }

    def _make_seed_pool(self, zero: bool):
        pool = torch.zeros(
            self.NUM_SLOTS,
            self.H,
            self.D,
            self.D,
            dtype=torch.float32,
            device=self.device,
        )
        if not zero:
            pool.normal_(std=0.05)
        return pool

    def _run_chunk_kda(self, inputs, cu_seqlens, pool, slot_indices, cp_context=None):
        # chunk_kda writes its output into the v buffer (o=v aliasing in
        # chunk_gla_fwd_o_gk) — clone per call so inputs stay reusable.
        return chunk_kda(
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"].clone(),
            g=inputs["g"],
            beta=inputs["beta"],
            initial_state=pool,
            initial_state_indices=slot_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"],
            cp_context=cp_context,
        )

    def _slice_rank_inputs(self, inputs, shard_ranges):
        sliced = {
            name: torch.cat(
                [inputs[name][:, lo:hi] for lo, hi in shard_ranges], dim=1
            ).contiguous()
            for name in ("q", "k", "v", "g", "beta")
        }
        sliced["A_log"] = inputs["A_log"]
        sliced["dt_bias"] = inputs["dt_bias"]
        return sliced

    def _scatter_shards(self, inputs, layouts, o_shards, total_tokens):
        o_full = inputs["q"].new_empty(1, total_tokens, self.H, self.D)
        for r, (_, ranges, _ids) in enumerate(layouts):
            offset = 0
            for lo, hi in ranges:
                o_full[:, lo:hi] = o_shards[r][:, offset : offset + (hi - lo)]
                offset += hi - lo
        return o_full

    def _run_sequential_chain(
        self, inputs, layouts, seed_pool, slot_indices, total_tokens
    ):
        pool = seed_pool.clone()
        o_shards = []
        for local_cu, ranges, seq_ids in layouts:
            cu = torch.tensor(local_cu, dtype=torch.int32, device=self.device)
            o_shards.append(
                self._run_chunk_kda(
                    self._slice_rank_inputs(inputs, ranges),
                    cu,
                    pool,
                    slot_indices[seq_ids],
                )
            )
        return self._scatter_shards(inputs, layouts, o_shards, total_tokens), pool

    def _run_cp_sim(
        self, inputs, layouts, seed_pool, slot_indices, world_size, total_tokens
    ):
        rank_inputs = [
            self._slice_rank_inputs(inputs, ranges) for _, ranges, _ids in layouts
        ]
        rank_cu = [
            torch.tensor(local_cu, dtype=torch.int32, device=self.device)
            for local_cu, _ranges, _ids in layouts
        ]
        num_global_seqs = len(slot_indices)
        gather = _RecordReplayGather(world_size)
        o_shards, pools = None, None
        with patch("torch.distributed.all_gather_into_tensor", new=gather):
            for do_replay in (False, True):
                o_shards, pools = [], []
                for r in range(world_size):
                    gather.current_rank = r
                    ctx = LinearAttnCPContext(
                        world_size=world_size,
                        rank=r,
                        group=object(),
                        num_global_seqs=num_global_seqs,
                        local_seq_ids=torch.tensor(
                            layouts[r][2], dtype=torch.int32, device=self.device
                        ),
                    )
                    pool_r = seed_pool.clone()
                    o_r = self._run_chunk_kda(
                        rank_inputs[r],
                        rank_cu[r],
                        pool_r,
                        slot_indices,
                        cp_context=ctx,
                    )
                    o_shards.append(o_r)
                    pools.append(pool_r)
                if not do_replay:
                    gather.build_replay()
        return (
            self._scatter_shards(inputs, layouts, o_shards, total_tokens),
            pools,
        )

    def _check_cp_matches_refs(self, seq_lens, world_size, zero_seed):
        total_tokens = sum(seq_lens)
        inputs = self._make_inputs(total_tokens)
        seed_pool = self._make_seed_pool(zero=zero_seed)
        slot_indices = torch.tensor(
            [3, 5, 1, 6][: len(seq_lens)], dtype=torch.int32, device=self.device
        )
        cu_vals = [0]
        for n in seq_lens:
            cu_vals.append(cu_vals[-1] + n)
        cu = torch.tensor(cu_vals, dtype=torch.int32, device=self.device)
        layouts = [
            build_cp_shard_layout(cu_vals, world_size, r) for r in range(world_size)
        ]

        mono_pool = seed_pool.clone()
        o_mono = self._run_chunk_kda(inputs, cu, mono_pool, slot_indices)
        o_chain, chain_pool = self._run_sequential_chain(
            inputs, layouts, seed_pool, slot_indices, total_tokens
        )
        o_cp, cp_pools = self._run_cp_sim(
            inputs, layouts, seed_pool, slot_indices, world_size, total_tokens
        )

        chain_ratio = _norm_ratio(o_cp, o_chain)
        self.assertLess(
            chain_ratio,
            CHAIN_TOL,
            f"CP{world_size} output vs sequential chain: {chain_ratio:.2e}",
        )
        mono_ratio = _norm_ratio(o_cp, o_mono)
        self.assertLess(
            mono_ratio,
            MONO_TOL,
            f"CP{world_size} output vs monolithic: {mono_ratio:.2e}",
        )
        for r, pool_r in enumerate(cp_pools):
            state_chain = _norm_ratio(pool_r[slot_indices], chain_pool[slot_indices])
            self.assertLess(
                state_chain,
                CHAIN_TOL,
                f"rank {r} final state vs chain: {state_chain:.2e}",
            )
            state_mono = _norm_ratio(pool_r[slot_indices], mono_pool[slot_indices])
            self.assertLess(
                state_mono,
                MONO_TOL,
                f"rank {r} final state vs monolithic: {state_mono:.2e}",
            )
        untouched = torch.ones(self.NUM_SLOTS, dtype=torch.bool)
        untouched[slot_indices.cpu()] = False
        for pool_r in cp_pools:
            torch.testing.assert_close(
                pool_r[untouched], seed_pool[untouched], rtol=0, atol=0
            )

    def test_cp4_fresh_prefill(self):
        # 1000 is not 64-aligned, 704 is: covers both shard-boundary layouts.
        self._check_cp_matches_refs(seq_lens=[1000, 704], world_size=4, zero_seed=True)

    def test_cp4_chunked_prefill_continuation(self):
        # Non-zero pool seed: the merge chain must start from the carried
        # state, and the writeback must land the correct global final state.
        self._check_cp_matches_refs(seq_lens=[831, 512], world_size=4, zero_seed=False)

    def test_cp8_empty_shards(self):
        # A 5-token sequence over CP8 leaves most ranks with an empty shard;
        # the layout must compact it away (base kernels corrupt on empty
        # sequences) while the merge still writes its final state everywhere.
        self._check_cp_matches_refs(seq_lens=[5, 640], world_size=8, zero_seed=False)

    def test_cp1_passthrough(self):
        # world_size == 1 must be a strict no-op passthrough (no scratch, no
        # gather), bit-identical to the non-CP path.
        total_tokens = 320
        inputs = self._make_inputs(total_tokens)
        seed_pool = self._make_seed_pool(zero=False)
        slot_indices = torch.tensor([2], dtype=torch.int32, device=self.device)
        cu = torch.tensor([0, total_tokens], dtype=torch.int32, device=self.device)

        ref_pool = seed_pool.clone()
        o_ref = self._run_chunk_kda(inputs, cu, ref_pool, slot_indices)

        cp_pool = seed_pool.clone()
        ctx = LinearAttnCPContext(world_size=1, rank=0, group=None)
        o_cp = self._run_chunk_kda(inputs, cu, cp_pool, slot_indices, cp_context=ctx)

        torch.testing.assert_close(o_cp, o_ref, rtol=0, atol=0)
        torch.testing.assert_close(cp_pool, ref_pool, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=3)
