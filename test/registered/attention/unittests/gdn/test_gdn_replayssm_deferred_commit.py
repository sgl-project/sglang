"""Deferred DFLASH ownership, copy/export, release and fold accounting."""

import unittest
from types import SimpleNamespace
from unittest import mock

import test_gdn_replayssm_direct_commit as fixture
import torch

from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_after_verify,
)
from sglang.srt.mem_cache.gdn_replayssm_commit import (
    DeferredGDNState,
    read_gdn_commit_stats,
)
from sglang.srt.mem_cache.kv.state_storage import StateStorage
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestDeferredGDNCommit(CustomTestCase):
    def setUp(self):
        patch = mock.patch.dict(fixture.__dict__, WIDTH=4)
        patch.start()
        self.addCleanup(patch.stop)
        torch.manual_seed(20261112)
        self.dispatcher = fixture.GDNKernelDispatcher(
            *[fixture.LinearAttnKernelBackend.TRITON] * 3
        )
        self.template = fixture._template(torch.bfloat16)
        self.layers = fixture._make_layers()
        self.rows = torch.arange(1, fixture.NUM_REAL + 1, device="cuda")
        self.slots = torch.arange(1, fixture.NUM_REAL + 1, device="cuda").int()
        self.padded_rows = torch.cat([self.rows, self.rows.new_zeros(1)])
        self.padded_slots = torch.cat([self.slots, self.slots.new_full((1,), -1)])
        self.qsl = torch.arange(
            0, fixture.NUM_SLOTS * 4 + 1, 4, device="cuda", dtype=torch.int32
        )
        self.batch = fixture._batch(fixture.NUM_SLOTS * 4, self.padded_rows)

    def arm(self, deferred=True):
        arm = fixture._Arm(
            self.template, replayssm=True, dispatcher=self.dispatcher, ring_len=16
        )
        arm.pool.mamba_cache = arm.state
        arm.pool.get_speculative_mamba2_params_all_layers = lambda: arm.state
        arm.pool._slot_siblings = []
        arm.pool.replayssm_write_pos = None
        arm.pool.debug_memory_pool = False
        arm.pool._should_fuse_slot_ops = lambda: False
        arm.pool._copy_temporal_slots = lambda **kw: MambaPool._copy_temporal_slots(
            arm.pool, **kw
        )
        arm.pool.replayssm_commit_stats = torch.zeros(
            fixture.NUM_ROWS, 9, device="cuda", dtype=torch.int64
        )
        arm.backend(self.qsl, self.padded_slots)
        if deferred:
            arm.pool.replayssm_deferred_state = DeferredGDNState(arm.pool)
            arm.pool.replayssm_deferred_state.bind(self.rows, self.slots)
        return arm

    def step(self, arms, *, steps=None, tracks=None, track_steps=None):
        inputs = fixture._dense_inputs()
        outputs = [
            fixture._verify(
                a.hybrid.linear_attn_backend, self.layers, self.batch, inputs
            )
            for a in arms
        ]
        if steps is None:
            steps = torch.zeros(fixture.NUM_REAL, device="cuda", dtype=torch.int64)
        for arm in arms:
            arm.commit(steps, self.rows, tracks, track_steps)
        return outputs

    def dirty(self, *arms):
        for _ in range(3):
            self.step(arms)
        for arm in arms:
            self.assertTrue(
                bool((arm.pool.replayssm_spec_write_pos[self.rows] == 3).all())
            )

    def test_copy_materializes_duplicate_active_sources_once(self):
        live, reference = self.arm(), self.arm()
        self.dirty(live, reference)
        reference.pool.replayssm_deferred_state.materialize(
            self.slots, reason="reference"
        )
        source = torch.cat([self.slots, self.slots[:1]])
        target = torch.tensor([7, 8, 9, 10, 11, 12, 13], device="cuda")
        MambaPool.copy_from(live.pool, source, target)
        self.assertTrue(
            torch.equal(
                live.state.temporal[:, target], reference.state.temporal[:, source]
            )
        )
        self.assertEqual(int(live.pool.replayssm_spec_write_pos.sum()), 0)
        self.assertEqual(read_gdn_commit_stats(live.pool)["forced_materializations"], 6)

    def test_active_cpu_export_restore_continues_without_ring_payload(self):
        live = self.arm()
        self.dirty(live)
        payload = MambaPool.get_cpu_copy(live.pool, self.slots)
        self.assertEqual(len(payload), 2)
        restored, reference = self.arm(), self.arm()
        self.dirty(restored)  # Restore must discard destination pending history.
        MambaPool.load_cpu_copy(restored.pool, payload, self.slots)
        reference.state.temporal[:, self.slots] = payload[1].cuda()
        reference.state.conv[0][:, self.slots] = payload[0][0].cuda()
        live.state.temporal.fill_(float("nan"))
        for name in fixture.RING_FIELDS:
            getattr(live.state, name).fill_(float("nan"))
        for _ in range(24):
            outputs = self.step([restored, reference])
            for a, b in zip(*outputs):
                self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(restored.state.temporal, reference.state.temporal))
        self.assertTrue(torch.equal(restored.state.conv[0], reference.state.conv[0]))

    def test_clear_slots_resets_only_their_owned_rows(self):
        arm = self.arm()
        self.dirty(arm)
        MambaPool.clear_slots(arm.pool, self.slots[:2])
        self.assertEqual(
            arm.pool.replayssm_spec_write_pos[self.rows].tolist(), [0, 0, 3, 3, 3, 3]
        )
        self.assertEqual(int(arm.state.temporal[:, self.slots[:2]].abs().sum()), 0)

    def test_release_unbinds_active_owner_before_slot_reuse(self):
        arm = self.arm()
        self.dirty(arm)
        storage = StateStorage.__new__(StateStorage)
        storage.pool = arm.pool
        storage.slots = SimpleNamespace(free=mock.Mock())
        req = SimpleNamespace(kv=SimpleNamespace(mamba_pool_idx=self.slots[0]))
        storage.release(SimpleNamespace(enable_mamba_extra_buffer=False), req)
        self.assertIsNone(req.kv.mamba_pool_idx)
        state = arm.pool.replayssm_deferred_state
        self.assertEqual(int(state.slot_to_row[self.slots[0]]), -1)
        self.assertEqual(int(state.row_to_slot[self.rows[0]]), -1)
        # The freed slot can become a tracked checkpoint; stale history must
        # not overwrite it when that checkpoint is copied or exported.
        arm.state.temporal[:, self.slots[0]].fill_(0.25)
        MambaPool.get_cpu_copy(arm.pool, self.slots[:1])
        self.assertTrue(bool((arm.state.temporal[:, self.slots[0]] == 0.25).all()))
        state.bind(self.rows[:1], self.slots[:1])
        self.assertEqual(int(arm.pool.replayssm_spec_write_pos[self.rows[0]]), 0)

    def test_fresh_row_reset_preserves_binding_and_other_history(self):
        arm = self.arm()
        self.dirty(arm)
        storage = StateStorage.__new__(StateStorage)
        storage.pool = arm.pool
        storage.reset_request_rows([int(self.rows[0])], device="cuda")
        self.assertEqual(
            arm.pool.replayssm_spec_write_pos[self.rows].tolist(), [0, 3, 3, 3, 3, 3]
        )
        state = arm.pool.replayssm_deferred_state
        self.assertEqual(int(state.slot_to_row[self.slots[0]]), int(self.rows[0]))
        self.assertEqual(int(state.row_to_slot[self.rows[0]]), int(self.slots[0]))
        state.clear()
        self.assertTrue(bool((state.row_to_slot == -1).all()))
        self.assertEqual(int(arm.pool.replayssm_spec_write_pos.sum()), 0)

    def test_acquire_binds_state_before_forward_row_reset(self):
        arm = self.arm()
        arm.pool.replayssm_deferred_state.clear()
        storage = StateStorage.__new__(StateStorage)
        storage.pool = arm.pool
        requests = [
            SimpleNamespace(
                kv=SimpleNamespace(
                    holds_mamba=True, mamba_pool_idx=slot, req_pool_idx=None
                )
            )
            for row, slot in zip(self.rows, self.slots)
        ]
        table = SimpleNamespace(
            device="cuda",
            enable_mamba_extra_buffer=False,
            req_index_to_mamba_index_mapping=torch.zeros(
                fixture.NUM_ROWS, device="cuda", dtype=torch.int32
            ),
        )
        storage.acquire(table, requests, self.rows.tolist())
        storage.reset_request_rows(self.rows.tolist(), device="cuda")
        state = arm.pool.replayssm_deferred_state
        self.assertTrue(torch.equal(state.row_to_slot[self.rows], self.slots))
        self.dirty(arm)
        MambaPool.get_cpu_copy(arm.pool, self.slots)
        self.assertEqual(int(arm.pool.replayssm_spec_write_pos.sum()), 0)

    def test_rejected_non_finite_drafts_across_capacity_folds(self):
        clean, poisoned = self.arm(), self.arm()
        steps = torch.tensor([0, 1, 2, 3, 0, 2], device="cuda")
        for iteration in range(32):
            inputs = fixture._dense_inputs()
            bad = [tuple(t.clone() for t in values) for values in inputs]
            if iteration % 4 == 2:
                for mixed, a, b in bad:
                    for row, step in enumerate(steps.tolist()):
                        rejected = slice(row * 4 + step + 1, (row + 1) * 4)
                        mixed[rejected] = float("nan")
                        a[rejected] = float("inf")
                        b[rejected] = float("nan")
            outputs = [
                fixture._verify(
                    arm.hybrid.linear_attn_backend, self.layers, self.batch, values
                )
                for arm, values in [(clean, inputs), (poisoned, bad)]
            ]
            for expected, actual in zip(*outputs):
                expected = expected.reshape(-1, fixture.V_HEADS, fixture.HEAD_DIM)
                actual = actual.reshape(-1, fixture.V_HEADS, fixture.HEAD_DIM)
                for row, step in enumerate(steps.tolist()):
                    accepted = slice(row * 4, row * 4 + step + 1)
                    self.assertTrue(
                        torch.equal(expected[accepted], actual[accepted]),
                        (iteration, row),
                    )
            for arm in [clean, poisoned]:
                arm.commit(steps, self.rows)
            self.assertTrue(torch.equal(clean.state.temporal, poisoned.state.temporal))
            self.assertTrue(torch.equal(clean.state.conv[0], poisoned.state.conv[0]))
            self.assertTrue(
                torch.equal(
                    clean.pool.replayssm_spec_write_pos,
                    poisoned.pool.replayssm_spec_write_pos,
                )
            )

    def test_deferred_verify_commit_cuda_graph_matches_eager_execution(self):
        arms = [self.arm(), self.arm()]
        inputs = fixture._dense_inputs()
        steps = torch.zeros(6, device="cuda", dtype=torch.int64)
        tracks = torch.arange(7, 13, device="cuda")
        track_steps = torch.tensor([0, -1, 0, -1, 0, -1], device="cuda")

        def run(index):
            out = fixture._verify(
                arms[index].hybrid.linear_attn_backend, self.layers, self.batch, inputs
            )
            arms[index].commit(steps, self.rows, tracks, track_steps)
            return out

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(0)
        torch.cuda.current_stream().wait_stream(stream)
        arms[0].state.temporal.copy_(self.template["temporal"])
        arms[0].state.conv[0].copy_(self.template["conv"])
        for name in fixture.RING_FIELDS:
            getattr(arms[0].state, name).zero_()
        for name in [
            "replayssm_spec_write_pos",
            "replayssm_cache_base",
            "replayssm_is_flush",
            "replayssm_commit_stats",
        ]:
            getattr(arms[0].pool, name).zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(0)
        for _ in range(32):
            for values in inputs:
                for tensor in values:
                    tensor.copy_(fixture._randn(*tensor.shape))
            steps.copy_(torch.randint(0, 4, (6,), device="cuda"))
            track_steps.copy_(
                torch.where(torch.arange(6, device="cuda") % 2 == 0, steps // 2, -1)
            )
            expected = run(1)
            graph.replay()
            for a, b in zip(captured, expected):
                self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(arms[0].state.temporal, arms[1].state.temporal))
            self.assertTrue(torch.equal(arms[0].state.conv[0], arms[1].state.conv[0]))
        self.assertTrue(
            torch.equal(
                arms[0].pool.replayssm_commit_stats, arms[1].pool.replayssm_commit_stats
            )
        )

    def test_tracked_cpu_checkpoint_is_independent_of_active_history(self):
        arm = self.arm()
        tracks = torch.arange(7, 13, device="cuda")
        zero = torch.zeros(6, device="cuda", dtype=torch.int64)
        for _ in range(3):
            self.step([arm], tracks=tracks, track_steps=zero)
        before = arm.state.temporal[:, tracks].clone()
        payload = MambaPool.get_cpu_copy(arm.pool, tracks)
        arm.pool.replayssm_deferred_state.reset(self.slots, forget=True)
        for name in fixture.RING_FIELDS:
            getattr(arm.state, name).fill_(float("nan"))
        restored = self.arm()
        MambaPool.load_cpu_copy(restored.pool, payload, self.slots)
        self.assertTrue(torch.equal(restored.state.temporal[:, self.slots], before))
        self.assertEqual(read_gdn_commit_stats(arm.pool)["forced_materializations"], 0)

    def test_fold_counts_and_early_flush_degeneracy(self):
        eager, deferred = self.arm(False), self.arm()
        for _ in range(20):
            self.step([eager, deferred])
        a, b = (read_gdn_commit_stats(x.pool) for x in [eager, deferred])
        self.assertEqual(a["active_fold_rows"], 120)
        self.assertEqual(b["active_fold_rows"], 12)
        self.assertEqual(b["accepted_tokens"], 120)
        self.assertEqual(b["max_history_depth"], 9)
        self.assertEqual(b["commit_kernel_launches"], 20)
        deferred.pool.replayssm_deferred_state.materialize(self.slots, reason="export")
        self.assertEqual(
            read_gdn_commit_stats(deferred.pool)["forced_materializations"], 6
        )
        arm = self.arm()
        for _ in range(4):
            commit_gdn_replayssm_after_verify(
                spec_state=arm.state,
                write_pos=arm.pool.replayssm_spec_write_pos,
                cache_base=arm.pool.replayssm_cache_base,
                is_flush=arm.pool.replayssm_is_flush,
                state_batch_indices=self.slots,
                replay_indices=self.rows,
                accept_lens=torch.ones_like(self.rows),
                last_correct_step_indices=torch.zeros_like(self.rows),
                fold_every_commit=False,
                max_spec_len=8,
                stats=arm.pool.replayssm_commit_stats,
            )
        self.assertEqual(read_gdn_commit_stats(arm.pool)["active_fold_rows"], 24)


if __name__ == "__main__":
    unittest.main()
