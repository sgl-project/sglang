# SPDX-License-Identifier: Apache-2.0
"""Unit tests for per-request TeaCache state in continuous batching.

Covers the explicit cache-state interface, A/B/A swap isolation on a shared
model, dual-transformer (per-phase) snapshots, drain/resume serialization,
and the packed per-row hit/miss plan.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.cache.teacache import (
    TeaCacheMixin,
    TeaCachePackedMember,
    TeaCachePackedPlan,
    TeaCacheRequestState,
    compute_l1_decision,
    resolve_teacache_phase_state,
)
from sglang.multimodal_gen.runtime.managers.continuous_batching import (
    ContinuousDenoisingCoordinator,
    TeaCacheStateIsolator,
)


class _FakeTeaCacheModel(TeaCacheMixin):
    """Minimal model exposing the mixin's state and explicit interface."""

    def __init__(self, prefix: str = "wan"):
        self.config = SimpleNamespace(prefix=prefix)
        self._init_teacache_state()


class _FakeTeaCacheParams:
    def __init__(self, thresh=0.2, start=1, end=100, use_ret_steps=False):
        self.teacache_thresh = thresh
        self._start = start
        self._end = end
        self.use_ret_steps = use_ret_steps

    def get_skip_boundaries(self, num_inference_steps, do_cfg):
        return self._start, self._end

    def get_coefficients(self):
        # Identity polynomial: rescaled L1 == raw relative L1.
        return [1.0, 0.0]


def _fill_state(state: TeaCacheRequestState, seed: int) -> TeaCacheRequestState:
    generator = torch.Generator().manual_seed(seed)
    state.cnt = seed
    state.accumulated_rel_l1_distance = 0.25 * seed
    state.previous_modulated_input = torch.randn(2, 3, generator=generator)
    state.previous_residual = torch.randn(2, 3, generator=generator)
    return state


class TestExplicitStateInterface(unittest.TestCase):
    def test_capture_install_round_trip(self):
        model = _FakeTeaCacheModel()
        model.cnt = 7
        model.accumulated_rel_l1_distance = 0.5
        model.previous_modulated_input = torch.ones(2, 2)
        model.previous_residual_negative = torch.zeros(3)

        snapshot = model.capture_teacache_state()
        self.assertIsInstance(snapshot, TeaCacheRequestState)
        self.assertEqual(snapshot.cnt, 7)
        self.assertIs(snapshot.previous_modulated_input, model.previous_modulated_input)

        model.reset_teacache_state()
        self.assertEqual(model.cnt, 0)
        self.assertIsNone(model.previous_modulated_input)

        model.install_teacache_state(snapshot)
        self.assertEqual(model.cnt, 7)
        self.assertEqual(model.accumulated_rel_l1_distance, 0.5)
        self.assertIs(model.previous_modulated_input, snapshot.previous_modulated_input)
        self.assertIs(
            model.previous_residual_negative, snapshot.previous_residual_negative
        )

    def test_install_none_resets(self):
        model = _FakeTeaCacheModel()
        model.cnt = 3
        model.previous_residual = torch.ones(4)
        model.install_teacache_state(None)
        self.assertEqual(model.cnt, 0)
        self.assertIsNone(model.previous_residual)

    def test_non_cfg_model_skips_negative_fields(self):
        model = _FakeTeaCacheModel(prefix="flux")
        self.assertFalse(model._supports_cfg_cache)
        snapshot = TeaCacheRequestState(cnt=2, previous_residual_negative=torch.ones(2))
        model.install_teacache_state(snapshot)
        self.assertEqual(model.cnt, 2)
        self.assertFalse(hasattr(model, "previous_residual_negative"))

    def test_isolator_prefers_explicit_interface(self):
        model = _FakeTeaCacheModel()
        model.cnt = 5
        snapshot = TeaCacheStateIsolator.capture(model)
        self.assertIsInstance(snapshot, TeaCacheRequestState)
        TeaCacheStateIsolator.install(model, None)
        self.assertEqual(model.cnt, 0)
        TeaCacheStateIsolator.install(model, snapshot)
        self.assertEqual(model.cnt, 5)

    def test_isolator_legacy_attr_scan(self):
        class _LegacyModel:
            def __init__(self):
                self.cnt = 4
                self.enable_teacache = True
                self.is_cfg_negative = False
                self.previous_residual = torch.ones(2)
                self.accumulated_rel_l1_distance = 0.75
                self.unrelated = "untouched"

            def reset_teacache_state(self):
                self.cnt = 0
                self.previous_residual = None
                self.accumulated_rel_l1_distance = 0.0

        model = _LegacyModel()
        snapshot = TeaCacheStateIsolator.capture(model)
        self.assertIsInstance(snapshot, dict)
        self.assertNotIn("unrelated", snapshot)
        TeaCacheStateIsolator.install(model, None)
        self.assertEqual(model.cnt, 0)
        TeaCacheStateIsolator.install(model, snapshot)
        self.assertEqual(model.cnt, 4)
        self.assertEqual(model.accumulated_rel_l1_distance, 0.75)


class _SwapHarness:
    """Drives coordinator swap in/out without a full coordinator."""

    def __init__(self, model, transformer_2=None):
        self.model = model
        self.transformer_2 = transformer_2

    def make_coordinator(self):
        coordinator = ContinuousDenoisingCoordinator.__new__(
            ContinuousDenoisingCoordinator
        )
        coordinator.allow_step_caches = True
        coordinator.server_args = SimpleNamespace(cb_packed_teacache=False)
        coordinator.denoising_stage = SimpleNamespace(transformer_2=self.transformer_2)
        return coordinator

    def make_state(self, model):
        return SimpleNamespace(
            req=SimpleNamespace(enable_teacache=True),
            current_step=SimpleNamespace(current_model=model),
            teacache_state=None,
        )


class TestABASwapIsolation(unittest.TestCase):
    def test_a_b_a_round_trip_is_isolated(self):
        model = _FakeTeaCacheModel()
        harness = _SwapHarness(model)
        coordinator = harness.make_coordinator()
        state_a = harness.make_state(model)
        state_b = harness.make_state(model)

        # A runs one step and stashes some cache state.
        swap = coordinator._swap_in_teacache_state(state_a)
        model.cnt = 3
        model.previous_residual = torch.full((2,), 3.0)
        model.accumulated_rel_l1_distance = 0.3
        coordinator._swap_out_teacache_state(state_a, swap)
        a_snapshot = state_a.teacache_state["transformer"]

        # B runs with completely different state.
        swap = coordinator._swap_in_teacache_state(state_b)
        self.assertEqual(model.cnt, 0)  # fresh reset for B
        model.cnt = 9
        model.previous_residual = torch.full((2,), 9.0)
        coordinator._swap_out_teacache_state(state_b, swap)

        # A again: model must see exactly A's state, untouched by B.
        swap = coordinator._swap_in_teacache_state(state_a)
        self.assertEqual(model.cnt, 3)
        self.assertEqual(model.accumulated_rel_l1_distance, 0.3)
        torch.testing.assert_close(model.previous_residual, torch.full((2,), 3.0))
        coordinator._swap_out_teacache_state(state_a, swap)
        torch.testing.assert_close(
            state_a.teacache_state["transformer"].previous_residual,
            a_snapshot.previous_residual,
        )
        self.assertEqual(state_a.teacache_state["transformer"].cnt, 3)
        self.assertEqual(state_b.teacache_state["transformer"].cnt, 9)

    def test_dual_transformer_phase_snapshots(self):
        model_1 = _FakeTeaCacheModel()
        model_2 = _FakeTeaCacheModel()
        harness = _SwapHarness(model_1, transformer_2=model_2)
        coordinator = harness.make_coordinator()
        state = harness.make_state(model_1)

        # Phase 1 on the high-noise expert.
        swap = coordinator._swap_in_teacache_state(state)
        model_1.cnt = 11
        model_1.previous_residual = torch.full((2,), 1.0)
        coordinator._swap_out_teacache_state(state, swap)

        # Cross the boundary onto the low-noise expert.
        state.current_step = SimpleNamespace(current_model=model_2)
        swap = coordinator._swap_in_teacache_state(state)
        # cnt carries so skip windows keep meaning; tensors never cross.
        self.assertEqual(model_2.cnt, 11)
        self.assertIsNone(model_2.previous_residual)
        model_2.cnt = 12
        model_2.previous_residual = torch.full((2,), 2.0)
        coordinator._swap_out_teacache_state(state, swap)

        phase_states = state.teacache_state
        self.assertEqual(set(phase_states), {"transformer", "transformer_2"})
        torch.testing.assert_close(
            phase_states["transformer"].previous_residual, torch.full((2,), 1.0)
        )
        torch.testing.assert_close(
            phase_states["transformer_2"].previous_residual, torch.full((2,), 2.0)
        )


class TestPhaseResolution(unittest.TestCase):
    def test_new_phase_seeds_counter_not_tensors(self):
        phase_states = {}
        first = resolve_teacache_phase_state(phase_states, "transformer", create=True)
        _fill_state(first, seed=6)
        second = resolve_teacache_phase_state(phase_states, "transformer_2")
        self.assertIsNotNone(second)
        self.assertEqual(second.cnt, 6)
        self.assertIsNone(second.previous_modulated_input)
        self.assertIsNone(second.previous_residual)
        # Resolving again returns the same object.
        self.assertIs(
            resolve_teacache_phase_state(phase_states, "transformer_2"), second
        )

    def test_create_materializes_missing_state(self):
        phase_states = {"transformer": None}
        state = resolve_teacache_phase_state(phase_states, "transformer", create=True)
        self.assertIsInstance(state, TeaCacheRequestState)
        self.assertIs(phase_states["transformer"], state)


class TestDrainResumeSerialization(unittest.TestCase):
    def test_payload_round_trip(self):
        state = _fill_state(TeaCacheRequestState(), seed=4)
        payload = state.to_payload()
        self.assertEqual(payload["cnt"], 4)
        restored = TeaCacheRequestState.from_payload(payload, device="cpu")
        self.assertEqual(restored.cnt, 4)
        torch.testing.assert_close(
            restored.previous_modulated_input, state.previous_modulated_input
        )
        torch.testing.assert_close(restored.previous_residual, state.previous_residual)

    def test_coordinator_export_import(self):
        request_state = SimpleNamespace(
            teacache_state={
                "transformer": _fill_state(TeaCacheRequestState(), seed=2),
                "transformer_2": None,
            }
        )
        payloads = ContinuousDenoisingCoordinator._export_teacache_states(request_state)
        self.assertEqual(set(payloads), {"transformer"})
        restored = ContinuousDenoisingCoordinator._import_teacache_states(
            payloads, torch.device("cpu")
        )
        self.assertEqual(restored["transformer"].cnt, 2)
        torch.testing.assert_close(
            restored["transformer"].previous_residual,
            request_state.teacache_state["transformer"].previous_residual,
        )
        self.assertIsNone(
            ContinuousDenoisingCoordinator._export_teacache_states(
                SimpleNamespace(teacache_state=None)
            )
        )


class TestPackedMemberDecisions(unittest.TestCase):
    def _mixin_reference_decisions(self, inputs, params, num_steps):
        """Reference decisions using the mixin's attr-based path."""
        model = _FakeTeaCacheModel()
        model.reset_teacache_state()
        decisions = []
        for step, modulated in enumerate(inputs):
            start, end = params.get_skip_boundaries(num_steps, False)
            is_boundary = model.cnt < start or model.cnt >= end
            should_calc = model._compute_teacache_decision(
                modulated_inp=modulated,
                is_boundary_step=is_boundary,
                coefficients=params.get_coefficients(),
                teacache_thresh=params.teacache_thresh,
            )
            decisions.append(should_calc)
            model.cnt += 1
        return decisions

    def test_member_matches_mixin_reference(self):
        torch.manual_seed(0)
        num_steps = 6
        params = _FakeTeaCacheParams(thresh=0.15, start=1, end=100)
        base = torch.randn(1, 4)
        # Small perturbations so some steps hit and some miss.
        inputs = [
            base + 0.01 * step * torch.ones_like(base) for step in range(num_steps)
        ]

        reference = self._mixin_reference_decisions(inputs, params, num_steps)

        member = TeaCachePackedMember(
            row_slice=slice(0, 1),
            state=TeaCacheRequestState(),
            step_index=0,
            num_inference_steps=num_steps,
            do_cfg=False,
            teacache_params=params,
        )
        packed = []
        for step, modulated in enumerate(inputs):
            member.step_index = step
            packed.append(member.decide(modulated, is_cfg_negative=False))
            member.advance()
        self.assertEqual(packed, reference)
        # At least one skip must occur for this test to be meaningful.
        self.assertIn(False, packed)

    def test_disabled_member_always_computes(self):
        member = TeaCachePackedMember(row_slice=slice(0, 1), state=None)
        self.assertTrue(member.decide(torch.ones(1, 2), is_cfg_negative=False))
        member.advance()  # no-op without state

    def test_first_step_resets_state(self):
        state = _fill_state(TeaCacheRequestState(), seed=9)
        member = TeaCachePackedMember(
            row_slice=slice(0, 1),
            state=state,
            step_index=0,
            num_inference_steps=4,
            do_cfg=False,
            teacache_params=_FakeTeaCacheParams(),
        )
        self.assertTrue(member.decide(torch.ones(1, 2), is_cfg_negative=False))
        self.assertEqual(state.cnt, 0)
        self.assertIsNone(state.previous_residual)

    def test_residual_stash_and_retrieve_per_branch(self):
        state = TeaCacheRequestState()
        member = TeaCachePackedMember(row_slice=slice(0, 1), state=state)
        pos = torch.ones(1, 2)
        neg = torch.zeros(1, 2)
        member.stash_residual(pos, is_cfg_negative=False)
        member.stash_residual(neg, is_cfg_negative=True)
        torch.testing.assert_close(member.cached_residual(False), pos)
        torch.testing.assert_close(member.cached_residual(True), neg)

    def test_plan_partition(self):
        params = _FakeTeaCacheParams(thresh=1000.0, start=0, end=100)
        # Warm member: baseline set, huge threshold -> skips.
        warm_state = TeaCacheRequestState(
            cnt=2,
            previous_modulated_input=torch.ones(1, 2),
            previous_residual=torch.zeros(1, 2),
        )
        warm = TeaCachePackedMember(
            row_slice=slice(0, 1),
            state=warm_state,
            step_index=2,
            num_inference_steps=8,
            teacache_params=params,
        )
        cold = TeaCachePackedMember(
            row_slice=slice(1, 2),
            state=TeaCacheRequestState(),
            step_index=2,
            num_inference_steps=8,
            teacache_params=params,
        )
        disabled = TeaCachePackedMember(row_slice=slice(2, 3), state=None)
        plan = TeaCachePackedPlan(members=[warm, cold, disabled])
        self.assertTrue(plan.any_enabled)

        modulated = torch.zeros(3, 2)
        compute, skip = plan.partition(modulated, is_cfg_negative=False)
        self.assertEqual([m.row_slice for m in skip], [slice(0, 1)])
        self.assertEqual([m.row_slice for m in compute], [slice(1, 2), slice(2, 3)])


class TestPackedGatherScatterForward(unittest.TestCase):
    """Row-level gather/scatter through the Wan block loop."""

    def _make_wan_stub(self):
        try:
            from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
                WanTransformer3DModel,
            )
        except Exception as e:  # pragma: no cover - env without heavy deps
            self.skipTest(f"wanvideo import unavailable: {e}")

        class _Block:
            def __call__(self, hidden, encoder, proj, freqs):
                # Row-local, deterministic transform.
                return hidden + 1.0 + encoder.sum(dim=1, keepdim=True) * 0.0

        stub = SimpleNamespace(blocks=[_Block(), _Block()])
        return WanTransformer3DModel._forward_blocks_with_packed_teacache, stub

    def test_skip_rows_use_cached_residual(self):
        forward, stub = self._make_wan_stub()
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            set_forward_context,
        )

        params = _FakeTeaCacheParams(thresh=1000.0, start=0, end=100)
        warm_state = TeaCacheRequestState(
            cnt=2,
            previous_modulated_input=torch.ones(1, 6),
            # Residual from a previous compute: out - in == +2 per block pair.
            previous_residual=torch.full((1, 4, 3), 2.0),
        )
        warm = TeaCachePackedMember(
            row_slice=slice(0, 1),
            state=warm_state,
            step_index=2,
            num_inference_steps=8,
            teacache_params=params,
        )
        cold = TeaCachePackedMember(
            row_slice=slice(1, 2),
            state=TeaCacheRequestState(),
            step_index=2,
            num_inference_steps=8,
            teacache_params=params,
        )
        plan = TeaCachePackedPlan(members=[warm, cold])

        hidden = torch.zeros(3, 4, 3)  # row 2 is uncovered padding
        encoder = torch.zeros(3, 5, 3)
        proj = torch.zeros(3, 6)
        temb = torch.zeros(3, 6)

        with set_forward_context(
            current_timestep=None,
            attn_metadata=None,
            forward_batch=SimpleNamespace(is_cfg_negative=False),
        ):
            output = forward(stub, plan, hidden, encoder, proj, temb, None)

        # Skip row: input + cached residual = 0 + 2.
        torch.testing.assert_close(output[0], torch.full((4, 3), 2.0))
        # Compute rows: two blocks add 1 each.
        torch.testing.assert_close(output[1], torch.full((4, 3), 2.0))
        torch.testing.assert_close(output[2], torch.full((4, 3), 2.0))
        # The cold member stashed its fresh residual and advanced.
        torch.testing.assert_close(
            cold.state.previous_residual, torch.full((1, 4, 3), 2.0)
        )
        self.assertEqual(cold.state.cnt, 1)
        self.assertEqual(warm.state.cnt, 3)

    def test_force_compute_keeps_state_but_never_skips(self):
        forward, stub = self._make_wan_stub()
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            set_forward_context,
        )

        params = _FakeTeaCacheParams(thresh=1000.0, start=0, end=100)
        warm_state = TeaCacheRequestState(
            cnt=2,
            previous_modulated_input=torch.ones(1, 6),
            previous_residual=torch.full((1, 4, 3), 5.0),
        )
        warm = TeaCachePackedMember(
            row_slice=slice(0, 1),
            state=warm_state,
            step_index=2,
            num_inference_steps=8,
            teacache_params=params,
        )
        plan = TeaCachePackedPlan(members=[warm])
        hidden = torch.zeros(1, 4, 3)
        encoder = torch.zeros(1, 5, 3)
        proj = torch.zeros(1, 6)
        temb = torch.zeros(1, 6)

        with set_forward_context(
            current_timestep=None,
            attn_metadata=None,
            forward_batch=SimpleNamespace(is_cfg_negative=False),
        ):
            output = forward(
                stub, plan, hidden, encoder, proj, temb, None, force_compute=True
            )

        # Blocks ran (no skip), residual refreshed from this compute.
        torch.testing.assert_close(output[0], torch.full((4, 3), 2.0))
        torch.testing.assert_close(
            warm.state.previous_residual, torch.full((1, 4, 3), 2.0)
        )


class TestPureDecisionFunction(unittest.TestCase):
    def test_missing_baseline_forces_compute(self):
        accum, should_calc = compute_l1_decision(
            torch.ones(2), None, 0.5, [1.0, 0.0], 0.1
        )
        self.assertEqual(accum, 0.0)
        self.assertTrue(should_calc)

    def test_threshold_crossing_resets_accumulator(self):
        baseline = torch.ones(4)
        accum, should_calc = compute_l1_decision(
            baseline * 2.0, baseline, 0.0, [1.0, 0.0], 0.5
        )
        # rel L1 = 1.0 >= 0.5 -> compute and reset.
        self.assertTrue(should_calc)
        self.assertEqual(accum, 0.0)

    def test_below_threshold_accumulates(self):
        baseline = torch.ones(4)
        accum, should_calc = compute_l1_decision(
            baseline * 1.1, baseline, 0.05, [1.0, 0.0], 0.5
        )
        self.assertFalse(should_calc)
        self.assertAlmostEqual(accum, 0.15, places=5)


if __name__ == "__main__":
    unittest.main()
