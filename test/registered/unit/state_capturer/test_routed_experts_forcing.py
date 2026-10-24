"""Routed-experts forcing with fake capture tables, and its forced top-k."""

import contextlib
import unittest
from array import array
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.layers.moe.topk import (
    TopK,
    TopKConfig,
    biased_topk_impl,
    fused_topk_softmax_torch_raw_logits,
    fused_topk_torch_native,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.runtime_context import get_forward
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.state_capturer.routed_experts_forcing import (
    REFUSED_DISABLED,
    REFUSED_NO_RETURN_ROUTED_EXPERTS,
    REFUSED_ROUTES_MISSING,
    RoutedExpertsForcing,
    _forced_routing_weights,
    admit_forcing_request,
    apply_forced_routing,
    forcing_forward,
    gate_mirror_problem,
    set_global_routed_experts_forcing,
)
from sglang.srt.state_capturer.routed_experts_store import RouteLayout, RouteSegment
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Layer 0 is dense: its capture column stays 0 and is never held.
LAYOUT = RouteLayout(moe_layer_ids=(1, 2), top_k=2, num_logical_experts=8)
GATE = TopKConfig(top_k=2, renormalize=True, scoring_func="softmax")
# The gate itself, run eagerly: its torch.compile wrapper needs a C++ compiler.
_biased_topk = biased_topk_impl._torchdynamo_orig_callable


class _Engine:
    """Capture tables and one softmax top-2 gate per MoE layer, whose logits depend
    on the weight version and the position. Every computed position writes its
    routes at a fresh KV slot, as the capturer does; slots are never reused."""

    def __init__(self):
        self.host = torch.zeros((512, 3, 2), dtype=torch.int32)
        self.req_to_token = torch.zeros((8, 64), dtype=torch.int64)
        self.forcing = RoutedExpertsForcing(
            layout=LAYOUT,
            max_tokens=10_000,
            host_table=self.host,
            pin_memory=False,
        )
        self.version = 0
        self._next_slot = 0
        self._next_pool_idx = 0

    def gate(self, positions, layer, version=None):
        version = self.version if version is None else version
        generator = torch.Generator().manual_seed(version * 1000 + layer)
        logits = torch.randn((64, 8), generator=generator)[positions]
        weights, ids = torch.softmax(logits, dim=-1).topk(2)
        return ids.int(), weights / weights.sum(-1, keepdim=True), logits

    def request(self, key, tokens, *, force_len, return_routed_experts=True):
        req = Req(
            f"rid-{self._next_pool_idx}",
            "",
            array("q", tokens),
            SamplingParams(),
            return_routed_experts=return_routed_experts,
            routed_experts_start_len=force_len,
        )
        return req, admit_forcing_request(req, key=key)

    def extend(self, req, start, stop):
        """A scheduled extend: positions [start, stop), forced where held."""
        batch = self.forcing.build_batch(
            reqs=[req], prefix_lens=[start], extend_lens=[stop - start]
        )
        slots = self._alloc(req, start, stop)
        with (
            contextlib.nullcontext()
            if batch is None
            else forcing_forward(batch, device="cpu")
        ):
            for layer in LAYOUT.moe_layer_ids:
                ids, weights, logits = self.gate(torch.arange(start, stop), layer)
                if (forced := get_forward().routed_experts_forced) is not None:
                    ids, weights = forced.force_routing(
                        topk_ids=ids,
                        topk_weights=weights,
                        router_logits=logits,
                        layer_id=layer,
                        topk_config=GATE,
                        packed=False,
                    )
                self.host[slots, layer] = ids
        return batch

    def decode(self, req, steps):
        """Decode steps, which compute only new positions and are never forced."""
        for _ in range(steps):
            position = len(req.origin_input_ids) + len(req.output_ids) - 1
            slots = self._alloc(req, position, position + 1)
            for layer in LAYOUT.moe_layer_ids:
                self.host[slots, layer] = self.gate(torch.tensor([position]), layer)[0]
            req.output_ids.append(100 + position)

    def retract(self, req):
        self.forcing.snapshot_on_retract(req, req_to_token=self.req_to_token)
        req.kv.req_pool_idx = None
        req.kv.kv_committed_len = 0

    def collect(self, req):
        """Collect rows [force_len, n - 1) as the result processor does."""
        n = len(req.origin_input_ids) + len(req.output_ids)
        slots = self.req_to_token[
            req.kv.req_pool_idx, req.routed_experts_start_len : n - 1
        ]
        req.routed_experts = self.host[slots]

    def finish(self, req):
        self.collect(req)
        self.forcing.commit_on_finish(req)

    def rows(self, req, start, stop):
        slots = self.req_to_token[req.kv.req_pool_idx, start:stop]
        return LAYOUT.narrow(self.host[slots].numpy())

    def gate_rows(self, start, stop, version):
        positions = torch.arange(start, stop)
        return np.stack(
            [self.gate(positions, layer, version)[0].numpy() for layer in (1, 2)], 1
        )

    def _alloc(self, req, start, stop):
        if req.kv.req_pool_idx is None:
            req.kv.req_pool_idx = self._next_pool_idx
            self._next_pool_idx += 1
        slots = torch.arange(self._next_slot, self._next_slot + stop - start)
        self._next_slot += stop - start
        self.req_to_token[req.kv.req_pool_idx, start:stop] = slots
        req.kv.kv_committed_len = stop
        return slots


def _same_sets(a, b):
    return np.array_equal(np.sort(a, axis=-1), np.sort(b, axis=-1))


class TestRoutedExpertsForcing(CustomTestCase):
    def setUp(self):
        super().setUp()
        enter_scope(self, published_topology())
        self.engine = _Engine()
        set_global_routed_experts_forcing(self.engine.forcing)

    def test_session_walk_through(self):
        """Turn 1 is retracted mid-decode and re-prefills under new weights; turn 2
        re-prefills under newer ones. Every recomputed held row keeps its experts."""
        engine = self.engine
        prompt = list(range(10, 16))
        turn1, refusal = engine.request("s", prompt, force_len=0)
        self.assertIsNone(refusal)
        self.assertEqual(turn1.extra_key, "routed_experts_session:s")
        self.assertIsNone(engine.extend(turn1, 0, 6))  # nothing held yet
        turn1.output_ids.append(99)
        engine.decode(turn1, 3)
        before = engine.rows(turn1, 0, 9)

        engine.retract(turn1)  # 6 prompt + 4 output tokens: rows [0, 9) were kept
        self.assertEqual(turn1.routed_experts_forcing_state.held.end, 9)
        engine.version = 1
        batch = engine.extend(turn1, 0, 10)
        self.assertEqual(int(batch.mask.sum()), 9)
        self.assertTrue(_same_sets(engine.rows(turn1, 0, 9), before))
        self.assertFalse(_same_sets(engine.gate_rows(0, 9, version=1), before))
        turn1.output_ids.append(98)
        engine.decode(turn1, 2)
        engine.finish(turn1)
        committed = engine.rows(turn1, 0, 12)
        self.assertEqual(
            turn1.routed_experts_forcing_state.meta_info(),
            {
                "held_tokens": 0,
                "forced_tokens": 9,
                "unforced_tokens": 0,
                "retract_snapshots": 1,
            },
        )

        tokens = prompt + list(turn1.output_ids) + [7, 8, 9]
        turn2, refusal = engine.request("s", tokens, force_len=12)
        self.assertIsNone(refusal)
        engine.version = 2
        self.assertEqual(int(engine.extend(turn2, 0, 16).mask.sum()), 12)
        self.assertTrue(_same_sets(engine.rows(turn2, 0, 12), committed))
        self.assertFalse(_same_sets(engine.gate_rows(0, 12, version=2), committed))
        self.assertEqual(
            turn2.routed_experts_forcing_state.meta_info(),
            {
                "held_tokens": 12,
                "forced_tokens": 12,
                "unforced_tokens": 0,
                "retract_snapshots": 0,
            },
        )

    def test_radix_namespace_composes_with_an_existing_extra_key(self):
        req = Req(
            "r",
            "",
            array("q", [1, 2]),
            SamplingParams(),
            extra_key="ns",
            lora_id="lora",
        )
        req.return_routed_experts = True
        self.assertIsNone(admit_forcing_request(req, key="s"))
        self.assertEqual(req.extra_key, "nslora|routed_experts_session:s")

    def test_refused_requests_never_run(self):
        """Any turn whose held rows fall short of its force length is refused
        before it computes anything, and says why in its meta_info."""
        engine = self.engine
        prompt = list(range(20, 32))
        turn1, _ = engine.request("s", prompt, force_len=0)
        engine.extend(turn1, 0, 12)
        turn1.output_ids.append(5)
        engine.finish(turn1)

        tokens = prompt + [5, 6]
        cases = [
            ("other", tokens, {}, REFUSED_ROUTES_MISSING, "held=0 requested=12"),
            (
                "s",
                prompt[:3] + [0] * 9 + [5, 6],
                {},
                REFUSED_ROUTES_MISSING,
                "held=3 requested=12",
            ),
            (
                "s",
                tokens,
                {"return_routed_experts": False},
                REFUSED_NO_RETURN_ROUTED_EXPERTS,
                "set return_routed_experts",
            ),
        ]
        for key, token_ids, kwargs, code, detail in cases:
            req, refusal = engine.request(key, token_ids, force_len=12, **kwargs)
            # The reason code is the message up to its first ":".
            self.assertEqual(refusal, f"{code}: {detail}")
            self.assertIsNone(req.extra_key)
            self.assertEqual(
                req.routed_experts_forcing_state.meta_info()["refused"], code
            )
        self.assertEqual(engine.forcing.refused_requests, 3)

        set_global_routed_experts_forcing(None)
        req, refusal = engine.request("s", prompt, force_len=0)
        self.assertEqual(
            refusal,
            f"{REFUSED_DISABLED}: start the server with --enable-routed-experts-forcing",
        )
        self.assertIsNone(req.routed_experts_forcing_state)

    def test_retract_snapshot_excludes_in_flight_rows(self):
        """Under overlap scheduling a retract can land while the request's last
        forward is in flight, before the capture writes its rows: those rows must
        be recomputed, never read."""
        engine = self.engine
        req, _ = engine.request("s", list(range(8)), force_len=0)
        engine.extend(req, 0, 8)
        # The prefill's rows are not written yet: an expert id that cannot exist.
        engine.host[engine.req_to_token[req.kv.req_pool_idx, :8]] = 200
        engine.retract(req)
        state = req.routed_experts_forcing_state
        self.assertEqual((state.held.end, state.retract_snapshots), (0, 0))

        engine.extend(req, 0, 8)
        req.output_ids.append(5)
        engine.decode(req, 2)  # positions 8 and 9; 3 tokens kept
        in_flight = engine._alloc(req, 10, 11)
        engine.host[in_flight] = 200
        engine.retract(req)
        self.assertEqual(state.held.end, 10)

    def test_failed_commit_refuses_next_turn(self):
        """A commit that comes after another attempt of its turn committed, or
        whose collected routes do not fit its tokens, drops the session's rows
        instead of raising in the scheduler: the next turn is refused."""
        prompt = list(range(30, 36))
        tokens = prompt + [1, 2, 3]
        for spoil in ("stale", "short", "missing"):
            with self.subTest(spoil):
                engine = _Engine()
                set_global_routed_experts_forcing(engine.forcing)
                turn1, _ = engine.request("s", prompt, force_len=0)
                engine.extend(turn1, 0, 6)
                turn1.output_ids.append(1)
                engine.finish(turn1)
                attempts = [engine.request("s", tokens, force_len=6)[0] for _ in "ab"]
                for req in attempts:
                    engine.extend(req, 0, 9)
                    req.output_ids.append(4)
                if spoil == "stale":
                    engine.finish(attempts[1])
                engine.collect(attempts[0])
                if spoil == "short":
                    attempts[0].routed_experts = attempts[0].routed_experts[:-1]
                elif spoil == "missing":
                    attempts[0].routed_experts = None
                engine.forcing.commit_on_finish(attempts[0])
                self.assertEqual(engine.forcing.dropped_commits, 1)
                _, refusal = engine.request("s", tokens + [4], force_len=9)
                self.assertEqual(
                    refusal, f"{REFUSED_ROUTES_MISSING}: held=0 requested=9"
                )

    def test_unchanged_set_keeps_the_gate_order_and_weights(self):
        """A held row whose experts the gate picks again in another order moves no
        bit, so forcing composes with captures that sort each row."""
        forcing = self.engine.forcing
        positions = torch.arange(4)
        ids, weights, logits = self.engine.gate(positions, layer=1)
        req, _ = self.engine.request("s", [1, 2, 3, 4, 5], force_len=0)
        held = ids.flip(-1).numpy().astype(np.uint8)
        state = req.routed_experts_forcing_state
        state.held = state.held.extended(
            RouteSegment(start=0, routes=np.stack([held, held], 1))
        )
        batch = forcing.build_batch(reqs=[req], prefix_lens=[0], extend_lens=[4])
        with forcing_forward(batch, device="cpu"):
            forced = get_forward().routed_experts_forced
            for layer in LAYOUT.moe_layer_ids:
                out_ids, out_weights = forced.force_routing(
                    topk_ids=ids,
                    topk_weights=weights,
                    router_logits=logits,
                    layer_id=layer,
                    topk_config=GATE,
                    packed=False,
                )
                self.assertTrue(torch.equal(out_ids, ids))
                self.assertTrue(torch.equal(out_weights, weights))

    def test_unforced_layer_raises(self):
        req, _ = self.engine.request("s", [1, 2, 3], force_len=0)
        state = req.routed_experts_forcing_state
        state.held = state.held.extended(
            RouteSegment(start=0, routes=np.zeros((2, 2, 2), dtype=np.uint8))
        )
        batch = self.engine.forcing.build_batch(
            reqs=[req], prefix_lens=[0], extend_lens=[3]
        )
        ids, weights, logits = self.engine.gate(torch.arange(3), layer=1)
        with self.assertRaisesRegex(RuntimeError, r"MoE layers \[2\]"):
            with forcing_forward(batch, device="cpu"):
                get_forward().routed_experts_forced.force_routing(
                    topk_ids=ids,
                    topk_weights=weights,
                    router_logits=logits,
                    layer_id=1,
                    topk_config=GATE,
                    packed=False,
                )


class TestCreate(CustomTestCase):
    def test_refuses_gates_it_cannot_force(self):
        """Forcing starts only for gates whose weights it reproduces, and its
        layout keeps the captured TopK layers of the loaded model."""
        capturer = SimpleNamespace(
            topk_size=2, host_cache=SimpleNamespace(buffer=torch.zeros((8, 4, 2)))
        )
        expert_location = SimpleNamespace(num_logical_experts=16)

        def create(*gate_kwargs):
            model = torch.nn.ModuleList(
                TopK(top_k=2, layer_id=layer_id, **kwargs)
                for layer_id, kwargs in zip((1, 3), gate_kwargs)
            )
            return RoutedExpertsForcing.create(
                model=model, capturer=capturer, expert_location=expert_location
            )

        with published_topology():
            self.assertEqual(
                create({}, {}).layout,
                RouteLayout(moe_layer_ids=(1, 3), top_k=2, num_logical_experts=16),
            )
            bias = {"correction_bias": torch.zeros(16)}
            with self.assertRaisesRegex(ValueError, "softmax gates with a correction"):
                create({}, bias)


class TestForcedRouting(CustomTestCase):
    def test_forced_weights_are_the_gates_own_for_its_choice(self):
        """For each gate config that gate_mirror_problem accepts, the forced
        weights equal the gate's own weights for the same expert set."""
        torch.manual_seed(0)
        logits = torch.randn(16, 64)
        hidden = torch.empty(16, 1)
        bias = torch.randn(64)
        gates = [
            # (scoring_func, renormalize, routed scaling on output, gate output)
            (
                "softmax",
                True,
                None,
                fused_topk_torch_native(
                    hidden_states=hidden, gating_output=logits, topk=8, renormalize=True
                ),
            ),
            # flashinfer_trtllm_routed's softmax over the chosen raw logits.
            (
                "softmax",
                True,
                None,
                fused_topk_softmax_torch_raw_logits(
                    hidden_states=hidden, gating_output=logits, topk=8, renormalize=True
                ),
            ),
            (
                "sigmoid",
                False,
                None,
                fused_topk_torch_native(
                    hidden_states=hidden,
                    gating_output=logits,
                    topk=8,
                    renormalize=False,
                    scoring_func="sigmoid",
                ),
            ),
            # A fused shared expert in the last column, which forcing leaves alone.
            (
                "sigmoid",
                True,
                2.5,
                _biased_topk(
                    hidden_states=hidden,
                    gating_output=logits,
                    correction_bias=bias,
                    topk=9,
                    renormalize=True,
                    num_fused_shared_experts=1,
                    routed_scaling_factor=2.5,
                    apply_routed_scaling_factor_on_output=True,
                ),
            ),
            *(
                (
                    "sqrtsoftplus",
                    renormalize,
                    None,
                    _biased_topk(
                        hidden_states=hidden,
                        gating_output=logits,
                        correction_bias=bias,
                        topk=8,
                        renormalize=renormalize,
                        scoring_func="sqrtsoftplus",
                    ),
                )
                for renormalize in (True, False)
            ),
        ]
        for scoring_func, renormalize, scaling, (weights, ids) in gates:
            with self.subTest(scoring_func=scoring_func, renormalize=renormalize):
                got = _forced_routing_weights(
                    logits,
                    ids[:, :8],
                    scoring_func=scoring_func,
                    renormalize=renormalize,
                    routed_scaling_factor=scaling,
                    apply_routed_scaling_factor_on_output=scaling is not None,
                )
                torch.testing.assert_close(got, weights[:, :8])

    def test_apply_touches_only_masked_rows_whose_expert_set_changed(self):
        torch.manual_seed(1)
        logits = torch.randn(6, 32)
        weights, ids = fused_topk_torch_native(
            hidden_states=logits, gating_output=logits, topk=4, renormalize=True
        )
        ids = ids.int()
        # A fused shared-expert column after the routed ones.
        topk_ids = torch.cat([ids, torch.full((6, 1), 32, dtype=torch.int32)], dim=1)
        topk_weights = torch.cat([weights, torch.full((6, 1), 0.4)], dim=1)

        forced = ids.clone()
        forced[1] = torch.tensor([0, 1, 2, 3])  # changed and masked
        forced[2] = forced[2].flip(0)  # same set, other order: untouched
        forced[4] = torch.tensor([5, 6, 7, 8])  # changed but not masked
        mask = torch.tensor([True, True, True, False, False, True])
        new_ids, new_weights = apply_forced_routing(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            router_logits=logits,
            forced_ids=forced,
            mask=mask,
            scoring_func="softmax",
            renormalize=True,
            routed_scaling_factor=None,
            apply_routed_scaling_factor_on_output=False,
        )
        for row in (0, 2, 3, 4, 5):
            self.assertTrue(torch.equal(new_ids[row], topk_ids[row]))
            self.assertTrue(torch.equal(new_weights[row], topk_weights[row]))
        self.assertEqual(new_ids[1, :4].tolist(), [0, 1, 2, 3])
        probs = logits.softmax(-1)[1, :4]
        torch.testing.assert_close(new_weights[1, :4], probs / probs.sum())
        self.assertTrue(torch.equal(new_ids[:, 4], topk_ids[:, 4]))
        self.assertTrue(torch.equal(new_weights[:, 4], topk_weights[:, 4]))
        self.assertEqual(topk_ids[1, :4].tolist(), ids[1].tolist(), "inputs unchanged")

        with self.assertRaisesRegex(ValueError, "forcing rows"):
            apply_forced_routing(
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                router_logits=logits,
                forced_ids=forced,
                mask=mask[:5],
                scoring_func="softmax",
                renormalize=True,
                routed_scaling_factor=None,
                apply_routed_scaling_factor_on_output=False,
            )

    def test_gate_mirror_problem_refuses_unmirrored_weights(self):
        def problem(**overrides):
            gate = dict(
                scoring_func="sigmoid",
                has_correction_bias=True,
                has_custom_routing=False,
                renormalize=True,
                apply_routed_scaling_factor_on_output=True,
                num_fused_shared_experts=1,
            )
            return gate_mirror_problem(**{**gate, **overrides})

        self.assertIsNone(problem())
        self.assertIsNone(
            problem(
                scoring_func="softmax",
                has_correction_bias=False,
                apply_routed_scaling_factor_on_output=False,
            )
        )
        self.assertIsNone(
            problem(
                renormalize=False,
                apply_routed_scaling_factor_on_output=False,
                num_fused_shared_experts=0,
            )
        )
        refused = (
            dict(has_custom_routing=True),
            dict(scoring_func="tanh"),
            dict(scoring_func="softmax", apply_routed_scaling_factor_on_output=False),
            dict(scoring_func="softmax", has_correction_bias=False),
            dict(
                scoring_func="softmax",
                has_correction_bias=False,
                apply_routed_scaling_factor_on_output=False,
                renormalize=False,
            ),
            dict(renormalize=False, num_fused_shared_experts=0),
            dict(renormalize=False, apply_routed_scaling_factor_on_output=False),
        )
        for overrides in refused:
            with self.subTest(**overrides):
                self.assertIsNotNone(problem(**overrides))


if __name__ == "__main__":
    unittest.main()
