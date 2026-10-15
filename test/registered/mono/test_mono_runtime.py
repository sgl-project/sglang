"""CPU tests of the model-agnostic monokernel runtime (sglang/kernels/ops/mono) against a fake spec and kernels."""

import types
import unittest
import unittest.mock
from pathlib import Path

import torch

from sglang.kernels.ops.mono import check, guards, runtime
from sglang.kernels.ops.mono.spec import (
    BITWISE,
    DeviceLimits,
    Exactness,
    Feature,
    LayerStateBinding,
    MonoContext,
    MonoSpec,
    Numerics,
    Point,
    Requirements,
    Span,
    check_spans,
    span_label,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeKernel:
    def __init__(self, rows):
        self.rows, self.launches = rows, []

    def launch(self, step, layer_id, *args, **kwargs):
        self.launches.append((step.rows, layer_id))
        return args


FFN = (Span("ffn", Point.ATTN_PARTIAL, Point.LAYER_OUT),)
WHOLE = (Span("layer", Point.LAYER_IN, Point.LAYER_OUT),)
LIMITS = DeviceLimits(widths=(1, 2, 6, 12), tp_sizes=(2, 4), resident_blocks=256)


class FakeModel:
    rows_per_request = (1,)

    def __init__(self, spans, make_kernel):
        self.spans, self._make_kernel = spans, make_kernel

    def kernel(self, rows):
        return self._make_kernel(rows)

    def span(self, route):
        return lambda *args: ("span", route.spans[0].name, route.step.rows, args)


def fake_spec(kernel=FakeKernel, **kw):
    def spans(cfg):
        return {0: FFN, 1: WHOLE, 3: FFN}

    base = dict(
        name="fake",
        devices={"gfx950": LIMITS},
        features=frozenset({Feature.SPEC_VERIFY}),
        requires=Requirements(),
        spans=spans,
        numerics=BITWISE,
        config_disable_reasons=lambda c, h: [],
        model_disable_reasons=lambda m: [],
        build=lambda model, ctx: FakeModel(spans(None), kernel),
    )
    base.update(kw)
    return MonoSpec(**base)


def make_runtime(spec):
    rt = runtime.MonoRuntime(spec, ctx(), spec.build(None, ctx()))
    rt.build_widths(LIMITS.widths)
    return rt


def cfg(**kw):
    base = dict(
        tp_size=4,
        nnodes=1,
        pp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        ep_size=1,
        enable_dp_attention=False,
        enable_eplb=False,
        speculative_algorithm="DSPARK",
        speculative_eagle_topk=1,
        enable_two_batch_overlap=False,
        enable_lora=False,
        disaggregation_mode="null",
        enable_return_routed_experts=False,
        enable_return_indexer_topk=False,
        enable_hierarchical_cache=False,
        enable_memory_saver=False,
        cpu_offload_gb=0,
        enable_torch_compile=False,
        enable_deterministic_inference=False,
        disable_cuda_graph=False,
        disable_decode_cuda_graph=False,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def ctx():
    return MonoContext(
        server_args=None,
        hf_config=None,
        tp_size=1,
        tp_rank=0,
        cpu_group=None,
        device=torch.device("cpu"),
        limits=LIMITS,
    )


def batch(mode: str, rows: int, bs: int):
    fm = types.SimpleNamespace(
        is_decode=lambda: mode == "decode", is_target_verify=lambda: mode == "verify"
    )
    return types.SimpleNamespace(
        forward_mode=fm, input_ids=torch.zeros(rows, dtype=torch.long), batch_size=bs
    )


class TestRefusals(CustomTestCase):
    def test_supported_config_passes(self):
        self.assertEqual(
            guards.config_refusals(cfg(), None, fake_spec(), isa="gfx950"), []
        )

    def test_every_reason_reported_at_once(self):
        spec = fake_spec(config_disable_reasons=lambda c, h: ["spec says no"])
        reasons = guards.config_refusals(
            cfg(
                enable_two_batch_overlap=True,
                enable_dp_attention=True,
                ep_size=8,
                enable_lora=True,
                tp_size=8,
            ),
            None,
            spec,
            isa="gfx942",
        )
        joined = "\n".join(reasons)
        for needle in (
            "two-batch overlap",
            "data-parallel attention",
            "expert parallelism",
            "LoRA",
            "tp_size 8",
            "gfx950",
            "spec says no",
        ):
            self.assertIn(needle, joined)
        with self.assertRaises(guards.MonoRefused):
            guards.raise_if_refused("config", spec.name, reasons)

    def test_default_deny_spec_verify(self):
        spec = fake_spec(features=frozenset())
        self.assertTrue(
            any(
                "speculative" in r
                for r in guards.config_refusals(cfg(), None, spec, isa="gfx950")
            )
        )

    def test_eager_is_allowed(self):
        self.assertEqual(
            guards.config_refusals(
                cfg(disable_cuda_graph=True), None, fake_spec(), isa="gfx950"
            ),
            [],
        )


class TestRouting(CustomTestCase):
    def test_decode_and_verify_at_widths(self):
        s, c = fake_spec(), ctx()
        self.assertIsNotNone(runtime.decide_for_batch(s, c, batch("decode", 2, 2)).step)
        d = runtime.decide_for_batch(s, c, batch("verify", 12, 2))
        self.assertTrue(d.step.is_verify and d.step.rows == 12)
        self.assertEqual(d.step.query_len, 6)
        self.assertEqual(
            runtime.decide_for_batch(s, c, batch("decode", 2, 2)).step.query_len, 1
        )

    def test_rows_must_split_evenly_over_the_batch(self):
        s, c = fake_spec(), ctx()
        self.assertEqual(
            runtime.decide_for_batch(s, c, batch("verify", 6, 4)).reason,
            "rows not a multiple of the batch",
        )

    def test_fallbacks(self):
        s, c = fake_spec(), ctx()
        self.assertEqual(
            runtime.decide_for_batch(s, c, batch("extend", 6, 1)).reason, "not decode"
        )
        self.assertEqual(
            runtime.decide_for_batch(s, c, batch("decode", 7, 7)).reason, "width"
        )
        no_verify = fake_spec(features=frozenset())
        self.assertEqual(
            runtime.decide_for_batch(no_verify, c, batch("verify", 6, 1)).reason,
            "verify unsupported",
        )
        picky = fake_spec(eligible=lambda c, step: "too big" if step.rows > 2 else None)
        self.assertEqual(
            runtime.decide_for_batch(picky, c, batch("decode", 6, 6)).reason, "too big"
        )

    def test_decision_is_pure(self):
        s, c = fake_spec(), ctx()
        a = runtime.decide_for_batch(s, c, batch("verify", 6, 1))
        b = runtime.decide_for_batch(s, c, batch("verify", 6, 1))
        self.assertEqual(a, b)


class TestSpecShape(CustomTestCase):
    def test_spans_must_be_contiguous(self):
        check_spans({0: FFN, 1: WHOLE})
        k1 = Span("k1", Point.LAYER_IN, Point.CORE_IN)
        k2 = Span("k2", Point.CORE_IN, Point.LAYER_OUT)
        check_spans({0: (k1, k2)})
        self.assertEqual(span_label((k1, k2)), "whole")
        self.assertEqual(span_label(FFN), "attn_partial..layer_out")
        with self.assertRaises(ValueError):
            check_spans({0: (k1, Span("k2", Point.MLP_IN, Point.LAYER_OUT))})
        with self.assertRaises(ValueError):
            check_spans({0: (k1, Span("k1", Point.CORE_IN, Point.LAYER_OUT))})
        with self.assertRaises(ValueError):
            Span("back", Point.LAYER_OUT, Point.LAYER_IN)

    def test_numerics_bounds(self):
        Numerics(Exactness.TOLERANCE, {"k2.hidden.max_rel": 0.2, "k1.kv.max_abs": 0})
        with self.assertRaises(ValueError):
            Numerics(Exactness.BITWISE, {"k1.kv.max_abs": 0})
        with self.assertRaises(ValueError):
            Numerics(Exactness.TOLERANCE, {"k1.kv": 0})
        with self.assertRaises(ValueError):
            Numerics(Exactness.TOLERANCE, {"k1.kv.mean": 0})

    def test_rfc_feature_list_refused(self):
        reasons = guards.config_refusals(
            cfg(
                nnodes=2,
                attn_cp_size=2,
                speculative_eagle_topk=4,
                enable_return_routed_experts=True,
                enable_return_indexer_topk=True,
                enable_hierarchical_cache=True,
                enable_memory_saver=True,
                cpu_offload_gb=8,
                enable_torch_compile=True,
            ),
            None,
            fake_spec(),
            isa="gfx950",
        )
        joined = "\n".join(reasons)
        for needle in (
            "multi-node",
            "context parallelism",
            "uneven verify",
            "routed-expert capture",
            "indexer top-k capture",
            "KV offload",
            "memory release",
            "CPU weight offload",
            "torch-compiled decode",
        ):
            self.assertIn(needle, joined)

    def test_unreadable_setting_counts_as_in_use(self):
        c = cfg()
        del c.enable_lora
        reasons = guards.config_refusals(c, None, fake_spec(), isa="gfx950")
        self.assertTrue(any("cannot read enable_lora" in r for r in reasons), reasons)

    def test_requirements(self):
        spec = fake_spec(
            requires=Requirements(
                attention_backends=frozenset({"dsv4"}), kv_cache=frozenset({"fp8_e4m3"})
            )
        )
        ok = cfg(attention_backend="dsv4", kv_cache_dtype="fp8_e4m3")
        self.assertEqual(guards.config_refusals(ok, None, spec, isa="gfx950"), [])
        bad = cfg(attention_backend="triton")
        reasons = guards.config_refusals(bad, None, spec, isa="gfx950")
        self.assertTrue(
            any("attention_backend='triton'" in r for r in reasons), reasons
        )
        self.assertTrue(
            any("cannot read kv_cache_dtype" in r for r in reasons), reasons
        )

    def test_unknown_isa_refused(self):
        reasons = guards.config_refusals(cfg(), None, fake_spec(), isa=None)
        self.assertTrue(any("built for gfx950" in r for r in reasons))


class TestRuntime(CustomTestCase):
    def test_routed_layers(self):
        rt = make_runtime(fake_spec())
        rt.begin_step(batch("decode", 6, 6))
        self.assertEqual(rt.routed(0).spans, FFN)
        self.assertEqual(rt.routed(1).spans, WHOLE)
        self.assertIsNone(rt.routed(2))
        route = rt.routed(3)
        self.assertEqual((route.kernel.rows, route.step.rows), (6, 6))
        self.assertIsNone(route.checker)
        rt.begin_step(batch("extend", 6, 1))
        self.assertIsNone(rt.routed(0))
        self.assertEqual(rt.counts["mono"], 1)
        self.assertEqual(rt.counts["fallback:not decode"], 1)

    def test_unbuilt_width_takes_the_normal_path(self):
        rt = runtime.MonoRuntime(fake_spec(), ctx(), fake_spec().build(None, ctx()))
        rt.build_widths((6,))
        self.assertEqual(
            rt.begin_step(batch("verify", 12, 2)).reason, "width not built"
        )
        self.assertIsNotNone(rt.begin_step(batch("verify", 6, 1)).step)
        with unittest.mock.patch(
            "torch.cuda.is_current_stream_capturing", return_value=True
        ):
            self.assertEqual(
                rt.begin_step(batch("decode", 2, 2)).reason, "width not built"
            )

    def test_switch(self):
        self.assertTrue(
            guards.mono_enabled(types.SimpleNamespace(enable_monokernel=True))
        )
        self.assertFalse(
            guards.mono_enabled(types.SimpleNamespace(enable_monokernel=False))
        )
        self.assertFalse(guards.mono_enabled(types.SimpleNamespace()))

    def test_attach_off_by_default(self):
        with unittest.mock.patch.object(
            runtime, "_server_args", return_value=types.SimpleNamespace()
        ):
            loop = types.SimpleNamespace()
            self.assertIsNone(runtime.attach(loop, "FakeForCausalLM"))
            self.assertFalse(hasattr(loop, "_mono_runtime"))
            w = torch.zeros(2)
            self.assertIs(runtime.load_tensor(loop, "x", w), w)

    def test_weight_updates_refused_through_the_updater(self):
        src = (
            Path(runtime.__file__).parents[3]
            / "srt/model_executor/model_runner_components/weight_updater.py"
        )
        self.assertIn(
            'getattr(module, "_derived_weight_cache_error", None)', src.read_text()
        )
        self.assertIn(
            "model._derived_weight_cache_error =", Path(runtime.__file__).read_text()
        )

    def test_attach_without_a_spec_refuses(self):
        on = types.SimpleNamespace(enable_monokernel=True)
        with unittest.mock.patch.object(runtime, "_server_args", return_value=on):
            with self.assertRaises(guards.MonoRefused):
                runtime.attach(types.SimpleNamespace(), "NoSuchForCausalLM")


class TestHooks(CustomTestCase):
    def test_route_and_kernel(self):
        loop = types.SimpleNamespace()
        self.assertIsNone(runtime.route(loop, batch("decode", 2, 2)))
        loop._mono_runtime = make_runtime(fake_spec())
        self.assertIsNone(runtime.route(loop, batch("extend", 6, 1)))
        plan = runtime.route(loop, batch("decode", 6, 6))
        self.assertIsNotNone(plan)
        self.assertEqual(
            runtime.kernel(plan, 1, "layer_in")(7), ("span", "layer", 6, (7,))
        )
        self.assertIsNone(
            runtime.kernel(plan, 0, "layer_in")
        )  # its span starts at attn_partial
        self.assertIsNotNone(runtime.kernel(plan, 0, "attn_partial"))
        self.assertIsNone(runtime.kernel(plan, 2, "layer_in"))  # not covered
        self.assertIsNone(runtime.kernel(None, 1, "layer_in"))

    def test_build_widths_up_front(self):
        rt = runtime.MonoRuntime(fake_spec(), ctx(), fake_spec().build(None, ctx()))
        rt.build_widths((2, 6))
        self.assertEqual(sorted(rt._kernels), [2, 6])

        def build(rows):
            if rows == 6:
                raise RuntimeError("does not fit")
            return FakeKernel(rows)

        with self.assertRaises(guards.MonoRefused):
            make_runtime(fake_spec(kernel=build))


class TestStateBindings(CustomTestCase):
    def test_routed_layer_waits_for_its_state(self):
        waits = []
        kv = torch.zeros(4)

        def bindings(ctx, model):
            return {1: LayerStateBinding(1, (kv,), waits.append)}

        spec = fake_spec(state_bindings=bindings)
        rt = make_runtime(spec)
        rt.begin_step(batch("decode", 2, 2))
        rt.routed(0)
        self.assertEqual(waits, [])
        rt.routed(1)
        self.assertEqual(waits, [1])
        rt.begin_step(batch("extend", 6, 1))
        self.assertIsNone(rt.routed(1))
        self.assertEqual(waits, [1])


class TestCheckMode(CustomTestCase):
    def test_first_steps_carry_the_checker_then_stop(self):
        env = {"SGLANG_TEST_MONO_CHECK": "1", "SGLANG_TEST_MONO_CHECK_STEPS": "2"}
        with unittest.mock.patch.dict("os.environ", env):
            rt = make_runtime(fake_spec())
        seen = []
        for _ in range(4):
            rt.begin_step(batch("decode", 2, 2))
            route = rt.routed(1)
            seen.append(route.checker is not None)
            if route.checker is not None:
                diffs = route.checker.compare(
                    1, 2, {"hidden": (torch.ones(2, 4), torch.zeros(2, 4))}
                )
                self.assertEqual(diffs, {"hidden": 1.0})
        self.assertEqual(seen, [True, True, False, False])

    def test_fallback_steps_do_not_count(self):
        with unittest.mock.patch.dict("os.environ", {"SGLANG_TEST_MONO_CHECK": "1"}):
            rt = make_runtime(fake_spec())
        rt.begin_step(batch("extend", 6, 1))
        self.assertEqual(rt.checker.steps, 0)
        rt.begin_step(batch("decode", 2, 2))
        self.assertIsNotNone(rt.routed(1).checker)

    def test_off_by_default(self):
        with unittest.mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(check.requested())
            rt = make_runtime(fake_spec())
        self.assertIsNone(rt.checker)


class TestDeadline(CustomTestCase):
    def test_half_the_watchdog_at_most_60_s(self):
        from sglang.kernels.ops.mono.common_hip import guard

        self.assertEqual(guard.deadline_s(300), 60)
        self.assertEqual(guard.deadline_s(30), 15)
        self.assertEqual(guard.deadline_s(None), 60)
        self.assertEqual(guard.deadline_ticks(20), 10 * guard.TICKS_PER_S)
        self.assertEqual(guard.deadline_refusals(30), [])
        self.assertTrue(guard.deadline_refusals(8))
        reasons = guards.config_refusals(
            cfg(watchdog_timeout=6), None, fake_spec(), isa="gfx950"
        )
        self.assertTrue(any("wait deadline" in r for r in reasons), reasons)

    def test_fault_injection_spec(self):
        from sglang.kernels.ops.mono.common_hip import guard

        env = {guard.FAULT_ENV: "1:attn:40"}
        with unittest.mock.patch.dict("os.environ", env):
            self.assertEqual(guard.fault_for_rank(1), ("attn", 40))
            self.assertEqual(guard.fault_for_rank(0), ("", 0))
        with unittest.mock.patch.dict("os.environ", {guard.FAULT_ENV: "1"}):
            with self.assertRaises(ValueError):
                guard.fault_from_env()
        self.assertNotEqual(guard.site_id("attn"), 0)


if __name__ == "__main__":
    unittest.main()
