"""CPU tests of the model-agnostic mono runtime (sglang/srt/layers/mono) against a fake spec and fake kernels."""

import types
import unittest
import unittest.mock

import torch

from sglang.srt.layers.mono import check, failstop, guards, layer, routing
from sglang.srt.layers.mono.spec import (
    Feature,
    LayerMode,
    LayerStateBinding,
    MonoContext,
    MonoSpec,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeKernel:
    def __init__(self, rows, health=None):
        self.rows, self._health, self.launches = rows, health, []

    def launch(self, step, layer_id, *args, **kwargs):
        self.launches.append((step.rows, layer_id))
        return args

    def health(self):
        return self._health


def fake_spec(**kw):
    base = dict(
        name="fake",
        architectures=("FakeForCausalLM",),
        layers=lambda cfg: {0: LayerMode.FFN, 1: LayerMode.WHOLE, 3: LayerMode.FFN},
        widths=(1, 2, 6, 12),
        supports=frozenset({Feature.SPEC_VERIFY}),
        supported_tp=(2, 4),
        build=lambda ctx, packed, rows: FakeKernel(rows),
    )
    base.update(kw)
    return MonoSpec(**base)


def cfg(**kw):
    """The server-args fields the guards read, every feature off."""
    base = dict(
        tp_size=4,
        ep_size=1,
        pp_size=1,
        disaggregation_mode="null",
        speculative_algorithm="DSPARK",
        enable_two_batch_overlap=False,
        enable_dp_attention=False,
        enable_lora=False,
        enable_hierarchical_cache=False,
        enable_eplb=False,
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
            guards.config_refusals(cfg(), None, fake_spec(), platform_ok=True), []
        )

    def test_every_reason_reported_at_once(self):
        spec = fake_spec(refuse_config=lambda c, h: ["spec says no"])
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
            platform_ok=False,
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
        spec = fake_spec(supports=frozenset())
        self.assertTrue(
            any(
                "speculative" in r
                for r in guards.config_refusals(cfg(), None, spec, platform_ok=True)
            )
        )

    def test_eager_is_allowed(self):
        self.assertEqual(
            guards.config_refusals(
                cfg(disable_cuda_graph=True), None, fake_spec(), platform_ok=True
            ),
            [],
        )


class TestRouting(CustomTestCase):
    def test_decode_and_verify_at_widths(self):
        s, c = fake_spec(), ctx()
        self.assertIsNotNone(routing.decide_for_batch(s, c, batch("decode", 2, 2)).step)
        d = routing.decide_for_batch(s, c, batch("verify", 12, 2))
        self.assertTrue(d.step.is_verify and d.step.rows == 12)
        self.assertEqual(d.step.query_len, 6)
        self.assertEqual(
            routing.decide_for_batch(s, c, batch("decode", 2, 2)).step.query_len, 1
        )

    def test_rows_must_split_evenly_over_the_batch(self):
        s, c = fake_spec(), ctx()
        self.assertEqual(
            routing.decide_for_batch(s, c, batch("verify", 6, 4)).reason,
            "rows not a multiple of the batch",
        )

    def test_fallbacks(self):
        s, c = fake_spec(), ctx()
        self.assertEqual(
            routing.decide_for_batch(s, c, batch("extend", 6, 1)).reason, "not decode"
        )
        self.assertEqual(
            routing.decide_for_batch(s, c, batch("decode", 7, 7)).reason, "width"
        )
        no_verify = fake_spec(supports=frozenset())
        self.assertEqual(
            routing.decide_for_batch(no_verify, c, batch("verify", 6, 1)).reason,
            "verify unsupported",
        )
        picky = fake_spec(eligible=lambda c, step: "too big" if step.rows > 2 else None)
        self.assertEqual(
            routing.decide_for_batch(picky, c, batch("decode", 6, 6)).reason, "too big"
        )

    def test_decision_is_pure(self):
        s, c = fake_spec(), ctx()
        a = routing.decide_for_batch(s, c, batch("verify", 6, 1))
        b = routing.decide_for_batch(s, c, batch("verify", 6, 1))
        self.assertEqual(a, b)

    def test_cache_rebind(self):
        b = routing.CacheBinding()
        kv = torch.zeros(8)
        self.assertTrue(b.changed({"kv": kv}))
        self.assertFalse(b.changed({"kv": kv}))
        self.assertTrue(b.changed({"kv": torch.zeros(8)}))
        self.assertTrue(b.changed({"kv": kv[:4]}))


class TestRuntime(CustomTestCase):
    def test_routed_layers_and_lazy_build(self):
        rt = layer.MonoRuntime(
            fake_spec(), ctx(), packed=None, modes=fake_spec().layers(None)
        )
        rt.begin_step(batch("decode", 6, 6))
        self.assertEqual(rt.routed(0)[0], LayerMode.FFN)
        self.assertEqual(rt.routed(1)[0], LayerMode.WHOLE)
        self.assertIsNone(rt.routed(2))
        route = rt.routed(3)
        self.assertEqual((route.kernel.rows, route.step.rows), (6, 6))
        self.assertIsNone(route.checker)
        rt.begin_step(batch("extend", 6, 1))
        self.assertIsNone(rt.routed(0))
        self.assertEqual(rt.counts["mono"], 1)
        self.assertEqual(rt.counts["fallback:not decode"], 1)

    def test_failed_build_refuses_width(self):
        def build(ctx, packed, rows):
            if rows == 12:
                raise RuntimeError("does not fit")
            return FakeKernel(rows)

        spec = fake_spec(build=build)
        rt = layer.MonoRuntime(spec, ctx(), packed=None, modes=spec.layers(None))
        self.assertEqual(
            rt.begin_step(batch("verify", 12, 2)).reason, "width refused at build"
        )
        self.assertEqual(
            rt.begin_step(batch("verify", 12, 2)).reason, "width refused at build"
        )
        self.assertIsNotNone(rt.begin_step(batch("verify", 6, 1)).step)

    def test_attach_off_by_default(self):
        self.assertIsNone(
            layer.attach(
                object(),
                server_args=types.SimpleNamespace(layer_backend="module"),
                hf_config=None,
                tp_rank=0,
                tp_size=1,
                cpu_group=None,
                device="cpu",
            )
        )

    def test_layer_backend_switch(self):
        on = types.SimpleNamespace(layer_backend="mono")
        self.assertTrue(guards.mono_enabled(on))
        self.assertFalse(
            guards.mono_enabled(types.SimpleNamespace(layer_backend="module"))
        )
        with self.assertRaises(guards.MonoRefused):
            layer.attach(
                object(),
                server_args=on,
                hf_config=types.SimpleNamespace(architectures=["NoSuchForCausalLM"]),
                tp_rank=0,
                tp_size=1,
                cpu_group=None,
                device="cpu",
            )

    def test_new_width_inside_a_capture_raises(self):
        rt = layer.MonoRuntime(
            fake_spec(), ctx(), packed=None, modes=fake_spec().layers(None)
        )
        rt.begin_step(batch("decode", 2, 2))
        with unittest.mock.patch(
            "torch.cuda.is_current_stream_capturing", return_value=True
        ):
            self.assertIsNotNone(rt.begin_step(batch("decode", 2, 2)).step)
            with self.assertRaises(RuntimeError):
                rt.begin_step(batch("decode", 6, 6))


class TestStateBindings(CustomTestCase):
    def test_routed_layer_waits_for_its_state(self):
        waits = []
        kv = torch.zeros(4)

        def bindings(ctx, packed):
            return {1: LayerStateBinding(1, (kv,), waits.append)}

        spec = fake_spec(state_bindings=bindings)
        rt = layer.MonoRuntime(spec, ctx(), packed=None, modes=spec.layers(None))
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
        env = {"SGLANG_MONO_CHECK": "1", "SGLANG_MONO_CHECK_STEPS": "2"}
        with unittest.mock.patch.dict("os.environ", env):
            rt = layer.MonoRuntime(
                fake_spec(), ctx(), packed=None, modes=fake_spec().layers(None)
            )
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
        with unittest.mock.patch.dict("os.environ", {"SGLANG_MONO_CHECK": "1"}):
            rt = layer.MonoRuntime(
                fake_spec(), ctx(), packed=None, modes=fake_spec().layers(None)
            )
        rt.begin_step(batch("extend", 6, 1))
        self.assertEqual(rt.checker.steps, 0)
        rt.begin_step(batch("decode", 2, 2))
        self.assertIsNotNone(rt.routed(1).checker)

    def test_off_by_default(self):
        with unittest.mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(check.requested())
            rt = layer.MonoRuntime(
                fake_spec(), ctx(), packed=None, modes=fake_spec().layers(None)
            )
        self.assertIsNone(rt.checker)


class TestFailStop(CustomTestCase):
    def test_clean_step_passes(self):
        failstop.check(torch.zeros(4, dtype=torch.int32), rank=0, step=1)
        failstop.check(None, rank=0, step=1)

    def test_error_word_raises(self):
        with self.assertRaises(failstop.MonoKernelError):
            failstop.check(torch.tensor([0, 3, 0], dtype=torch.int32), rank=1, step=7)

    def test_unsafe_warn_keeps_serving(self):
        import os

        os.environ["SGLANG_MONO_UNSAFE_WARN"] = "1"
        try:
            failstop.check(torch.tensor([1], dtype=torch.int32), rank=0, step=1)
        finally:
            del os.environ["SGLANG_MONO_UNSAFE_WARN"]

    def test_runtime_health_gathers_kernels(self):
        spec = fake_spec(
            build=lambda c, p, rows: FakeKernel(
                rows, torch.tensor([0, rows], dtype=torch.int32)
            )
        )
        rt = layer.MonoRuntime(spec, ctx(), packed=None, modes=spec.layers(None))
        rt.begin_step(batch("decode", 2, 2))
        rt.begin_step(batch("decode", 6, 6))
        self.assertEqual(rt.health().tolist(), [0, 2, 0, 6])


if __name__ == "__main__":
    unittest.main()
