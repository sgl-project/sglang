# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the monokernel plugin: refusal, deadlines, the span hook, numerics, spans with a gap, routing from
the graph key, the load-hook order, and agreement across ranks (one rank failing a check or a build)."""

import math
import multiprocessing as std_mp
import socket
import types
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as tmp

from sglang.experimental.monokernels import guards, runtime
from sglang.experimental.monokernels.spec import (
    DeviceLimits,
    Exactness,
    Feature,
    MonoSpec,
    Numerics,
    Point,
    Requirements,
    Span,
    SpanPlan,
    StepKind,
    check_layer_spans,
    measure,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

A = Span("a", Point.LAYER_IN, Point.CORE_IN)
B = Span("b", Point.ATTN_PARTIAL, Point.LAYER_OUT)


def _spec(**kw):
    base = dict(
        name="fake",
        devices={
            "gfx950": DeviceLimits(max_rows=48, tp_sizes=(2, 4), resident_blocks=256)
        },
        features=frozenset(),
        requires=Requirements(),
        span_names=("a", "b"),
        spans=lambda owner, kind: SpanPlan({0: (A, B), 1: (B,)}),
        numerics=Numerics(Exactness.TOLERANCE, {"a.out.max_abs": None}),
        config_disable_reasons=lambda cfg, hf: [],
        model_disable_reasons=lambda owner: [],
        build=lambda ctx, owner, state: None,
    )
    base.update(kw)
    return MonoSpec(**base)


def _cfg(**over):
    fields = dict(
        tp_size=4,
        pp_size=1,
        nnodes=1,
        enable_dp_attention=False,
        ep_size=1,
        enable_eplb=False,
        attn_cp_size=1,
        dcp_size=1,
        rl_on_policy_target=None,
        debug_cuda_graph=False,
        cuda_graph_config=types.SimpleNamespace(
            decode=types.SimpleNamespace(backend="full")
        ),
        speculative_algorithm=None,
        speculative_eagle_topk=None,
        enable_two_batch_overlap=False,
        enable_lora=None,
        lora_paths=None,
        disaggregation_mode="null",
        enable_hierarchical_cache=False,
        enable_memory_saver=False,
        cpu_offload_gb=0,
        enable_weights_cpu_backup=False,
        enable_torch_compile=False,
        enable_deterministic_inference=False,
        disable_cuda_graph=False,
        enable_return_routed_experts=False,
        enable_return_indexer_topk=False,
        enable_return_hidden_states=False,
        watchdog_timeout=300,
    )
    fields.update(over)
    return types.SimpleNamespace(**fields)


class TestRefusal(unittest.TestCase):
    def test_default_config_accepted(self):
        self.assertEqual(guards.feature_reasons(_cfg(), _spec()), [])

    def test_every_reason_at_once(self):
        reasons = guards.feature_reasons(
            _cfg(
                enable_dp_attention=True,
                ep_size=8,
                enable_lora=True,
                disable_cuda_graph=True,
            ),
            _spec(),
        )
        self.assertEqual(len(reasons), 4, reasons)
        for f in (
            Feature.DP_ATTENTION,
            Feature.EXPERT_PARALLEL,
            Feature.LORA,
            Feature.EAGER,
        ):
            self.assertTrue(any(f.value in r for r in reasons), (f, reasons))

    def test_listed_feature_accepted(self):
        spec = _spec(features=frozenset({Feature.SPECULATIVE}))
        self.assertEqual(
            guards.feature_reasons(_cfg(speculative_algorithm="DSPARK"), spec), []
        )
        self.assertEqual(
            len(guards.feature_reasons(_cfg(speculative_algorithm="DSPARK"), _spec())),
            1,
        )

    def test_unreadable_setting_counts_as_used(self):
        cfg = _cfg()
        del cfg.enable_eplb
        reasons = guards.feature_reasons(cfg, _spec())
        self.assertEqual(len(reasons), 1)
        self.assertIn("cannot be read", reasons[0])

    def test_decode_graph_backend(self):
        def graph(backend, **over):
            return guards.feature_reasons(
                _cfg(
                    cuda_graph_config=types.SimpleNamespace(
                        decode=types.SimpleNamespace(backend=backend)
                    ),
                    **over,
                ),
                _spec(),
            )

        self.assertEqual(graph("full"), [])
        self.assertEqual(graph("disabled"), [f"{Feature.EAGER.value} is not supported"])
        self.assertEqual(
            graph("breakable"),
            [f"{Feature.BREAKABLE_DECODE_GRAPH.value} is not supported"],
        )
        self.assertEqual(
            graph("full", debug_cuda_graph=True),
            [f"{Feature.BREAKABLE_DECODE_GRAPH.value} is not supported"],
        )

    def test_decode_context_parallel_and_rl_on_policy(self):
        reasons = guards.feature_reasons(
            _cfg(dcp_size=2, rl_on_policy_target="fsdp"), _spec()
        )
        self.assertEqual(
            sorted(reasons),
            sorted(
                f"{f.value} is not supported"
                for f in (Feature.DECODE_CONTEXT_PARALLEL, Feature.RL_ON_POLICY)
            ),
        )

    def test_bitwise_only_modes(self):
        for f in (Feature.RL_ON_POLICY, Feature.DETERMINISTIC):
            with self.assertRaises(ValueError):
                _spec(features=frozenset({f}))
            _spec(
                features=frozenset({f}),
                numerics=Numerics(Exactness.BITWISE, {"a.out.max_abs": 0.0}),
            )

    def test_fault_injection(self):
        sited = _spec(wait_sites=("ffn", "pmix"))

        def parse(raw, spec=sited, tp=4):
            with mock.patch.object(
                guards.envs.SGLANG_TEST_MONO_FAULT_INJECT, "get", return_value=raw
            ):
                return guards.fault_injection(spec, tp)

        self.assertEqual(parse(""), (None, []))
        self.assertEqual(parse("1:ffn:4000"), ((1, "ffn", 4000), []))
        self.assertEqual(parse("0:pmix"), ((0, "pmix", 0), []))
        for bad in ("x:ffn", "1", "1:ffn:-3", "1::4", "4:ffn", "1:nowhere"):
            fault, reasons = parse(bad)
            self.assertIsNone(fault, bad)
            self.assertEqual(len(reasons), 1, bad)
        self.assertEqual(parse("1:anything", spec=_spec()), ((1, "anything", 0), []))
        with mock.patch.object(
            guards.envs.SGLANG_TEST_MONO_FAULT_INJECT, "get", return_value="9:ffn"
        ):
            self.assertEqual(len(guards.config_refusals(_cfg(), None, sited)), 1)

    def test_deadline(self):
        self.assertEqual(guards.deadline_s(300), 60.0)
        self.assertEqual(guards.deadline_s(30), 15.0)
        self.assertEqual(guards.deadline_s(None), 60.0)
        self.assertEqual(guards.deadline_reasons(10), [])
        self.assertEqual(len(guards.deadline_reasons(8)), 1)

    def test_span_hook(self):
        spec = _spec()
        with mock.patch.object(
            guards.envs.SGLANG_TEST_MONO_SPANS, "get", return_value=""
        ):
            self.assertEqual(guards.span_selection(spec), (frozenset({"a", "b"}), []))
        with mock.patch.object(
            guards.envs.SGLANG_TEST_MONO_SPANS, "get", return_value="b"
        ):
            self.assertEqual(guards.span_selection(spec), (frozenset({"b"}), []))
        with mock.patch.object(
            guards.envs.SGLANG_TEST_MONO_SPANS, "get", return_value="b,zz"
        ):
            selected, reasons = guards.span_selection(spec)
            self.assertEqual(selected, frozenset())
            self.assertEqual(len(reasons), 1)
            self.assertIn("zz", reasons[0])

    def test_raise_refused_lists_all(self):
        with self.assertRaises(guards.MonoRefused) as cm:
            guards.raise_refused("fake", "startup", ["one", "two"])
        self.assertIn("one", str(cm.exception))
        self.assertIn("two", str(cm.exception))
        guards.raise_refused("fake", "startup", [])


class TestSpecTypes(unittest.TestCase):
    def test_points_and_spans(self):
        self.assertEqual(Point.of("layer_in"), Point.LAYER_IN)
        with self.assertRaises(ValueError):
            Point.of("nowhere")
        with self.assertRaises(ValueError):
            Span("bad", Point.CORE_IN, Point.LAYER_IN)

    def test_gap_allowed_overlap_refused(self):
        check_layer_spans(0, (A, B))
        with self.assertRaises(ValueError):
            check_layer_spans(0, (Span("x", Point.LAYER_IN, Point.MLP_IN), B))

    def test_numerics_keys(self):
        Numerics(Exactness.TOLERANCE, {"a.out.max_rel": 1e-2, "a.kv.max_abs": 0.0})
        with self.assertRaises(ValueError):
            Numerics(Exactness.TOLERANCE, {"a.out.l2": 1.0})
        with self.assertRaises(ValueError):
            Numerics(Exactness.BITWISE, {"a.out.max_abs": 1e-3})
        with self.assertRaises(ValueError):
            Numerics(Exactness.TOLERANCE, {"a.out.max_abs": math.nan})

    def test_measure(self):
        ref = torch.tensor([1.0, -4.0, 2.0])
        m = measure(torch.tensor([1.5, -4.0, 2.0]), ref)
        self.assertEqual(m["max_abs"], 0.5)
        self.assertEqual(m["max_rel"], 0.125)
        self.assertEqual(measure(ref, ref), {"max_abs": 0.0, "max_rel": 0.0})
        zero = torch.zeros(3)
        self.assertEqual(measure(zero, zero)["max_rel"], 0.0)
        self.assertEqual(
            measure(torch.tensor([0.0, 1e-3, 0.0]), zero)["max_rel"], math.inf
        )
        self.assertEqual(
            measure(torch.tensor([1.0, math.nan, 2.0]), ref)["max_abs"], math.inf
        )
        self.assertEqual(measure(torch.zeros(2), ref)["max_abs"], math.inf)


def _runtime(widths, verify_tokens=None):
    ctx = runtime.MonoContext(
        0, 1, None, torch.device("cpu"), "gfx950", 60.0, frozenset({"a", "b"}), None
    )
    rt = runtime.MonoRuntime(_spec(), ctx, None, runtime.LoadState(_spec()))
    rt.widths = widths
    rt.verify_tokens = verify_tokens
    for kind in (StepKind.DECODE, StepKind.TARGET_VERIFY):
        rt.plans[kind] = {0: (B,), 1: (A, B), 2: (B,)}
        rt.firsts[kind] = {"b": 0, "a": 1}
    rt.model = types.SimpleNamespace(span=lambda plan, layer, span: (layer, span.name))
    return rt


def _batch(mode, rows, bs):
    fm = types.SimpleNamespace(
        is_decode=lambda: mode == "decode", is_target_verify=lambda: mode == "verify"
    )
    return types.SimpleNamespace(
        forward_mode=fm, input_ids=torch.zeros(rows), batch_size=bs
    )


class TestRouting(unittest.TestCase):
    def test_decode_built_width(self):
        rt = _runtime({StepKind.DECODE: (1, 2, 4, 8)})
        plan = rt.route(_batch("decode", 8, 8))
        self.assertEqual((plan.kind, plan.rows), (StepKind.DECODE, 8))
        self.assertEqual(plan.first, {"b": 0, "a": 1})
        self.assertEqual(runtime.kernel(plan, 1, "layer_in"), (1, "a"))
        self.assertEqual(runtime.kernel(plan, 0, "attn_partial"), (0, "b"))
        self.assertIsNone(runtime.kernel(plan, 0, "layer_in"))
        self.assertIsNone(runtime.kernel(plan, 5, "attn_partial"))
        self.assertIsNone(runtime.kernel(None, 0, "layer_in"))

    def test_unbuilt_width_and_other_modes_take_the_model_path(self):
        rt = _runtime({StepKind.DECODE: (1, 2, 4, 8)}, verify_tokens=6)
        self.assertIsNone(rt.route(_batch("decode", 3, 3)))
        self.assertIsNone(rt.route(_batch("extend", 8, 1)))
        self.assertIsNone(rt.route(_batch("verify", 12, 2)))

    def test_rows_must_match_the_graph_key(self):
        rt = _runtime(
            {StepKind.DECODE: (8,), StepKind.TARGET_VERIFY: (12,)}, verify_tokens=6
        )
        self.assertEqual(rt.route(_batch("verify", 12, 2)).rows, 12)
        for mode, rows, bs in (("decode", 8, 4), ("verify", 13, 2), ("verify", 12, 3)):
            with self.assertRaises(runtime.MonokernelError, msg=(mode, rows, bs)):
                rt.route(_batch(mode, rows, bs))

    def test_widths_rule(self):
        sizes = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 40, 48, 56, 64]
        self.assertEqual(
            runtime.widths_for(sizes, None, 48),
            {StepKind.DECODE: (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 40, 48)},
        )
        self.assertEqual(
            runtime.widths_for(sizes, 6, 48),
            {StepKind.TARGET_VERIFY: (6, 12, 18, 24, 30, 36, 42, 48)},
        )
        with self.assertRaises(guards.MonoRefused):
            runtime.widths_for(sizes, 0, 48)


class TestLoadOrder(unittest.TestCase):
    def setUp(self):
        self.seen = []
        spec = _spec(
            load_tensor=lambda state, owner, name, t: self.seen.append(
                ("tensor", name)
            ),
            loaded=lambda state, owner: self.seen.append(("loaded",)),
        )
        self.patches = [
            mock.patch.object(runtime, "enabled", return_value=True),
            mock.patch.object(runtime, "load_spec", return_value=spec),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_order(self):
        owner = types.SimpleNamespace()
        t = torch.zeros(1)
        self.assertIs(runtime.load_tensor(owner, "w", t), t)
        runtime.loaded(owner)
        self.assertEqual(self.seen, [("tensor", "w"), ("loaded",)])
        with self.assertRaises(runtime.MonokernelError):
            runtime.load_tensor(owner, "late", t)
        with self.assertRaises(runtime.MonokernelError):
            runtime.loaded(owner)

    def test_attach_before_loaded_refused(self):
        owner = types.SimpleNamespace()
        runtime.load_tensor(owner, "w", torch.zeros(1))
        with self.assertRaises(runtime.MonokernelError):
            runtime.attach(owner)

    def test_switch_off_does_nothing(self):
        with mock.patch.object(runtime, "enabled", return_value=False):
            owner = types.SimpleNamespace()
            runtime.load_tensor(owner, "w", torch.zeros(1))
            runtime.loaded(owner)
            runtime.attach(owner)
        self.assertEqual(self.seen, [])


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rank_main(rank, world, port, case, out):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    ctx = runtime.MonoContext(
        rank,
        world,
        dist.group.WORLD,
        torch.device("cpu"),
        "gfx950",
        60.0,
        frozenset({"a", "b"}),
        None,
    )

    def stage():
        if case == "shape" and rank == 1:
            raise RuntimeError("layer 3: weight (7,), want (8,)")
        if case == "build" and rank == 0:
            raise RuntimeError("width 8: build failed")
        return [], "ok"

    def build_widths(widths):
        for w in widths:
            if case == "attach_width" and rank == 1 and w == 4:
                raise RuntimeError(f"width {w}: build failed")
        return {w: 0.0 for w in widths}

    try:
        runtime.agree(ctx, "fake", "a stage", stage)
        if case == "plan":
            spec = _spec(
                spans=lambda owner, kind: SpanPlan(
                    {0: (A, B)} if rank == 0 else {0: (B,)}
                )
            )
            rt = runtime.MonoRuntime(spec, ctx, None, runtime.LoadState(spec))
            rt._plan(StepKind.DECODE)
        if case in ("attach_ok", "attach_width", "no_width"):
            model = types.SimpleNamespace(
                allocate=lambda widths: {}, build_widths=build_widths
            )
            spec = _spec(build=lambda ctx, owner, state: model)
            rt = runtime.MonoRuntime(
                spec, ctx, types.SimpleNamespace(), runtime.LoadState(spec)
            )
            widths = {StepKind.DECODE: () if case == "no_width" else (1, 2, 4, 8)}
            with (
                mock.patch.object(runtime, "reachable_widths", return_value=widths),
                mock.patch.object(runtime, "_verify_tokens", return_value=None),
            ):
                rt.attach()
        out.put((rank, "passed"))
    except guards.MonoRefused as err:
        out.put((rank, str(err)))
    finally:
        dist.destroy_process_group()


def _run_ranks(case, world=2):
    ctx = tmp.get_context("spawn")
    out = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_rank_main, args=(r, world, port, case, out))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    got = dict(out.get(timeout=120) for _ in procs)
    for p in procs:
        p.join(timeout=60)
    return got


class TestAgreement(unittest.TestCase):
    def test_all_pass(self):
        self.assertEqual(_run_ranks("ok"), {0: "passed", 1: "passed"})

    def test_one_rank_fails_a_shape_check(self):
        got = _run_ranks("shape")
        for r in (0, 1):
            self.assertIn("rank 1: RuntimeError: layer 3", got[r])

    def test_one_rank_fails_a_build(self):
        got = _run_ranks("build")
        for r in (0, 1):
            self.assertIn("rank 0: RuntimeError: width 8", got[r])

    def test_plans_differ(self):
        got = _run_ranks("plan")
        for r in (0, 1):
            self.assertIn("the decode plan", got[r])

    def test_attach_on_every_rank(self):
        self.assertEqual(_run_ranks("attach_ok"), {0: "passed", 1: "passed"})

    def test_attach_one_rank_fails_one_width(self):
        got = _run_ranks("attach_width")
        for r in (0, 1):
            self.assertIn("kernel builds", got[r])
            self.assertIn("rank 1: RuntimeError: width 4", got[r])

    def test_attach_refuses_with_no_reachable_width(self):
        got = _run_ranks("no_width")
        for r in (0, 1):
            self.assertIn("refused at widths", got[r])


if __name__ == "__main__":
    std_mp.set_start_method("spawn", force=True)
    unittest.main()
