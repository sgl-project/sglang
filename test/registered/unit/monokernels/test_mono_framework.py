"""The monokernel framework on CPU, with a fake spec: the plugin registry, the
startup refusals, the decode capture sizes, the forward switch, the load hook and
the wait deadlines."""

import inspect
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch
from msgspec.structs import replace

from sglang.srt import monokernels as mono
from sglang.srt.monokernels import guards, runtime
from sglang.srt.monokernels import spec as registry
from sglang.srt.monokernels.guards import (
    MAX_DEADLINE_S,
    covered_capture_sizes,
    deadline_s,
    features_in_use,
    monokernel_disable_reasons,
)
from sglang.srt.monokernels.spec import (
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, stage="base-a", runner_config="cpu")

ARCH = "FakeMonoForCausalLM"
BACKEND = "fake"
WIDTHS = (1, 2, 4, 6, 12)


class FakeBuilt:
    """A built spec: serves every key but decode steps of 4 rows, and records each
    routed forward's rows and plan state."""

    def __init__(self):
        self.asked = 0
        self.steps = []

    def serves(self, key):
        self.asked += 1
        return (key.forward_mode, key.rows) != ("decode", 4)

    def forward(self, plan, input_ids, positions, forward_batch, input_embeds=None):
        plan.state["layers"] = plan.state.get("layers", 0) + 1
        self.steps.append((plan.key.rows, dict(plan.state)))
        return "mono"


def _load_tensor(copies, model, name, tensor):
    if name.endswith("shared.weight"):
        copies["shared expert", name] = tensor.clone()


SPEC = MonoSpec(
    name="fake",
    devices={
        "gfx950": DeviceLimits(widths=WIDTHS, tp_sizes=(2, 4), resident_blocks=256)
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(kv_cache_dtypes=frozenset({"fp8_e4m3"})),
    numerics=Numerics(kind=NumericsKind.BITWISE),
    config_disable_reasons=lambda cfg, hf: (
        ["context_length too long"] if cfg.context_length == 7 else []
    ),
    model_disable_reasons=lambda model, hf, copies: [],
    build=lambda model, hf, ctx: FakeBuilt(),
    api=registry.API_VERSION,
    load_tensor=_load_tensor,
)
OLD_SPEC = replace(SPEC, api=registry.API_VERSION - 1)


def server_args(**kw):
    defaults = {
        "model_path": "dummy",
        "tp_size": 4,
        "kv_cache_dtype": "fp8_e4m3",
        "monokernel_backend": BACKEND,
    }
    return ServerArgs(**{**defaults, **kw})


def reasons_for(cfg, arch=ARCH, isa="gfx950"):
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[arch]),
        dtype="bfloat16",
        quantization=None,
    )
    return monokernel_disable_reasons(cfg=cfg, model_config=model_config, isa=isa)


def batch(mode, bs, rows):
    forward_mode = SimpleNamespace(
        is_decode=lambda: mode == "decode",
        is_target_verify=lambda: mode == "target_verify",
    )
    return SimpleNamespace(
        forward_mode=forward_mode,
        batch_size=bs,
        input_ids=torch.zeros(rows, dtype=torch.int64),
    )


class Lm(torch.nn.Module):
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        return "normal"


def switched():
    """An ``Lm`` as attach leaves it, for a spec built at ``WIDTHS``, 6 tokens a
    verify request."""
    lm, built = Lm(), FakeBuilt()
    served = runtime._served(built=built, widths=WIDTHS, tokens=6)
    runtime._switch(model=lm, built=built, served=served)
    return lm, built


class TestMonokernelFramework(CustomTestCase):
    def setUp(self):
        registered = patch.dict(registry._REGISTRY, clear=True)
        registered.start()
        self.addCleanup(registered.stop)
        registry.register(backend=BACKEND, architecture=ARCH, path=f"{__name__}:SPEC")

    def test_find_spec_refuses_with_a_reason(self):
        register = partial(registry.register, backend=BACKEND)
        register(architecture="Broken", path="no_such_module:SPEC")
        register(architecture="Old", path=f"{__name__}:OLD_SPEC")
        find = partial(registry.find_spec, backend=BACKEND)
        self.assertIs(find(architectures=[ARCH], isa="gfx950")[0], SPEC)
        api = f"API {registry.API_VERSION - 1}, this SGLang has {registry.API_VERSION}"
        for archs, isa, why in (
            ([ARCH], "gfx942", "has no spec"),
            (["Other"], "gfx950", "has no spec"),
            (["Broken"], "gfx950", "does not import: ModuleNotFoundError"),
            (["Old"], "gfx950", api),
        ):
            with self.subTest(archs=archs, isa=isa):
                spec, reason = find(architectures=archs, isa=isa)
                self.assertIsNone(spec)
                self.assertIn(why, reason)
        spec, reason = registry.find_spec(
            backend="nope", architectures=[ARCH], isa="gfx950"
        )
        self.assertIsNone(spec)
        self.assertIn("no monokernel backend 'nope' is registered", reason)
        self.assertIn(repr(BACKEND), reason)  # the registered ones are listed

    def test_a_backend_has_one_spec_an_architecture(self):
        # plugins register in every process: the same spec again is a no-op, and
        # another plugin's spec for the architecture is an error, not a silent swap
        registry.register(backend=BACKEND, architecture=ARCH, path=f"{__name__}:SPEC")
        with self.assertRaises(ValueError):
            registry.register(backend=BACKEND, architecture=ARCH, path="other:SPEC")
        self.assertIs(
            registry.find_spec(backend=BACKEND, architectures=[ARCH], isa="gfx950")[0],
            SPEC,
        )

    def test_features_in_use_are_refused_unless_accepted(self):
        # every probe is off at SGLang's defaults: one stuck on refuses every server
        self.assertEqual(features_in_use(ServerArgs(model_path="dummy")), set())
        _, reasons = reasons_for(server_args(pp_size=2, speculative_algorithm="DSPARK"))
        self.assertEqual(len(reasons), 1, reasons)  # SPEC accepts speculative decoding
        self.assertIn(Feature.PIPELINE_PARALLEL.value, reasons[0])

    def test_every_refusal_reason_is_listed(self):
        spec, reasons = reasons_for(
            server_args(
                tp_size=8,
                kv_cache_dtype="auto",
                enable_eplb=True,
                context_length=7,
                watchdog_timeout=6,
            )
        )
        self.assertIs(spec, SPEC)
        for want in (
            "tp_size 8",
            "--kv-cache-dtype",
            "EPLB",
            "context_length",
            "--watchdog-timeout",
        ):
            self.assertTrue(any(want in r for r in reasons), (want, reasons))
        self.assertEqual(reasons_for(server_args())[1], [])
        self.assertIsNone(reasons_for(server_args(), arch="Other")[0])
        self.assertEqual(
            reasons_for(server_args(), isa=None), (None, ["no GPU is visible"])
        )

    def test_exact_modes_take_only_bitwise_specs(self):
        modes = {"enable_deterministic_inference": True, "rl_on_policy_target": "fsdp"}
        self.assertEqual(reasons_for(server_args(**modes))[1], [])  # SPEC is BITWISE
        tolerance = replace(
            SPEC,
            numerics=Numerics(kind=NumericsKind.TOLERANCE, bounds={"k1.q.rel": 5e-3}),
        )
        with patch.object(guards, "find_spec", return_value=(tolerance, "")):
            _, reasons = reasons_for(server_args(**modes))
        for flag in ("--enable-deterministic-inference", "--rl-on-policy-target"):
            self.assertTrue(any(flag in r for r in reasons), (flag, reasons))

    def test_capture_sizes_are_the_widths_up_to_the_largest(self):
        # decode: rows are requests
        sizes = covered_capture_sizes(
            widths=WIDTHS, tokens_per_request=1, sizes=[1, 2, 3, 4, 8, 16, 32]
        )
        self.assertEqual(sizes, [1, 2, 4, 6, 12, 16, 32])
        # verify at 6 tokens a request: widths 6 and 12 are 1 and 2 requests
        sizes = covered_capture_sizes(
            widths=WIDTHS, tokens_per_request=6, sizes=[1, 2, 3, 4, 8]
        )
        self.assertEqual(sizes, [1, 2, 3, 4, 8])

    def test_the_switch_routes_by_graph_key(self):
        lm, built = switched()
        # SGLang's graph runners choose what to pass from the forward's parameters
        self.assertEqual(inspect.signature(lm.forward), inspect.signature(Lm().forward))
        asked = built.asked
        x = torch.zeros(1)
        self.assertEqual(lm(x, None, batch("decode", 2, 2)), "mono")
        self.assertEqual(
            lm(x, None, forward_batch=batch("target_verify", 2, 12)), "mono"
        )
        for normal in (
            batch("decode", 3, 3),  # a width not built
            batch("decode", 4, 4),  # a key the spec does not serve
            batch("decode", 64, 64),  # over the largest width
            batch("target_verify", 2, 7),  # uneven rows a request
            batch("extend", 2, 9),
        ):
            self.assertEqual(lm(x, None, normal), "normal")
        # the served keys were fixed at attach: routing runs none of the spec's code
        self.assertEqual(built.asked, asked)

    def test_each_forward_has_its_own_plan_state(self):
        # the spec's forward keeps what its layers share in plan.state; a plan kept
        # across forwards would hand one step's values to the next
        lm, built = switched()
        for _ in range(2):
            lm(torch.zeros(1), None, batch("decode", 2, 2))
        self.assertEqual(built.steps, [(2, {"layers": 1}), (2, {"layers": 1})])

    def test_watched_weights_pass_through_and_keep_copies(self):
        model, other = torch.nn.Module(), torch.nn.Module()
        model.config = SimpleNamespace(architectures=[ARCH])
        other.config = SimpleNamespace(architectures=["Other"])
        weights = [
            ("layers.0.shared.weight", torch.ones(16, dtype=torch.uint8)),
            ("layers.0.other.weight", torch.ones(16)),
        ]
        with (
            patch.object(runtime, "_backend", return_value=BACKEND),
            patch.object(runtime, "device_isa", return_value="gfx950"),
        ):
            watched = list(mono.watch_weights(model=model, weights=iter(weights)))
            self.assertIs(mono.watch_weights(model=other, weights=weights), weights)
        # every checkpoint tensor reaches the model's loader, as it is and in order
        self.assertEqual(
            [(n, id(t)) for n, t in watched], [(n, id(t)) for n, t in weights]
        )
        self.assertEqual(
            list(model.__dict__["_mono_copies"]),
            [("shared expert", "layers.0.shared.weight")],
        )
        with patch.object(runtime, "_backend", return_value=None):
            self.assertIs(mono.watch_weights(model=model, weights=weights), weights)

    def test_deadlines_stay_below_the_watchdog(self):
        self.assertEqual(deadline_s(None), MAX_DEADLINE_S)
        self.assertEqual(deadline_s(3600), MAX_DEADLINE_S)
        self.assertEqual(deadline_s(30), 15.0)


if __name__ == "__main__":
    unittest.main()
