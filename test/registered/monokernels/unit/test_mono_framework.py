"""The monokernel plugin on CPU with a fake spec and a fake forward (RFC #43949):
the registry and its entry points, the API version, feature refusal, the config
checks, the modes that take only BITWISE specs, capture sizes, routing by graph
key from the keys fixed at attach, the normal path's reasons, the spec's forward
bound to its plan, the load hook's weight copies, and the deadlines."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from sglang.experimental import monokernels as mono
from sglang.experimental.monokernels import guards, runtime
from sglang.experimental.monokernels import spec as registry
from sglang.experimental.monokernels.guards import (
    MAX_DEADLINE_S,
    covered_capture_sizes,
    deadline_s,
    features_in_use,
    monokernel_disable_reasons,
)
from sglang.experimental.monokernels.runtime import shape_key
from sglang.experimental.monokernels.spec import (
    BuildContext,
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
    ShapeKey,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

ARCH = "FakeMonoForCausalLM"


class FakeModel:
    """Serves every built width, except decode steps of 4 rows (a key the spec
    leaves to the normal forward)."""

    def __init__(self, widths):
        self.widths = widths
        self.log = []
        self.asked = 0

    def serves(self, key):
        self.asked += 1
        return (key.rows, key.forward_mode) != (4, "decode")

    def forward(self, plan, input_ids, positions=None, forward_batch=None):
        self.log.append((plan.key.rows, plan.state.get("tag")))
        return input_ids + 1


def _load_tensor(copies, model, name, tensor):
    if name.endswith("shared.weight"):
        copies["shared expert", name] = tensor.clone()


SPEC = MonoSpec(
    name="fake",
    devices={
        "gfx950": DeviceLimits(
            widths=(1, 2, 4, 6, 12), tp_sizes=(2, 4), resident_blocks=256
        )
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(kv_cache_dtypes=frozenset({"fp8_e4m3"})),
    numerics=Numerics(NumericsKind.BITWISE),
    config_disable_reasons=lambda cfg, hf: (
        ["context_length too long"] if cfg.context_length == 7 else []
    ),
    model_disable_reasons=lambda model, hf, copies: [],
    build=lambda model, hf, ctx: FakeModel(ctx.widths),
    api=registry.API_VERSION,
    load_tensor=_load_tensor,
)


OLD_SPEC = replace(SPEC, api=0)


@pytest.fixture(autouse=True)
def _registered():
    registry.register(ARCH, f"{__name__}:SPEC")
    yield
    registry.unregister(ARCH, f"{__name__}:SPEC")


def cfg(**kw):
    base = dict(
        pp_size=1, nnodes=1, enable_dp_attention=False, attn_dp_size=1, ep_size=1,
        enable_eplb=False, attn_cp_size=1, dcp_size=1, enable_prefill_cp=False,
        speculative_algorithm=None, speculative_num_draft_tokens=None,
        enable_two_batch_overlap=False, enable_single_batch_overlap=False,
        enable_lora=None, lora_paths=None, disaggregation_mode="null",
        enable_return_routed_experts=False, enable_return_indexer_topk=False,
        enable_hierarchical_cache=False, enable_memory_saver=False, cpu_offload_gb=0,
        offload_group_size=-1, enable_torch_compile=False, tp_size=4, dtype="auto",
        kv_cache_dtype="fp8_e4m3", attention_backend="dsv4", moe_runner_backend="auto",
        context_length=None, watchdog_timeout=300,
        enable_deterministic_inference=False, rl_on_policy_target=None,
    )  # fmt: skip
    base.update(kw)
    return SimpleNamespace(**base)


def model_config(arch=ARCH):
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[arch]), quantization=None
    )


def batch(mode, bs, rows):
    fm = SimpleNamespace(
        is_decode=lambda: mode == "decode",
        is_target_verify=lambda: mode == "target_verify",
    )
    return SimpleNamespace(
        forward_mode=fm, batch_size=bs, input_ids=torch.zeros(rows, dtype=torch.int64)
    )


def attached(widths=None, tokens=6):
    """A model as attach() leaves it: ``mono`` holds the runtime."""
    widths = widths or SPEC.widths("gfx950")
    built = SPEC.build(None, None, BuildContext("gfx950", widths, 30.0, {}))
    rt = runtime.MonoRuntime(built, widths, tokens)
    return SimpleNamespace(mono=rt), built


def test_registry_reports_broken_and_mismatched_specs(monkeypatch):
    registry.register("Broken", "no_such_module:SPEC")
    registry.register("Old", f"{__name__}:OLD_SPEC")
    try:
        spec, why = registry.find_spec(["Broken"], "gfx950")
        assert spec is None and "does not import: ModuleNotFoundError" in why
        spec, why = registry.find_spec(["Old"], "gfx950")
        assert spec is None and f"API 0, this SGLang has {registry.API_VERSION}" in why
    finally:
        registry.unregister("Broken", "no_such_module:SPEC")
        registry.unregister("Old", f"{__name__}:OLD_SPEC")
    # installed packages list their specs as entry points
    monkeypatch.setattr(
        registry, "_entry_points", lambda: {"Pkg": (f"{__name__}:SPEC",)}
    )
    assert registry.find_spec(["Pkg"], "gfx950")[0] is SPEC


def test_registry_finds_the_spec_for_the_isa():
    spec, why = registry.find_spec([ARCH], "gfx950")
    assert spec is SPEC and why == ""
    spec, why = registry.find_spec([ARCH], "gfx942")
    assert spec is None and "gfx942" in why
    spec, why = registry.find_spec(["Other"], "gfx950")
    assert spec is None and "no monokernel spec" in why


def test_features_refused_unless_accepted():
    assert features_in_use(cfg()) == set()
    used = features_in_use(
        cfg(pp_size=2, speculative_algorithm="DSPARK", enable_torch_compile=True)
    )
    assert used == {
        Feature.PIPELINE_PARALLEL,
        Feature.SPECULATIVE_DECODING,
        Feature.TORCH_COMPILE,
    }
    _, reasons = monokernel_disable_reasons(
        cfg(pp_size=2, speculative_algorithm="DSPARK"), model_config(), "gfx950"
    )
    assert len(reasons) == 1 and "pipeline" in reasons[0]


def test_a_failing_probe_counts_as_in_use():
    broken = cfg()
    del broken.pp_size
    assert Feature.PIPELINE_PARALLEL in features_in_use(broken)


def test_config_reasons_are_all_listed():
    spec, reasons = monokernel_disable_reasons(
        cfg(
            tp_size=8,
            kv_cache_dtype="auto",
            enable_eplb=True,
            context_length=7,
            watchdog_timeout=6,
        ),
        model_config(),
        "gfx950",
    )
    assert spec is SPEC
    text = " | ".join(reasons)
    for want in (
        "tp_size 8",
        "--kv-cache-dtype",
        "EPLB",
        "context_length",
        "--watchdog-timeout",
    ):
        assert want in text, text
    _, reasons = monokernel_disable_reasons(cfg(), model_config(), "gfx950")
    assert reasons == []
    spec, reasons = monokernel_disable_reasons(cfg(), model_config("Other"), "gfx950")
    assert spec is None and reasons
    _, reasons = monokernel_disable_reasons(cfg(), model_config(), None)
    assert reasons == ["no GPU is visible"]


def test_exact_modes_take_only_bitwise_specs(monkeypatch):
    modes = {"enable_deterministic_inference": True, "rl_on_policy_target": "fsdp"}
    for name, value in modes.items():
        _, reasons = monokernel_disable_reasons(
            cfg(**{name: value}), model_config(), "gfx950"
        )
        assert reasons == []  # SPEC is BITWISE
    tol = replace(SPEC, numerics=Numerics(NumericsKind.TOLERANCE, {"k1.q.rel": 5e-3}))
    monkeypatch.setattr(guards, "find_spec", lambda arch, isa: (tol, None))
    _, reasons = monokernel_disable_reasons(cfg(**modes), model_config(), "gfx950")
    text = " | ".join(reasons)
    assert "--enable-deterministic-inference" in text, text
    assert "--rl-on-policy-target" in text, text


def test_capture_sizes_are_the_widths_up_to_the_largest():
    # decode: rows are requests
    sizes = covered_capture_sizes((1, 2, 4, 6, 12), 1, [1, 2, 3, 4, 8, 16, 32])
    assert sizes == [1, 2, 4, 6, 12, 16, 32]
    # verify at 6 tokens a request: widths 6 and 12 are 1 and 2 requests
    sizes = covered_capture_sizes((1, 2, 4, 6, 12), 6, [1, 2, 3, 4, 8])
    assert sizes == [1, 2, 3, 4, 8]


def test_route_reads_only_the_graph_key():
    model, _ = attached()
    rt = model.mono
    assert shape_key(batch("decode", 2, 2)) == ShapeKey("decode", 2, 1, 2)
    assert shape_key(batch("extend", 2, 2)) is None
    assert runtime.route(model, batch("decode", 2, 2)).key.rows == 2
    assert runtime.route(model, batch("decode", 3, 3)) is None  # not a built width
    assert runtime.route(model, batch("decode", 4, 4)) is None  # no span for the key
    assert runtime.route(model, batch("target_verify", 1, 6)).key == ShapeKey(
        "target_verify", 1, 6, 6
    )
    assert runtime.route(SimpleNamespace(), batch("decode", 2, 2)) is None  # detached
    # every rank derives the same decision text from the key
    assert rt.decision(ShapeKey("decode", 4, 1, 4)).endswith(": normal")
    assert rt.decision(ShapeKey("decode", 3, 1, 3)).endswith(": normal")
    assert rt.decision(ShapeKey("decode", 2, 1, 2)).endswith(": mono")


def test_routing_runs_no_spec_code():
    model, built = attached()
    asked = built.asked
    for _ in range(3):
        assert runtime.route(model, batch("decode", 2, 2)) is not None
        assert runtime.route(model, batch("decode", 64, 64)) is None
    assert built.asked == asked  # the served keys were fixed at attach


def test_the_normal_path_has_a_reason():
    model, _ = attached()
    extend = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: False, is_target_verify=lambda: False, name="EXTEND"
        ),
        batch_size=2,
        input_ids=torch.zeros(9, dtype=torch.int64),
    )
    uneven = batch("target_verify", 2, 7)
    uneven.forward_mode.name = "TARGET_VERIFY"
    for b in (extend, uneven, batch("decode", 64, 64), batch("decode", 4, 4)):
        assert runtime.route(model, b) is None
    runtime.route(model, batch("decode", 2, 2))
    assert model.mono.counts == {
        "extend steps": 1,
        "target_verify with uneven rows a request": 1,
        "decode over 12 rows": 1,
        "decode of 4 rows, which the spec does not serve": 1,
        "mono": 1,
    }


def test_only_the_built_widths_route():
    model, _ = attached(widths=(6, 12))
    assert runtime.route(model, batch("decode", 2, 2)) is None
    assert runtime.route(model, batch("target_verify", 2, 12)) is not None


def test_each_forward_has_its_own_plan_state():
    model, built = attached()
    plan = runtime.route(model, batch("decode", 2, 2), tag="step 1")
    assert built.forward(plan, torch.zeros(1)).item() == 1
    assert built.log[-1] == (2, "step 1")
    assert runtime.route(model, batch("decode", 2, 2)).state == {}


def test_tolerance_numerics_need_per_output_bounds():
    with pytest.raises(ValueError):
        Numerics(NumericsKind.TOLERANCE)
    with pytest.raises(ValueError):
        Numerics(NumericsKind.TOLERANCE, {"layer_out.cosine": 0.9999})  # no output
    n = Numerics(NumericsKind.TOLERANCE, {"k1.kv.max_abs": 0.0, "k1.q.rel": 5e-3})
    assert replace(SPEC, numerics=n).numerics.bound("k1.q.rel") == 5e-3


def test_watched_weights_pass_through_and_keep_copies(monkeypatch):
    monkeypatch.setattr(runtime, "enabled", lambda: True)
    monkeypatch.setattr(runtime, "device_isa", lambda: "gfx950")
    model = torch.nn.Module()
    model.config = SimpleNamespace(architectures=[ARCH])
    weights = [
        ("layers.0.shared.weight", torch.ones(16, dtype=torch.uint8)),
        ("layers.0.other.weight", torch.ones(16)),
    ]
    assert list(mono.watch_weights(model, iter(weights))) == weights
    copies = model.__dict__["_mono_copies"]
    assert list(copies) == [("shared expert", "layers.0.shared.weight")]
    assert runtime._sizes(copies) == "shared expert 1 x 0.0 MiB"
    # a model no spec covers, and the switch off: the weights as they are
    other = torch.nn.Module()
    other.config = SimpleNamespace(architectures=["Other"])
    assert mono.watch_weights(other, weights) is weights
    monkeypatch.setattr(runtime, "enabled", lambda: False)
    assert mono.watch_weights(model, weights) is weights


class Lm(torch.nn.Module):
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        return "normal"


def test_the_switch_routes_each_call():
    model, built = attached()
    lm = Lm()
    lm.mono = model.mono
    runtime._switch(lm)
    x = torch.zeros(1)
    # a routed step runs the spec's forward with the model's inputs
    assert lm(x, None, batch("decode", 2, 2)).item() == 1
    assert built.log[-1] == (2, None)
    assert lm(x, None, forward_batch=batch("target_verify", 1, 6)).item() == 1
    # any other step, the model's own forward
    assert lm(x, None, batch("decode", 4, 4)) == "normal"
    assert lm.forward(x, None, batch("decode", 64, 64)) == "normal"


def test_deadlines_stay_below_the_watchdog():
    assert deadline_s(None) == MAX_DEADLINE_S
    assert deadline_s(3600) == MAX_DEADLINE_S
    assert deadline_s(30) == 15.0
