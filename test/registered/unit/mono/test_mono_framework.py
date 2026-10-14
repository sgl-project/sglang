"""The monokernel plugin on CPU with a fake spec and a fake kernel (RFC #43949):
spans, the registry, feature refusal, the config checks, the modes that take
only BITWISE specs, capture sizes, routing by graph key, kernels bound to their
plan, the load hook's weight copies, and the deadlines."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops import mono
from sglang.kernels.ops.mono import guards, runtime
from sglang.kernels.ops.mono import spec as registry
from sglang.kernels.ops.mono.guards import (
    MAX_DEADLINE_S,
    covered_capture_sizes,
    deadline_s,
    features_in_use,
    monokernel_disable_reasons,
)
from sglang.kernels.ops.mono.runtime import shape_key
from sglang.kernels.ops.mono.spec import (
    BuildContext,
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
    ShapeKey,
    Span,
    SpanPoint,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

ARCH = "FakeMonoForCausalLM"
LAYERS = 4
WHOLE = Span("layer", SpanPoint.LAYER_IN, SpanPoint.LAYER_OUT)
FFN = Span("ffn", SpanPoint.ATTN_PARTIAL, SpanPoint.LAYER_OUT, reason="backend_state")


class FakeModel:
    """Layer 0 takes only the FFN span; the others the whole layer, except the
    last layer in verify steps (a span turned off by step type). Decode steps of
    4 rows keep the normal path (a key turned off)."""

    def __init__(self, widths):
        self.widths = widths
        self.log = []

    def kernel(self, layer_id, point, key):
        if key.rows not in self.widths or (key.rows, key.forward_mode) == (4, "decode"):
            return None
        if layer_id == 0 and point == SpanPoint.ATTN_PARTIAL:
            return self._run
        if layer_id == LAYERS - 1 and key.forward_mode == "target_verify":
            return None
        if layer_id > 0 and point == SpanPoint.LAYER_IN:
            return self._run
        return None

    def _run(self, plan, x):
        self.log.append((plan.key.rows, plan.state.get("tag")))
        return x + 1


def _load_tensor(copies, model, name, tensor):
    if name.endswith("shared.weight"):
        copies["shared expert", name] = tensor.clone()


def _spans(_cfg):
    return {0: (FFN,), **{i: (WHOLE,) for i in range(1, LAYERS)}}


SPEC = MonoSpec(
    name="fake",
    devices={
        "gfx950": DeviceLimits(
            widths=(1, 2, 4, 6, 12), tp_sizes=(2, 4), resident_blocks=256
        )
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(kv_cache_dtypes=frozenset({"fp8_e4m3"})),
    spans=_spans,
    numerics=Numerics(NumericsKind.BITWISE),
    config_disable_reasons=lambda cfg, hf: (
        ["context_length too long"] if cfg.context_length == 7 else []
    ),
    model_disable_reasons=lambda model, hf, copies: [],
    build=lambda model, hf, ctx: FakeModel(ctx.widths),
    load_tensor=_load_tensor,
)


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


def attached(widths=None):
    """A model as attach() leaves it: ``mono`` holds the runtime."""
    widths = widths or SPEC.widths("gfx950")
    built = SPEC.build(None, None, BuildContext("gfx950", widths, 30.0, {}))
    rt = runtime.MonoRuntime(built, range(LAYERS), widths)
    return SimpleNamespace(mono=rt), built


def test_partial_span_needs_a_reason():
    with pytest.raises(ValueError):
        Span("half", SpanPoint.LAYER_IN, SpanPoint.CORE_IN)
    with pytest.raises(ValueError):
        Span("backwards", SpanPoint.LAYER_OUT, SpanPoint.LAYER_IN)
    Span("half", SpanPoint.LAYER_IN, SpanPoint.CORE_IN, reason="backend_state")


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
    assert mono.route(model, batch("decode", 2, 2)).key.rows == 2
    assert mono.route(model, batch("decode", 3, 3)) is None  # not a built width
    assert mono.route(model, batch("decode", 4, 4)) is None  # no span for the key
    assert mono.route(model, batch("target_verify", 1, 6)).key == ShapeKey(
        "target_verify", 1, 6, 6
    )
    assert mono.route(SimpleNamespace(), batch("decode", 2, 2)) is None  # detached
    # every rank derives the same decision text from the key
    assert rt.decision(ShapeKey("decode", 4, 1, 4)).endswith(": normal")
    assert rt.decision(ShapeKey("decode", 3, 1, 3)).endswith(": normal")
    assert rt.decision(ShapeKey("decode", 2, 1, 2)).endswith(
        ": 0.attn_partial 1.layer_in 2.layer_in 3.layer_in"
    )


def test_only_the_built_widths_route():
    model, _ = attached(widths=(6, 12))
    assert mono.route(model, batch("decode", 2, 2)) is None
    assert mono.route(model, batch("target_verify", 2, 12)) is not None


def test_kernels_follow_the_spans_bound_to_their_plan():
    model, built = attached()
    plan = mono.route(model, batch("decode", 2, 2), tag="step 1")
    assert mono.kernel(plan, 0, "layer_in") is None
    assert mono.kernel(plan, 0, SpanPoint.ATTN_PARTIAL)(torch.zeros(1)) == 1
    x = torch.zeros(1)
    for i in range(1, LAYERS):
        x = mono.kernel(plan, i, "layer_in")(x)
    assert x.item() == LAYERS - 1
    # route's keywords reach the kernels; each forward has its own plan state
    assert built.log[-1] == (2, "step 1")
    assert mono.route(model, batch("decode", 2, 2)).state == {}
    assert mono.kernel(None, 1, "layer_in") is None


def test_spans_can_turn_off_by_step_type():
    model, _ = attached()
    plan = mono.route(model, batch("target_verify", 1, 6))
    assert mono.kernel(plan, LAYERS - 1, "layer_in") is None
    assert mono.kernel(plan, LAYERS - 2, "layer_in") is not None


def test_tolerance_numerics_need_per_output_bounds():
    with pytest.raises(ValueError):
        Numerics(NumericsKind.TOLERANCE)
    with pytest.raises(ValueError):
        Numerics(NumericsKind.TOLERANCE, {"layer_out.cosine": 0.9999})  # no output
    n = Numerics(NumericsKind.TOLERANCE, {"k1.kv.max_abs": 0.0, "k1.q.rel": 5e-3})
    assert replace(SPEC, numerics=n).numerics.bound("k1.q.rel") == 5e-3


def test_load_hook_keeps_copies_by_kind(monkeypatch):
    monkeypatch.setattr(runtime, "enabled", lambda: True)
    monkeypatch.setattr(runtime, "device_isa", lambda: "gfx950")
    model = torch.nn.Module()
    model.config = SimpleNamespace(architectures=[ARCH])
    mono.load_tensor(model, "layers.0.shared.weight", torch.ones(16, dtype=torch.uint8))
    mono.load_tensor(model, "layers.0.other.weight", torch.ones(16))
    copies = model.__dict__["_mono_copies"]
    assert list(copies) == [("shared expert", "layers.0.shared.weight")]
    assert runtime._sizes(copies) == "shared expert 1 x 0.0 MiB"
    # off: the hook does nothing
    monkeypatch.setattr(runtime, "enabled", lambda: False)
    other = torch.nn.Module()
    other.config = model.config
    mono.load_tensor(other, "layers.0.shared.weight", torch.ones(16))
    assert "_mono_copies" not in other.__dict__


def test_deadlines_stay_below_the_watchdog():
    assert deadline_s(None) == MAX_DEADLINE_S
    assert deadline_s(3600) == MAX_DEADLINE_S
    assert deadline_s(30) == 15.0
