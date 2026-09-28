# SPDX-License-Identifier: Apache-2.0
"""DPCache predictor, planner, DiT hook, request lifecycle and batching; CPU only."""

import itertools
import json
import math
from types import SimpleNamespace

import msgspec
import pytest
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.cache.dpcache import (
    DPCacheMixin,
    DPCacheRequestSignature,
    DPCacheState,
    build_schedule_artifact,
    check_schedule_matches,
    checkpoint_identity,
    config_digest,
    load_schedule_dir,
    pact_costs,
    pact_step_errors,
    plan_schedule,
    predict_feature,
    schedule_cost,
    validate_schedule,
)
from sglang.multimodal_gen.runtime.layers.lora.linear import BaseLayerWithLoRA
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    denoising as stage_module,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)


def linear_features(num_steps, shape=(2, 3), dtype=torch.bfloat16):
    # quarter-integer slopes keep every intermediate exactly representable
    base = torch.arange(6, dtype=torch.float32).reshape(shape) - 2
    slope = torch.tensor([0.25, -0.5, 1.0, 0.75, -1.25, 2.0]).reshape(shape)
    return [(base + slope * step).to(dtype) for step in range(num_steps)]


def collapsed_plan(costs, num_full_steps, lead):
    """Reference-style planner: one predecessor per (budget, current key)."""
    num_steps = costs.shape[0]
    best = {lead - 1: (0.0, [*range(lead)])}
    for _ in range(lead, num_full_steps):
        level = {}
        for j, (value, keys) in sorted(best.items()):
            for k in range(j + 1, num_steps):
                candidate = value + costs[keys[-2], j, k].item()
                if candidate < level.get(k, (math.inf,))[0]:
                    level[k] = (candidate, keys + [k])
        best = level
    return min(
        (value + costs[keys[-2], j, num_steps].item(), keys)
        for j, (value, keys) in best.items()
    )


def brute_force(costs, num_full_steps, lead, force_last_full=False):
    num_steps = costs.shape[0]
    best = (math.inf, None)
    for rest in itertools.combinations(range(lead, num_steps), num_full_steps - lead):
        keys = [*range(lead), *rest]
        if force_last_full and keys[-1] != num_steps - 1:
            continue
        best = min(best, (schedule_cost(costs, keys, num_steps), keys))
    return best


def random_costs(num_steps, generator):
    errors = torch.full((num_steps,) * 3, math.nan, dtype=torch.float64)
    for i, j in itertools.combinations(range(num_steps), 2):
        for t in range(j + 1, num_steps):
            errors[i, j, t] = torch.rand((), generator=generator, dtype=torch.float64)
    return pact_costs(errors)


def test_predictor_extrapolates_linear_features_over_irregular_gaps():
    features = linear_features(12)
    for i, j, t in [(0, 1, 2), (1, 4, 5), (2, 5, 11), (0, 3, 9), (4, 7, 8)]:
        actual = predict_feature(features[i], features[j], i, j, t)
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, features[t], atol=0, rtol=0)


def test_predictor_stays_in_feature_dtype_and_order():
    previous = torch.tensor([1.0, 3.0], dtype=torch.bfloat16)
    last = torch.tensor([1.5, 2.0], dtype=torch.bfloat16)
    actual = predict_feature(previous, last, 3, 6, 10)
    slope = (last - previous) / 3
    expected = last + slope * 4
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with pytest.raises(ValueError):
        predict_feature(previous, last, 6, 3, 10)
    with pytest.raises(ValueError):
        predict_feature(previous, last, 3, 6, 6)


def test_step_errors_vanish_on_linear_trajectories():
    errors = pact_step_errors(linear_features(7))
    for i, j, t in itertools.product(range(7), repeat=3):
        if i < j < t:
            assert errors[i, j, t].item() == 0.0
        else:
            assert math.isnan(errors[i, j, t].item())


def test_step_errors_are_mean_abs_of_the_runtime_prediction():
    torch.manual_seed(0)
    features = [torch.randn(4, 5).bfloat16() for _ in range(5)]
    errors = pact_step_errors(features)
    predicted = predict_feature(features[1], features[3], 1, 3, 4)
    expected = (predicted.float() - features[4].float()).abs().double().mean()
    assert errors[1, 3, 4].item() == pytest.approx(expected.item(), rel=1e-12)


def test_costs_score_only_predicted_steps():
    generator = torch.Generator().manual_seed(1)
    num_steps = 6
    errors = torch.full((num_steps,) * 3, math.nan, dtype=torch.float64)
    for i, j in itertools.combinations(range(num_steps), 2):
        for t in range(j + 1, num_steps):
            errors[i, j, t] = torch.rand((), generator=generator, dtype=torch.float64)
    costs = pact_costs(errors)
    for i, j in itertools.combinations(range(num_steps), 2):
        assert costs[i, j, j + 1].item() == 0.0  # adjacent keys predict nothing
        for k in range(j + 2, num_steps + 1):
            expected = sum(errors[i, j, t].item() for t in range(j + 1, k))
            assert costs[i, j, k].item() == pytest.approx(expected, rel=1e-12)
        # the terminal sentinel T scores steps through T-1 and nothing at T
        assert costs[i, j, num_steps].item() == pytest.approx(
            errors[i, j, j + 1 :].sum().item(), rel=1e-12
        )
    assert math.isinf(costs[3, 2, 4].item())


@pytest.mark.parametrize("num_steps", [5, 7, 9])
@pytest.mark.parametrize("lead", [2, 3])
def test_exact_planner_matches_exhaustive_enumeration(num_steps, lead):
    generator = torch.Generator().manual_seed(num_steps * 10 + lead)
    for _ in range(4):
        costs = random_costs(num_steps, generator)
        for budget in range(lead, num_steps + 1):
            mandatory = tuple(range(lead))
            keys, value = plan_schedule(costs, budget, mandatory)
            expected_value, _ = brute_force(costs, budget, lead)
            assert len(keys) == budget and keys[:lead] == list(mandatory)
            assert value == pytest.approx(expected_value, abs=1e-12)
            assert schedule_cost(costs, keys, num_steps) == pytest.approx(
                value, abs=1e-12
            )
            if budget > lead:
                forced, forced_value = plan_schedule(
                    costs, budget, mandatory, force_last_full=True
                )
                assert forced[-1] == num_steps - 1
                assert forced_value == pytest.approx(
                    brute_force(costs, budget, lead, force_last_full=True)[0],
                    abs=1e-12,
                )


def test_predecessor_collapsed_dp_is_not_optimal_but_exact_dp_is():
    generator = torch.Generator().manual_seed(7)
    for _ in range(200):
        costs = random_costs(7, generator)
        collapsed, _ = collapsed_plan(costs, 5, 3)
        exact_keys, exact = plan_schedule(costs, 5)
        optimum, _ = brute_force(costs, 5, 3)
        assert exact == pytest.approx(optimum, abs=1e-12)
        if collapsed > optimum + 1e-9:
            break
    else:
        pytest.fail("no counterexample for the predecessor-collapsed planner")


def test_planner_is_deterministic_on_ties():
    costs = pact_costs(torch.zeros(6, 6, 6, dtype=torch.float64))
    assert plan_schedule(costs, 4) == plan_schedule(costs, 4) == ([0, 1, 2, 3], 0.0)


def test_planner_rejects_infeasible_budgets_and_bad_mandatory_steps():
    costs = random_costs(6, torch.Generator().manual_seed(0))
    for budget in (2, 7, -1):
        with pytest.raises(ValueError):
            plan_schedule(costs, budget)
    with pytest.raises(ValueError):
        plan_schedule(costs, 3, force_last_full=True)
    with pytest.raises(TypeError):
        plan_schedule(costs, 4.0)
    for mandatory in [(0,), (1, 2), (0, 2), ()]:
        with pytest.raises(ValueError):
            plan_schedule(costs, 4, mandatory)
    assert plan_schedule(costs, 6) == ([0, 1, 2, 3, 4, 5], 0.0)
    unreachable = costs.clone()
    unreachable[:, :, 6] = math.inf
    with pytest.raises(ValueError, match="no finite schedule"):
        plan_schedule(unreachable, 4)


def brute_force_gap(costs, num_full_steps, lead, max_gap):
    num_steps = costs.shape[0]
    best = math.inf
    for rest in itertools.combinations(range(lead, num_steps), num_full_steps - lead):
        keys = [*range(lead), *rest, num_steps]
        if all(b - a <= max_gap for a, b in zip(keys, keys[1:])):
            best = min(best, schedule_cost(costs, keys[:-1], num_steps))
    return best


@pytest.mark.parametrize("max_gap", [2, 3, 4])
def test_gap_bounded_planner_matches_exhaustive_enumeration(max_gap):
    """Every key gap, the terminal one included, must respect max_gap."""
    generator = torch.Generator().manual_seed(max_gap)
    num_steps = 9
    for _ in range(3):
        raw = torch.full((num_steps,) * 3, math.nan, dtype=torch.float64)
        for i, j in itertools.combinations(range(num_steps), 2):
            if j - i <= max_gap:
                for t in range(j + 1, min(num_steps, j + max_gap)):
                    raw[i, j, t] = torch.rand(
                        (), generator=generator, dtype=torch.float64
                    )
        costs = pact_costs(raw, max_gap=max_gap)
        for budget in range(3, num_steps + 1):
            expected = brute_force_gap(costs, budget, 3, max_gap)
            if math.isinf(expected):
                with pytest.raises(ValueError, match="no finite schedule"):
                    plan_schedule(costs, budget, max_gap=max_gap)
                continue
            keys, value = plan_schedule(costs, budget, max_gap=max_gap)
            gaps = [b - a for a, b in zip(keys, keys[1:] + [num_steps])]
            assert max(gaps) <= max_gap
            assert value == pytest.approx(expected, abs=1e-12)


def test_gap_bound_applies_to_the_terminal_segment_even_with_unmasked_costs():
    costs = pact_costs(torch.zeros(6, 6, 6, dtype=torch.float64))
    with pytest.raises(ValueError, match="no finite schedule"):
        plan_schedule(costs, 3, max_gap=2)
    keys, _ = plan_schedule(costs, 4, max_gap=2)
    assert 6 - keys[-1] <= 2 and max(b - a for a, b in zip(keys, keys[1:])) <= 2


def test_gap_bounded_errors_score_only_usable_triples():
    features = [
        torch.randn(3, generator=torch.Generator().manual_seed(s)).bfloat16()
        for s in range(8)
    ]
    errors = pact_step_errors(features, max_gap=3)
    for i, j, t in itertools.product(range(8), repeat=3):
        usable = i < j < t and j - i <= 3 and t < j + 3
        assert math.isfinite(errors[i, j, t].item()) == usable
    costs = pact_costs(errors, max_gap=3)
    assert math.isinf(costs[0, 4, 5].item())  # anchor gap 4 > 3
    assert math.isinf(costs[1, 2, 6].item())  # key gap 4 > 3
    assert math.isfinite(costs[4, 5, 8].item())  # terminal gap 3


def test_pact_costs_reject_nonfinite_errors():
    errors = torch.zeros(4, 4, 4, dtype=torch.float64)
    errors[0, 1, 3] = math.inf
    with pytest.raises(ValueError, match="nonfinite"):
        pact_costs(errors)


def test_state_uses_only_full_step_anchors():
    features = linear_features(8)
    state = DPCacheState([0, 1, 2, 5], num_steps=8)
    for step in (0, 1, 2):
        state.record(step, features[step])
    torch.testing.assert_close(state.predict(3), features[3], atol=0, rtol=0)
    torch.testing.assert_close(state.predict(4), features[4], atol=0, rtol=0)
    # a full step whose feature departs from the line resets the anchors to (2, 5)
    kink = features[5] + 1
    state.record(5, kink)
    expected = predict_feature(features[2], kink, 2, 5, 6)
    torch.testing.assert_close(state.predict(6), expected, atol=0, rtol=0)
    assert (state.num_full, state.num_predicted) == (4, 3)


def test_state_stores_a_detached_copy():
    feature = torch.ones(3, dtype=torch.bfloat16)
    state = DPCacheState([0, 1], num_steps=3)
    state.record(0, feature)
    feature.fill_(5)
    state.record(1, torch.full((3,), 2.0, dtype=torch.bfloat16))
    torch.testing.assert_close(state.predict(2), torch.full((3,), 3.0).bfloat16())


def test_state_rejects_out_of_order_unscheduled_and_non_bf16_steps():
    zero = torch.zeros(1, dtype=torch.bfloat16)
    state = DPCacheState([0, 1, 3], num_steps=4)
    with pytest.raises(RuntimeError, match="BF16 features only"):
        state.record(0, torch.zeros(1))
    with pytest.raises(RuntimeError, match="not a scheduled full step"):
        state.record(2, zero)
    state.record(0, zero)
    with pytest.raises(RuntimeError, match="fewer than two"):
        DPCacheState([0, 2], num_steps=3).predict(1)
    with pytest.raises(RuntimeError, match="must increase"):
        state.record(0, zero)
    with pytest.raises(RuntimeError, match="is a scheduled full step"):
        state.predict(1)
    with pytest.raises(ValueError, match="outside"):
        state.predict(4)


def test_states_are_isolated_per_branch():
    positive = DPCacheState([0, 1], num_steps=3)
    negative = DPCacheState([0, 1], num_steps=3)
    for step in (0, 1):
        positive.record(step, torch.full((2,), float(step), dtype=torch.bfloat16))
        negative.record(step, torch.full((2,), -2.0 * step, dtype=torch.bfloat16))
    assert positive.predict(2).tolist() == [2.0, 2.0]
    assert negative.predict(2).tolist() == [-4.0, -4.0]


SIGNATURE = DPCacheRequestSignature(
    pipeline="SimpleNamespace",
    checkpoint="790c92633540aa0cb11d9abf19eb46d861714758",
    num_inference_steps=6,
    height=64,
    width=64,
    guidance_scale=1.0,
    do_classifier_free_guidance=False,
    quality="lossless",
    attention_backend="torch_sdpa",
    dtype="bfloat16",
    scheduler="SimpleNamespace",
    scheduler_config_sha256=config_digest({"shift": 1.0}),
    timesteps=(1000.0, 800.0, 600.0, 400.0, 200.0, 100.0),
    sigmas=(1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0),
)


def artifact(full_steps=(0, 1, 2, 4), max_gap=None, **overrides):
    result = build_schedule_artifact(
        signature=SIGNATURE,
        full_steps=list(full_steps),
        mandatory=(0, 1, 2),
        calibrated_cost=0.5,
        calibration={"manifest_sha256": "0" * 64, "num_samples": 1},
        source_commit="deadbeef",
        max_gap=max_gap,
    )
    request = overrides.pop("request", {})
    result.update(overrides)
    result["request"].update(request)
    return result


def test_schedule_artifact_roundtrip_and_request_match():
    schedule = json.loads(json.dumps(artifact()))
    assert validate_schedule(schedule) == (0, 1, 2, 4)
    assert check_schedule_matches(schedule, SIGNATURE) == (0, 1, 2, 4)


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema": "sglang-dpcache-schedule-v1"},
        {"full_steps": [0, 1, 4, 2], "num_full_steps": 4},
        {"full_steps": [0, 2, 3, 4], "num_full_steps": 4},
        {"full_steps": [0, 1, 2, 6], "num_full_steps": 4},
        {"full_steps": [0, 1, 2, 2], "num_full_steps": 4},
        {"full_steps": [0, True, 2, 4], "num_full_steps": 4},
        {"num_full_steps": 5},
        {"mandatory_full_steps": [1, 2]},
        {"mandatory_full_steps": [0, True, 2]},
        {"predictor": {"name": "taylor", "order": 2, "arithmetic": "fp32"}},
        {"objective": "endpoint-inclusive"},
        {"force_last_full": True},
        {"max_gap": 1},
        {"max_gap": 0},
        {"source_commit": ""},
        {"calibration": {"num_samples": 1}},
        {"calibrated_cost": float("nan")},
        {"request": {"timesteps": [1.0]}},
        {"request": {"sigmas": [1.0, 0.8, 0.6, 0.4, 0.2, float("inf"), 0.0]}},
        {"request": {"height": 0}},
        {"request": {"dtype": "float32"}},
        {"request": {"extra": 1}},
    ],
)
def test_malformed_schedules_are_rejected(overrides):
    with pytest.raises(ValueError):
        validate_schedule(artifact(**overrides))


def test_schedule_gaps_include_the_terminal_segment():
    assert validate_schedule(artifact(full_steps=(0, 1, 2, 4), max_gap=2))
    with pytest.raises(ValueError, match="max_gap"):
        validate_schedule(artifact(full_steps=(0, 1, 2, 3), max_gap=2))


def test_schedule_without_required_fields_is_rejected():
    schedule = artifact()
    del schedule["calibration"]
    with pytest.raises(ValueError, match="missing"):
        validate_schedule(schedule)
    with pytest.raises(TypeError):
        validate_schedule([0, 1, 2])


@pytest.mark.parametrize(
    "field, value",
    [
        ("height", 128),
        ("num_inference_steps", 7),
        ("guidance_scale", 4.0),
        ("do_classifier_free_guidance", True),
        ("quality", "high"),
        ("attention_backend", "fa"),
        ("dtype", "float16"),
        ("checkpoint", "other-revision"),
        ("scheduler", "OtherScheduler"),
        ("scheduler_config_sha256", config_digest({"shift": 3.0})),
        ("sigmas", (1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01)),
        ("timesteps", (1000.0, 800.0, 600.0, 400.0, 200.0, 99.0)),
    ],
)
def test_mismatched_request_is_rejected(field, value):
    signature = msgspec.structs.replace(SIGNATURE, **{field: value})
    with pytest.raises(ValueError, match="does not match"):
        check_schedule_matches(artifact(), signature)


def test_checkpoint_identity_uses_snapshot_revision(tmp_path):
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    assert checkpoint_identity(str(snapshot)) == "abc123"
    assert checkpoint_identity("Qwen/Qwen-Image-2.1") == "Qwen/Qwen-Image-2.1"


class _TinyDiT(DPCacheMixin, torch.nn.Module):
    """A block stack that doubles its input, behind the DPCache hook."""

    _supports_dpcache = True

    def __init__(self):
        super().__init__()
        self.proj_out = torch.nn.Linear(1, 1, dtype=torch.bfloat16)
        self.block_calls = 0

    def forward(self, x):
        def run_blocks():
            self.block_calls += 1
            return x * 2

        return self.dpcache_blocks(run_blocks)


def forward_batch(states=None, budget=None, is_cfg_negative=False, is_warmup=False):
    return SimpleNamespace(
        dpcache_states=states,
        dpcache_budget=budget,
        is_cfg_negative=is_cfg_negative,
        is_warmup=is_warmup,
    )


def run_dit(dit, step, x, batch):
    with set_forward_context(
        current_timestep=step, attn_metadata=None, forward_batch=batch
    ):
        return dit(x)


def test_hook_runs_full_steps_and_predicts_the_rest():
    dit = _TinyDiT()
    batch = forward_batch({False: DPCacheState([0, 1], num_steps=3)})
    x = torch.ones(2, dtype=torch.bfloat16)
    assert run_dit(dit, 0, x * 0, batch).tolist() == [0.0, 0.0]
    assert run_dit(dit, 1, x, batch).tolist() == [2.0, 2.0]
    assert run_dit(dit, 2, x * 100, batch).tolist() == [4.0, 4.0]
    assert dit.block_calls == 2


def test_hook_keeps_one_state_per_cfg_branch():
    dit = _TinyDiT()
    states = {
        False: DPCacheState([0, 1], num_steps=3),
        True: DPCacheState([0, 1], num_steps=3),
    }
    for step in (0, 1):
        run_dit(
            dit,
            step,
            torch.full((1,), float(step), dtype=torch.bfloat16),
            forward_batch(states),
        )
        run_dit(
            dit,
            step,
            torch.full((1,), -float(step), dtype=torch.bfloat16),
            forward_batch(states, is_cfg_negative=True),
        )
    x = torch.zeros(1, dtype=torch.bfloat16)
    assert run_dit(dit, 2, x, forward_batch(states)).tolist() == [4.0]
    negative = forward_batch(states, is_cfg_negative=True)
    assert run_dit(dit, 2, x, negative).tolist() == [-4.0]


@pytest.mark.parametrize(
    "batch",
    [None, forward_batch(), forward_batch(budget=4, is_warmup=True)],
)
def test_hook_without_state_runs_the_blocks(batch):
    dit = _TinyDiT()
    x = torch.ones(1, dtype=torch.bfloat16)
    assert run_dit(dit, 0, x, batch).tolist() == [2.0]
    assert dit.block_calls == 1


def test_hook_refuses_a_requested_budget_without_state():
    """A requested budget must never silently run uncached."""
    dit = _TinyDiT()
    with pytest.raises(RuntimeError, match="no state is attached"):
        run_dit(dit, 0, torch.ones(1), forward_batch(budget=4))
    assert dit.block_calls == 0


@pytest.fixture(autouse=True)
def single_gpu(monkeypatch):
    monkeypatch.setattr(stage_module, "get_sp_world_size", lambda: 1)
    monkeypatch.setattr(stage_module, "get_tp_world_size", lambda: 1)


class _FakeLoRA(BaseLayerWithLoRA):
    def __init__(self, disable_lora):
        torch.nn.Module.__init__(self)
        self.merged, self.disable_lora = False, disable_lora


def make_server_args(**overrides):
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(),
        model_path="790c92633540aa0cb11d9abf19eb46d861714758",
        attention_backend="torch_sdpa",
        enable_torch_compile=False,
        enable_breakable_cuda_graph=False,
        enable_cfg_parallel=False,
        dpcache_schedule_dir=None,
        dpcache_default_budget=None,
    )
    server_args.__dict__.update(overrides)
    return server_args


def make_stage(
    dtype=torch.bfloat16,
    lora=None,
    supported=True,
    schedules=None,
    **server_overrides,
):
    stage = object.__new__(DenoisingStage)
    transformer = _TinyDiT().to(dtype)
    transformer._supports_dpcache = supported
    if lora is not None:
        transformer.lora = _FakeLoRA(disable_lora=not lora)
    stage.transformer = transformer
    stage.transformer_2 = None
    stage._cache_dit_enabled = False
    stage._offloaded_dit_modules_for_compile = []
    # K=4 for the 64x64, 6-step SIGNATURE request
    stage._dpcache_schedules = [artifact()] if schedules is None else schedules
    server_args = make_server_args(**server_overrides)
    stage.server_args = server_args
    return stage, server_args


def make_batch(budget, height=64, **overrides):
    fields = dict(
        dpcache_budget=budget,
        dpcache_calibration=None,
        seed=42,
        enable_cache_dit=False,
        enable_teacache=False,
        enable_spectrum=False,
        skip_softmax_params=None,
        attention_backend_override=None,
        do_classifier_free_guidance=False,
        guidance_scale=1.0,
        quality="lossless",
        condition_image=None,
        image_path=None,
        prompt="a cat",
        num_outputs_per_prompt=1,
    )
    fields.update(overrides)
    return SimpleNamespace(
        sampling_params=SimpleNamespace(**fields),
        timesteps=torch.tensor(SIGNATURE.timesteps),
        scheduler=SimpleNamespace(
            sigmas=torch.tensor(SIGNATURE.sigmas), config={"shift": 1.0}
        ),
        height=height,
        width=64,
        is_warmup=False,
        dpcache_states=None,
        **fields,
    )


def denoise_must_not_run(monkeypatch):
    monkeypatch.setattr(DenoisingStage, "_denoise", lambda *a: pytest.fail("ran"))


def record_states(monkeypatch):
    """Stand-in denoise that records which states the request carried."""
    seen = []
    monkeypatch.setattr(
        DenoisingStage,
        "_denoise",
        lambda self, batch, server_args: seen.append(batch.dpcache_states) or batch,
    )
    return seen


def test_stage_attaches_fresh_state_per_branch_and_cleans_up_on_error(monkeypatch):
    stage, server_args = make_stage()
    batch = make_batch(4)
    seen = {}

    def failing_denoise(self, batch, server_args):
        seen.update(batch.dpcache_states)
        raise RuntimeError("boom")

    monkeypatch.setattr(DenoisingStage, "_denoise", failing_denoise)
    with pytest.raises(RuntimeError, match="boom"):
        stage.forward(batch, server_args)
    assert list(seen) == [False]
    assert seen[False].full_steps == frozenset({0, 1, 2, 4})
    assert batch.dpcache_states is None

    monkeypatch.setattr(DenoisingStage, "_denoise", lambda self, b, s: b)
    stage.forward(batch, server_args)
    assert batch.dpcache_states is None


@pytest.mark.parametrize("budget", [None, 0])
def test_stage_without_a_budget_is_untouched(monkeypatch, budget):
    stage, server_args = make_stage(lora=True, supported=False)
    batch = make_batch(budget, do_classifier_free_guidance=True)
    seen = record_states(monkeypatch)
    assert stage.forward(batch, server_args) is batch
    assert seen == [None]


def test_server_default_budget_applies_to_requests_without_one(monkeypatch):
    stage, server_args = make_stage(dpcache_default_budget=4)
    seen = record_states(monkeypatch)
    stage.forward(make_batch(None), server_args)
    assert seen[0][False].full_steps == frozenset({0, 1, 2, 4})
    stage.forward(make_batch(0), server_args)
    assert seen[1] is None


@pytest.mark.parametrize(
    "batch_overrides, stage_overrides",
    [
        ({"height": 128}, {}),
        ({"do_classifier_free_guidance": True}, {}),
        ({}, {"supported": False}),
        ({}, {"schedules": []}),
    ],
)
def test_server_default_budget_falls_back_to_native(
    monkeypatch, batch_overrides, stage_overrides
):
    """A default must not fail requests it was never calibrated for."""
    stage, server_args = make_stage(dpcache_default_budget=4, **stage_overrides)
    batch = make_batch(None)
    batch.__dict__.update(batch_overrides)
    seen = record_states(monkeypatch)
    assert stage.forward(batch, server_args) is batch
    assert seen == [None]


def test_warmup_request_runs_without_the_schedule(monkeypatch):
    """A warmup copy keeps the budget but runs fewer steps than any schedule."""
    stage, server_args = make_stage()
    batch = make_batch(4)
    batch.is_warmup = True
    batch.timesteps = batch.timesteps[:1]
    seen = record_states(monkeypatch)
    assert stage.forward(batch, server_args) is batch
    assert seen == [None]


def test_explicit_budget_without_served_schedules_fails(monkeypatch):
    stage, server_args = make_stage(schedules=[])
    denoise_must_not_run(monkeypatch)
    with pytest.raises(ValueError, match="--dpcache-schedule-dir"):
        stage.forward(make_batch(4), server_args)


def test_stage_rejects_a_model_without_the_hook(monkeypatch):
    stage, server_args = make_stage(supported=False)
    denoise_must_not_run(monkeypatch)
    with pytest.raises(ValueError, match="not supported by _TinyDiT"):
        stage.forward(make_batch(4), server_args)


@pytest.mark.parametrize(
    "stage_overrides, batch_overrides",
    [
        ({"enable_torch_compile": True}, {}),
        ({"enable_breakable_cuda_graph": True}, {}),
        ({"enable_cfg_parallel": True}, {}),
        ({"dtype": torch.float32}, {}),
        ({"lora": True}, {}),
        ({}, {"enable_teacache": True}),
        ({}, {"enable_spectrum": True}),
        ({}, {"enable_cache_dit": True}),
        ({}, {"skip_softmax_params": {"threshold": 0.1}}),
        ({}, {"attention_backend_override": "fa"}),
        ({}, {"do_classifier_free_guidance": True}),
        ({}, {"quality": "high"}),
        ({}, {"condition_image": "ref.png"}),
        ({}, {"image_path": ["ref.png"]}),
        ({}, {"prompt": ["a", "b"]}),
        ({}, {"num_outputs_per_prompt": 2}),
    ],
)
def test_stage_rejects_unsupported_compositions(
    monkeypatch, stage_overrides, batch_overrides
):
    stage, server_args = make_stage(**stage_overrides)
    batch = make_batch(4, **batch_overrides)
    denoise_must_not_run(monkeypatch)
    with pytest.raises(ValueError, match="cannot be combined"):
        stage.forward(batch, server_args)


def test_stage_rejects_dual_transformers(monkeypatch):
    stage, server_args = make_stage()
    stage.transformer_2 = _TinyDiT()
    denoise_must_not_run(monkeypatch)
    with pytest.raises(ValueError, match="dual transformers"):
        stage.forward(make_batch(4), server_args)


def test_stage_accepts_loaded_but_disabled_lora(monkeypatch):
    stage, server_args = make_stage(lora=False)
    batch = make_batch(4)
    monkeypatch.setattr(DenoisingStage, "_denoise", lambda self, b, s: b)
    assert stage.forward(batch, server_args) is batch


def test_stale_cache_dit_mount_is_unmounted_instead_of_rejected(monkeypatch):
    """A DPCache request after a Cache-DiT request must not fail on the old mount."""
    stage, server_args = make_stage()
    stage._cache_dit_enabled = True
    unmounted = []
    monkeypatch.setattr(
        DenoisingStage,
        "_unmount_cache_dit",
        lambda self: (
            unmounted.append(True) or setattr(self, "_cache_dit_enabled", False)
        ),
    )
    monkeypatch.setattr(DenoisingStage, "_denoise", lambda self, b, s: b)
    batch = make_batch(4)
    assert stage.forward(batch, server_args) is batch
    assert unmounted == [True]

    stage, server_args = make_stage()
    stage._cache_dit_enabled = True
    batch = make_batch(4, enable_cache_dit=True)
    denoise_must_not_run(monkeypatch)
    with pytest.raises(ValueError, match="Cache-DiT"):
        stage.forward(batch, server_args)


@pytest.mark.parametrize("budget, height", [(5, 64), (4, 128)])
def test_stage_rejects_a_budget_or_request_without_a_schedule(
    monkeypatch, budget, height
):
    stage, server_args = make_stage()
    batch = make_batch(budget, height=height)
    denoise_must_not_run(monkeypatch)
    with pytest.raises(ValueError, match="no DPCache schedule") as info:
        stage.forward(batch, server_args)
    assert "available: K=4 (64x64, 6 steps, guidance 1.0)" in str(info.value)


def test_stage_selects_the_schedule_calibrated_for_the_request(monkeypatch):
    tall = artifact(full_steps=(0, 1, 2, 5), request={"height": 128})
    stage, server_args = make_stage(schedules=[tall, artifact()])
    seen = record_states(monkeypatch)
    stage.forward(make_batch(4, height=128), server_args)
    stage.forward(make_batch(4), server_args)
    assert [s[False].full_steps for s in seen] == [
        frozenset({0, 1, 2, 5}),
        frozenset({0, 1, 2, 4}),
    ]


def write_schedules(directory, *schedules):
    directory.mkdir(exist_ok=True)
    for index, schedule in enumerate(schedules):
        (directory / f"K{index}.json").write_text(json.dumps(schedule))
    return str(directory)


SERVED = dict(
    pipeline=SIGNATURE.pipeline,
    checkpoint=SIGNATURE.checkpoint,
    attention_backend=SIGNATURE.attention_backend,
)


def test_schedule_dir_loads_every_schedule(tmp_path):
    tall = artifact(request={"height": 128})
    path = write_schedules(
        tmp_path / "s", artifact(), artifact(full_steps=(0, 1, 2, 3, 5)), tall
    )
    schedules = load_schedule_dir(path, **SERVED)
    assert [s["num_full_steps"] for s in schedules] == [4, 5, 4]


@pytest.mark.parametrize(
    "served, match",
    [
        ({"pipeline": "OtherPipelineConfig"}, "pipeline"),
        ({"checkpoint": "other-revision"}, "checkpoint"),
        ({"attention_backend": "fa"}, "attention_backend"),
    ],
)
def test_schedule_dir_rejects_schedules_for_another_server(tmp_path, served, match):
    path = write_schedules(tmp_path / "s", artifact())
    with pytest.raises(ValueError, match=match):
        load_schedule_dir(path, **{**SERVED, **served})


def test_schedule_dir_rejects_ambiguous_invalid_and_empty_sets(tmp_path):
    path = write_schedules(
        tmp_path / "dup", artifact(), artifact(full_steps=(0, 1, 2, 3))
    )
    with pytest.raises(ValueError, match="same budget"):
        load_schedule_dir(path, **SERVED)
    path = write_schedules(tmp_path / "bad", artifact(schema="v0"))
    with pytest.raises(ValueError, match="K0.json"):
        load_schedule_dir(path, **SERVED)
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no DPCache schedules"):
        load_schedule_dir(str(tmp_path / "empty"), **SERVED)


def test_server_start_checks_the_schedule_dir(tmp_path):
    stage, _ = make_stage()
    stage.server_args = make_server_args(
        dpcache_schedule_dir=write_schedules(tmp_path / "s", artifact())
    )
    assert len(stage._load_dpcache_schedules()) == 1
    stage.transformer._supports_dpcache = False
    with pytest.raises(ValueError, match="does not support DPCache"):
        stage._load_dpcache_schedules()
    stage.server_args = make_server_args()
    assert stage._load_dpcache_schedules() == []


@pytest.mark.parametrize("budget", [1, -1, True, 2.0])
def test_request_budget_must_be_off_or_a_budget(budget):
    with pytest.raises(ValueError, match="dpcache_budget"):
        SamplingParams(dpcache_budget=budget)
    assert SamplingParams(dpcache_budget=0).dpcache_budget == 0


def scheduler_request(budget, calibration=None):
    sampling = SamplingParams(
        prompt="a cat", dpcache_budget=budget, dpcache_calibration=calibration
    )
    return SimpleNamespace(
        is_warmup=False,
        realtime_session_id=None,
        session=None,
        prompt=sampling.prompt,
        image_path=None,
        return_file_paths_only=False,
        num_outputs_per_prompt=1,
        sampling_params=sampling,
        dpcache_budget=budget,
        dpcache_calibration=calibration,
    )


def make_scheduler(default_budget=None):
    scheduler = object.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            supports_sequential_multi_output_inference=lambda: False,
            supports_sequential_dit_inference=lambda: False,
        ),
        dpcache_default_budget=default_budget,
    )
    return scheduler


@pytest.mark.parametrize(
    "budgets, default_budget",
    [((4, 4), None), ((4, None), None), ((None, 4), None), ((None, None), 4)],
)
def test_dpcache_requests_are_not_dynamically_batched(budgets, default_budget):
    """A merged request would carry several prompts, which DPCache rejects."""
    scheduler = make_scheduler(default_budget)
    base, candidate = (scheduler_request(budget) for budget in budgets)
    assert not scheduler._can_dynamic_batch(base, candidate)
    assert scheduler._get_dynamic_batch_reject_reason(base, candidate) == "dpcache"


def test_opted_out_requests_still_batch_under_a_default_budget():
    scheduler = make_scheduler(default_budget=4)
    request = scheduler_request(0)
    assert not scheduler._uses_dpcache(request)
