"""Canonical validation reuse must preserve before-work bytes and errors."""

import json
from dataclasses import asdict, replace

import pytest
from sglang_simulator.simulation.sglang import dspark
from sglang_simulator.time_predictor.base import PredictorError


@pytest.fixture(autouse=True)
def clear_validation_cache():
    dspark._cached_canonical_budget.cache_clear()
    yield
    dspark._cached_canonical_budget.cache_clear()


def request(**changes):
    fields = dict(
        rid="request-a",
        target_prefix_len=512,
        extend_len=6,
        draft_valid_from=0,
        draft_valid_to=512,
        output_tokens_before=1,
        max_new_tokens=20,
        sampling_params_json='{"max_new_tokens":20,"temperature":1.0}',
    )
    fields.update(changes)
    return dspark.DSparkRequest(**fields)


def count_json_roundtrips(monkeypatch):
    calls = dict(loads=0, dumps=0)
    for name in calls:
        original = getattr(dspark.json, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(dspark.json, name, counted)
    return calls


def test_replacements_reuse_validation_without_changing_serialized_inputs(monkeypatch):
    calls = count_json_roundtrips(monkeypatch)
    before_request = request()
    group = dspark.DSparkGroup(
        0, "TARGET_VERIFY", (before_request,), '{"capture":{"batch":1}}'
    )
    step = dspark.DSparkStep(
        0, dspark.DSparkTopology(4, 1), (group,), '{"runtime":{"source":"same"}}'
    )
    updated_request = replace(before_request, output_tokens_before=2)
    updated_group = replace(group, requests=(updated_request,))
    updated_step = replace(step, forward_id=1, groups=(updated_group,))
    assert calls == dict(loads=3, dumps=3)
    assert updated_request.remaining_output_tokens == 18
    assert updated_step.required_phases == step.required_phases
    assert updated_request.sampling_params_json == before_request.sampling_params_json
    assert updated_group.execution_json == group.execution_json
    assert updated_step.runtime_profile_json == step.runtime_profile_json
    assert asdict(replace(updated_step, forward_id=0, groups=(group,))) == asdict(step)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"max_new_tokens": 21}, "output budget"),
        ({"output_tokens_before": 20}, "before-state"),
        ({"draft_valid_to": 513}, "before-state"),
        ({"emits_output": 1}, "chunk completion"),
        ({"max_new_tokens": True}, "integer"),
    ],
)
def test_cached_recipe_never_bypasses_current_request_checks(changes, message):
    before = request()
    with pytest.raises(PredictorError, match=message):
        replace(before, **changes)


@pytest.mark.parametrize("budget", [True, 20.0, "20", None, [20], {"value": 20}])
def test_recipe_budget_keeps_exact_integer_type(budget):
    value = json.dumps(
        {"max_new_tokens": budget}, sort_keys=True, separators=(",", ":")
    )
    for _ in range(2):
        with pytest.raises(PredictorError, match="output budget"):
            request(sampling_params_json=value)


@pytest.mark.parametrize(
    "value",
    [
        "not JSON",
        "{}",
        "[]",
        "null",
        '{"x": 1}',
        '{"z":1,"a":2}',
        '{"x":1,"x":1}',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":"\u00e9"}',
    ],
)
def test_failed_validation_preserves_original_error_and_is_not_cached(value):
    with pytest.raises(PredictorError) as original:
        dspark._canonical_object(value, "execution_json")
    for _ in range(2):
        with pytest.raises(PredictorError) as current:
            dspark._canonical_budget(value, "execution_json")
        assert current.value.args == original.value.args
        assert type(current.value.__cause__) is type(original.value.__cause__)
    assert dspark._cached_canonical_budget.cache_info().currsize == 0


def test_exact_full_string_and_label_are_retained_in_the_cache_key(monkeypatch):
    calls = count_json_roundtrips(monkeypatch)
    first = '{"max_new_tokens":20,"unknown":{"x":1}}'
    second = '{"max_new_tokens":20,"unknown":{"x":2}}'
    assert dspark._canonical_budget(first, "sampling_params_json") == 20
    assert dspark._canonical_budget(second, "sampling_params_json") == 20
    assert dspark._canonical_budget(first, "execution_json") == 20
    assert calls == dict(loads=3, dumps=3)
    assert dspark._cached_canonical_budget.cache_info().currsize == 3
    # The original parser's mutable return value cannot mutate cached metadata.
    parsed = dspark._canonical_object(first, "sampling_params_json")
    parsed["max_new_tokens"] = 99
    assert dspark._canonical_budget(first, "sampling_params_json") == 20


def test_unhashable_nonbuiltin_values_keep_the_original_validation_path():
    class UnhashableString(str):
        __hash__ = None

    value = UnhashableString('{"max_new_tokens":20}')
    with pytest.raises(PredictorError, match="must be canonical JSON"):
        dspark._canonical_budget(value, "sampling_params_json")
    # Labels were not restricted by the original parser; do not hash or reject
    # an otherwise accepted non-builtin label at the optimization boundary.
    assert dspark._canonical_budget(str(value), UnhashableString("recipe")) == 20
    assert dspark._cached_canonical_budget.cache_info().currsize == 0


@pytest.mark.parametrize("value", [None, True, 1, b"{}", {}, []])
def test_nonstring_values_keep_the_original_type_error(value):
    with pytest.raises(PredictorError, match="must be canonical JSON"):
        dspark._canonical_budget(value, "execution_json")
    assert dspark._cached_canonical_budget.cache_info().currsize == 0


def test_oversize_valid_objects_are_accepted_without_retention(monkeypatch):
    value = json.dumps({"source": "x" * (1024 * 1024)}, separators=(",", ":"))
    calls = count_json_roundtrips(monkeypatch)
    for _ in range(2):
        assert dspark._canonical_budget(value, "runtime_profile_json") is None
    assert calls == dict(loads=2, dumps=2)
    assert dspark._cached_canonical_budget.cache_info().currsize == 0


def test_successful_validation_retention_is_bounded():
    for index in range(33):
        dspark._canonical_budget(f'{{"source":{index}}}', "runtime_profile_json")
    info = dspark._cached_canonical_budget.cache_info()
    assert info.maxsize == info.currsize == 32
    dspark._canonical_budget('{"source":0}', "runtime_profile_json")
    assert dspark._cached_canonical_budget.cache_info().misses == 34
