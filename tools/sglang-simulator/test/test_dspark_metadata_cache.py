"""Metadata validation survives batch pressure without rechecking static JSON."""

import json
from collections import Counter

import pytest
from sglang_simulator.simulation.sglang import dspark
from sglang_simulator.time_predictor.base import PredictorError


@pytest.fixture(autouse=True)
def clear_metadata():
    caches = (
        dspark._cached_canonical_budget,
        dspark._cached_canonical_profile,
        dspark._cached_canonical_sampling,
    )
    for cache in caches:
        cache.cache_clear()
    yield
    for cache in caches:
        cache.cache_clear()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize("batch_size", [1, 16, 32, 64])
def test_changing_batches_do_not_evict_immutable_run_or_request_facts(
    monkeypatch, batch_size
):
    original = dspark._read_canonical_budget
    calls = Counter()

    def read(value, label):
        calls[label] += 1
        return original(value, label)

    monkeypatch.setattr(dspark, "_read_canonical_budget", read)
    profile = encoded({"source": "fixture", "padding": "x" * 512_000})
    recipes = [encoded({"max_new_tokens": 100 + i}) for i in range(batch_size)]
    for step in range(80):
        for _ in range(2):
            for i, recipe in enumerate(recipes):
                assert (
                    dspark._canonical_budget(recipe, "sampling_params_json") == 100 + i
                )
            assert (
                dspark._canonical_budget(
                    encoded({"seq_lens_cpu": [step] * batch_size}), "execution_json"
                )
                is None
            )
            assert dspark._canonical_budget(profile, "runtime_profile_json") is None
    assert calls["runtime_profile_json"] == 1
    assert calls["sampling_params_json"] == batch_size
    assert calls["execution_json"] == 80


@pytest.mark.parametrize(
    "label", ["sampling_params_json", "runtime_profile_json", "execution_json"]
)
def test_cache_never_accepts_changed_noncanonical_or_malformed_text(label):
    assert dspark._canonical_budget('{"max_new_tokens":12}', label) == 12
    assert dspark._canonical_budget('{"max_new_tokens":13}', label) == 13
    for invalid in ('{"max_new_tokens": 12}', '{"a":1,"a":1}', '{"x":NaN}', "[]", "{"):
        with pytest.raises(PredictorError):
            dspark._canonical_budget(invalid, label)


@pytest.mark.parametrize(
    ("label", "padding"),
    [("sampling_params_json", 16 * 1024), ("execution_json", 64 * 1024)],
)
def test_large_documents_remain_validated_without_retaining_them(
    monkeypatch, label, padding
):
    original = dspark._read_canonical_budget
    calls = []

    def read(value, current_label):
        calls.append(current_label)
        return original(value, current_label)

    monkeypatch.setattr(dspark, "_read_canonical_budget", read)
    value = encoded({"max_new_tokens": 123, "padding": "x" * padding})
    assert dspark._canonical_budget(value, label) == 123
    assert dspark._canonical_budget(value, label) == 123
    assert calls == [label, label]
