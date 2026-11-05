import copy
import json
from dataclasses import asdict, replace

import pytest

from sglang.srt.mem_cache.storage.lmcache.load_cost_profile import (
    LoadCostKey,
    LoadCostProfile,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

KEY = LoadCostKey("model", 2, 256, 2048, 256, 512)


def profile_data():
    return {
        "version": 1,
        "metric": "request_elapsed_ms",
        "profiles": [dict(asdict(KEY), load_ms=10, recompute_ms=8)],
    }


@pytest.mark.parametrize(
    "load,recompute,margin,should_load",
    [
        (10, 8, 0.05, False),
        (10, 9.5, 0.05, False),
        (10, 9.6, 0.05, True),
        (10, 10, 0, True),
        (10, 11, 0, True),
        (10, 9, 0.2, True),
    ],
)
def test_cost_savings_margin(load, recompute, margin, should_load):
    data = profile_data()
    data["profiles"][0].update(load_ms=load, recompute_ms=recompute)
    data["min_savings"] = margin
    assert LoadCostProfile(data).should_load(KEY) is should_load


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_name", "other"),
        ("world_size", 1),
        ("chunk_size", 128),
        ("lookup_tokens", 2304),
        ("engine_computed_tokens", 0),
        ("external_tokens", 256),
    ],
)
def test_unmeasured_geometry_keeps_loading(field, value):
    policy = LoadCostProfile(profile_data())
    assert policy.should_load(replace(KEY, **{field: value}))
    assert not policy.should_load(KEY)


@pytest.mark.parametrize(
    "field,value",
    [
        ("load_ms", float("inf")),
        ("recompute_ms", float("nan")),
        ("load_ms", True),
        ("load_ms", "10"),
        ("recompute_ms", 0),
        ("load_ms", -1),
        ("load_ms", 10**1000),
        ("world_size", True),
        ("world_size", 0),
        ("chunk_size", 0),
        ("lookup_tokens", 255),
        ("engine_computed_tokens", -1),
        ("external_tokens", 0),
        ("external_tokens", 2048),
        ("model_name", ""),
    ],
)
def test_reject_invalid_measurements(field, value):
    data = profile_data()
    data["profiles"][0][field] = value
    with pytest.raises(ValueError):
        LoadCostProfile(data)


@pytest.mark.parametrize("margin", [-0.1, 1, True, float("nan"), float("inf")])
def test_reject_invalid_margin(margin):
    data = profile_data()
    data["min_savings"] = margin
    with pytest.raises(ValueError, match="min_savings"):
        LoadCostProfile(data)


def test_duplicate_or_unknown_fields_do_not_silently_change_policy():
    data = profile_data()
    data["profiles"].append(copy.deepcopy(data["profiles"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        LoadCostProfile(data)
    data = profile_data()
    data["profiles"][0]["recomputation_ms"] = 1
    with pytest.raises(ValueError, match="exact geometry"):
        LoadCostProfile(data)
    data = profile_data()
    data["minimum_savings"] = 0.5
    with pytest.raises(ValueError, match="Unknown"):
        LoadCostProfile(data)


@pytest.mark.parametrize(
    "field,value", [("version", 2), ("metric", "tokens_per_second"), ("profiles", [])]
)
def test_reject_unsupported_schema(field, value):
    data = profile_data()
    data[field] = value
    with pytest.raises(ValueError):
        LoadCostProfile(data)


def test_profile_file_is_loaded_once(tmp_path):
    path = tmp_path / "costs.json"
    path.write_text(json.dumps(profile_data()), encoding="utf-8")
    policy = LoadCostProfile.from_file(str(path))
    path.unlink()
    assert not policy.should_load(KEY)
    assert policy.should_load(replace(KEY, external_tokens=256))
