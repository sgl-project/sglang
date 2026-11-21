"""Keep ordinary forward and explicit native profiles on their own SDK routes."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from sglang_simulator.simulation.manager.config import ConfigManager


@pytest.mark.parametrize("native", [False, True])
def test_combined_factory_preserves_provider_selection_and_field_guards(
    monkeypatch, native
):
    calls = []

    def provider(route):
        def construct(*args, **kwargs):
            calls.append((route, args, kwargs))
            return SimpleNamespace(route=route)

        return construct

    ordinary = ModuleType("sglang_simulator.time_predictor.simopus")
    ordinary.SimOpusTimePredictor = provider("ordinary")
    staged = ModuleType("sglang_simulator.time_predictor.simopus_dspark")
    staged.SimOpusDSparkTimePredictor = provider("native")
    monkeypatch.setitem(sys.modules, ordinary.__name__, ordinary)
    monkeypatch.setitem(sys.modules, staged.__name__, staged)
    options = dict(
        name="simopus",
        model_id="deepseek-ai/DeepSeek-V4.1-Flash"
        if native
        else "deepseek-ai/DeepSeek-R1",
        model_revision="a" * 40,
        provider_revision="b" * 40,
    )
    if native:
        options["native_segment_profile_id"] = "c" * 64
    monkeypatch.setattr(ConfigManager, "_raw_config", dict(predictor=options))
    result = ConfigManager.get_inference_time_predictor(None, None, None)
    assert result.route == ("native" if native else "ordinary")
    assert len(calls) == 1 and calls[0][0] == result.route
    assert ("native_segment_profile_id" in calls[0][2]) is native
    options["unrecognized_field"] = True
    with pytest.raises(ValueError, match="unknown .*SimOpus"):
        ConfigManager.get_inference_time_predictor(None, None, None)
    assert len(calls) == 1
