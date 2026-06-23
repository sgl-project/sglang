import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.eagle_utils import (
    get_draft_input_from_target_hidden_dim,
    get_draft_recurrent_hidden_state_spec,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _eagle_config_get(eagle_config, key, default=None):
    if isinstance(eagle_config, dict):
        return eagle_config.get(key, default)
    return getattr(eagle_config, key, default)


def _old_draft_extend_hidden_size(model_runner):
    model_config = model_runner.model_config
    if not model_runner.spec_algorithm.is_eagle3():
        return model_config.spec_hidden_size

    hf_config = model_config.hf_config
    eagle_config = getattr(hf_config, "eagle_config", None) or {}
    if not _eagle_config_get(eagle_config, "use_aux_hidden_state", True):
        return model_config.spec_hidden_size

    num_aux = getattr(hf_config, "num_aux_hidden_states", None)
    if num_aux is None:
        layer_ids = _eagle_config_get(eagle_config, "eagle_aux_hidden_state_layer_ids")
        if layer_ids is None:
            layer_ids = getattr(hf_config, "eagle_aux_hidden_state_layer_ids", None)
        num_aux = len(layer_ids) if layer_ids else 3

    target_hidden = getattr(hf_config, "target_hidden_size", model_config.hidden_size)
    return target_hidden * num_aux


def _runner(
    *,
    spec_algorithm,
    hidden_size,
    spec_hidden_size,
    hf_config,
    fc_in_features=None,
    dtype=torch.bfloat16,
):
    inner = SimpleNamespace()
    if fc_in_features is not None:
        inner.fc = SimpleNamespace(in_features=fc_in_features)

    return SimpleNamespace(
        model=SimpleNamespace(model=inner),
        model_config=SimpleNamespace(
            hidden_size=hidden_size,
            spec_hidden_size=spec_hidden_size,
            hf_config=hf_config,
            dtype=dtype,
        ),
        spec_algorithm=spec_algorithm,
    )


@pytest.mark.parametrize(
    "model_runner",
    [
        _runner(
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            hidden_size=8192,
            spec_hidden_size=8192,
            hf_config=SimpleNamespace(
                target_hidden_size=8192,
                eagle_config={
                    "use_aux_hidden_state": True,
                    "eagle_aux_hidden_state_layer_ids": [2, 23, 43],
                },
            ),
            fc_in_features=24576,
        ),
        _runner(
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            hidden_size=5120,
            spec_hidden_size=5120,
            hf_config=SimpleNamespace(
                target_hidden_size=5120,
                num_aux_hidden_states=4,
                eagle_config={"use_aux_hidden_state": True},
            ),
            fc_in_features=20480,
        ),
        _runner(
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            hidden_size=4096,
            spec_hidden_size=4096,
            hf_config=SimpleNamespace(
                target_hidden_size=4096,
                eagle_aux_hidden_state_layer_ids=[1, 7],
                eagle_config={"use_aux_hidden_state": True},
            ),
            fc_in_features=8192,
        ),
        _runner(
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            hidden_size=4096,
            spec_hidden_size=4096,
            hf_config=SimpleNamespace(
                eagle_config={"use_aux_hidden_state": False},
            ),
        ),
        _runner(
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            hidden_size=4096,
            spec_hidden_size=4096,
            hf_config=SimpleNamespace(),
            fc_in_features=8192,
        ),
    ],
)
def test_get_draft_input_from_target_hidden_dim_matches_old_oss_draft_extend_rules(
    model_runner,
):
    assert get_draft_input_from_target_hidden_dim(
        model_runner
    ) == _old_draft_extend_hidden_size(model_runner)


@pytest.mark.parametrize(
    "model_runner",
    [
        _runner(
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            hidden_size=8192,
            spec_hidden_size=8192,
            hf_config=SimpleNamespace(
                target_hidden_size=8192,
                eagle_config={
                    "use_aux_hidden_state": True,
                    "eagle_aux_hidden_state_layer_ids": [2, 23, 43],
                },
            ),
            fc_in_features=24576,
        ),
        _runner(
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            hidden_size=5120,
            spec_hidden_size=5120,
            hf_config=SimpleNamespace(
                target_hidden_size=5120,
                num_aux_hidden_states=4,
                eagle_config={"use_aux_hidden_state": True},
            ),
            fc_in_features=20480,
        ),
        _runner(
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            hidden_size=4096,
            spec_hidden_size=4096,
            hf_config=SimpleNamespace(
                target_hidden_size=4096,
                eagle_aux_hidden_state_layer_ids=[1, 7],
                eagle_config={"use_aux_hidden_state": True},
            ),
            fc_in_features=8192,
        ),
        _runner(
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            hidden_size=4096,
            spec_hidden_size=4096,
            hf_config=SimpleNamespace(
                target_hidden_size=4096,
                eagle_config={
                    "use_aux_hidden_state": True,
                    "eagle_aux_hidden_state_layer_ids": [2, 18, 30],
                },
            ),
            fc_in_features=16384,
        ),
    ],
)
def test_get_draft_input_from_target_hidden_dim_matches_oss_eagle3_projection_width(
    model_runner,
):
    assert (
        get_draft_input_from_target_hidden_dim(model_runner)
        == model_runner.model.model.fc.in_features
    )


@pytest.mark.parametrize(
    ("spec_algorithm", "expected"),
    [
        (SpeculativeAlgorithm.STANDALONE, (None, None)),
        (SpeculativeAlgorithm.EAGLE, (4096, torch.bfloat16)),
        (SpeculativeAlgorithm.EAGLE3, (4096, torch.bfloat16)),
    ],
)
def test_get_draft_recurrent_hidden_state_spec(spec_algorithm, expected):
    model_runner = _runner(
        spec_algorithm=spec_algorithm,
        hidden_size=8192,
        spec_hidden_size=4096,
        hf_config=SimpleNamespace(),
    )

    assert get_draft_recurrent_hidden_state_spec(model_runner) == expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
