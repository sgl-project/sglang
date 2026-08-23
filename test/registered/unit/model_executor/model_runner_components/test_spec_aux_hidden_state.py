import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    SpecAuxHiddenStateConfig,
    _map_muse_target_layer_ids,
    _resolve_eagle_aux_hidden_state,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    ("target_model_type", "draft_architecture", "expected"),
    [
        ("muse_glimmer", "MuseGlimmerAssistantModel", [2, 14, 26, 38, 50]),
        ("muse_glimmer", "DFlash2DraftModel", [2, 14, 26, 38, 50]),
        ("muse_glimmer", "DFlashDraftModel", [1, 13, 25, 37, 49]),
        ("qwen3", "DFlash2DraftModel", [1, 13, 25, 37, 49]),
        ("qwen3", "MuseGlimmerAssistantModel", [1, 13, 25, 37, 49]),
    ],
)
def test_muse_target_layer_id_mapping(target_model_type, draft_architecture, expected):
    """The +1 belongs to Muse targets, which report layer outputs where the rest
    report layer inputs. The draft architecture alone does not earn it."""
    assert (
        _map_muse_target_layer_ids(
            target_hf_config=SimpleNamespace(model_type=target_model_type),
            draft_hf_config=SimpleNamespace(architectures=[draft_architecture]),
            layer_ids=[1, 13, 25, 37, 49],
        )
        == expected
    )


@patch(
    "sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state."
    "get_spec"
)
@patch(
    "sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state."
    "ModelConfig.from_server_args"
)
def test_integrated_mtp_resolves_draft_layer_count(
    mock_from_server_args, mock_get_spec
):
    mock_from_server_args.return_value = SimpleNamespace(
        num_nextn_predict_layers=1,
        is_hybrid_swa=False,
        is_deepseek_v4_arch=False,
    )
    mock_get_spec.return_value = SimpleNamespace(
        speculative_draft_model_path=None,
        speculative_draft_model_revision=None,
    )
    server_args = SimpleNamespace()
    spec_algorithm = SimpleNamespace(
        is_eagle=lambda: True,
        is_standalone=lambda: False,
        is_eagle3=lambda: False,
    )
    config = SpecAuxHiddenStateConfig()

    _resolve_eagle_aux_hidden_state(
        config=config,
        server_args=server_args,
        spec_algorithm=spec_algorithm,
        is_draft_worker=False,
    )

    assert config.eagle_draft_num_layers == 1
    mock_from_server_args.assert_called_once_with(
        server_args,
        model_path=None,
        model_revision=None,
        is_draft_model=True,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
