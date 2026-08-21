import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    SpecAuxHiddenStateConfig,
    _map_muse_target_layer_ids,
    _resolve_dflash_aux_hidden_state,
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


def test_explicit_dspark_target_layers_define_draft_stage_count():
    """The sidecar must cover every DSpark stage, not the HF MTP count."""
    server_args = SimpleNamespace(
        speculative_draft_model_path="/model",
        speculative_draft_model_revision=None,
    )
    target_hf_config = SimpleNamespace(
        num_hidden_layers=43, model_type="deepseek_v4"
    )
    model_config = SimpleNamespace(
        hf_config=target_hf_config,
        hf_text_config=target_hf_config,
    )
    draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_hidden_layers=43,
            num_nextn_predict_layers=1,
            dspark_block_size=5,
            dspark_markov_rank=256,
            dspark_noise_token_id=128799,
            dspark_target_layer_ids=[40, 41, 42],
        ),
        num_nextn_predict_layers=1,
    )
    spec_algorithm = MagicMock()
    spec_algorithm.is_dflash_family.return_value = True
    spec_algorithm.is_dspark.return_value = True
    config = SpecAuxHiddenStateConfig()

    with (
        patch(
            "sglang.srt.model_executor.model_runner_components."
            "spec_aux_hidden_state.ModelConfig.from_server_args",
            return_value=draft_model_config,
        ),
        patch(
            "sglang.srt.model_executor.model_runner_components."
            "spec_aux_hidden_state._resolve_dflash_draft_cell_size",
            return_value=123,
        ),
    ):
        _resolve_dflash_aux_hidden_state(
            config=config,
            server_args=server_args,
            model_config=model_config,
            spec_algorithm=spec_algorithm,
            is_draft_worker=False,
        )

    assert config.dflash_draft_num_layers == 3
    assert config.dflash_target_layer_ids == [40, 41, 42]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
