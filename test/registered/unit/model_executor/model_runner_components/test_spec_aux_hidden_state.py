import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from sglang.srt.model_executor.model_runner_components import spec_aux_hidden_state
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(
    est_time=1,
    suite="base-a-test-cpu",
    nightly=False,
    disabled=None,
)


@pytest.fixture(autouse=True)
def _published_config():
    with get_context().override_server_args(
        speculative_draft_model_path=None,
        speculative_draft_model_revision=None,
    ):
        yield


def _resolve_kwargs():
    spec_algorithm = MagicMock()
    spec_algorithm.is_eagle.return_value = True
    spec_algorithm.is_standalone.return_value = False
    spec_algorithm.is_eagle3.return_value = False
    spec_algorithm.is_dflash_family.return_value = False
    return dict(
        server_args=SimpleNamespace(
            speculative_draft_model_path=None,
            speculative_draft_model_revision=None,
        ),
        model_config=SimpleNamespace(
            num_nextn_predict_layers=1,
            hf_config=SimpleNamespace(eagle_config=None),
            is_hybrid_swa=False,
            is_deepseek_v4_arch=False,
            swa_attention_layer_ids=[],
        ),
        spec_algorithm=spec_algorithm,
    )


def test_native_mtp_uses_target_nextn_layer_count(monkeypatch):
    from_server_args = MagicMock()
    monkeypatch.setattr(
        spec_aux_hidden_state.ModelConfig, "from_server_args", from_server_args
    )
    config = spec_aux_hidden_state.resolve_spec_aux_hidden_state_config(
        **_resolve_kwargs(), is_draft_worker=False
    )
    assert config.eagle_draft_num_layers == 1
    from_server_args.assert_not_called()


def test_draft_worker_does_not_budget_itself():
    config = spec_aux_hidden_state.resolve_spec_aux_hidden_state_config(
        **_resolve_kwargs(), is_draft_worker=True
    )
    assert config.eagle_draft_num_layers is None


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
        spec_aux_hidden_state._map_muse_target_layer_ids(
            target_hf_config=SimpleNamespace(model_type=target_model_type),
            draft_hf_config=SimpleNamespace(architectures=[draft_architecture]),
            layer_ids=[1, 13, 25, 37, 49],
        )
        == expected
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
