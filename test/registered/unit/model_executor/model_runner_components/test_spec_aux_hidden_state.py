import sys
from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.model_runner_components import spec_aux_hidden_state
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    SpecAuxHiddenStateConfig,
    _map_muse_target_layer_ids,
    _resolve_eagle_aux_hidden_state,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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


def test_an_mtp_depth_named_mtp_num_hidden_layers_counts_as_the_eagle_draft(monkeypatch):
    """Qwen3.5 names its MTP depth mtp_num_hidden_layers. Read as no draft, the target's
    KV budget leaves the draft's pool, of the target's token count, outside
    mem_fraction_static."""
    monkeypatch.setattr(
        spec_aux_hidden_state,
        "get_spec",
        lambda: SimpleNamespace(speculative_draft_model_path=None),
    )
    config = SpecAuxHiddenStateConfig()

    _resolve_eagle_aux_hidden_state(
        config=config,
        server_args=None,
        model_config=SimpleNamespace(
            num_nextn_predict_layers=None,
            hf_text_config=SimpleNamespace(mtp_num_hidden_layers=1),
            is_hybrid_swa=False,
            is_deepseek_v4_arch=False,
        ),
        spec_algorithm=SpeculativeAlgorithm.EAGLE,
        is_draft_worker=False,
    )

    assert config.eagle_draft_num_layers == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
