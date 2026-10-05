import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.aux_hidden_states import (
    AuxHiddenStatePacker,
    pack_aux_hidden_states,
)
from sglang.srt.model_executor.model_runner_components.misc_utils import (
    resolve_aux_hidden_states_width,
)
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    _map_muse_target_layer_ids,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


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


def test_packer_writes_into_graph_buffer_like_list_path():
    """A graph runner hands every graph size a row slice of one buffer; the
    packed output must equal the list path's concat and live in that storage,
    and a buffer that does not fit the captures must be rejected."""
    captures = [torch.randn(5, 4) for _ in range(3)]
    shared = torch.full((8, 12), float("nan"))
    packer = AuxHiddenStatePacker.for_batch(
        SimpleNamespace(aux_hidden_states_buffer=shared[:5]), 3
    )
    for hidden in captures:
        packer.append(hidden)
    packed = packer.finalize()

    torch.testing.assert_close(packed, pack_aux_hidden_states(list(captures)))
    assert packed.data_ptr() == shared.data_ptr()
    assert shared[5:].isnan().all()

    too_small = SimpleNamespace(aux_hidden_states_buffer=shared[:4])
    with pytest.raises(ValueError):
        AuxHiddenStatePacker.for_batch(too_small, 3).append(captures[0])


class _SharedAuxModel:
    def get_aux_hidden_states_width(self) -> int:
        return 12


@pytest.mark.parametrize(
    ("model", "dflash", "is_draft_worker", "expected"),
    [
        (_SharedAuxModel(), True, False, 12),
        # Only the DFlash family is known to consume them within the step.
        (_SharedAuxModel(), False, False, 0),
        (_SharedAuxModel(), True, True, 0),
        (object(), True, False, 0),
    ],
)
def test_only_dflash_target_models_share_aux_output(
    model, dflash, is_draft_worker, expected
):
    assert (
        resolve_aux_hidden_states_width(
            model=model,
            spec_algorithm=SimpleNamespace(is_dflash_family=lambda: dflash),
            is_draft_worker=is_draft_worker,
        )
        == expected
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
