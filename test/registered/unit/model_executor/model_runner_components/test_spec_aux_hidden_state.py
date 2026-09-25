import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.aux_hidden_states import (
    AuxHiddenStatePacker,
    pack_aux_hidden_states,
)
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    _map_muse_target_layer_ids,
)
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


def test_packer_writes_into_external_buffer_like_list_path():
    """A graph runner hands every graph size a row slice of one buffer; the
    packed output must equal the list path's concat and live in that storage,
    and a buffer that does not fit the captures must be rejected."""
    captures = [torch.randn(5, 4) for _ in range(3)]
    shared = torch.full((8, 12), float("nan"))
    packer = AuxHiddenStatePacker(3, out=shared[:5])
    for hidden in captures:
        packer.append(hidden)
    packed = packer.finalize()

    torch.testing.assert_close(packed, pack_aux_hidden_states(list(captures)))
    assert packed.data_ptr() == shared.data_ptr()
    assert shared[5:].isnan().all()

    with pytest.raises(ValueError):
        AuxHiddenStatePacker(3, out=shared[:4]).append(captures[0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
