from types import SimpleNamespace

import pytest

from sglang.srt.speculative.dlite_utils import (
    build_dlite_target_layer_ids,
    parse_dlite_draft_config,
    validate_dlite_target_compatibility,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _draft_config(**overrides):
    values = {
        "architectures": ["DLiteDraftModel"],
        "block_size": 8,
        "dlite_config": {
            "architecture_version": "dlite_v1",
            "chs_num_layers": 4,
            "mask_token_id": 99,
            "model_role": "student",
            "sequential_head": "rnn",
            "sequential_rank": 32,
            "target_layer_ids": [0, 1, 6, 7],
        },
        "hidden_size": 64,
        "num_target_layers": 8,
        "vocab_size": 128,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _target_config(**overrides):
    values = {
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 64,
        "model_type": "qwen3",
        "num_hidden_layers": 8,
        "vocab_size": 128,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_parse_released_dlite_v1_contract():
    parsed = parse_dlite_draft_config(_draft_config())

    assert parsed.block_size == 8
    assert parsed.proposal_length == 7
    assert parsed.target_layer_ids == (0, 1, 6, 7)
    assert parsed.sequential_rank == 32
    assert parsed.query_position_offset == 0


def test_public_qwen3_checkpoints_use_fixed_chs_layers():
    assert build_dlite_target_layer_ids(36, 14) == (
        0,
        1,
        4,
        7,
        10,
        13,
        16,
        19,
        22,
        25,
        28,
        31,
        34,
        35,
    )
    assert build_dlite_target_layer_ids(36, 12) == (
        0,
        1,
        5,
        8,
        12,
        16,
        19,
        23,
        27,
        30,
        34,
        35,
    )


def test_parse_released_dlite_v2_contract():
    draft = _draft_config()
    draft.dlite_config = {
        **draft.dlite_config,
        "architecture_version": "dlite_v2",
        "model_role": "pivot_q_student",
    }

    parsed = parse_dlite_draft_config(draft)

    assert parsed.architecture_version == "dlite_v2"
    assert parsed.model_role == "pivot_q_student"
    assert parsed.query_position_offset == 1


def test_validate_target_contract():
    parsed = validate_dlite_target_compatibility(
        target_hf_config=_target_config(),
        draft_hf_config=_draft_config(),
    )
    assert parsed.mask_token_id == 99


@pytest.mark.parametrize(
    ("target_overrides", "message"),
    [
        ({"num_hidden_layers": 9}, "num_hidden_layers mismatch"),
        ({"hidden_size": 80}, "hidden_size mismatch"),
        ({"vocab_size": 96}, "vocab_size mismatch"),
        (
            {"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
            "Qwen3ForCausalLM",
        ),
    ],
)
def test_rejects_incompatible_target(target_overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_dlite_target_compatibility(
            target_hf_config=_target_config(**target_overrides),
            draft_hf_config=_draft_config(),
        )


def test_rejects_unsupported_architecture_version():
    draft = _draft_config()
    draft.dlite_config = {**draft.dlite_config, "architecture_version": "dlite_v3"}
    with pytest.raises(ValueError, match="architecture_version"):
        parse_dlite_draft_config(draft)


def test_rejects_unsorted_target_layers():
    draft = _draft_config()
    draft.dlite_config = {
        **draft.dlite_config,
        "target_layer_ids": [0, 6, 3, 7],
    }
    with pytest.raises(ValueError, match="sorted and unique"):
        parse_dlite_draft_config(draft)


def test_rejects_non_checkpoint_target_layer_selection():
    draft = _draft_config()
    draft.dlite_config = {
        **draft.dlite_config,
        "target_layer_ids": [0, 2, 6, 7],
    }
    with pytest.raises(ValueError, match="fixed first/last"):
        parse_dlite_draft_config(draft)
