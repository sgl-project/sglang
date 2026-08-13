from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.dllm.algorithm.joint_threshold_indel import (
    JointThresholdInDel,
)
from sglang.srt.dllm.algorithm.joint_threshold_indel import (
    _graph_step as run_graph_step,
)
from sglang.srt.dllm.algorithm.joint_threshold_indel import (
    _init_graph_state,
)
from sglang.srt.dllm.config import DllmConfig

BLOCK_SIZE = 8
MASK_ID = 97
DELETE_ID = 98
SPLIT_ID = 99
VOCAB_SIZE = 100


def _config(max_post_edit_steps=8):
    return SimpleNamespace(
        block_size=BLOCK_SIZE,
        mask_id=MASK_ID,
        delete_token_id=DELETE_ID,
        split_token_id=SPLIT_ID,
        first_done_first_out_mode=False,
        algorithm_config={
            "threshold": 0.5,
            "edit_threshold": 0,
            "max_post_edit_steps": max_post_edit_steps,
        },
    )


def _logits(predictions, second_predictions=None):
    logits = torch.full((BLOCK_SIZE, VOCAB_SIZE), -14.0)
    rows = torch.arange(BLOCK_SIZE)
    logits[rows, predictions] = 8.0 + torch.linspace(0.0, 2.0, BLOCK_SIZE)
    if second_predictions is not None:
        logits[rows, second_predictions] = 7.0
    return logits


def _graph_step(input_ids, logits, state, max_post_edit_steps=8):
    return run_graph_step(
        input_ids,
        logits.unsqueeze(0),
        state,
        mask_id=MASK_ID,
        delete_token_id=DELETE_ID,
        split_token_id=SPLIT_ID,
        threshold=0.5,
        edit_threshold=0,
        max_post_edit_steps=max_post_edit_steps,
        max_regular_update_steps=BLOCK_SIZE + max_post_edit_steps,
    )


def _paired_states(algorithm, initial):
    batch = SimpleNamespace(batch_size=1, input_ids=initial.clone())
    graph_ids = initial.view(1, BLOCK_SIZE).clone()
    return SimpleNamespace(
        batch=batch,
        state=algorithm.init_step_state(batch),
        graph_ids=graph_ids,
        graph_state=_init_graph_state(
            graph_ids,
            mask_id=MASK_ID,
            vocab_size=VOCAB_SIZE,
            max_steps=algorithm.max_steps(BLOCK_SIZE),
        ),
    )


def _step_both(algorithm, paired, logits, max_post_edit_steps=8):
    eager_done = algorithm.step_vectorized(paired.batch, logits.clone(), paired.state)
    graph_done = _graph_step(
        paired.graph_ids,
        logits,
        paired.graph_state,
        max_post_edit_steps=max_post_edit_steps,
    )
    assert eager_done[0] == bool(graph_done[0])
    assert torch.equal(paired.batch.input_ids, paired.graph_ids.view(-1))
    return eager_done, graph_done


def test_graph_step_matches_vectorized_mask_and_edit_path():
    algorithm = JointThresholdInDel(_config())
    initial = torch.full((BLOCK_SIZE,), MASK_ID, dtype=torch.long)
    initial[:2] = torch.tensor([11, 12])
    paired = _paired_states(algorithm, initial)

    for step in range(4):
        predictions = (torch.arange(BLOCK_SIZE) + 20 + step * BLOCK_SIZE) % MASK_ID
        if step == 2:
            predictions[3] = DELETE_ID
            predictions[4] = SPLIT_ID
        logits = _logits(predictions)
        _step_both(algorithm, paired, logits)
        assert torch.equal(
            paired.state[0]["is_orig_mask"],
            paired.graph_state["is_orig_mask"][0],
        )


def test_graph_step_matches_vectorized_cycle_guard():
    algorithm = JointThresholdInDel(_config())
    initial = torch.arange(10, 10 + BLOCK_SIZE)
    paired = _paired_states(algorithm, initial)
    paired.state[0]["prompt_mask"].zero_()
    paired.state[0]["is_orig_mask"].zero_()
    paired.graph_state["prompt_mask"].zero_()
    paired.graph_state["is_orig_mask"].zero_()

    alternate = torch.arange(30, 30 + BLOCK_SIZE)
    third = torch.arange(50, 50 + BLOCK_SIZE)
    for prediction in (alternate, initial, alternate):
        _step_both(algorithm, paired, _logits(prediction, third))


def test_force_finish_scrubs_mask_and_waits_for_kv_persist():
    algorithm = JointThresholdInDel(_config())
    initial = torch.arange(10, 10 + BLOCK_SIZE)
    paired = _paired_states(algorithm, initial)
    paired.state[0]["prompt_mask"].zero_()
    paired.state[0]["is_orig_mask"].zero_()
    paired.state[0]["post_edit_steps"] = 7
    paired.graph_state["prompt_mask"].zero_()
    paired.graph_state["is_orig_mask"].zero_()
    paired.graph_state["post_edit_steps"].fill_(7)

    predictions = torch.full((BLOCK_SIZE,), MASK_ID)
    alternatives = torch.arange(30, 30 + BLOCK_SIZE)
    logits = _logits(predictions, alternatives)

    eager_done, graph_done = _step_both(algorithm, paired, logits)

    assert eager_done == [False]
    assert graph_done.tolist() == [False]
    assert MASK_ID not in paired.graph_ids

    eager_done, graph_done = _step_both(algorithm, paired, logits)

    assert eager_done == [True]
    assert graph_done.tolist() == [True]


def test_original_mask_prediction_uses_non_mask_fallback():
    algorithm = JointThresholdInDel(_config())
    initial = torch.tensor([11, MASK_ID, 13, 14, 15, 16, 17, 18])
    batch = SimpleNamespace(batch_size=1, input_ids=initial.clone())
    state = algorithm.init_step_state(batch)
    predictions = initial.clone()
    predictions[1] = MASK_ID
    alternatives = predictions.clone()
    alternatives[1] = 23

    done = algorithm.step_vectorized(batch, _logits(predictions, alternatives), state)

    assert done == [False]
    assert batch.input_ids[1].item() == 23


def test_prompt_is_only_the_contiguous_prefix():
    algorithm = JointThresholdInDel(_config())
    initial = torch.tensor([11, MASK_ID, 13, MASK_ID, 15, 16, 17, 18])
    batch = SimpleNamespace(batch_size=1, input_ids=initial)

    state = algorithm.init_step_state(batch)[0]

    assert state["prompt_mask"].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_max_steps_reserves_cleanup_and_persist_rounds():
    algorithm = JointThresholdInDel(_config())

    assert algorithm.max_steps(BLOCK_SIZE) == BLOCK_SIZE + 8 + 2


def test_zero_post_edit_budget_cleans_and_persists():
    algorithm = JointThresholdInDel(_config(max_post_edit_steps=0))
    initial = torch.full((BLOCK_SIZE,), MASK_ID, dtype=torch.long)
    initial[0] = 11
    paired = _paired_states(algorithm, initial)
    predictions = torch.full((BLOCK_SIZE,), MASK_ID)
    alternatives = torch.arange(20, 20 + BLOCK_SIZE)
    logits = _logits(predictions, alternatives)

    for _ in range(algorithm.max_steps(BLOCK_SIZE)):
        eager_done, graph_done = _step_both(
            algorithm, paired, logits, max_post_edit_steps=0
        )
        if MASK_ID not in paired.graph_ids:
            break

    assert eager_done == [False]
    assert graph_done.tolist() == [False]
    assert MASK_ID not in paired.graph_ids

    eager_done, graph_done = _step_both(
        algorithm, paired, logits, max_post_edit_steps=0
    )

    assert eager_done == [True]
    assert graph_done.tolist() == [True]


def test_hard_regular_budget_forces_cleanup():
    algorithm = JointThresholdInDel(_config())
    initial = torch.full((BLOCK_SIZE,), MASK_ID, dtype=torch.long)
    initial[0] = 11
    paired = _paired_states(algorithm, initial)
    paired.state[0]["num_update_steps"] = algorithm.max_regular_update_steps
    paired.graph_state["num_update_steps"].fill_(algorithm.max_regular_update_steps)
    predictions = torch.full((BLOCK_SIZE,), MASK_ID)
    alternatives = torch.arange(20, 20 + BLOCK_SIZE)
    logits = _logits(predictions, alternatives)

    eager_done, graph_done = _step_both(algorithm, paired, logits)

    assert eager_done == [False]
    assert graph_done.tolist() == [False]
    assert MASK_ID not in paired.graph_ids


@patch("sglang.srt.dllm.config.ModelConfig.from_server_args")
def test_indel_token_ids_come_from_checkpoint_config(mock_model_config):
    mock_model_config.return_value.hf_config = SimpleNamespace(
        architectures=["LLaDA2MoeModelLM"],
        delete_token_id=DELETE_ID,
        split_token_id=SPLIT_ID,
    )

    def server_args(model_path):
        return SimpleNamespace(
            dllm_algorithm="JointThresholdInDel",
            model_path=model_path,
            revision=None,
            max_running_requests=None,
            dllm_algorithm_config=None,
            dllm_fdfo=False,
        )

    config = DllmConfig.from_server_args(server_args("block-routing-model"))
    assert config.delete_token_id == DELETE_ID
    assert config.split_token_id == SPLIT_ID

    mock_model_config.return_value.hf_config.delete_token_id = None
    with pytest.raises(RuntimeError, match="delete_token_id and split_token_id"):
        DllmConfig.from_server_args(server_args("block-routing-model"))
