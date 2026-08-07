from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from types import SimpleNamespace

import torch

from sglang.srt.dllm.algorithm.joint_threshold_indel import JointThresholdInDel
from sglang.srt.dllm.algorithm.joint_threshold_indel_graph import (
    graph_step,
    init_graph_state,
)

BLOCK_SIZE = 8
MASK_ID = 97
DELETE_ID = 98
SPLIT_ID = 99
VOCAB_SIZE = 100


def _config():
    return SimpleNamespace(
        block_size=BLOCK_SIZE,
        mask_id=MASK_ID,
        delete_token_id=DELETE_ID,
        split_token_id=SPLIT_ID,
        first_done_first_out_mode=False,
        algorithm_config={
            "threshold": 0.5,
            "edit_threshold": 0,
            "max_post_edit_steps": 8,
            "min_mask_transfer": 1,
        },
    )


def _logits(predictions, second_predictions=None):
    logits = torch.full((BLOCK_SIZE, VOCAB_SIZE), -14.0)
    rows = torch.arange(BLOCK_SIZE)
    logits[rows, predictions] = 8.0 + torch.linspace(0.0, 2.0, BLOCK_SIZE)
    if second_predictions is not None:
        logits[rows, second_predictions] = 7.0
    return logits


def _graph_step(input_ids, logits, state):
    return graph_step(
        input_ids,
        logits.unsqueeze(0),
        state,
        mask_id=MASK_ID,
        delete_token_id=DELETE_ID,
        split_token_id=SPLIT_ID,
        threshold=0.5,
        edit_threshold=0,
        post_edit_threshold=0,
        max_post_edit_steps=8,
    )


def test_graph_step_matches_vectorized_mask_and_edit_path():
    algorithm = JointThresholdInDel(_config())
    initial = torch.full((BLOCK_SIZE,), MASK_ID, dtype=torch.long)
    initial[:2] = torch.tensor([11, 12])
    baseline_batch = SimpleNamespace(batch_size=1, input_ids=initial.clone())
    baseline_state = algorithm.init_step_state(baseline_batch)
    graph_ids = initial.view(1, BLOCK_SIZE).clone()
    graph_state = init_graph_state(
        graph_ids,
        mask_id=MASK_ID,
        vocab_size=VOCAB_SIZE,
        max_steps=17,
    )

    for step in range(4):
        predictions = (torch.arange(BLOCK_SIZE) + 20 + step * BLOCK_SIZE) % MASK_ID
        if step == 2:
            predictions[3] = DELETE_ID
            predictions[4] = SPLIT_ID
        logits = _logits(predictions)
        baseline_done = algorithm.step_vectorized(
            baseline_batch, logits.clone(), baseline_state
        )
        graph_done = _graph_step(graph_ids, logits, graph_state)

        assert baseline_done[0] == bool(graph_done[0])
        assert torch.equal(baseline_batch.input_ids, graph_ids.view(-1))
        assert torch.equal(
            baseline_state[0]["is_orig_mask"],
            graph_state["is_orig_mask"][0],
        )


def test_graph_step_matches_vectorized_cycle_guard():
    algorithm = JointThresholdInDel(_config())
    initial = torch.arange(10, 10 + BLOCK_SIZE)
    baseline_batch = SimpleNamespace(batch_size=1, input_ids=initial.clone())
    baseline_state = algorithm.init_step_state(baseline_batch)
    baseline_state[0]["prompt_mask"].zero_()
    baseline_state[0]["is_orig_mask"].zero_()

    graph_ids = initial.view(1, BLOCK_SIZE).clone()
    graph_state = init_graph_state(
        graph_ids,
        mask_id=MASK_ID,
        vocab_size=VOCAB_SIZE,
        max_steps=17,
    )
    graph_state["prompt_mask"].zero_()
    graph_state["is_orig_mask"].zero_()

    alternate = torch.arange(30, 30 + BLOCK_SIZE)
    third = torch.arange(50, 50 + BLOCK_SIZE)
    for prediction in (alternate, initial, alternate):
        logits = _logits(prediction, third)
        baseline_done = algorithm.step_vectorized(
            baseline_batch, logits.clone(), baseline_state
        )
        graph_done = _graph_step(graph_ids, logits, graph_state)

        assert baseline_done[0] == bool(graph_done[0])
        assert torch.equal(baseline_batch.input_ids, graph_ids.view(-1))
