"""Unit tests for JointThresholdInDel without a server or model weights."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.dllm.algorithm import get_algorithm
from sglang.srt.dllm.algorithm.joint_threshold_indel import JointThresholdInDel
from sglang.srt.dllm.config import DllmConfig

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


MASK = 0
DELETE = 1
SPLIT = 2
VOCAB_SIZE = 12


def make_config(
    block_size=6,
    *,
    threshold=0.5,
    edit_threshold=0,
    max_post_edit_steps=2,
    delete_token_id=DELETE,
    split_token_id=SPLIT,
):
    return DllmConfig(
        algorithm="JointThresholdInDel",
        algorithm_config={
            "threshold": threshold,
            "edit_threshold": edit_threshold,
            "max_post_edit_steps": max_post_edit_steps,
        },
        block_size=block_size,
        mask_id=MASK,
        max_running_requests=8,
        delete_token_id=delete_token_id,
        split_token_id=split_token_id,
    )


def make_batch(blocks):
    input_ids = torch.tensor(blocks, dtype=torch.long)
    return SimpleNamespace(
        batch_size=input_ids.shape[0],
        input_ids=input_ids.flatten(),
    )


def make_logits(rows, vocab_size=VOCAB_SIZE):
    """Build logits from rows of token IDs ordered from most to least likely."""
    logits = torch.full((len(rows), vocab_size), -20.0)
    for row_index, candidates in enumerate(rows):
        for rank, token_id in enumerate(candidates):
            logits[row_index, token_id] = 10.0 - rank
    return logits


class TestJointThresholdInDelInitialization(CustomTestCase):
    def test_initial_state_and_step_limits(self):
        algorithm = JointThresholdInDel(
            make_config(block_size=6, max_post_edit_steps=3)
        )
        batch = make_batch([[3, 4, MASK, MASK, MASK, MASK]])

        state = algorithm.init_step_state(batch)[0]

        self.assertEqual(state["prompt_len"], 2)
        self.assertEqual(
            state["prompt_index"].tolist(), [True, True, False, False, False, False]
        )
        self.assertEqual(
            state["is_orig_mask"].tolist(),
            [False, False, True, True, True, True],
        )
        self.assertEqual(state["num_update_steps"], 0)
        self.assertEqual(algorithm.max_regular_update_steps, 9)
        self.assertEqual(algorithm.max_steps(6), 11)

    def test_prompt_is_the_contiguous_prefix_not_every_initial_non_mask(self):
        algorithm = JointThresholdInDel(make_config(block_size=5))
        batch = make_batch([[3, MASK, 4, MASK, MASK]])

        state = algorithm.init_step_state(batch)[0]

        self.assertEqual(state["prompt_len"], 1)
        self.assertEqual(
            state["prompt_index"].tolist(), [True, False, False, False, False]
        )

    def test_missing_edit_token_ids_preserves_constructor_error(self):
        config = make_config(delete_token_id=None, split_token_id=None)

        with self.assertRaisesRegex(
            RuntimeError, "requires delete_token_id and split_token_id"
        ):
            get_algorithm(config)

    def test_unknown_algorithm_has_specific_error(self):
        config = make_config()
        config.algorithm = "DoesNotExist"

        with self.assertRaisesRegex(
            RuntimeError, "Unknown diffusion LLM algorithm: DoesNotExist"
        ):
            get_algorithm(config)

    @patch("sglang.srt.dllm.config.ModelConfig.from_server_args")
    def test_llada_config_provides_edit_token_ids(self, mock_model_config):
        mock_model_config.return_value.hf_config.architectures = ["LLaDA2MoeModelLM"]
        server_args = SimpleNamespace(
            dllm_algorithm="JointThresholdInDel",
            model_path="model",
            revision=None,
            max_running_requests=None,
            dllm_algorithm_config=None,
            dllm_fdfo=True,
        )

        config = DllmConfig.from_server_args(server_args)

        self.assertEqual(config.delete_token_id, 156930)
        self.assertEqual(config.split_token_id, 156931)
        self.assertEqual(config.max_running_requests, 1)
        self.assertTrue(config.first_done_first_out_mode)

    @patch("sglang.srt.dllm.config.ModelConfig.from_server_args")
    def test_non_indel_model_has_no_edit_token_ids(self, mock_model_config):
        mock_model_config.return_value.hf_config.architectures = ["SDARForCausalLM"]
        server_args = SimpleNamespace(
            dllm_algorithm="JointThreshold",
            model_path="model",
            revision=None,
            max_running_requests=4,
            dllm_algorithm_config=None,
            dllm_fdfo=False,
        )

        config = DllmConfig.from_server_args(server_args)

        self.assertIsNone(config.delete_token_id)
        self.assertIsNone(config.split_token_id)
        self.assertEqual(config.max_running_requests, 4)


class TestJointThresholdInDelStructuralEdits(CustomTestCase):
    def setUp(self):
        self.algorithm = JointThresholdInDel(make_config(block_size=8))

    def test_mixed_edits_preserve_prompt_and_original_mask_flags(self):
        old_block_tokens = [DELETE, SPLIT, 3, 4, 5, 6, MASK, MASK]
        block_tokens = [DELETE, SPLIT, 3, DELETE, SPLIT, 6, MASK, MASK]
        is_orig_mask = [False, False, True, False, True, False, False, False]

        tokens, result_is_orig_mask = self.algorithm._apply_edit_operations(
            block_tokens, old_block_tokens, is_orig_mask, prompt_len=2
        )

        self.assertEqual(tokens, [DELETE, SPLIT, 3, MASK, 5, 6, MASK, MASK])
        self.assertEqual(
            result_is_orig_mask,
            [False, False, True, False, True, False, False, False],
        )

    def test_delete_pads_only_the_generated_suffix(self):
        algorithm = JointThresholdInDel(make_config(block_size=6))
        old_block_tokens = [DELETE, SPLIT, 3, 4, 5, 6]
        block_tokens = [DELETE, SPLIT, DELETE, DELETE, DELETE, DELETE]

        tokens, result_is_orig_mask = algorithm._apply_edit_operations(
            block_tokens, old_block_tokens, [False] * 6, prompt_len=2
        )

        self.assertEqual(tokens, [DELETE, SPLIT, MASK, MASK, MASK, MASK])
        self.assertEqual(result_is_orig_mask, [False] * 6)

    def test_split_truncates_only_the_generated_suffix(self):
        algorithm = JointThresholdInDel(make_config(block_size=6))
        old_block_tokens = [DELETE, SPLIT, 3, 4, 5, 6]
        block_tokens = [DELETE, SPLIT, SPLIT, SPLIT, SPLIT, SPLIT]
        initial_is_orig_mask = [False, False, True, False, True, False]

        tokens, result_is_orig_mask = algorithm._apply_edit_operations(
            block_tokens, old_block_tokens, initial_is_orig_mask, prompt_len=2
        )

        self.assertEqual(tokens, [DELETE, SPLIT, MASK, 3, MASK, 4])
        self.assertEqual(result_is_orig_mask, [False, False, False, True, False, False])

    def test_step_never_interprets_reserved_prompt_tokens_as_edits(self):
        algorithm = JointThresholdInDel(make_config(block_size=5))
        batch = make_batch([[DELETE, SPLIT, MASK, MASK, MASK]])
        state = algorithm.init_step_state(batch)[0]
        logits = make_logits(
            [
                [5, DELETE],
                [6, SPLIT],
                [DELETE, 7],
                [8],
                [9],
            ]
        )

        algorithm.step(batch, logits, [state])

        self.assertEqual(batch.input_ids[:2].tolist(), [DELETE, SPLIT])
        self.assertEqual(state["prompt_len"], 2)
        self.assertEqual(
            state["prompt_index"].tolist(), [True, True, False, False, False]
        )


class TestJointThresholdInDelSelection(CustomTestCase):
    def test_original_mask_to_mask_uses_non_mask_fallback(self):
        algorithm = JointThresholdInDel(make_config(block_size=3, threshold=1.1))
        batch = make_batch([[3, MASK, 4]])
        state = algorithm.init_step_state(batch)[0]
        logits = make_logits([[3], [MASK, 5], [4]])

        done = algorithm.step(batch, logits, [state])

        self.assertEqual(done, [False])
        self.assertEqual(batch.input_ids.tolist(), [3, 5, 4])

    def test_fallback_confidence_is_recomputed_for_threshold_selection(self):
        algorithm = JointThresholdInDel(make_config(block_size=4, threshold=0.6))
        batch = make_batch([[3, MASK, MASK, 4]])
        state = algorithm.init_step_state(batch)[0]
        logits = torch.zeros((4, VOCAB_SIZE))
        logits[0, 3] = 10
        logits[1, MASK] = 10
        logits[1, 5] = 0
        logits[2, 6] = 4
        logits[3, 4] = 10

        algorithm.step(batch, logits, [state])

        self.assertEqual(batch.input_ids.tolist(), [3, MASK, 6, 4])

    def test_introduced_mask_quota_selects_an_unresolved_original_mask(self):
        algorithm = JointThresholdInDel(make_config(block_size=5, threshold=1.1))
        batch = make_batch([[3, MASK, MASK, MASK, MASK]])
        state = algorithm.init_step_state(batch)[0]
        state["is_orig_mask"] = torch.tensor([False, True, True, False, False])
        logits = torch.zeros((5, VOCAB_SIZE))
        logits[0, 3] = 10
        for position, score in zip([3, 4, 1, 2], [9, 8, 7, 6]):
            logits[position, position + 4] = score

        algorithm.step(batch, logits, [state])

        self.assertEqual(batch.input_ids.tolist(), [3, 5, MASK, 7, 8])

    def test_split_can_preserve_one_unresolved_original_mask(self):
        algorithm = JointThresholdInDel(make_config(block_size=4, threshold=1.1))
        batch = make_batch([[3, MASK, 4, 4]])
        state = algorithm.init_step_state(batch)[0]
        logits = make_logits([[3], [SPLIT, 5], [4], [4]])
        unresolved_original_mask_count_before = (
            state["is_orig_mask"] & (batch.input_ids == MASK)
        ).sum()

        done = algorithm.step(batch, logits, [state])

        self.assertEqual(done, [False])
        self.assertEqual(batch.input_ids.tolist(), [3, MASK, MASK, 4])
        self.assertEqual(
            state["is_orig_mask"].tolist(),
            [False, False, True, False],
        )
        unresolved_original_mask_count_after = (
            state["is_orig_mask"] & (batch.input_ids == MASK)
        ).sum()
        self.assertEqual(unresolved_original_mask_count_before.item(), 1)
        self.assertEqual(unresolved_original_mask_count_after.item(), 1)
        self.assertEqual(state["num_update_steps"], 1)
        self.assertFalse(state["finished"])

    def test_ordinary_introduced_mask_to_mask_can_wait_for_final_cleanup(self):
        algorithm = JointThresholdInDel(
            make_config(block_size=4, max_post_edit_steps=2)
        )
        batch = make_batch([[3, MASK, MASK, MASK]])
        state = algorithm.init_step_state(batch)[0]
        state["is_orig_mask"] = torch.zeros(4, dtype=torch.bool)
        logits = make_logits([[3], [MASK, 5], [MASK, 6], [MASK, 7]])

        done = algorithm.step(batch, logits, [state])

        self.assertEqual(done, [False])
        self.assertEqual(batch.input_ids.tolist(), [3, MASK, MASK, MASK])
        self.assertEqual(state["post_edit_steps"], 1)

    def test_ordinary_round_fills_all_introduced_masks_with_token_predictions(self):
        algorithm = JointThresholdInDel(make_config(block_size=4))
        batch = make_batch([[3, MASK, MASK, MASK]])
        state = algorithm.init_step_state(batch)[0]
        state["is_orig_mask"] = torch.zeros(4, dtype=torch.bool)
        logits = make_logits([[3], [5], [6], [7]])

        algorithm.step(batch, logits, [state])

        self.assertEqual(batch.input_ids.tolist(), [3, 5, 6, 7])

    def test_t2t_respects_prompt_and_edit_threshold(self):
        algorithm = JointThresholdInDel(make_config(block_size=4, edit_threshold=0.9))
        batch = make_batch([[3, MASK, 4, MASK]])
        state = algorithm.init_step_state(batch)[0]
        logits = torch.zeros((4, VOCAB_SIZE))
        logits[0, 5] = 10
        logits[1, 7] = 10
        logits[2, 6] = 0.1
        logits[3, 8] = 10

        algorithm.step(batch, logits, [state])

        self.assertEqual(batch.input_ids.tolist(), [3, 7, 4, 8])

    def test_t2t_edits_confident_generated_token(self):
        algorithm = JointThresholdInDel(make_config(block_size=4, edit_threshold=0.9))
        batch = make_batch([[3, MASK, 4, MASK]])
        state = algorithm.init_step_state(batch)[0]
        logits = make_logits([[5], [7], [6], [8]])

        algorithm.step(batch, logits, [state])

        self.assertEqual(batch.input_ids.tolist(), [3, 7, 6, 8])


class TestJointThresholdInDelTermination(CustomTestCase):
    def test_post_edit_boundary_then_final_cleanup_scrubs_all_reserved_tokens(self):
        algorithm = JointThresholdInDel(
            make_config(block_size=4, max_post_edit_steps=1)
        )
        batch = make_batch([[3, MASK, MASK, 4]])
        state = algorithm.init_step_state(batch)[0]
        state["is_orig_mask"] = torch.zeros(4, dtype=torch.bool)
        ordinary_logits = make_logits([[3], [MASK, 5], [MASK, 6], [4]])

        done = algorithm.step(batch, ordinary_logits, [state])

        self.assertEqual(done, [False])
        self.assertEqual(state["post_edit_steps"], 1)
        self.assertFalse(state["finished"])
        self.assertEqual(batch.input_ids.tolist(), [3, MASK, MASK, 4])

        logits = make_logits(
            [
                [DELETE, 8],
                [MASK, 5],
                [DELETE, 6],
                [SPLIT, 7],
            ]
        )

        done = algorithm.step(batch, logits, [state])

        self.assertEqual(done, [False])
        self.assertEqual(batch.input_ids.tolist(), [3, 5, 6, 7])
        self.assertTrue(state["finished"])
        self.assertNotIn(MASK, batch.input_ids[1:].tolist())
        self.assertNotIn(DELETE, batch.input_ids[1:].tolist())
        self.assertNotIn(SPLIT, batch.input_ids[1:].tolist())

        confirmed_ids = batch.input_ids.clone()
        done = algorithm.step(batch, make_logits([[3], [5], [6], [7]]), [state])
        self.assertEqual(done, [True])
        self.assertTrue(torch.equal(batch.input_ids, confirmed_ids))

    def test_hard_limit_runs_one_separate_final_cleanup_update(self):
        algorithm = JointThresholdInDel(
            make_config(block_size=3, max_post_edit_steps=2)
        )
        batch = make_batch([[3, MASK, MASK]])
        state = algorithm.init_step_state(batch)[0]
        state["num_update_steps"] = algorithm.max_regular_update_steps
        logits = make_logits([[3], [MASK, SPLIT, 5], [MASK, DELETE, 6]])

        done = algorithm.step(batch, logits, [state])

        self.assertEqual(
            state["num_update_steps"], algorithm.max_regular_update_steps + 1
        )
        self.assertEqual(done, [False])
        self.assertEqual(batch.input_ids.tolist(), [3, 5, 6])
        self.assertTrue(state["finished"])

    def test_stable_terminal_step_finishes_immediately(self):
        algorithm = JointThresholdInDel(make_config(block_size=3))
        batch = make_batch([[3, 4, 5]])
        state = algorithm.init_step_state(batch)[0]
        logits = make_logits([[3], [4], [5]])

        done = algorithm.step(batch, logits, [state])

        self.assertEqual(done, [True])
        self.assertTrue(state["finished"])

    def test_batch_requests_finish_independently(self):
        algorithm = JointThresholdInDel(
            make_config(block_size=3, max_post_edit_steps=0)
        )
        batch = make_batch([[3, 4, 5], [3, MASK, 4]])
        states = algorithm.init_step_state(batch)
        states[1]["is_orig_mask"] = torch.zeros(3, dtype=torch.bool)
        logits = make_logits(
            [
                [3],
                [4],
                [5],
                [3],
                [MASK, 6],
                [4],
            ]
        )

        done = algorithm.step(batch, logits, states)

        self.assertEqual(done, [True, False])
        self.assertTrue(states[0]["finished"])
        self.assertTrue(states[1]["finished"])
        self.assertEqual(batch.input_ids.view(2, 3)[1].tolist(), [3, 6, 4])


class TestJointThresholdInDelAntiLoop(CustomTestCase):
    def test_repeated_original_mask_state_never_resamples_mask(self):
        algorithm = JointThresholdInDel(make_config(block_size=3, threshold=1.1))
        batch = make_batch([[3, MASK, 4]])
        state = algorithm.init_step_state(batch)[0]
        state["seen_input_keys"].add((3, MASK, 4))
        state["sampled_history"][1] = {5}
        logits = make_logits([[3], [5, MASK, 6], [4]])

        algorithm.step(batch, logits, [state])

        self.assertEqual(batch.input_ids.tolist(), [3, 6, 4])
        self.assertEqual(state["sampled_history"][1], {5, 6})

    def test_split_and_delete_move_original_mask_flags_in_step(self):
        algorithm = JointThresholdInDel(make_config(block_size=5))
        batch = make_batch([[3, MASK, MASK, MASK, MASK]])
        state = algorithm.init_step_state(batch)[0]
        logits = make_logits(
            [
                [3],
                [SPLIT, 5],
                [DELETE, 6],
                [7],
                [8],
            ]
        )

        algorithm.step(batch, logits, [state])

        self.assertEqual(batch.input_ids.tolist(), [3, MASK, MASK, 7, 8])
        self.assertEqual(
            state["is_orig_mask"].tolist(),
            [False, False, True, True, True],
        )


if __name__ == "__main__":
    unittest.main()
