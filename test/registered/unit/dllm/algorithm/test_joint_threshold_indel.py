"""Unit tests for JointThresholdInDel without a server or model weights."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
LLADA2_MASK_ID = 156895
LLADA2_VOCAB_SIZE = 157184


def make_config(
    block_size=6,
    *,
    threshold=0.5,
    edit_threshold=0,
    max_post_edit_steps=2,
    fdfo=False,
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
        first_done_first_out_mode=fdfo,
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


def make_model_output(logits):
    return SimpleNamespace(
        logits_output=SimpleNamespace(full_logits=logits),
        can_run_graph=False,
    )


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

    def test_invalid_max_post_edit_steps_is_rejected(self):
        for value in (-34, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "max_post_edit_steps must be a non-negative integer"
                ):
                    JointThresholdInDel(
                        make_config(block_size=32, max_post_edit_steps=value)
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
    def test_checkpoint_metadata_enables_indel(self, mock_model_config):
        mock_model_config.return_value.vocab_size = LLADA2_VOCAB_SIZE
        mock_model_config.return_value.hf_config = SimpleNamespace(
            architectures=["LLaDA2MoeModelLM"],
            delete_token_id=9,
            split_token_id=10,
        )
        server_args = SimpleNamespace(
            dllm_algorithm="JointThresholdInDel",
            model_path="local/indel-checkpoint",
            revision=None,
            max_running_requests=None,
            dllm_algorithm_config=None,
            dllm_fdfo=True,
        )

        config = DllmConfig.from_server_args(server_args)

        self.assertEqual(config.delete_token_id, 9)
        self.assertEqual(config.split_token_id, 10)
        self.assertEqual(config.max_running_requests, 1)
        self.assertTrue(config.first_done_first_out_mode)

    @patch("sglang.srt.dllm.config.ModelConfig.from_server_args")
    def test_invalid_checkpoint_edit_token_ids_are_rejected(self, mock_model_config):
        mock_model_config.return_value.vocab_size = LLADA2_VOCAB_SIZE
        server_args = SimpleNamespace(
            dllm_algorithm="JointThresholdInDel",
            model_path="local/indel-checkpoint",
            revision=None,
            max_running_requests=4,
            dllm_algorithm_config=None,
            dllm_fdfo=False,
        )
        test_cases = [
            ("boolean", True, 10, "delete_token_id must be an integer token ID"),
            ("non_integer", 1.5, 10, "delete_token_id must be an integer token ID"),
            ("negative", -1, 10, "delete_token_id must be within"),
            ("upper_bound", 9, LLADA2_VOCAB_SIZE, "split_token_id must be within"),
            ("same_edit_ids", 9, 9, "token IDs must be distinct"),
            ("delete_is_mask", LLADA2_MASK_ID, 10, "token IDs must be distinct"),
            ("split_is_mask", 9, LLADA2_MASK_ID, "token IDs must be distinct"),
        ]

        for case_name, delete_token_id, split_token_id, error_pattern in test_cases:
            with self.subTest(case=case_name):
                mock_model_config.return_value.hf_config = SimpleNamespace(
                    architectures=["LLaDA2MoeModelLM"],
                    delete_token_id=delete_token_id,
                    split_token_id=split_token_id,
                )

                with self.assertRaisesRegex(ValueError, error_pattern):
                    DllmConfig.from_server_args(server_args)

    @patch("sglang.srt.dllm.config.ModelConfig.from_server_args")
    def test_incomplete_checkpoint_edit_token_ids_are_rejected(self, mock_model_config):
        test_cases = [
            ("inclusionAI/LLaDA2.2-flash", None, None),
            ("inclusionAI/LLaDA2.0-mini", None, None),
            ("local/incomplete-indel-checkpoint", 9, None),
            ("local/incomplete-indel-checkpoint", None, 10),
        ]

        for model_path, delete_token_id, split_token_id in test_cases:
            with self.subTest(
                model_path=model_path,
                delete_token_id=delete_token_id,
                split_token_id=split_token_id,
            ):
                hf_config = {"architectures": ["LLaDA2MoeModelLM"]}
                if delete_token_id is not None:
                    hf_config["delete_token_id"] = delete_token_id
                if split_token_id is not None:
                    hf_config["split_token_id"] = split_token_id
                mock_model_config.return_value.hf_config = SimpleNamespace(**hf_config)
                server_args = SimpleNamespace(
                    dllm_algorithm="JointThresholdInDel",
                    model_path=model_path,
                    revision=None,
                    max_running_requests=4,
                    dllm_algorithm_config=None,
                    dllm_fdfo=False,
                )

                with self.assertRaises(RuntimeError) as context:
                    DllmConfig.from_server_args(server_args)

                message = str(context.exception)
                self.assertIn(model_path, message)
                self.assertIn("delete_token_id and split_token_id", message)
                self.assertIn("--json-model-override-args", message)

    @patch("sglang.srt.dllm.config.ModelConfig.from_server_args")
    def test_non_indel_algorithm_does_not_require_edit_token_ids(
        self, mock_model_config
    ):
        mock_model_config.return_value.hf_config = SimpleNamespace(
            architectures=["LLaDA2MoeModelLM"]
        )
        server_args = SimpleNamespace(
            dllm_algorithm="JointThreshold",
            model_path="inclusionAI/LLaDA2.0-mini",
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
    def test_budget_one_final_cleanup_scrubs_all_reserved_tokens(self):
        algorithm = JointThresholdInDel(
            make_config(block_size=4, max_post_edit_steps=1)
        )
        batch = make_batch([[3, MASK, MASK, 4]])
        state = algorithm.init_step_state(batch)[0]
        state["is_orig_mask"] = torch.zeros(4, dtype=torch.bool)
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

    def test_hard_limit_final_cleanup_resolves_all_mask_tokens(self):
        algorithm = JointThresholdInDel(
            make_config(block_size=3, max_post_edit_steps=2)
        )
        batch = make_batch([[3, MASK, MASK]])
        state = algorithm.init_step_state(batch)[0]
        state["num_update_steps"] = algorithm.max_regular_update_steps
        logits = torch.zeros((3, VOCAB_SIZE))
        logits[0, 3] = 1.0
        logits[1, MASK] = 0.3
        logits[1, SPLIT] = 0.2
        logits[1, 5] = 0.1
        logits[2, MASK] = 0.4
        logits[2, DELETE] = 0.3
        logits[2, 6] = 0.2

        mask_confidence = logits[1:].softmax(dim=-1).max(dim=-1).values
        self.assertTrue((mask_confidence < algorithm.threshold).all())

        done = algorithm.step(batch, logits, [state])

        self.assertEqual(
            state["num_update_steps"], algorithm.max_regular_update_steps + 1
        )
        self.assertEqual(done, [False])
        self.assertEqual(batch.input_ids.tolist(), [3, 5, 6])
        self.assertNotIn(MASK, batch.input_ids.tolist())
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


class TestJointThresholdInDelFdfo(CustomTestCase):
    def test_fdfo_budget_zero_cleans_up_while_resolving_last_original_mask(self):
        algorithm = JointThresholdInDel(
            make_config(block_size=3, max_post_edit_steps=0, fdfo=True)
        )
        model_runner = Mock()
        model_runner.forward.side_effect = [
            make_model_output(
                make_logits(
                    [
                        [3],
                        [4],
                        [5],
                        [3],
                        [SPLIT, 6],
                        [DELETE, 7],
                    ]
                )
            ),
            make_model_output(make_logits([[3], [6], [7]])),
        ]

        batch = make_batch([[3, 4, 5], [3, MASK, 4]])
        _, next_token_ids, accept_lengths, algo_states, _ = algorithm.run(
            model_runner, batch
        )

        self.assertEqual(accept_lengths, [3, 0])
        self.assertIsNone(algo_states[0])
        self.assertEqual(next_token_ids[1], [3, 6, 7])
        surviving_state = algo_states[1]
        self.assertEqual(surviving_state["num_update_steps"], 1)
        self.assertEqual(surviving_state["post_edit_steps"], 0)
        self.assertTrue(surviving_state["finished"])

        batch = make_batch([next_token_ids[1]])
        _, next_token_ids, accept_lengths, algo_states, _ = algorithm.run(
            model_runner, batch, [surviving_state]
        )

        self.assertEqual(accept_lengths, [3])
        self.assertEqual(next_token_ids, [[3, 6, 7]])
        self.assertEqual(algo_states, [None])
        self.assertEqual(model_runner.forward.call_count, 2)

    def test_fdfo_budget_one_uses_first_post_edit_round_for_cleanup(self):
        algorithm = JointThresholdInDel(
            make_config(block_size=3, max_post_edit_steps=1, fdfo=True)
        )
        model_runner = Mock()
        model_runner.forward.side_effect = [
            make_model_output(make_logits([[3], [6], [4]])),
            make_model_output(make_logits([[3], [DELETE, 7], [SPLIT, 8]])),
            make_model_output(make_logits([[3], [7], [8]])),
        ]

        batch = make_batch([[3, MASK, 4]])
        _, next_token_ids, accept_lengths, algo_states, _ = algorithm.run(
            model_runner, batch
        )

        self.assertEqual(accept_lengths, [0])
        self.assertEqual(next_token_ids, [[3, 6, 4]])
        state = algo_states[0]
        self.assertEqual(state["post_edit_steps"], 0)
        self.assertFalse(state["finished"])

        batch = make_batch(next_token_ids)
        _, next_token_ids, accept_lengths, algo_states, _ = algorithm.run(
            model_runner, batch, algo_states
        )

        self.assertEqual(accept_lengths, [0])
        self.assertEqual(next_token_ids, [[3, 7, 8]])
        self.assertIs(algo_states[0], state)
        self.assertEqual(state["post_edit_steps"], 1)
        self.assertTrue(state["finished"])

        batch = make_batch(next_token_ids)
        _, next_token_ids, accept_lengths, algo_states, _ = algorithm.run(
            model_runner, batch, algo_states
        )

        self.assertEqual(accept_lengths, [3])
        self.assertEqual(next_token_ids, [[3, 7, 8]])
        self.assertEqual(algo_states, [None])
        self.assertEqual(model_runner.forward.call_count, 3)


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


if __name__ == "__main__":
    unittest.main()
