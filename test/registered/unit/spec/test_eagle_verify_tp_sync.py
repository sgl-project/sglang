"""Regression test for synchronizing EAGLE verify results across TP ranks."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.speculative.eagle_utils as eagle_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _FakeForwardMode:
    def is_idle(self):
        return False


class _FakeTpGroup:
    world_size = 2

    def __init__(self):
        self.broadcasts = []

    def broadcast(self, tensor, src):
        self.broadcasts.append((tensor, src))


class TestEagleVerifyTpSync(unittest.TestCase):
    def test_hip_greedy_verify_results_are_broadcast(self):
        tp_group = _FakeTpGroup()
        logits = torch.tensor(
            [
                [1.0, 3.0, 2.0],
                [5.0, 4.0, 1.0],
                [0.0, 2.0, 6.0],
            ],
            dtype=torch.float32,
        )
        batch = SimpleNamespace(
            device=torch.device("cpu"),
            forward_mode=_FakeForwardMode(),
            seq_lens=torch.tensor([3], dtype=torch.int32),
            sampling_info=SimpleNamespace(
                is_all_greedy=False,
                acc_additive_penalties=None,
                acc_scaling_penalties=None,
                logit_bias=None,
            ),
        )
        verify_input = SimpleNamespace(
            draft_token=torch.tensor([10, 11, 12], dtype=torch.int64),
            draft_token_num=3,
            max_tree_depth=3,
            retrieve_index=torch.zeros((1, 3), dtype=torch.int32),
            retrieve_next_token=torch.full((3,), -1, dtype=torch.int32),
            retrieve_next_sibling=torch.full((3,), -1, dtype=torch.int32),
            tree_topk=1,
            grammar=None,
        )
        logits_output = SimpleNamespace(next_token_logits=logits)

        def fake_verify_tree_greedy_func(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            target_predict,
            topk,
        ):
            predicts.copy_(target_predict.flatten().to(torch.int32))
            accept_index[0, 0] = 0
            accept_token_num.fill_(1)
            return predicts, accept_index, accept_token_num

        with (
            patch.object(eagle_utils, "_is_cuda", False),
            patch.object(eagle_utils, "_is_hip", True),
            patch.object(eagle_utils, "_is_musa", False),
            patch.object(eagle_utils, "_is_npu", False),
            patch.object(
                eagle_utils, "verify_tree_greedy_func", fake_verify_tree_greedy_func
            ),
            patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=False,
            ),
            patch("sglang.srt.distributed.get_tp_group", return_value=tp_group),
        ):
            predict, num_correct_tokens, accept_index = eagle_utils.eagle_sample(
                verify_input, batch, logits_output
            )

        self.assertEqual(num_correct_tokens.tolist(), [2])
        self.assertEqual(predict.tolist(), [1, 0, 2])
        self.assertEqual(accept_index.tolist(), [[0, -1, -1]])
        self.assertEqual([src for _, src in tp_group.broadcasts], [0, 0, 0])
        self.assertIs(tp_group.broadcasts[0][0], predict)
        self.assertIs(tp_group.broadcasts[1][0], accept_index)
        self.assertEqual(tp_group.broadcasts[2][0].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
