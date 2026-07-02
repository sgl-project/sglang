import unittest

import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner.eager_runner import (
    _get_cp_v2_input_embeds,
    _prepare_cp_v2_logits_inputs,
    _trim_cp_v2_padding,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestEagerRunnerCPV2(CustomTestCase):
    def test_get_cp_v2_input_embeds_prefers_model_helper(self):
        input_ids = torch.tensor([1, 2])

        class Model:
            def get_input_embedding(self, ids):
                return ids + 10

            def get_input_embeddings(self):
                raise AssertionError("raw embedding module should not be used")

        self.assertTrue(
            torch.equal(_get_cp_v2_input_embeds(Model(), input_ids), input_ids + 10)
        )

    def test_prepare_cp_v2_logits_inputs_gathers_mimo_hidden_states_before_norm(self):
        hidden_states = torch.tensor([[1.0], [2.0]])
        hidden_states_before_norm = torch.tensor([[3.0], [4.0]])
        gathered = []

        def gather(x):
            gathered.append(x)
            return x + 10

        result = _prepare_cp_v2_logits_inputs(
            (hidden_states, hidden_states_before_norm),
            capture_aux_hidden_states=False,
            gather_fn=gather,
        )

        self.assertEqual(len(gathered), 2)
        self.assertIs(gathered[0], hidden_states)
        self.assertIs(gathered[1], hidden_states_before_norm)
        self.assertTrue(torch.equal(result.hidden_states, hidden_states + 10))
        self.assertIsNone(result.aux_hidden_states)
        self.assertTrue(
            torch.equal(
                result.hidden_states_before_norm,
                hidden_states_before_norm + 10,
            )
        )

    def test_prepare_cp_v2_logits_inputs_preserves_aux_hidden_states(self):
        hidden_states = torch.tensor([[1.0], [2.0]])
        aux_hidden_states = torch.tensor([[5.0], [6.0]])

        result = _prepare_cp_v2_logits_inputs(
            (hidden_states, aux_hidden_states),
            capture_aux_hidden_states=True,
            gather_fn=lambda x: x + 10,
        )

        self.assertTrue(torch.equal(result.hidden_states, hidden_states + 10))
        self.assertTrue(torch.equal(result.aux_hidden_states, aux_hidden_states))
        self.assertIsNone(result.hidden_states_before_norm)

    def test_trim_cp_v2_padding_uses_real_token_count(self):
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=torch.arange(8),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([5], dtype=torch.int32),
            out_cache_loc=torch.arange(100, 108),
            seq_lens_sum=5,
            positions=torch.arange(8),
            input_embeds=torch.arange(16, dtype=torch.float32).view(8, 2),
            extend_num_tokens=8,
            extend_seq_lens=torch.tensor([5], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([0], dtype=torch.int32),
            extend_seq_lens_cpu=[5],
            extend_prefix_lens_cpu=[0],
            num_token_non_padded=torch.tensor(5, dtype=torch.int32),
            num_token_non_padded_cpu=5,
            original_global_num_tokens_cpu=[5],
            global_num_tokens_cpu=[8],
            global_num_tokens_gpu=torch.tensor([8], dtype=torch.int64),
        )

        trimmed = _trim_cp_v2_padding(forward_batch)

        self.assertIsNot(trimmed, forward_batch)
        self.assertTrue(torch.equal(trimmed.input_ids, torch.arange(5)))
        self.assertTrue(torch.equal(trimmed.positions, torch.arange(5)))
        self.assertTrue(torch.equal(trimmed.out_cache_loc, torch.arange(100, 105)))
        self.assertTrue(
            torch.equal(
                trimmed.input_embeds,
                torch.arange(10, dtype=torch.float32).view(5, 2),
            )
        )
        self.assertEqual(trimmed.extend_num_tokens, 5)
        self.assertEqual(trimmed.global_num_tokens_cpu, [5])
        self.assertTrue(torch.equal(trimmed.global_num_tokens_gpu, torch.tensor([5])))
        self.assertEqual(trimmed.num_token_non_padded_cpu, 5)
        self.assertEqual(trimmed.num_token_non_padded.item(), 5)


if __name__ == "__main__":
    unittest.main()
