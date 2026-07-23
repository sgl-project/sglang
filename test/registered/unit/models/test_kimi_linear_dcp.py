import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.kimi_k3_hook import (
    apply_kimi_k3_spec_backend_defaults,
)
from sglang.srt.models.kimi_linear import (
    KimiLinearForCausalLM,
    _get_kda_local_num_heads,
    _materialize_residual_stream,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestKimiLinearDCP(CustomTestCase):
    def test_kda_heads_use_global_tp(self):
        self.assertEqual(_get_kda_local_num_heads(64, 4), 16)
        with self.assertRaisesRegex(ValueError, "divisible"):
            _get_kda_local_num_heads(63, 4)

    @patch("sglang.srt.models.kimi_linear.prepare_decode_context_parallel_metadata")
    def test_planner_hook_delegates_all_arguments(self, planner):
        planner.return_value = object()
        kwargs = dict(
            seq_lens=torch.tensor([8]),
            extend_prefix_lens=torch.tensor([4]),
            extend_prefix_lens_cpu=torch.tensor([4]),
            extend_seq_lens=torch.tensor([4]),
            req_pool_indices=torch.tensor([0]),
            req_to_token=torch.arange(16).view(1, 16),
            seq_lens_sum=8,
            kv_buffer_shape=torch.Size([16, 1, 576]),
            kv_cache_dtype=torch.bfloat16,
            kv_cache_device=torch.device("cpu"),
            create_chunked_prefix_cache_kv_indices_fn=object(),
        )
        model = object.__new__(KimiLinearForCausalLM)
        got = model.prepare_context_parallel_metadata_for_dcp(**kwargs)
        self.assertIs(got, planner.return_value)
        planner.assert_called_once_with(**kwargs)


class TestKimiLinearDSpark(CustomTestCase):
    def test_capture_materializes_post_layer_residual_stream(self):
        hidden_states = torch.tensor([[1.0, 2.0]])
        residual = torch.tensor([[3.0, 5.0]])
        torch.testing.assert_close(
            _materialize_residual_stream(hidden_states, residual),
            torch.tensor([[4.0, 7.0]]),
        )
        self.assertIs(_materialize_residual_stream(hidden_states, None), hidden_states)

    def test_set_dspark_layers_to_capture(self):
        model = object.__new__(KimiLinearForCausalLM)
        model.pp_group = SimpleNamespace(world_size=1, is_last_rank=True)
        embed_tokens = object()
        model.model = SimpleNamespace(
            dspark_layers_to_capture=None, embed_tokens=embed_tokens
        )
        model.capture_aux_hidden_states = False

        model.set_dspark_layers_to_capture([2, 5, 8])

        self.assertTrue(model.capture_aux_hidden_states)
        self.assertEqual(model.model.dspark_layers_to_capture, [2, 5, 8])
        self.assertIs(model.get_input_embeddings(), embed_tokens)

    def test_dspark_capture_rejects_pipeline_parallelism(self):
        model = object.__new__(KimiLinearForCausalLM)
        model.pp_group = SimpleNamespace(world_size=2, is_last_rank=True)
        with self.assertRaisesRegex(NotImplementedError, "PP=1"):
            model.set_dspark_layers_to_capture([1])

    @patch("sglang.srt.utils.is_sm100_supported", return_value=True)
    def test_dspark_defaults_use_ragged_capable_kda_and_dense_draft(self, _):
        args = SimpleNamespace(
            speculative_algorithm="DSPARK",
            linear_attn_verify_backend=None,
            speculative_draft_attention_backend=None,
        )
        apply_kimi_k3_spec_backend_defaults(args)
        self.assertEqual(args.linear_attn_verify_backend, "triton")
        self.assertEqual(args.speculative_draft_attention_backend, "trtllm_mha")


if __name__ == "__main__":
    unittest.main()
