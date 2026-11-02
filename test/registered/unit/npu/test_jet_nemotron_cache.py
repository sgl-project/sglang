import copy
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci
from torch import nn
from torch.nn import functional as F

register_npu_ci(est_time=30, suite="stage-a-unit-test-npu")


class TestJetCacheResolution(unittest.TestCase):
    def test_npu_jet_selects_extra_buffer_without_disabling_overlap(self):
        from sglang.srt.arg_groups import overrides

        for architecture in (
            "JetNemotronForCausalLM",
            "JetVLMForConditionalGeneration",
        ):
            view = SimpleNamespace(
                _model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(architectures=[architecture])
                ),
                disable_radix_cache=False,
                disable_overlap_schedule=False,
                linear_attn_backend="triton",
                mamba_radix_cache_strategy="auto",
                page_size=128,
            )
            with self.subTest(architecture=architecture):
                with patch.object(
                    overrides, "get_platform", return_value=SimpleNamespace(is_npu=True)
                ):
                    self.assertEqual(
                        overrides._mamba_radix_cache_resolution(view),
                        {
                            "uses_mamba_radix_cache": True,
                            "mamba_radix_cache_strategy": "extra_buffer",
                        },
                    )
                with patch.object(
                    overrides,
                    "get_platform",
                    return_value=SimpleNamespace(is_npu=False),
                ):
                    self.assertFalse(
                        overrides.supports_mamba_cache_extra_buffer(
                            view, view._model_config.hf_config
                        )
                    )


@unittest.skipUnless(is_npu(), "Requires an Ascend NPU")
class TestJetDynamicConvolution(unittest.TestCase):
    def test_ragged_outputs_and_prefix_states_match_sequence_reference(self):
        from sglang.srt.models.jet_nemotron import DynamicShortConvolution

        torch.manual_seed(2026)
        module = DynamicShortConvolution.__new__(DynamicShortConvolution)
        nn.Module.__init__(module)
        module.hidden_size = 8
        module.kernel_size = 4
        module.kernel_generator = nn.Linear(6, 32)
        device_module = copy.deepcopy(module).npu()

        for lengths, checkpoints in (
            ([1, 5, 2], [0, 3, 1]),
            ([1, 1, 1], [1, 0, 1]),
            ([128, 129], [128, 128]),
        ):
            with self.subTest(lengths=lengths):
                x = torch.randn(sum(lengths), 8)
                generator_input = torch.randn(sum(lengths), 6)
                history = torch.randn(len(lengths), 8, 3)
                kernels = module.kernel_generator(generator_input).reshape(-1, 8, 4)
                outputs, final_states, checkpoint_states = [], [], []
                start = 0
                for row, (length, checkpoint) in enumerate(zip(lengths, checkpoints)):
                    sequence = torch.cat((history[row].T, x[start : start + length]))
                    for token in range(length):
                        outputs.append(
                            F.silu(
                                (
                                    sequence[token : token + 4].T
                                    * kernels[start + token]
                                ).sum(-1)
                            )
                        )
                    final_states.append(sequence[-3:].T)
                    checkpoint_states.append(sequence[checkpoint : checkpoint + 3].T)
                    start += length
                with torch.no_grad():
                    actual = device_module(
                        x.npu(),
                        conv_state=history.npu(),
                        generator_input=generator_input.npu(),
                        seq_lens=torch.tensor(lengths, device="npu"),
                        checkpoint_lens=torch.tensor(checkpoints, device="npu"),
                    )
                for result, reference in zip(
                    actual, (outputs, final_states, checkpoint_states)
                ):
                    torch.testing.assert_close(
                        result.cpu(), torch.stack(reference), rtol=1e-4, atol=1e-4
                    )


@unittest.skipUnless(is_npu(), "Requires an Ascend NPU")
class TestJetRecurrentCheckpoints(unittest.TestCase):
    def test_checkpoint_restore_matches_uninterrupted_recurrence(self):
        from sglang.kernels.ops.attention.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule_update,
        )

        torch.manual_seed(42)
        lengths = [5, 3, 4]
        checkpoint_lens = [2, 3, -1]
        active = [0, 2, 3]
        destinations = [4, 5, -1]
        heads, key_dim, value_dim = 2, 96, 256
        q = torch.randn(1, sum(lengths), heads, key_dim, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(1, sum(lengths), heads, value_dim, dtype=torch.bfloat16)
        g = -torch.rand(1, sum(lengths), heads)
        beta = torch.rand(1, sum(lengths), heads, dtype=torch.bfloat16)
        original = torch.randn(6, heads, value_dim, key_dim) * 0.1
        pool = original.npu()
        output = fused_recurrent_gated_delta_rule_update(
            q.npu(),
            k.npu(),
            v.npu(),
            g.npu(),
            beta.npu(),
            initial_state_source=pool,
            initial_state_indices=torch.tensor(active, dtype=torch.int32, device="npu"),
            cu_seqlens=torch.tensor([0, 5, 8, 12], dtype=torch.int32, device="npu"),
            use_qk_l2norm_in_kernel=True,
            checkpoint_seq_lens=torch.tensor(checkpoint_lens, device="npu"),
            checkpoint_state_indices=torch.tensor(destinations, device="npu"),
        )
        expected_pool = original.clone()
        expected_output = torch.empty_like(v)
        start = 0
        for row, length in enumerate(lengths):
            state = original[active[row]].clone()
            for token in range(length):
                idx = start + token
                qt, kt = q[0, idx].float(), k[0, idx].float()
                qt = qt / torch.sqrt(qt.square().sum(-1, keepdim=True) + 1e-6)
                kt = kt / torch.sqrt(kt.square().sum(-1, keepdim=True) + 1e-6)
                state = state * g[0, idx].exp()[:, None, None]
                delta = v[0, idx].float() - (state * kt[:, None, :]).sum(-1)
                state = (
                    state
                    + (delta * beta[0, idx].float()[:, None])[:, :, None]
                    * kt[:, None, :]
                )
                expected_output[0, idx] = (
                    state * (qt / math.sqrt(key_dim))[:, None, :]
                ).sum(-1)
                if token + 1 == checkpoint_lens[row]:
                    expected_pool[destinations[row]] = state
            expected_pool[active[row]] = state
            start += length
        torch.testing.assert_close(output.cpu(), expected_output, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(pool.cpu(), expected_pool, rtol=2e-3, atol=2e-3)
        # Resume row zero from its middle-prefix snapshot. Its suffix must
        # match the original forward, including the final recurrent state.
        resumed = fused_recurrent_gated_delta_rule_update(
            q[:, 2:5].contiguous().npu(),
            k[:, 2:5].contiguous().npu(),
            v[:, 2:5].contiguous().npu(),
            g[:, 2:5].contiguous().npu(),
            beta[:, 2:5].contiguous().npu(),
            initial_state_source=pool,
            initial_state_indices=torch.tensor([4], dtype=torch.int32, device="npu"),
            cu_seqlens=torch.tensor([0, 3], dtype=torch.int32, device="npu"),
            use_qk_l2norm_in_kernel=True,
        )
        torch.testing.assert_close(resumed.cpu(), output[:, 2:5].cpu())
        torch.testing.assert_close(pool[4].cpu(), pool[0].cpu())


@unittest.skipUnless(is_npu(), "Requires an Ascend NPU")
class TestJetPagedSlidingWindow(unittest.TestCase):
    def test_fia_window_matches_reference_with_cached_prefix(self):
        import torch_npu

        torch.manual_seed(21)
        page_size, context_len, window = 128, 1300, 1152
        heads, kv_heads, head_dim = 12, 2, 128
        pages = torch.randperm(11)
        key_pool = torch.randn(11, page_size, kv_heads, head_dim, dtype=torch.bfloat16)
        value_pool = torch.randn_like(key_pool)
        key = key_pool[pages].reshape(-1, kv_heads, head_dim)[:context_len]
        value = value_pool[pages].reshape(-1, kv_heads, head_dim)[:context_len]
        mask = torch.ones(2048, 2048, dtype=torch.bool).triu(1).npu()
        for query_len in (11, 128, context_len):
            with self.subTest(query_len=query_len):
                query = torch.randn(query_len, heads, head_dim, dtype=torch.bfloat16)
                output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                    query=query.npu(),
                    key=key_pool.reshape(11, page_size, -1).npu(),
                    value=value_pool.reshape(11, page_size, -1).npu(),
                    block_table=pages[None].to(dtype=torch.int32, device="npu"),
                    block_size=page_size,
                    input_layout="TND",
                    num_query_heads=heads,
                    num_key_value_heads=kv_heads,
                    actual_seq_qlen=[query_len],
                    actual_seq_kvlen=[context_len],
                    atten_mask=mask,
                    sparse_mode=4,
                    pre_tokens=window,
                    next_tokens=0,
                    softmax_scale=head_dim**-0.5,
                )
                q_positions = torch.arange(context_len - query_len, context_len)[
                    :, None
                ]
                k_positions = torch.arange(context_len)[None, :]
                allowed = (k_positions <= q_positions) & (
                    k_positions >= q_positions - window
                )
                expected = F.scaled_dot_product_attention(
                    query.float().transpose(0, 1)[None],
                    key.float()
                    .repeat_interleave(heads // kv_heads, dim=1)
                    .transpose(0, 1)[None],
                    value.float()
                    .repeat_interleave(heads // kv_heads, dim=1)
                    .transpose(0, 1)[None],
                    attn_mask=allowed,
                )[0].transpose(0, 1)
                torch.testing.assert_close(
                    output.cpu().float(), expected, rtol=2e-2, atol=2e-3
                )


@unittest.skipUnless(is_npu(), "Requires an Ascend NPU")
class TestJetInputLogprobs(unittest.TestCase):
    def test_fast_input_topk_uses_native_npu_reductions(self):
        from sglang.srt.layers.logprob_processor import InputLogprobProcessor

        torch.manual_seed(19)
        logits = torch.randn(4, 151936)
        metadata = SimpleNamespace(
            sample_indices_cpu=[3],
            input_logprob_indices_cpu=[0, 1, 2],
            extend_return_top_logprob=True,
            extend_token_ids_logprob=False,
            top_logprobs_nums=[3],
            extend_logprob_pruned_lens_cpu=[3],
            extend_input_logprob_token_ids_gpu=torch.tensor([1, 2, 3], device="npu"),
        )
        processor = InputLogprobProcessor(vocab_size=logits.shape[-1])
        processor.enable_fast_input_logprobs = True
        with patch.object(
            processor, "_can_borrow_logprob_memory", side_effect=AssertionError
        ):
            result, sampled = processor.forward(
                logits.npu(),
                sample_indices=torch.tensor([3], device="npu"),
                input_logprob_indices=torch.tensor([0, 1, 2], device="npu"),
                token_to_seq_idx=[0, 0, 0, 0],
                lm_head=None,
                get_logits_fn=lambda states, *_args, **_kwargs: states,
                logits_metadata=metadata,
            )
        reference = logits.log_softmax(-1)
        torch.testing.assert_close(
            result.token_logprobs.cpu(),
            reference[torch.arange(3), torch.tensor([1, 2, 3])],
        )
        expected_values, expected_indices = reference[:3].topk(3, dim=-1)
        torch.testing.assert_close(
            torch.tensor(result.top_logprobs_val[0]), expected_values
        )
        self.assertEqual(result.top_logprobs_idx[0], expected_indices.tolist())
        torch.testing.assert_close(sampled.cpu(), logits[3:4])


if __name__ == "__main__":
    unittest.main()
