import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    SampleStepTokens,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.models.dflash import DFlashAttention
from sglang.srt.models.dspark import DSparkDraftMixin, VanillaMarkov
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=25, suite="stage-a-unit-test-npu")


class TestNpuDsparkSampling(unittest.TestCase):
    def test_vanilla_markov_vocab_shards_match_replicated_logits(self):
        device = torch.device("npu")
        batch_size, vocab_size, markov_rank, tp_size = 2, 4096, 64, 4
        head = VanillaMarkov(vocab_size=vocab_size, markov_rank=markov_rank).to(
            device=device, dtype=torch.bfloat16
        )
        base_logits = torch.randn(
            batch_size, vocab_size, device=device, dtype=torch.bfloat16
        )
        prev_tokens = torch.randint(vocab_size, (batch_size,), device=device)

        expected = base_logits + head.compute_step_bias(prev_tokens, None)
        latent = head.get_prev_embeddings(prev_tokens)
        width = vocab_size // tp_size
        local_steps = []
        for rank in range(tp_size):
            start, end = rank * width, (rank + 1) * width
            local_bias = F.linear(
                latent.to(head.markov_w2.weight.dtype),
                head.markov_w2.weight[start:end],
            )
            local_steps.append(base_logits[:, start:end] + local_bias)
        actual = torch.cat(local_steps, dim=-1)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            actual.argmax(dim=-1), expected.argmax(dim=-1), rtol=0, atol=0
        )

    def test_tp_sharded_greedy_block_matches_replicated_graph(self):
        device = torch.device("npu")
        batch_size, gamma, vocab_size = 2, 3, 4096
        head = VanillaMarkov(vocab_size=vocab_size, markov_rank=64).to(
            device=device, dtype=torch.bfloat16
        )
        base_logits = torch.randn(
            batch_size,
            gamma,
            vocab_size,
            device=device,
            dtype=torch.bfloat16,
        )
        anchors = torch.randint(vocab_size, (batch_size,), device=device)
        expected, _ = head.sample_block(
            base_logits,
            first_prev_tokens=anchors,
            hidden_states=None,
            sampler=lambda logits, _step_idx: logits.argmax(dim=-1),
            return_corrected_logits=False,
        )

        class _IdentityGroup:
            @staticmethod
            def all_gather(value, dim=-1):
                del dim
                return value

        head._tp_shard = SimpleNamespace(
            tp_size=1,
            org_vocab_start=0,
            org_vocab_end=vocab_size,
            num_embeddings_per_partition=vocab_size,
            num_embeddings_padded=vocab_size,
        )
        head._shard_group = _IdentityGroup()
        actual = head.sample_greedy_block_sharded(
            base_logits, first_prev_tokens=anchors
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            graph_actual = head.sample_greedy_block_sharded(
                base_logits, first_prev_tokens=anchors
            )
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(graph_actual, expected, rtol=0, atol=0)

    def test_greedy_near_tie_uses_logits_argmax(self):
        device = torch.device("npu")
        logits = torch.tensor([[0.0, 1.0e-8, -1.0]], device=device)
        expected = logits.argmax(dim=-1)
        actual = SampleStepTokens.execute(
            step_logits=logits,
            temperatures=torch.ones(1, device=device),
            greedy_mask=torch.ones(1, dtype=torch.bool, device=device),
            exp_noise=torch.ones_like(logits),
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_greedy_skips_corrected_logits_store(self):
        device = torch.device("npu")
        logits = torch.randn(2, 4097, device=device)
        corrected = torch.full_like(logits, 17.0)
        actual = SampleStepTokens.execute(
            step_logits=logits,
            temperatures=torch.ones(2, device=device),
            greedy_mask=torch.ones(2, dtype=torch.bool, device=device),
            exp_noise=torch.ones_like(logits),
            corrected_logits_out=corrected,
            write_corrected_logits=torch.zeros((), dtype=torch.int32, device=device),
        )

        torch.testing.assert_close(actual, logits.argmax(dim=-1), rtol=0, atol=0)
        self.assertTrue(torch.all(corrected == 17.0).item())

    def test_folded_proposal_graph_matches_eager(self):
        device = torch.device("npu")
        # Exercise a production-sized combine reduction and the gamma-strided
        # corrected-logit destination, not only a one-tile toy vocabulary.
        batch_size, gamma, hidden_size, vocab_size = 2, 3, 32, 163840

        class _Model:
            def __init__(self):
                weight = torch.randn(
                    vocab_size, hidden_size, dtype=torch.bfloat16, device=device
                )
                self.lm_head = SimpleNamespace(weight=weight, org_vocab_size=vocab_size)
                self.markov_head = VanillaMarkov(
                    vocab_size=vocab_size, markov_rank=8
                ).to(device=device, dtype=torch.bfloat16)

            def compute_base_logits(self, hidden_states):
                return F.linear(hidden_states, self.lm_head.weight), None

        model = _Model()
        sampler = DsparkDraftSampler(
            model=model,
            gamma=gamma,
            max_bs=batch_size,
            device=device,
            folded_sampling=True,
        )
        hidden = torch.randn(
            batch_size * gamma,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        input_ids = torch.randint(
            vocab_size, (batch_size * gamma,), dtype=torch.int64, device=device
        )
        base_logits, _ = model.compute_base_logits(hidden)
        expected_greedy_tokens, _ = model.markov_head.sample_block(
            base_logits.view(batch_size, gamma, vocab_size),
            first_prev_tokens=input_ids.view(batch_size, gamma)[:, 0],
            hidden_states=hidden.view(batch_size, gamma, hidden_size),
            sampler=lambda step_logits, _step_idx: step_logits.argmax(dim=-1),
        )

        # Production capture also runs a warmup pass; compile the Ascend
        # Triton kernel before entering NPUGraph capture.
        sampler(hidden, input_ids)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            sampler(hidden, input_ids)

        # Production captures with greedy defaults, then can stage a mixed
        # replay into the same graph. Force row 1 toward token zero so a branch
        # accidentally frozen to greedy cannot pass by chance.
        mixed_greedy_mask = torch.tensor([True, False], device=device)
        mixed_noise = torch.ones(batch_size, vocab_size, device=device)
        mixed_noise[1, 0] = 1.0e-30
        sampler.greedy_mask[:batch_size].copy_(mixed_greedy_mask)
        sampler.kernel_greedy_mask[:batch_size].copy_(mixed_greedy_mask.to(torch.int32))
        sampler.temperatures[:batch_size].fill_(1.0)
        sampler.exp_noise[:batch_size].copy_(mixed_noise)
        sampler.write_corrected_logits.fill_(1)
        expected_tokens, expected_logits = model.markov_head.sample_block(
            base_logits.view(batch_size, gamma, vocab_size),
            first_prev_tokens=input_ids.view(batch_size, gamma)[:, 0],
            hidden_states=hidden.view(batch_size, gamma, hidden_size),
            sampler=lambda step_logits, _step_idx: SampleStepTokens.torch(
                step_logits=step_logits,
                temperatures=sampler.temperatures[:batch_size],
                greedy_mask=mixed_greedy_mask,
                exp_noise=mixed_noise,
            ),
        )
        self.assertTrue(torch.all(expected_tokens[1] == 0).item())
        graph.replay()
        torch.npu.synchronize()

        actual_tokens = sampler.out[: batch_size * gamma].view(batch_size, gamma)
        actual_logits = sampler.corrected_out[: batch_size * gamma].view(
            batch_size, gamma, vocab_size
        )
        torch.testing.assert_close(actual_tokens, expected_tokens, rtol=0, atol=0)
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)

        # The same captured graph must suppress the full-vocabulary store when
        # the next request is greedy; the device flag is staged before replay.
        sampler.corrected_out.fill_(17.0)
        sampler.greedy_mask.fill_(True)
        sampler.kernel_greedy_mask.fill_(1)
        sampler.write_corrected_logits.zero_()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            sampler.out[: batch_size * gamma].view(batch_size, gamma),
            expected_greedy_tokens,
            rtol=0,
            atol=0,
        )
        self.assertTrue(torch.all(sampler.corrected_out == 17.0).item())

    def test_markov_block_can_skip_corrected_logits_materialization(self):
        device = torch.device("npu")
        batch_size, gamma, vocab_size = 2, 3, 64
        head = VanillaMarkov(vocab_size=vocab_size, markov_rank=8).to(
            device=device, dtype=torch.bfloat16
        )
        base_logits = torch.randn(
            batch_size, gamma, vocab_size, dtype=torch.bfloat16, device=device
        )
        anchors = torch.randint(vocab_size, (batch_size,), device=device)

        expected_tokens, _ = head.sample_block(
            base_logits,
            first_prev_tokens=anchors,
            hidden_states=None,
            sampler=lambda logits, _step_idx: logits.argmax(dim=-1),
        )
        actual_tokens, corrected_logits = head.sample_block(
            base_logits,
            first_prev_tokens=anchors,
            hidden_states=None,
            sampler=lambda logits, _step_idx: logits.argmax(dim=-1),
            return_corrected_logits=False,
        )

        self.assertIsNone(corrected_logits)
        torch.testing.assert_close(actual_tokens, expected_tokens, rtol=0, atol=0)

    def test_sampling_noise_is_staged_only_for_stochastic_batches(self):
        device = torch.device("npu")
        lm_head = SimpleNamespace(org_vocab_size=32, weight=torch.empty(0))
        model = SimpleNamespace(lm_head=lm_head, markov_head=object())
        sampler = DsparkDraftSampler(
            model=model,
            gamma=2,
            max_bs=4,
            device=device,
            folded_sampling=True,
        )
        sampler.exp_noise.fill_(7.0)

        all_greedy = SimpleNamespace(
            temperatures=torch.ones(2, device=device),
            top_ks=torch.ones(2, dtype=torch.int32, device=device),
            is_all_greedy=True,
        )
        sampler.stage_sampling_params(bs=2, sampling_info=all_greedy)
        self.assertTrue(torch.all(sampler.exp_noise == 7.0).item())
        self.assertTrue(torch.all(sampler.greedy_mask).item())
        self.assertTrue(torch.all(sampler.kernel_greedy_mask == 1).item())

        mixed = SimpleNamespace(
            temperatures=torch.ones(2, device=device),
            top_ks=torch.tensor([1, 8], dtype=torch.int32, device=device),
            is_all_greedy=False,
        )
        sampler.greedy_mask[2:].fill_(False)
        sampler.stage_sampling_params(bs=2, sampling_info=mixed)
        self.assertTrue(torch.all(sampler.exp_noise > 0).item())
        self.assertFalse(torch.all(sampler.exp_noise == 7.0).item())
        self.assertTrue(torch.equal(sampler.greedy_mask[:2], mixed.top_ks <= 1))
        self.assertTrue(
            torch.equal(
                sampler.kernel_greedy_mask[:2],
                (mixed.top_ks <= 1).to(torch.int32),
            )
        )
        self.assertTrue(torch.all(sampler.greedy_mask[2:]).item())
        self.assertTrue(torch.all(sampler.kernel_greedy_mask[2:] == 1).item())

        # Returning to greedy restores both graph buffers once; subsequent
        # greedy requests reuse the resident state and leave RNG untouched.
        mixed_noise = sampler.exp_noise.clone()
        sampler.stage_sampling_params(bs=2, sampling_info=all_greedy)
        sampler.stage_sampling_params(bs=2, sampling_info=all_greedy)
        self.assertTrue(torch.all(sampler.greedy_mask).item())
        self.assertTrue(torch.all(sampler.kernel_greedy_mask == 1).item())
        self.assertEqual(sampler.write_corrected_logits.item(), 0)
        torch.testing.assert_close(sampler.exp_noise, mixed_noise, rtol=0, atol=0)

    def test_mixed_rows_match_exponential_race_reference(self):
        device = torch.device("npu")
        generator = torch.Generator(device=device).manual_seed(19)
        logits = torch.randn(4, 5003, device=device, generator=generator)
        temperatures = torch.tensor([0.7, 1.0, 1.3, 0.5], device=device)
        greedy_mask = torch.tensor([True, False, True, False], device=device)
        exp_noise = torch.empty_like(logits).exponential_(1, generator=generator)

        noise = torch.where(greedy_mask[:, None], 1.0, exp_noise)
        expected = (logits.float() - temperatures[:, None] * noise.log()).argmax(dim=-1)
        actual = SampleStepTokens.execute(
            step_logits=logits,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
            exp_noise=exp_noise,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TestNpuDsparkOptimizedPaths(unittest.TestCase):
    def test_embedding_graph_matches_eager(self):
        device = torch.device("npu")
        owner = SimpleNamespace(
            embed_tokens=torch.nn.Embedding(
                128, 64, dtype=torch.bfloat16, device=device
            )
        )
        input_ids = torch.arange(12, dtype=torch.int64, device=device)
        expected = DSparkDraftMixin.forward_embed(owner, input_ids)

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = DSparkDraftMixin.forward_embed(owner, input_ids)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_flattened_kv_only_rope_uses_npu_3d_contract(self):
        device = torch.device("npu")
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", device="npu")
        )
        tokens, num_heads, head_dim = 5, 2, 64
        rotary = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=4096,
            base=10000.0,
            is_neox_style=True,
        ).to(device)
        positions = torch.arange(tokens, dtype=torch.int64, device=device)
        flattened = torch.randn(
            tokens,
            num_heads * head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        shaped = flattened.view(tokens, num_heads, head_dim)
        _, expected = rotary(positions, torch.empty_like(shaped), shaped)

        owner = SimpleNamespace(head_dim=head_dim, rotary_emb=rotary)
        actual = DFlashAttention.apply_k_rope(owner, positions, flattened)
        torch.testing.assert_close(
            actual, expected.reshape_as(flattened), rtol=0, atol=0
        )

    def test_stacked_ctx_kv_matches_per_layer(self):
        device = torch.device("npu")
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", device="npu")
        )
        tokens, hidden_size = 5, 128
        num_layers, num_kv_heads, head_dim = 3, 2, 64
        kv_size = num_kv_heads * head_dim
        eps = 1.0e-6
        rotary = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=4096,
            base=10000.0,
            is_neox_style=True,
        ).to(device)

        layers = []
        weights = []
        norm_weights = []
        for _ in range(num_layers):
            weight = torch.randn(
                2 * kv_size,
                hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )
            k_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=torch.bfloat16)
            attn = SimpleNamespace(
                kv_size=kv_size,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                rotary_emb=rotary,
                k_norm=k_norm,
            )
            layers.append(SimpleNamespace(self_attn=attn))
            weights.append(weight)
            norm_weights.append(k_norm.weight)

        owner = SimpleNamespace(layers=layers)
        stacked = {
            "weight": torch.cat(weights, dim=0),
            "bias": None,
            "k_norm_weight": torch.stack(norm_weights).float(),
            "eps": eps,
        }
        hidden = torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device=device)
        positions = torch.arange(tokens, dtype=torch.int64, device=device)

        expected_k, expected_v = [], []
        for layer, weight in zip(layers, weights):
            kv = F.linear(hidden, weight)
            k, v = kv.split((kv_size, kv_size), dim=-1)
            k = layer.self_attn.k_norm(k.reshape(-1, head_dim)).view_as(k)
            k = k.view(tokens, num_kv_heads, head_dim)
            dummy_q = torch.empty_like(k)
            _, k = rotary(positions, dummy_q, k)
            expected_k.append(k)
            expected_v.append(v.view(tokens, num_kv_heads, head_dim))

        actual_k, actual_v = DSparkDraftMixin._project_ctx_kv_stacked(
            owner, ctx_hidden=hidden, positions=positions, stacked=stacked
        )
        for layer in range(num_layers):
            torch.testing.assert_close(
                actual_k[layer], expected_k[layer], rtol=2.0e-2, atol=2.0e-2
            )
            torch.testing.assert_close(
                actual_v[layer], expected_v[layer], rtol=2.0e-2, atol=2.0e-2
            )


if __name__ == "__main__":
    unittest.main()
