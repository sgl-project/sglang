"""Gemma 4 vocabulary ownership, loading, scaling, and logits contracts."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig

from sglang.srt.environ import envs
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.gemma3_causal import Gemma3TextScaledWordEmbedding
from sglang.srt.models.gemma4_causal import (
    Gemma4ForCausalLM,
    Gemma4TextModel,
    Gemma4TextScaledWordEmbedding,
    shared_vocab_weight,
)
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
from sglang.srt.models.gemma4_mtp import Gemma4AssistantForCausalLM
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def text_config(**overrides):
    values = dict(
        vocab_size=259,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=2,
        global_head_dim=2,
        layer_types=["full_attention"] * 2,
        max_position_embeddings=64,
        sliding_window=16,
        enable_moe_block=False,
        hidden_size_per_layer_input=0,
        pad_token_id=0,
        tie_word_embeddings=True,
        final_logit_softcapping=30.0,
    )
    values.update(overrides)
    return Gemma4TextConfig(**values)


def multimodal_config(config):
    vision = Gemma4VisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=2,
        patch_size=2,
    )
    return Gemma4Config(
        text_config=config.to_dict(), vision_config=vision.to_dict(), audio_config=None
    )


@contextmanager
def parallel_scope(rank=0, size=4, dp=1, pp=1, pp_rank=0, dp_head=False):
    with get_context().override_server_args(
        model_path="dummy",
        device="cpu",
        tp_size=size,
        pp_size=pp,
        attn_dp_size=dp,
        enable_dp_attention=dp > 1,
        enable_dp_lm_head=dp_head,
    ):
        tp = SimpleNamespace(rank_in_group=rank, world_size=size, cpu_group=None)
        attn = SimpleNamespace(
            rank_in_group=rank % (size // dp), world_size=size // dp, cpu_group=None
        )
        pipeline = SimpleNamespace(
            rank_in_group=pp_rank,
            world_size=pp,
            is_first_rank=pp_rank == 0,
            is_last_rank=pp_rank == pp - 1,
        )
        with get_parallel().override(
            tp_rank=rank,
            tp_size=size,
            tp_group=tp,
            attn_tp_rank=attn.rank_in_group,
            attn_tp_size=attn.world_size,
            attn_tp_group=attn,
            attn_dp_size=dp,
            pp_rank=pp_rank,
            pp_group=pipeline,
            enable_dp_lm_head=dp_head,
        ):
            yield tp, attn


class TestGemma4VocabParallel(CustomTestCase):
    def setUp(self):
        self.env_override = envs.SGLANG_DISABLE_FUSIONS.override(False)
        self.env_override.__enter__()
        self.addCleanup(self.env_override.__exit__, None, None, None)
        # Exercise the unquantized tied bank independently of CPU AMX packing.
        amx = patch("sglang.srt.models.gemma4_mm._is_cpu_amx_available", False)
        amx.start()
        self.addCleanup(amx.stop)
        self.weight = torch.randn(259, 8, generator=torch.Generator().manual_seed(42))

    def test_shard_boundaries_padding_and_scaling(self):
        ids = torch.tensor([[0, 79, 80, 159], [160, 239, 240, 258]])
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            weight = self.weight.to(dtype)
            local_outputs = []
            layers = []
            for rank in range(4):
                with parallel_scope(rank):
                    layer = Gemma4TextScaledWordEmbedding(259, 8, 0, 8**0.5).to(dtype)
                    layer.weight.weight_loader(layer.weight, weight)
                    layers.append(layer)
                    local_outputs.append(layer._embed_local_shard(ids))
                    start = rank * 80
                    end = min(start + 80, 259)
                    torch.testing.assert_close(
                        layer.weight[: end - start], weight[start:end]
                    )
                    self.assertEqual(
                        layer.weight[end - start :].count_nonzero().item(), 0
                    )
            full_lookup = torch.stack(local_outputs).sum(0)
            expected = F.embedding(ids, weight) * 8**0.5
            for rank, layer in enumerate(layers):
                with (
                    parallel_scope(rank),
                    patch(
                        "sglang.srt.layers.vocab_parallel_embedding.tensor_model_parallel_all_reduce",
                        return_value=full_lookup,
                    ),
                ):
                    actual = layer(ids)
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(layer.weight.shape, (80, 8))

    def test_checkpoint_loading_tying_and_fusion_switch(self):
        for multimodal in (False, True):
            for disabled in (False, True):
                with self.subTest(multimodal=multimodal, disabled=disabled):
                    with (
                        parallel_scope(2),
                        envs.SGLANG_DISABLE_FUSIONS.override(disabled),
                    ):
                        cfg = text_config()
                        model = (
                            Gemma4ForConditionalGeneration(multimodal_config(cfg))
                            if multimodal
                            else Gemma4ForCausalLM(cfg)
                        )
                        key = (
                            "model.language_model.embed_tokens.weight"
                            if multimodal
                            else "model.embed_tokens.weight"
                        )
                        model.load_weights([(key, self.weight)])
                        embed = model.get_input_embeddings()
                        self.assertIs(model.lm_head, embed)
                        self.assertIs(model.lm_head.weight, embed.weight)
                        expected = self.weight if disabled else self.weight[160:240]
                        torch.testing.assert_close(
                            embed.weight, expected, rtol=0, atol=0
                        )
                        self.assertIsInstance(
                            embed,
                            Gemma3TextScaledWordEmbedding
                            if disabled
                            else Gemma4TextScaledWordEmbedding,
                        )
                        if not disabled:
                            shared, head = model.get_embed_and_head()
                            self.assertIs(shared, embed.weight)
                            self.assertIs(head, shared)
        self.assertTrue(issubclass(Gemma3TextScaledWordEmbedding, torch.nn.Embedding))

    def test_tp1_and_per_layer_embedding_stay_scaled(self):
        with parallel_scope(size=1):
            cfg = text_config(
                hidden_size_per_layer_input=4, vocab_size_per_layer_input=259
            )
            model = Gemma4TextModel(cfg)
            embed = model.embed_tokens
            embed.weight.weight_loader(embed.weight, self.weight)
            ids = torch.tensor([0, 258])
            torch.testing.assert_close(
                embed(ids), F.embedding(ids, self.weight) * 8**0.5
            )
            self.assertEqual(embed.tp_size, 1)
            self.assertIsInstance(
                model.embed_tokens_per_layer, Gemma3TextScaledWordEmbedding
            )
            ple = torch.randn_like(model.embed_tokens_per_layer.weight)
            model.embed_tokens_per_layer.weight.data.copy_(ple)
            torch.testing.assert_close(
                model.get_per_layer_inputs(ids),
                (F.embedding(ids, ple) * 2).reshape(2, 2, 4),
            )

    def test_pp_loads_the_tied_checkpoint_on_the_head_stage(self):
        for multimodal in (False, True):
            for stage in (0, 1):
                with parallel_scope(3, pp=2, pp_rank=stage):
                    cfg = text_config()
                    model = (
                        Gemma4ForConditionalGeneration(multimodal_config(cfg))
                        if multimodal
                        else Gemma4ForCausalLM(cfg)
                    )
                    key = (
                        "model.language_model.embed_tokens.weight"
                        if multimodal
                        else "model.embed_tokens.weight"
                    )
                    model.load_weights([(key, self.weight)])
                    owned = (
                        model.get_input_embeddings() if stage == 0 else model.lm_head
                    )
                    self.assertIsInstance(owned, VocabParallelEmbedding)
                    torch.testing.assert_close(owned.weight[:19], self.weight[240:])
                    self.assertEqual(owned.weight[19:].count_nonzero().item(), 0)
                    with self.assertRaises(NotImplementedError):
                        model.get_embed_and_head()

    def test_attention_group_logits_reconstruct_padded_vocabulary_once(self):
        for dp_head in (False, True):
            with parallel_scope(3, dp=2, dp_head=dp_head):
                model = Gemma4ForCausalLM(text_config())
                embed = model.get_input_embeddings()
                embed.weight.weight_loader(embed.weight, self.weight)
                proc = model.logits_processor
                self.assertTrue(embed.use_attn_tp_group)
                self.assertEqual(embed.tp_size, 2)
                self.assertTrue(proc.use_attn_tp_group)
                self.assertFalse(proc.do_tensor_parallel_all_gather_dp_attn)
                hidden = torch.randn(2, 8, generator=torch.Generator().manual_seed(123))
                full_logits = F.linear(hidden, F.pad(self.weight, (0, 0, 0, 61)))
                blocks = full_logits.reshape(2, 2, 160).permute(1, 0, 2).contiguous()
                calls = []

                def gather(output, local):
                    torch.testing.assert_close(local, blocks[1])
                    output.copy_(blocks)
                    calls.append(tuple(local.shape))

                metadata = SimpleNamespace(
                    next_token_logits_buffer=None, forward_mode=ForwardMode.DECODE
                )
                with (
                    patch(
                        "sglang.srt.layers.logits_processor.attn_tp_all_gather_into_tensor",
                        side_effect=gather,
                    ),
                    patch(
                        "sglang.srt.layers.logits_processor.fused_softcap",
                        side_effect=lambda logits, cap: logits.copy_(
                            cap * torch.tanh(logits / cap)
                        ),
                    ),
                ):
                    actual = proc._get_logits(hidden, embed, metadata)
                expected = 30 * torch.tanh(full_logits[:, :259] / 30)
                torch.testing.assert_close(actual, expected)
                self.assertEqual(actual.shape, (2, 259))
                self.assertEqual(calls, [(2, 160)])

    def test_tp_logits_and_unshared_head_keep_vocabulary_order(self):
        for tied in (True, False):
            with parallel_scope(2):
                model = Gemma4ForCausalLM(text_config(tie_word_embeddings=tied))
                head = model.lm_head
                head.weight.weight_loader(head.weight, self.weight)
                hidden = torch.randn(2, 8, generator=torch.Generator().manual_seed(1))
                full_logits = F.linear(hidden, F.pad(self.weight, (0, 0, 0, 61)))
                calls = []

                def gather(local):
                    torch.testing.assert_close(local, full_logits[:, 160:240])
                    calls.append(local.shape)
                    return full_logits.clone()

                proc = model.logits_processor
                metadata = SimpleNamespace(
                    next_token_logits_buffer=None, forward_mode=ForwardMode.DECODE
                )
                with (
                    patch.object(proc, "_logits_gatherer", side_effect=gather),
                    patch(
                        "sglang.srt.layers.logits_processor.fused_softcap",
                        side_effect=lambda logits, cap: logits.copy_(
                            cap * torch.tanh(logits / cap)
                        ),
                    ),
                ):
                    actual = proc._get_logits(hidden, head, metadata)
                torch.testing.assert_close(
                    actual, 30 * torch.tanh(full_logits[:, :259] / 30)
                )
                self.assertEqual(len(calls), 1)
                if not tied:
                    self.assertIsNot(head.weight, model.get_input_embeddings().weight)
                    self.assertIs(model.get_embed_and_head()[1], head.weight)

    def test_one_rank_per_attention_replica_skips_logits_gather(self):
        with parallel_scope(3, dp=4):
            model = Gemma4ForCausalLM(text_config())
            head = model.lm_head
            self.assertEqual(head.tp_size, 1)
            self.assertTrue(head.use_attn_tp_group)
            self.assertFalse(model.logits_processor.do_tensor_parallel_all_gather)
            self.assertFalse(
                model.logits_processor.do_tensor_parallel_all_gather_dp_attn
            )

    def test_speculative_sharing_reuses_or_repartitions_once(self):
        with parallel_scope(1) as (group, _):
            source = Gemma4TextScaledWordEmbedding(259, 8, 0)
            source.weight.weight_loader(source.weight, self.weight)
            draft = VocabParallelEmbedding(259, 8)
            self.assertIs(
                shared_vocab_weight(source, draft_embedding=draft), source.weight
            )
            replicated = Gemma3TextScaledWordEmbedding(259, 8, 0)
            padded = F.pad(self.weight, (0, 0, 0, 61))
            group.all_gather_into_tensor = lambda output, local: output.copy_(padded)
            torch.testing.assert_close(
                shared_vocab_weight(source, draft_embedding=replicated), self.weight
            )
            with parallel_scope(1, dp=2):
                attention_draft = VocabParallelEmbedding(259, 8, use_attn_tp_group=True)
            remapped = shared_vocab_weight(source, draft_embedding=attention_draft)
            torch.testing.assert_close(remapped[:99], self.weight[160:])
            self.assertEqual(remapped[99:].count_nonzero().item(), 0)
            cfg = text_config(backbone_hidden_size=8, use_ordered_embeddings=False)
            assistant = Gemma4AssistantForCausalLM(cfg)
            self.assertIsInstance(
                assistant.model.embed_tokens, Gemma3TextScaledWordEmbedding
            )
            self.assertFalse(assistant.logits_processor.do_tensor_parallel_all_gather)


if __name__ == "__main__":
    unittest.main()
