"""Gemma embedding exports retain their source or recipient layout."""

import socket
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import (
    Gemma3Config,
    Gemma3TextConfig,
    Gemma4TextConfig,
    LlamaConfig,
    SiglipVisionConfig,
)

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.models.gemma3_causal import EmbeddingGemmaModel, Gemma3ForCausalLM
from sglang.srt.models.gemma3_mm import Gemma3ForConditionalGeneration
from sglang.srt.models.gemma4_causal import Gemma4ForCausalLM
from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.pp_draft_embedding import (
    find_draft_embedding_param,
    resolve_draft_embed_and_head,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=18, stage="stage-b", runner_config="1-gpu-small")


def build_source(kind, *, tied=True, vocab=512):
    extra = {}
    if kind == "gemma4":
        extra.update(
            global_head_dim=16,
            vocab_size_per_layer_input=vocab,
            hidden_size_per_layer_input=8,
        )
    cls = Gemma4TextConfig if kind == "gemma4" else Gemma3TextConfig
    cfg = cls(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        vocab_size=vocab,
        max_position_embeddings=32,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
        tie_word_embeddings=tied,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **extra,
    )
    if kind == "gemma3_mm":
        vision = SiglipVisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=32,
            patch_size=16,
        )
        cfg = Gemma3Config(
            text_config=cfg.to_dict(),
            vision_config=vision.to_dict(),
            mm_tokens_per_image=4,
        )
    model_cls = {
        "gemma3": Gemma3ForCausalLM,
        "gemma4": Gemma4ForCausalLM,
        "gemma3_mm": Gemma3ForConditionalGeneration,
    }[kind]
    with torch.device("cuda"):
        return model_cls(cfg)


def build_draft(vocab=512, *, embedding_only=False):
    if embedding_only:
        draft = torch.nn.Module()
        draft.model = torch.nn.Module()
        with torch.device("cuda"):
            draft.model.embed_tokens = VocabParallelEmbedding(vocab, 128)
        return draft
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        vocab_size=vocab,
        max_position_embeddings=32,
        tie_word_embeddings=True,
        draft_vocab_size=vocab,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    with torch.device("cuda"):
        return LlamaForCausalLMEagle3(config)


def source_weights(model):
    source = getattr(model, "language_model", model)
    return source.model.embed_tokens.weight, source.lm_head.weight


def fill_source(model, version=0):
    embed, head = source_weights(model)
    for index, weight in enumerate((embed, head)):
        if index == 1 and weight is embed:
            continue
        values = (
            (
                torch.arange(weight.numel(), device=weight.device).reshape(weight.shape)
                % 127
            )
            - 63
            + version
            + index
        ) / 64
        with torch.no_grad():
            weight.copy_(values)
    return embed, head


def share(model, draft):
    return resolve_draft_embed_and_head(
        target_model=model,
        draft_model=draft,
        model_path="/nonexistent",
        revision=None,
        load_config=LoadConfig(load_format=LoadFormat.DUMMY),
    )


def changed_scope(changed):
    if not changed:
        return nullcontext()
    return get_parallel().override(
        tp_size=4,
        tp_rank=3,
        tp_group=None,
        attn_tp_size=4,
        attn_tp_rank=3,
        attn_tp_group=None,
        attn_dp_size=1,
        attn_dp_rank=0,
        attn_cp_size=1,
        attn_cp_rank=0,
        moe_tp_size=4,
        moe_tp_rank=3,
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_ep_group=None,
        moe_dp_size=1,
        moe_dp_rank=0,
    )


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestGemmaSharedVocabLayout(CustomTestCase):
    def setUp(self):
        if torch.distributed.is_initialized():
            self.skipTest("requires an isolated distributed test process")
        reset_context()
        self.addCleanup(reset_context)
        original = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, original)
        torch.set_default_dtype(torch.bfloat16)
        server = ServerArgs(model_path="dummy", device="cuda", tp_size=1)
        publish(server, role="test", ranks=SpawnRanks(world_rank=0, gpu_id=0))
        torch.cuda.set_device(0)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
        )
        self.addCleanup(destroy_distributed_environment)
        initialize_model_parallel()
        self.addCleanup(destroy_model_parallel)
        initialize_dp_attention(server)

    def test_default_exports_keep_construction_layout(self):
        for kind in ("gemma3", "gemma4", "gemma3_mm"):
            for tied in (True, False):
                with self.subTest(kind=kind, tied=tied):
                    model = build_source(kind, tied=tied)
                    expected = fill_source(model)
                    with changed_scope(True):
                        actual = model.get_embed_and_head()
                        if hasattr(model, "get_embed"):
                            self.assertIs(model.get_embed(), expected[0])
                    for result, weight in zip(actual, expected):
                        self.assertIs(result, weight)

    def test_sharing_uses_constructed_draft_after_scope_exit(self):
        for kind in ("gemma3", "gemma4", "gemma3_mm"):
            model = build_source(kind)
            expected = fill_source(model)
            draft = build_draft()
            with changed_scope(True):
                actual = share(model, draft)
            for result, weight in zip(actual, expected):
                self.assertIs(result, weight)

    def test_draft_rank_and_width_override_source_export(self):
        for kind in ("gemma3", "gemma4", "gemma3_mm"):
            model = build_source(kind)
            embed, head = fill_source(model)
            with changed_scope(True):
                draft = build_draft(embedding_only=True)
            actual = share(model, draft)
            for result, weight in zip(actual, (embed, head)):
                torch.testing.assert_close(result, weight[384:512], rtol=0, atol=0)
                self.assertEqual(
                    result.untyped_storage().data_ptr(),
                    weight.untyped_storage().data_ptr(),
                )
            self.assertIs(
                find_draft_embedding_param(draft)[1], draft.model.embed_tokens.weight
            )

    def test_shared_views_see_source_weight_updates(self):
        model = build_source("gemma3")
        fill_source(model)
        with changed_scope(True):
            draft = build_draft(embedding_only=True)
        embed, head = share(model, draft)
        full_embed, full_head = fill_source(model, version=13)
        torch.testing.assert_close(embed, full_embed[384:512], rtol=0, atol=0)
        torch.testing.assert_close(head, full_head[384:512], rtol=0, atol=0)

    def test_embedding_subclass_keeps_its_export_layout(self):
        config = build_source("gemma3").config
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "modules.json").write_text("[]")
            config._name_or_path = directory
            with torch.device("cuda"):
                model = EmbeddingGemmaModel(config)
            with changed_scope(True):
                self.assertIs(model.get_embed(), model.model.embed_tokens.weight)

    def test_ambiguous_recipient_fails_without_scope_fallback(self):
        model = build_source("gemma3")
        with self.assertRaisesRegex(ValueError, "no single VocabParallelEmbedding"):
            share(model, torch.nn.Module())


if __name__ == "__main__":
    unittest.main()
