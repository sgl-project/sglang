"""Gemma embedding exports retain their source or recipient layout."""

import copy
import socket
import unittest
from contextlib import nullcontext

import torch
from transformers import (
    Gemma3Config,
    Gemma3TextConfig,
    Gemma4AssistantConfig,
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
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
from sglang.srt.models.gemma3_causal import Gemma3ForCausalLM
from sglang.srt.models.gemma3_mm import Gemma3ForConditionalGeneration
from sglang.srt.models.gemma4_causal import Gemma4ForCausalLM
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
from sglang.srt.models.gemma4_mtp import Gemma4AssistantForCausalLM
from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.pp_draft_embedding import (
    find_draft_embedding_param,
    resolve_draft_embed_and_head,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=18, stage="base-b", runner_config="1-gpu-small")


def build_source(kind, *, tied=True, vocab=512):
    extra = {}
    if kind in ("gemma4", "gemma4_mm"):
        extra.update(
            global_head_dim=16,
            vocab_size_per_layer_input=vocab,
            hidden_size_per_layer_input=8,
        )
    cls = Gemma4TextConfig if kind in ("gemma4", "gemma4_mm") else Gemma3TextConfig
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
    if kind == "gemma4_mm":
        cfg = Gemma4Config(
            text_config=cfg,
            vision_config=Gemma4VisionConfig(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=16,
                patch_size=16,
                position_embedding_size=32,
            ),
            audio_config=None,
        )
    model_cls = {
        "gemma3": Gemma3ForCausalLM,
        "gemma4": Gemma4ForCausalLM,
        "gemma3_mm": Gemma3ForConditionalGeneration,
        "gemma4_mm": Gemma4ForConditionalGeneration,
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
    if isinstance(model, Gemma4ForConditionalGeneration):
        embed = model.language_model.embed_tokens.weight
        return embed, embed
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


def changed_scope(changed, *, size=4, rank=3):
    if not changed:
        return nullcontext()
    return get_parallel().override(
        tp_size=size,
        tp_rank=rank,
        tp_group=None,
        attn_tp_size=size,
        attn_tp_rank=rank,
        attn_tp_group=None,
        attn_dp_size=1,
        attn_dp_rank=0,
        attn_cp_size=1,
        attn_cp_rank=0,
        moe_tp_size=size,
        moe_tp_rank=rank,
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

    def test_native_gemma4_mtp_keeps_full_target_embedding(self):
        """Replicated MTP recipients keep the full embedding after scope exit."""
        for kind in ("gemma4", "gemma4_mm"):
            for tied in (True, False):
                with self.subTest(kind=kind, tied=tied):
                    model = build_source(kind, tied=tied, vocab=513)
                    embed, head = fill_source(model)
                    text_config = (
                        model.config.text_config
                        if kind == "gemma4_mm"
                        else model.config
                    )
                    assistant_text = copy.deepcopy(text_config)
                    assistant_text.hidden_size_per_layer_input = 0
                    assistant_text.vocab_size_per_layer_input = 0
                    assistant_text.enable_moe_block = False
                    assistant_text.use_double_wide_mlp = False
                    assistant_text.num_kv_shared_layers = (
                        assistant_text.num_hidden_layers
                    )
                    config = Gemma4AssistantConfig(
                        text_config=assistant_text,
                        backbone_hidden_size=text_config.hidden_size,
                    )
                    with changed_scope(True), torch.device("cuda"):
                        draft = Gemma4AssistantForCausalLM(config)
                    actual_embed, actual_head = share(model, draft)
                    self.assertIs(actual_embed, embed)
                    self.assertIs(actual_head, head)
                    draft.set_embed_and_head(actual_embed, actual_head)
                    ids = torch.tensor([0, 512], device="cuda")
                    torch.testing.assert_close(
                        torch.nn.functional.embedding(ids, draft.target_embed_weight),
                        embed[ids],
                        rtol=0,
                        atol=0,
                    )
                    self.assertIsNone(find_draft_embedding_param(draft))

    def test_shared_views_see_source_weight_updates(self):
        model = build_source("gemma3")
        fill_source(model)
        with changed_scope(True):
            draft = build_draft(embedding_only=True)
        embed, head = share(model, draft)
        full_embed, full_head = fill_source(model, version=13)
        torch.testing.assert_close(embed, full_embed[384:512], rtol=0, atol=0)
        torch.testing.assert_close(head, full_head[384:512], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
