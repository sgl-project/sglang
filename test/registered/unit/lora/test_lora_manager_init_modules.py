# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""LoRA attachment and target-validation contracts."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.lora import lora_manager as lora_manager_module
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.utils import auto_detect_lora_target_modules
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# A VL model whose vision tower nests under `layers.<i>.` just like the decoder,
# so get_layer_id() resolves both to layer 0 and only should_apply_lora can tell
# them apart.
LANGUAGE_QKV = "language_model.model.layers.0.mixer.qkv_proj"
VISION_QKV = "vision_model.encoder.layers.0.attn.attn.qkv_proj"
LANGUAGE_EMBED_TOKENS = "language_model.model.embed_tokens"


class _TiedEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.org_vocab_size = 8
        self.embedding_dim = 4
        self.weight = torch.nn.Parameter(torch.randn(8, 4))


class _ParallelLMHead(torch.nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        params_dtype,
        org_num_embeddings,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=params_dtype)
        )


class TestTiedLMHeadLoRA(CustomTestCase):
    def test_tied_head_gets_independent_wrapper_with_shared_base_weight(self):
        """An lm_head-only adapter must survive tied input/output embeddings."""
        model = torch.nn.Module()
        tied_embedding = _TiedEmbedding()
        model.embed_tokens = tied_embedding
        model.lm_head = tied_embedding

        manager = LoRAManager.__new__(LoRAManager)
        manager.base_model = model
        manager.base_hf_config = SimpleNamespace(num_hidden_layers=0)
        manager.target_modules = {"lm_head"}
        wrapped_lm_head = object()

        inkling_module = types.ModuleType("sglang.srt.models.inkling_common.dense_mlp")
        inkling_module.InklingBatchDenseMLP = type("InklingBatchDenseMLP", (), {})

        with (
            patch.object(lora_manager_module, "ParallelLMHead", _ParallelLMHead),
            patch.object(
                manager, "set_lora_module", return_value=wrapped_lm_head
            ) as set_lora_module,
            patch.dict(
                sys.modules,
                {"sglang.srt.models.inkling_common.dense_mlp": inkling_module},
            ),
        ):
            manager.init_lora_modules()

        self.assertIsNot(model.lm_head, model.embed_tokens)
        self.assertIs(model.lm_head.weight, model.embed_tokens.weight)
        self.assertIs(manager.lm_head_module, wrapped_lm_head)
        set_lora_module.assert_called_once_with("lm_head", model.lm_head)


class TestShouldApplyLoRAGate(CustomTestCase):
    """init_lora_modules wraps only modules that should_apply_lora admits."""

    @staticmethod
    def _manager(modules, target_modules):
        manager = LoRAManager.__new__(LoRAManager)
        manager.base_model = SimpleNamespace(
            named_modules=lambda: modules,
            should_apply_lora=lambda name: name.startswith(
                "language_model.model.layers."
            ),
        )
        manager.base_hf_config = SimpleNamespace(num_hidden_layers=1)
        manager.target_modules = target_modules
        return manager

    def test_vision_tower_module_sharing_a_target_name_is_skipped(self):
        """The vision tower's qkv_proj is skipped while the decoder's is wrapped,
        and the gate stays below the embed_tokens / lm_head handling, which sits
        outside the gated prefix and must still be wrapped."""
        embed_tokens = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
        manager = self._manager(
            modules=[
                (LANGUAGE_QKV, torch.nn.Module()),
                (VISION_QKV, torch.nn.Module()),
                (LANGUAGE_EMBED_TOKENS, embed_tokens),
            ],
            target_modules={"qkv_proj", "embed_tokens"},
        )

        with patch.object(manager, "set_lora_module", side_effect=lambda name, _: name):
            manager.init_lora_modules()

        self.assertEqual(list(manager.lora_modules[0]), [LANGUAGE_QKV])
        self.assertEqual(manager.embed_tokens_module, LANGUAGE_EMBED_TOKENS)


class TestModelLoRAScope(CustomTestCase):
    def test_model_scope_keeps_decoder_projections(self):
        """Decoder FFN, expert, and hybrid projections stay inside LoRA scope;
        encoders stay outside (per-model `should_apply_lora` snapshot). A
        projection a model stops admitting is served without LoRA, silently."""
        from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
        from sglang.srt.models.mllama4 import Llama4ForConditionalGeneration
        from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration

        for model_class, prefix, projections, encoders in (
            (
                Llama4ForConditionalGeneration,
                "language_model.model.layers.0.",
                (
                    "feed_forward.down_proj",
                    "feed_forward.experts",
                    "feed_forward.shared_expert.gate_up_proj",
                ),
                ("vision_model", "multi_modal_projector"),
            ),
            (
                Gemma4ForConditionalGeneration,
                "language_model.layers.0.",
                ("moe.experts",),
                ("vision_tower", "audio_tower"),
            ),
            (
                Qwen4ExpForConditionalGeneration,
                "model.layers.0.",
                (
                    "qkv_proj",
                    "o_proj",
                    "linear_attn.in_proj_qkvz",
                    "mlp.experts",
                    "mlp.shared_expert.down_proj",
                ),
                ("visual",),
            ),
        ):
            with self.subTest(model=model_class.__name__):
                model = object.__new__(model_class)
                torch.nn.Module.__init__(model)
                for projection in projections:
                    name = prefix + projection
                    self.assertTrue(model.should_apply_lora(name), name)
                for encoder in encoders:
                    name = encoder + ".layers.0.qkv_proj"
                    self.assertFalse(model.should_apply_lora(name), name)


class _MockLinearBase(torch.nn.Module):
    pass


class _MockFusedMoE(torch.nn.Module):
    pass


class _MockParallelLMHead(torch.nn.Module):
    pass


class TestAutoDetectTargets(CustomTestCase):
    def test_auto_detect_lora_target_modules(self):
        """A Qwen3-8B-like dense module tree resolves to the expected target set;
        a silent rename of an internal projection would make auto-detection
        skip that module and LoRA would never be applied there."""
        model = torch.nn.Module()
        inner = torch.nn.Module()
        layer = torch.nn.Module()
        attn = torch.nn.Module()
        attn.qkv_proj = _MockLinearBase()
        attn.o_proj = _MockLinearBase()
        layer.self_attn = attn
        mlp = torch.nn.Module()
        mlp.gate_up_proj = _MockLinearBase()
        mlp.down_proj = _MockLinearBase()
        layer.mlp = mlp
        inner.layers = torch.nn.ModuleList([layer])
        inner.embed_tokens = torch.nn.Embedding(10, 8)  # not a LinearBase: excluded
        model.model = inner
        model.lm_head = _MockParallelLMHead()

        with (
            patch("sglang.srt.layers.linear.LinearBase", _MockLinearBase),
            patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer.FusedMoE", _MockFusedMoE
            ),
            patch(
                "sglang.srt.layers.vocab_parallel_embedding.ParallelLMHead",
                _MockParallelLMHead,
            ),
        ):
            detected = auto_detect_lora_target_modules(model)

        self.assertEqual(
            detected, {"qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head"}
        )


class TestTargetValidation(CustomTestCase):
    """init_state rejects unsupported targets before any module is wrapped."""

    @staticmethod
    def _manager(adapter_targets=None):
        manager = LoRAManager.__new__(LoRAManager)
        modules = [("model.layers.0.experts", FusedMoE.__new__(FusedMoE))]
        manager.base_model = SimpleNamespace(named_modules=lambda: modules)
        manager.configs = {}
        if adapter_targets is not None:
            manager.configs["adapter"] = LoRAConfig.from_dict(
                {"target_modules": adapter_targets, "r": 8, "lora_alpha": 8}
            )
        manager.lora_refs = {
            "adapter": SimpleNamespace(lora_name="adapter", lora_path="/unused")
        }
        manager.init_lora_adapters = Mock()
        manager.lora_backend = Mock()
        manager.init_lora_modules = Mock()
        manager.init_memory_pool = Mock()
        manager.lora_added_tokens_size = None
        return manager

    @staticmethod
    def _replicated_q():
        return patch.object(
            lora_manager_module,
            "get_parallel",
            return_value=SimpleNamespace(dcp_replicate_q_proj=True),
        )

    def test_normalized_q_target_rejects_replication_before_attachment(self):
        """Explicit and adapter-inferred q_proj targets hit the same check."""
        explicit = self._manager()
        inferred = self._manager("all-linear")
        inferred.base_model.named_modules().append(
            ("model.layers.0.q_proj", ReplicatedLinear.__new__(ReplicatedLinear))
        )
        for manager, kwargs in (
            (explicit, dict(target_modules={"q_proj"})),
            (inferred, dict(lora_paths=[object()])),
        ):
            with (
                self.subTest(**kwargs),
                self._replicated_q(),
                self.assertRaisesRegex(ValueError, "Replicated-Q"),
            ):
                manager.init_state(max_lora_rank=8, **kwargs)
            self.assertIn("qkv_proj", manager.target_modules)
            manager.init_lora_modules.assert_not_called()
            manager.init_memory_pool.assert_not_called()

    def test_indexer_target_requires_unfused_projection(self):
        manager = self._manager()
        manager.base_model.named_modules().append(
            (
                "model.layers.0.self_attn.indexer",
                SimpleNamespace(use_dsa_indexer_fusion=True),
            )
        )
        with self.assertRaisesRegex(ValueError, "DSA indexer Q/K fusion"):
            manager.init_lora_shapes(max_lora_rank=8, target_modules={"wk"})


if __name__ == "__main__":
    unittest.main()
