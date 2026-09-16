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

"""Module-scan regressions in LoRAManager.init_lora_modules."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.lora import lora_manager as lora_manager_module
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

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
    """Regression for #21864, which deleted the should_apply_lora call site.

    Without the gate a VL model's vision projections are wrapped and bound to
    the language model's buffers, and the first request naming an adapter
    aborts the scheduler on a LoRA buffer/weight shape mismatch.
    """

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
        manager = self._manager(
            modules=[
                (LANGUAGE_QKV, torch.nn.Module()),
                (VISION_QKV, torch.nn.Module()),
            ],
            target_modules={"qkv_proj"},
        )

        with patch.object(manager, "set_lora_module", side_effect=lambda name, _: name):
            manager.init_lora_modules()

        self.assertEqual(list(manager.lora_modules[0]), [LANGUAGE_QKV])

    def test_embed_tokens_outside_the_gated_prefix_is_still_wrapped(self):
        """The gate stays below the embed_tokens / lm_head handling."""
        embed_tokens = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
        manager = self._manager(
            modules=[(LANGUAGE_EMBED_TOKENS, embed_tokens)],
            target_modules={"embed_tokens"},
        )

        with patch.object(manager, "set_lora_module", side_effect=lambda name, _: name):
            manager.init_lora_modules()

        self.assertEqual(manager.embed_tokens_module, LANGUAGE_EMBED_TOKENS)


if __name__ == "__main__":
    unittest.main()
