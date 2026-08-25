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

"""Regression for #18634: LoRA wrapping of an object-shared lm_head."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.lora import lora_manager as lora_manager_module
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


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


if __name__ == "__main__":
    unittest.main()
