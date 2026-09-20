# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.cache.conditioning import ConditioningCache
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import Cosmos3LanguageModel
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_neo_chat import (
    NEOChatModel,
)


class UndLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden, cos_sin, positions):
        self.calls += 1
        return hidden, hidden + 2, hidden + 3


@torch.no_grad()
def test_cosmos_understanding_cache_restores_private_kv():
    model = Cosmos3LanguageModel.__new__(Cosmos3LanguageModel)
    torch.nn.Module.__init__(model)
    model.embed_tokens = torch.nn.Embedding(8, 4)
    model.rotary_emb = SimpleNamespace(
        build_rope_cache_inputs=lambda positions, cache_dtype: (None, positions)
    )
    layer = UndLayer()
    model.layers = torch.nn.ModuleList([layer])
    model.eval()
    ids = torch.tensor([[1, 2]])
    mask = torch.ones_like(ids)
    positions = torch.zeros(3, 1, 2, dtype=torch.long)
    with ConditioningCache(4096).scope():
        first = model(ids, mask, positions)
        expected = first[0][0].clone()
        first[0][0].zero_()
        hit = model(ids, mask, positions)
        torch.testing.assert_close(hit[0][0], expected, rtol=0, atol=0)
        assert layer.calls == 1
        model(ids, mask, positions + 1)
        model(ids, mask * 0, positions)
        assert layer.calls == 3


class Vision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, pixel_values, **kwargs):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=pixel_values * 2)


@torch.no_grad()
def test_sensenova_caches_references_but_runs_noisy_vision_each_step():
    model = NEOChatModel.__new__(NEOChatModel)
    torch.nn.Module.__init__(model)
    model.vision_model = Vision()
    model.fm_modules = torch.nn.ModuleDict({"vision_model_mot_gen": Vision()})
    model.eval()
    pixels = torch.ones(2, 4)
    with ConditioningCache(4096).scope():
        for _ in range(2):
            model.extract_feature(pixels)
            model.extract_feature(pixels, gen_model=True)
    assert model.vision_model.calls == 1
    assert model.fm_modules["vision_model_mot_gen"].calls == 2
