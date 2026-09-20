# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.runtime.cache.conditioning import (
    ConditioningCache,
    invalidate_conditioning_caches,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import Cosmos3LanguageModel
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_neo_chat import (
    NEOChatModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime.text_encoding import (
    RealtimeTextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import (
    VLAPrefixEncodingStage,
)
from sglang.multimodal_gen.runtime.realtime.session import RealtimeSession


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


@pytest.mark.parametrize("disabled, capacity", [(False, 512), (True, 512), (False, 0)])
def test_realtime_text_respects_disable_and_weight_invalidation(
    monkeypatch, disabled, capacity
):
    calls = []

    def encode(self, batch, server_args):
        calls.append(batch.prompt)
        batch.prompt_embeds = [torch.ones(1)]
        return batch

    monkeypatch.setattr(TextEncodingStage, "forward", encode)
    stage = RealtimeTextEncodingStage([], [])
    session = RealtimeSession()
    args = SimpleNamespace(
        disable_conditioning_cache=disabled,
        conditioning_cache_max_size_mb=capacity,
    )
    for _ in range(2):
        stage.forward(Req(prompt="a cat", session=session), args)
    expected = 2 if disabled or capacity == 0 else 1
    assert len(calls) == expected
    invalidate_conditioning_caches()
    stage.forward(Req(prompt="a cat", session=session), args)
    assert len(calls) == expected + 1


@pytest.mark.parametrize("disabled, capacity", [(False, 512), (True, 512), (False, 0)])
def test_vla_prefix_respects_disable_and_weight_invalidation(disabled, capacity):
    stage = VLAPrefixEncodingStage.__new__(VLAPrefixEncodingStage)
    stage.policy_model = Mock()
    stage.policy_model.build_prefix_cache_key.return_value = "observation"
    entries = {}
    stage.prefix_cache = entries
    args = SimpleNamespace(
        disable_conditioning_cache=disabled,
        conditioning_cache_max_size_mb=capacity,
        pipeline_config=SimpleNamespace(enable_global_prefix_cache=True),
    )
    batch = Req(extra={"vla": {}})
    key, cached = stage.get_cached_context(batch, args, None)
    assert cached is None
    if disabled or capacity == 0:
        assert key is None
        stage.policy_model.build_prefix_cache_key.assert_not_called()
    else:
        context = object()
        entries[key] = context
        assert stage.get_cached_context(batch, args, None)[1] is context
        invalidate_conditioning_caches()
        assert stage.get_cached_context(batch, args, None)[1] is None
