# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from torch import nn

import sglang.multimodal_gen.runtime.models.vlas.pi05_policy as pi05_policy_module
from sglang.multimodal_gen.configs.pipeline_configs.pi05 import Pi05PipelineConfig
from sglang.multimodal_gen.runtime.cache.vla_prefix_cache import (
    PrefixContext,
    VLADensePrefixCache,
)
from sglang.multimodal_gen.runtime.distributed.vla import VLASplitGroup
from sglang.multimodal_gen.runtime.models.vlas.pi05_core import (
    Pi05SiglipAttention,
    patch_siglip_vision_attention_to_native,
)
from sglang.multimodal_gen.runtime.models.vlas.pi05_policy import (
    Pi05CheckpointManifest,
    Pi05PolicyModel,
)
from sglang.multimodal_gen.runtime.utils.vla_denoise_graph import (
    VLADenoiseGraphRunner,
    _CapturedDenoiseGraph,
)


def _prefix_context(value: float, digest: str | None) -> PrefixContext:
    keys = torch.full((1, 1, 2, 4), value)
    values = torch.full((1, 1, 2, 4), value)
    return PrefixContext(
        past_key_values=VLADensePrefixCache(((keys, values, None),)),
        prefix_pad_masks=torch.ones(1, 2, dtype=torch.bool),
        prefix_position_ids=torch.arange(2).unsqueeze(0),
        prefix_len=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        cache_key_digest=digest,
    )


def test_vla_split_group_marks_all_action_ranks():
    split = VLASplitGroup(
        group=SimpleNamespace(world_size=2),
        prefix_root=0,
        action_root=1,
        action_ranks=(0, 1),
        rank=0,
    )

    assert split.enabled
    assert split.is_prefix_rank
    assert split.is_action_rank
    assert split.uses_action_sp


def test_denoise_graph_skips_prefix_copy_for_same_digest(monkeypatch):
    runner = VLADenoiseGraphRunner(enabled=True)
    static_context = _prefix_context(1.0, "same")
    captured = _CapturedDenoiseGraph(
        graph=object(),
        static_prefix_context=static_context,
        static_x_t=torch.empty(1, 2, 4),
        static_timestep=torch.empty(1),
        static_output=torch.empty(1, 2, 4),
        current_context_id=123,
        current_context_digest="same",
    )

    def fail_copy(*args, **kwargs):
        raise AssertionError("PrefixContext should not be copied on digest hit")

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.utils.vla_denoise_graph._copy_prefix_context_",
        fail_copy,
    )

    runner._sync_context_if_needed(captured, _prefix_context(2.0, "same"))

    assert captured.static_prefix_context.past_key_values[0][0].eq(1.0).all()


def test_runai_direct_gpu_loader_does_not_reject_split_roles(monkeypatch):
    class FakeSafeOpen:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def keys(self):
            return ["action.weight"]

    monkeypatch.setattr(
        pi05_policy_module,
        "safe_open",
        lambda *args, **kwargs: FakeSafeOpen(),
    )

    model = Pi05PolicyModel.__new__(Pi05PolicyModel)
    model.device = torch.device("cuda")
    model.runtime_role = "action"
    model.config = Pi05PipelineConfig()
    model.manifest = Pi05CheckpointManifest(
        model_path="fake",
        safetensor_files=["fake.safetensors"],
    )
    model._should_read_source_key = lambda key: True
    model._resolve_target_key = lambda source_key, target_state: source_key
    target_state = {
        "action.weight": SimpleNamespace(device=SimpleNamespace(type="cuda")),
    }

    assert model._should_stream_weights_to_gpu(target_state)

    model.runtime_role = "idle"
    assert model._should_stream_weights_to_gpu(target_state)


def test_action_parallel_info_reports_single_rank_without_process_group():
    model = Pi05PolicyModel.__new__(Pi05PolicyModel)
    model.runtime_role = "all"

    info = model.action_parallel_info(prefix_context=None)

    assert info == {
        "split_group": False,
        "runtime_role": "all",
        "action_sequence_parallel": False,
    }


class _FakeSiglipAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 8
        self.num_heads = 2
        self.head_dim = 4
        self.scale = self.head_dim**-0.5
        self.dropout = 0.0
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)


def test_siglip_attention_patch_uses_native_wrapper_once():
    layer = SimpleNamespace(self_attn=_FakeSiglipAttention())
    vision_model = SimpleNamespace(encoder=SimpleNamespace(layers=[layer]))

    patch_siglip_vision_attention_to_native(vision_model)
    first = layer.self_attn
    patch_siglip_vision_attention_to_native(vision_model)

    assert isinstance(first, Pi05SiglipAttention)
    assert layer.self_attn is first
