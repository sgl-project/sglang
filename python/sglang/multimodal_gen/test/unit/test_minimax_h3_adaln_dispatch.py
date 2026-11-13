# SPDX-License-Identifier: Apache-2.0
"""Layer-specific AdaLN dispatch through a stack that shares forward kwargs."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MiniMaxH3DiTArchConfig,
)
from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as h3
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3DiTBlock,
    MiniMaxH3DiTModel,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_adaln_cache import (
    MiniMaxH3AdalnCache,
)
from sglang.multimodal_gen.test.unit.host_sync_guard import forbid_host_syncs

pytestmark = pytest.mark.usefixtures("single_process_model_parallel")

_ARCH = MiniMaxH3DiTArchConfig(num_layers=3, hidden_size=4, time_embed_dim=3)


class _Identity(nn.Module):
    qkv_proj = fc1 = None

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, **kwargs):
        return x


def test_block_constructor_preserves_layer_index(monkeypatch):
    monkeypatch.setattr(h3, "MiniMaxH3Attention", _Identity)
    monkeypatch.setattr(h3, "MiniMaxH3MLP", _Identity)
    block = MiniMaxH3DiTBlock(
        _ARCH, None, prefix="blocks.2", use_adaln_cache=True, layer_index=2
    )
    assert block.layer_index == 2


class _SharedKwargsStack(nn.Module):
    """Cache-DiT Pattern_3's full-compute contract: reuse kwargs in every block."""

    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x, **kwargs):
        for block in self.blocks:
            x = block(x, **kwargs)
        return x


class _FinalLayer(nn.Module):
    def forward(self, x, **kwargs):
        return x, x


def _blocks():
    blocks = nn.ModuleList()
    for index in range(_ARCH.num_layers):
        block = MiniMaxH3DiTBlock.__new__(MiniMaxH3DiTBlock)
        nn.Module.__init__(block)
        block.layer_index = index
        block.norm1 = block.norm2 = nn.Identity()
        block.attn = _Identity()
        block.mlp = _Identity()
        block.adaln_proj = None
        block.preserve_input_for_cache_dit = True
        blocks.append(block)
    return blocks


def _cache(tmp_path):
    width = 6 * MINIMAX_H3_ADALN_MODALITY_NUM * _ARCH.hidden_size
    path = tmp_path / "adaln.safetensors"
    # Every plan, timestep, layer, and modality must be distinguishable.
    params = torch.arange(2 * 2 * _ARCH.num_layers * width, dtype=torch.float32)
    save_file(
        {
            "plan_timesteps": torch.tensor([[0.25, 0.0], [0.5, 0.75]]),
            "plan_lengths": torch.tensor([1, 2], dtype=torch.int64),
            "block_params": (params.reshape(2, 2, _ARCH.num_layers, width) / 4096).to(
                torch.bfloat16
            ),
            "final_params": torch.zeros(
                2, 2, 2 * _ARCH.hidden_size, dtype=torch.bfloat16
            ),
        },
        path,
        metadata={"format_version": "2", "model_variant": "fl2va"},
    )
    cache = MiniMaxH3AdalnCache(_ARCH, path=str(path), model_variant="fl2va")
    cache.load(torch.device("cpu"))
    return cache


@torch.inference_mode()
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
@pytest.mark.parametrize("stack", ["plain", "shared_kwargs", "cache_dit"])
def test_adaln_dispatch_matches_per_layer_reference_across_steps(
    tmp_path, stack, device, monkeypatch, request
):
    device = torch.device(device)
    cache = _cache(tmp_path).to(device)
    blocks = _blocks().to(device)
    original_blocks = tuple(blocks)
    # Real norms keep their output separate from the residual for CUDA's
    # in-place modulation, and give the model a parameter-backed device.
    for block in blocks:
        block.norm1 = nn.RMSNorm(_ARCH.hidden_size, eps=_ARCH.norm_eps).to(
            device=device, dtype=torch.bfloat16
        )
        block.norm2 = nn.RMSNorm(_ARCH.hidden_size, eps=_ARCH.norm_eps).to(
            device=device, dtype=torch.bfloat16
        )
    model = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.arch = _ARCH
    model.blocks = (
        nn.ModuleList([_SharedKwargsStack(blocks)])
        if stack == "shared_kwargs"
        else blocks
    )
    model.adaln_cache = cache
    model.adaln_t_table = None
    model.final_layer = _FinalLayer()
    model._resolve_attention_backend_once = lambda: None
    model._h3_spectrum_requested = lambda **kwargs: False
    model.materialize_mps_non_layer_weights = lambda *args: None
    model.release_mps_non_layer_weights = lambda *args: None
    initial = torch.ones(4, _ARCH.hidden_size, device=device, dtype=torch.bfloat16)
    model._embed = lambda **kw: (
        initial.clone(),
        torch.zeros(
            kw["unique_timesteps"].numel(), _ARCH.time_embed_dim, device=device
        ),
    )
    wrapper_calls = []
    if stack == "cache_dit":
        pytest.importorskip("cache_dit")
        from cache_dit.caching.cache_blocks.pattern_3_4_5 import (
            CachedBlocks_Pattern_3_4_5,
        )

        from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
            CacheDitConfig,
            disable_cache_on_transformer,
            enable_cache_on_transformer,
        )

        wrapper_forward = CachedBlocks_Pattern_3_4_5.forward

        def observed_wrapper_forward(self, *args, **kwargs):
            wrapper_calls.append(type(self).__name__)
            return wrapper_forward(self, *args, **kwargs)

        monkeypatch.setattr(
            CachedBlocks_Pattern_3_4_5, "forward", observed_wrapper_forward
        )
        enable_cache_on_transformer(
            model,
            CacheDitConfig(enabled=True, num_inference_steps=4, max_warmup_steps=4),
        )
        request.addfinalizer(lambda: disable_cache_on_transformer(model))

    cu_seqlens = torch.tensor([0, 4], device=device, dtype=torch.int32)
    packed = SimpleNamespace(
        cu_seqlens_q=cu_seqlens, cu_seqlens_q_host=(0, 4), max_seqlen_q=4
    )
    token_tags = torch.tensor([0, 1, 2, 2], device=device)
    # Switch plans and widths, then return to a previous plan to catch stale rows.
    for slot, timesteps in [(0, [0.25]), (1, [0.5, 0.75]), (0, [0.25])]:
        inverse = torch.tensor(
            [0, 0, len(timesteps) - 1, len(timesteps) - 1], device=device
        )
        combined = inverse * MINIMAX_H3_ADALN_MODALITY_NUM + token_tags
        block_kwargs = dict(
            adaln_input=torch.zeros(
                len(timesteps), _ARCH.time_embed_dim, device=device
            ),
            combined_indices=combined,
            rope_cache=(),
            cu_seqlens=cu_seqlens,
            max_seqlen=4,
        )
        all_params = cache.block_all(
            cache_plan_index=torch.tensor(slot, device=device),
            num_timesteps=len(timesteps),
        )
        expected = initial.clone()
        wrong = initial.clone()
        for block, params in zip(original_blocks, all_params):
            expected = block(expected, adaln_params=params, **block_kwargs)
            wrong = block(wrong, adaln_params=all_params[0], **block_kwargs)
        assert not torch.equal(expected, wrong), "The oracle must expose layer-0 reuse"

        inputs = dict(
            x=torch.zeros(1, 4, 1, device=device),
            audio_x=torch.zeros(1, 1, 1, device=device),
            img_position_ids=torch.zeros(1, 4, 3, device=device),
            unique_timesteps=torch.tensor(timesteps, device=device),
            inverse_indices=inverse,
            update_mask=torch.ones(2, device=device, dtype=torch.bfloat16),
            token_tags=token_tags,
            prompt_embeds=torch.zeros(1, 1, 1, device=device),
            img_pos_info={"position_ids": torch.tensor([2, 3], device=device)},
            audio_pos_info={"position_ids": torch.tensor([1], device=device)},
            text_pos_info={"position_ids": torch.tensor([0], device=device)},
            img_pos_for_infer_output_info={
                "position_ids": torch.tensor([2, 3], device=device)
            },
            packed_seq_params=packed,
            refiner_packed_seq_params=packed,
            rope_cache=(),
            adaln_cache_slot=torch.tensor(slot, device=device),
        )
        with forbid_host_syncs():
            video, audio = model(**inputs)
        torch.testing.assert_close(video, expected[2:], rtol=0, atol=0)
        torch.testing.assert_close(audio, expected[1:2], rtol=0, atol=0)
    if stack == "cache_dit":
        assert len(wrapper_calls) == 3
