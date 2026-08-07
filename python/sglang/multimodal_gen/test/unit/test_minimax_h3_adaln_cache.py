# SPDX-License-Identifier: Apache-2.0

import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3AdalnCache,
)


def test_minimax_h3_adaln_cache_matches_bf16_embedding(tmp_path):
    arch = MiniMaxH3DiTArchConfig(
        num_layers=2,
        hidden_size=4,
        time_embed_dim=3,
    )
    cache_path = tmp_path / "adaln.safetensors"
    adaln_inputs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).bfloat16()
    block_params = (
        torch.arange(2 * 2 * 72, dtype=torch.float32).reshape(2, 2, 72).bfloat16()
    )
    final_params = torch.arange(16, dtype=torch.float32).reshape(2, 8).bfloat16()
    save_file(
        {
            "adaln_inputs": adaln_inputs,
            "block_params": block_params,
            "final_params": final_params,
        },
        cache_path,
        metadata={"format_version": "1", "model_variant": "fl2va"},
    )

    cache = MiniMaxH3AdalnCache(
        arch,
        path=str(cache_path),
        model_variant="fl2va",
    )
    cache.load(torch.device("cpu"))

    cache_indices = cache.lookup(adaln_inputs.flip(0))
    block = cache.block(1, cache_indices)
    final = cache.final(cache_indices)

    assert torch.equal(torch.cat(block, dim=-1), block_params.flip(0)[:, 1])
    assert torch.equal(torch.cat(final, dim=-1), final_params.flip(0))
