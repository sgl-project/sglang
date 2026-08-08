# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import torch.nn as nn

from sglang.multimodal_gen.configs.models.dits.ltx_2 import (
    LTX2ArchConfig,
    LTX2Config,
)
from sglang.multimodal_gen.runtime.models.dits import ltx_2


class _FakeParallelLinear(nn.Module):
    def __init__(
        self,
        *_args,
        quant_config=None,
        prefix: str = "",
        **_kwargs,
    ) -> None:
        super().__init__()
        self.quant_config = quant_config
        self.prefix = prefix


class _FakeAttention(nn.Module):
    def __init__(self, *_args, prefix: str = "", **_kwargs) -> None:
        super().__init__()
        self.prefix = prefix


def _tiny_ltx2_config() -> LTX2Config:
    return LTX2Config(
        arch_config=LTX2ArchConfig(
            num_attention_heads=1,
            attention_head_dim=4,
            in_channels=4,
            out_channels=4,
            num_layers=2,
            cross_attention_dim=4,
            caption_channels=4,
            positional_embedding_max_pos=[2, 8, 8],
            audio_num_attention_heads=1,
            audio_attention_head_dim=4,
            audio_in_channels=4,
            audio_out_channels=4,
            audio_cross_attention_dim=4,
            audio_positional_embedding_max_pos=[2],
            connector_num_attention_heads=1,
            audio_connector_num_attention_heads=1,
            apply_gated_attention=True,
        )
    )


def test_ltx2_quantized_linears_use_checkpoint_module_paths():
    quant_config = object()

    with (
        patch.object(ltx_2, "get_tp_world_size", return_value=1),
        patch.object(ltx_2, "ColumnParallelLinear", _FakeParallelLinear),
        patch.object(ltx_2, "RowParallelLinear", _FakeParallelLinear),
        patch.object(ltx_2, "USPAttention", _FakeAttention),
        patch.object(ltx_2, "LocalAttention", _FakeAttention),
    ):
        model = ltx_2.LTX2VideoTransformer3DModel(
            _tiny_ltx2_config(),
            {
                "patch_size": 1,
                "patch_size_t": 1,
                "audio_patch_size": 1,
                "audio_patch_size_t": 1,
            },
            quant_config=quant_config,
        )

    quantized_prefixes = {
        name: module.prefix
        for name, module in model.named_modules()
        if isinstance(module, _FakeParallelLinear)
        and module.quant_config is quant_config
    }

    assert len(quantized_prefixes) == 72
    assert quantized_prefixes == {name: name for name in quantized_prefixes}
