# SPDX-License-Identifier: Apache-2.0
"""Load real PEFT safetensors through Kandinsky6's native adapter mapping."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6 import (
    Kandinsky6TI2VAPipelineConfig,
)
from sglang.multimodal_gen.runtime.pipelines.kandinsky6_pipeline import (
    Kandinsky6TI2VAPipeline,
)


@pytest.mark.parametrize(
    "prefix", ["", "transformer.", "base_model.model.transformer."]
)
@pytest.mark.parametrize(
    "source,target",
    [
        (
            "visual_transformer_blocks.0.video_dec_block.feed_forward.net.0.proj",
            "visual_transformer_blocks.0.videoT.feed_forward.mlp.fc_in",
        ),
        (
            "visual_transformer_blocks.0.audio_dec_block.feed_forward.net.2",
            "visual_transformer_blocks.0.audioT.feed_forward.mlp.fc_out",
        ),
        (
            "video_text_transformer_blocks.1.feed_forward.net.0.proj",
            "video_text_transformer_blocks.1.feed_forward.mlp.fc_in",
        ),
        (
            "audio_text_transformer_blocks.1.attn.to_query",
            "audio_text_transformer_blocks.1.self_attention.to_query",
        ),
        (
            "video_time_embeddings.timestep_embedder.linear_1",
            "video_time_embeddings.in_layer",
        ),
        (
            "audio_time_embeddings.timestep_embedder.linear_2",
            "audio_time_embeddings.out_layer",
        ),
        (
            "visual_transformer_blocks.0.videoT.self_attention.to_key",
            "visual_transformer_blocks.0.videoT.self_attention.to_key",
        ),
    ],
)
def test_peft_adapter_maps_to_native_layers(tmp_path, prefix, source, target):
    pipeline = object.__new__(Kandinsky6TI2VAPipeline)
    pipeline.server_args = SimpleNamespace(
        pipeline_config=Kandinsky6TI2VAPipelineConfig(), lora_path=None
    )
    pipeline.modules = {
        "transformer": SimpleNamespace(
            param_names_mapping={}, lora_param_names_mapping={}
        )
    }
    pipeline.device = torch.device("cpu")
    pipeline.lora_adapters = defaultdict(dict)
    pipeline.loaded_adapter_paths = {}
    pipeline.loaded_adapter_alphas = {}
    values = {
        "lora_A": torch.arange(16, dtype=torch.float32).reshape(2, 8),
        "lora_B": torch.arange(24, dtype=torch.float32).reshape(12, 2),
        "alpha": torch.tensor(4.0),
    }
    path = tmp_path / "adapter.safetensors"
    save_file(
        {
            f"{prefix}{source}.{key}.default.weight"
            if key != "alpha"
            else f"{prefix}{source}.alpha": value
            for key, value in values.items()
        },
        path,
    )
    pipeline.load_lora_adapter(str(path), "probe", rank=0)
    actual = pipeline.lora_adapters["probe"]
    assert set(actual) == {f"{target}.{key}" for key in values}
    for key, expected in values.items():
        torch.testing.assert_close(actual[f"{target}.{key}"], expected, rtol=0, atol=0)
