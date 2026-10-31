# SPDX-License-Identifier: Apache-2.0
"""Check every released Pika projection against the unfused PEFT equations."""

from collections import defaultdict
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    PikaH3FlexPipelineConfig,
)
from sglang.multimodal_gen.runtime.layers.lora.linear import _compute_lora_delta
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
    PikaH3FlexPipeline,
)
from sglang.multimodal_gen.runtime.utils.model_overlay import resolve_model_overlay


class _AdapterLoader(PikaH3FlexPipeline):
    def create_pipeline_stages(self, server_args):
        raise AssertionError("Component accuracy only loads the adapter")


@torch.no_grad()
def check_pika_adapter_accuracy(variant: str) -> None:
    path = hf_hub_download(
        "Pika-Labs/Pika-H3-Flex",
        f"{variant}/adapter.safetensors",
        revision=resolve_model_overlay("Pika-Labs/Pika-H3-Flex")["source_revision"],
    )
    source = load_file(path)
    assert len(source) == 624
    loader = _AdapterLoader.__new__(_AdapterLoader)
    loader.server_args = SimpleNamespace(
        lora_path=path,
        lora_weight_name=None,
        pipeline_config=PikaH3FlexPipelineConfig(),
    )
    loader.device = torch.device("cpu")
    loader.modules = {
        "transformer": SimpleNamespace(
            param_names_mapping={}, lora_param_names_mapping={}
        )
    }
    loader.lora_adapters = defaultdict(dict)
    loader.loaded_adapter_paths = {}
    loader.loaded_adapter_files = {}
    loader.loaded_adapter_configs = {}
    loader.loaded_adapter_fingerprints = {}
    loader.loaded_adapter_alphas = {}
    loader.load_lora_adapter(path, variant, rank=0, lora_alpha=128)
    mapped = loader.lora_adapters[variant]
    assert len(mapped) == 416
    assert all(weight.dtype == torch.bfloat16 for weight in mapped.values())
    checked = set()
    generator = torch.Generator(device="cuda").manual_seed(42)
    prefixes = [
        *[(f"transformer_blocks.{i}", f"blocks.{i}") for i in range(50)],
        *[
            (f"token_refiner.refiner_blocks.{i}", f"token_refiner.blocks.{i}")
            for i in range(2)
        ],
    ]
    projections = (
        ("attn.qkv_proj", ("attn.to_q", "attn.to_k", "attn.to_v")),
        ("attn.out_proj", ("attn.to_out.0",)),
        ("mlp.fc1", ("ff.net.0.proj",)),
        ("mlp.fc2", ("ff.net.2",)),
    )
    for original, native in prefixes:
        for projection, original_projections in projections:
            name = f"{native}.{projection}"
            a = mapped[f"{name}.lora_A"].to(device="cuda")
            b = mapped[f"{name}.lora_B"].to(device="cuda")
            x = torch.randn(
                17,
                a.shape[-1],
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            expected_parts = []
            for original_projection in original_projections:
                prefix = f"{original}.{original_projection}"
                source_a = source[f"{prefix}.lora_A.default.weight"].to(a)
                source_b = source[f"{prefix}.lora_B.default.weight"].to(b)
                assert source_a.shape[0] == source_b.shape[1] == 128
                expected_parts.append(F.linear(F.linear(x, source_a), source_b))
            expected = torch.cat(expected_parts, dim=-1)
            if projection == "mlp.fc1":
                value, gate = expected.chunk(2, dim=-1)
                expected = torch.cat((gate, value), dim=-1)
            actual = _compute_lora_delta(x, a, b)
            relative_error = (
                actual.float() - expected.float()
            ).norm() / expected.float().norm()
            cosine = F.cosine_similarity(
                actual.float().flatten(), expected.float().flatten(), dim=0
            )
            assert relative_error < 0.01, (variant, name, relative_error.item())
            assert cosine > 0.99998, (variant, name, cosine.item())
            checked.update((f"{name}.lora_A", f"{name}.lora_B"))
    assert checked == set(mapped)
