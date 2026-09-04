from collections import defaultdict
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.lora.linear import (
    BaseLayerWithLoRA,
    _use_owned_base_snapshot,
    wrap_with_lora_layer,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_lora

_RANK_PATCH = "sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline.dist.get_rank"


class _TestLoRAPipeline(LoRAPipeline):
    def create_pipeline_stages(self, server_args):
        return None


def _make_layer() -> BaseLayerWithLoRA:
    return BaseLayerWithLoRA(torch.nn.Linear(2, 2, bias=False))


def _make_pipeline(layer: BaseLayerWithLoRA) -> _TestLoRAPipeline:
    pipeline = object.__new__(_TestLoRAPipeline)
    pipeline.modules = {"transformer": torch.nn.Module()}
    pipeline.server_args = SimpleNamespace(
        lora_alpha=None,
        lora_merge_mode="dynamic",
        model_path="/model",
    )
    pipeline.lora_initialized = True
    pipeline.lora_adapters = defaultdict(dict)
    pipeline.loaded_adapter_paths = {"adapter": "/adapter"}
    pipeline.loaded_adapter_alphas = {"adapter": None}
    pipeline.cur_adapter_name = {}
    pipeline.cur_adapter_path = {}
    pipeline.cur_adapter_strength = {}
    pipeline.cur_adapter_config = {}
    pipeline.lora_layers = {"linear": layer}
    pipeline.lora_layers_transformer_2 = {}
    pipeline.lora_layers_critic = {}
    pipeline.is_lora_merged = {}

    pipeline.lora_adapters["adapter"]["linear.lora_A"] = torch.ones(1, 2)
    pipeline.lora_adapters["adapter"]["linear.lora_B"] = torch.ones(2, 1)
    return pipeline


def test_merge_cache_only_accepts_cpu_backed_weights():
    pipeline = _make_pipeline(_make_layer())
    cpu_cache = pipeline._merge_cache_for(
        "transformer",
        pipeline.lora_layers,
        ["/adapter"],
        [1.0],
        enabled=True,
    )
    assert cpu_cache is not None

    resident_layer = BaseLayerWithLoRA(
        torch.nn.Linear(2, 2, bias=False, device="meta"), snapshot_base=False
    )
    resident_cache = pipeline._merge_cache_for(
        "transformer",
        {"linear": resident_layer},
        ["/adapter"],
        [1.0],
        enabled=True,
    )
    assert resident_cache is None


def test_zero_copy_snapshot_is_limited_to_cpu_backed_layers():
    assert not _use_owned_base_snapshot(False, "cpu")
    assert not _use_owned_base_snapshot(False, "meta")
    assert _use_owned_base_snapshot(False, "cuda")
    assert _use_owned_base_snapshot(True, "cpu")

    cpu_layer = wrap_with_lora_layer(
        torch.nn.Linear(2, 2, bias=False), snapshot_base=False
    )
    assert cpu_layer is not None
    assert cpu_layer._base_is_view

    meta_layer = wrap_with_lora_layer(
        torch.nn.Linear(2, 2, bias=False, device="meta"), snapshot_base=False
    )
    assert meta_layer is not None
    assert meta_layer._base_is_view


def test_quantized_base_uses_dynamic_lora_in_auto_mode():
    with patch(
        "sglang.multimodal_gen.runtime.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        base_layer = ReplicatedLinear(
            2,
            2,
            bias=False,
            quant_config=Fp8Config(is_checkpoint_fp8_serialized=True),
        )
    layer = BaseLayerWithLoRA(base_layer)
    pipeline = _make_pipeline(layer)

    assert not pipeline._should_merge_lora_for_layers(
        "transformer", {"linear": layer}, "auto"
    )
    with pytest.raises(ValueError, match="use merge mode 'dynamic'"):
        pipeline._should_merge_lora_for_layers(
            "transformer", {"linear": layer}, "merge"
        )


def test_dynamic_lora_reactivates_cached_layers_without_weight_update_context():
    layer = _make_layer()
    pipeline = _make_pipeline(layer)
    context_calls = 0

    @contextmanager
    def counted_context(*args, **kwargs):
        nonlocal context_calls
        context_calls += 1
        yield []

    pipeline._temporarily_disable_offload = counted_context

    with patch(_RANK_PATCH, return_value=0):
        pipeline.set_lora(
            "adapter",
            "/adapter",
            target="transformer",
            strength=0.75,
            merge_mode="dynamic",
        )

    first_lora_a = layer.lora_A
    first_lora_b = layer.lora_B
    assert context_calls == 0
    assert not layer.disable_lora

    pipeline._temporarily_disable_offload = lambda *args, **kwargs: nullcontext([])
    pipeline.deactivate_lora_weights("transformer")
    assert layer.disable_lora

    def fail_apply(*args, **kwargs):
        raise AssertionError("cached dynamic LoRA should not rebuild weights")

    context_calls = 0
    pipeline._temporarily_disable_offload = counted_context
    pipeline._apply_lora_to_layers = fail_apply

    with patch(_RANK_PATCH, return_value=0):
        pipeline.set_lora(
            "adapter",
            None,
            target="transformer",
            strength=0.75,
            merge_mode="dynamic",
        )

    assert context_calls == 0
    assert not layer.disable_lora
    assert layer.lora_A is first_lora_a
    assert layer.lora_B is first_lora_b


def test_merged_lora_still_uses_weight_update_context():
    layer = _make_layer()
    pipeline = _make_pipeline(layer)
    context_calls = 0

    @contextmanager
    def counted_context(*args, **kwargs):
        nonlocal context_calls
        context_calls += 1
        yield []

    pipeline._temporarily_disable_offload = counted_context

    with patch(_RANK_PATCH, return_value=0):
        pipeline.set_lora(
            "adapter",
            "/adapter",
            target="transformer",
            strength=1.0,
            merge_mode="merge",
        )

    assert context_calls == 1
    assert layer.merged
    assert pipeline.is_lora_merged["transformer"]


def test_lora_alpha_override_updates_cached_adapter_scale():
    layer = _make_layer()
    pipeline = _make_pipeline(layer)

    with patch(_RANK_PATCH, return_value=0):
        pipeline.set_lora(
            "adapter",
            None,
            target="transformer",
            strength=1.0,
            merge_mode="dynamic",
            lora_alpha=8,
        )

    assert pipeline.loaded_adapter_alphas["adapter"] == 8
    assert layer.lora_rank == 1
    assert layer.lora_alpha == 8


def test_lora_tree_url_selects_one_pinned_weight(tmp_path):
    weight_name = "adapter-v4.safetensors"
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    weight_path = adapter_dir / weight_name
    weight_path.touch()
    model_info = SimpleNamespace(
        sha="immutable-sha",
        siblings=[
            SimpleNamespace(rfilename="adapters/adapter-v3.safetensors"),
            SimpleNamespace(rfilename="adapters/adapter-v4.safetensors"),
        ],
    )

    download_target = (
        "sglang.multimodal_gen.runtime.utils.hf_diffusers_utils.maybe_download_model"
    )
    with (
        patch(
            "sglang.multimodal_gen.runtime.weights.source.HfApi.model_info",
            return_value=model_info,
        ),
        patch(download_target, return_value=str(tmp_path)) as download,
    ):
        actual = maybe_download_lora(
            "https://huggingface.co/org/multi-adapter/tree/main/adapters",
            weight_name=weight_name,
        )

    assert actual == str(weight_path)
    assert download.call_args.args[0] == "org/multi-adapter"
    assert download.call_args.kwargs["revision"] == "immutable-sha"
    assert download.call_args.kwargs["allow_patterns"] == [
        "*.json",
        "adapters/*.json",
        f"adapters/{weight_name}",
    ]


def test_lora_exact_file_url_needs_no_weight_name(tmp_path):
    weight_path = tmp_path / "adapter.safetensors"
    weight_path.touch()
    model_info = SimpleNamespace(
        sha="immutable-sha",
        siblings=[
            SimpleNamespace(rfilename="adapter.safetensors"),
            SimpleNamespace(rfilename="other.safetensors"),
        ],
    )

    download_target = (
        "sglang.multimodal_gen.runtime.utils.hf_diffusers_utils.maybe_download_model"
    )
    with (
        patch(
            "sglang.multimodal_gen.runtime.weights.source.HfApi.model_info",
            return_value=model_info,
        ),
        patch(download_target, return_value=str(tmp_path)) as download,
    ):
        actual = maybe_download_lora(
            "https://huggingface.co/org/multi-adapter/resolve/main/adapter.safetensors"
        )

    assert actual == str(weight_path)
    assert download.call_args.kwargs["allow_patterns"] == [
        "*.json",
        "adapter.safetensors",
    ]
