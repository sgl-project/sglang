from collections import defaultdict
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from prometheus_client import CollectorRegistry

from sglang.multimodal_gen.runtime.cache.conditioning import ConditioningCache
from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.lora.linear import (
    BaseLayerWithLoRA,
    _use_owned_base_snapshot,
    wrap_with_lora_layer,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.observability.metrics import DiffusionMetrics
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
    pipeline.loaded_adapter_files = {}
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


def test_worker_metrics_count_individual_adapters_in_multi_lora():
    pipeline = _make_pipeline(_make_layer())
    pipeline._temporarily_disable_offload = lambda *args, **kwargs: nullcontext([])
    pipeline.loaded_adapter_paths["second"] = "/second"
    pipeline.loaded_adapter_alphas["second"] = None
    pipeline.lora_adapters["second"] = pipeline.lora_adapters["adapter"]
    registry = CollectorRegistry()
    worker = GPUWorker.__new__(GPUWorker)
    worker.pipeline = pipeline
    worker.metrics = DiffusionMetrics(role="monolithic", replica="0", registry=registry)
    with patch(_RANK_PATCH, return_value=0):
        worker.set_lora(
            ["adapter", "second"],
            [None, None],
            target="transformer",
            strength=[0.5, 0.5],
            merge_mode="merge",
        )
    assert (
        registry.get_sample_value(
            "sglang:diffusion_lora_active_adapters",
            {"role": "monolithic", "replica": "0"},
        )
        == 2
    )
    assert pipeline.get_lora_status()["active"]["transformer"][0]["nicknames"] == [
        "adapter",
        "second",
    ]


@pytest.mark.parametrize("operation", ["set", "merge", "unmerge", "deactivate"])
@torch.no_grad()
def test_lora_mutations_preserve_independent_conditioning(operation):
    layer = _make_layer()
    pipeline = _make_pipeline(layer)
    pipeline.modules["transformer"].add_module("linear", layer)
    pipeline._temporarily_disable_offload = lambda *args, **kwargs: nullcontext([])
    encoder = torch.nn.Linear(2, 2).eval()
    cache = ConditioningCache(1024)
    x = torch.ones(1, 2)
    with patch(_RANK_PATCH, return_value=0):
        pipeline.set_lora("adapter", merge_mode="dynamic")
        cache.run(encoder, "forward", (x,), {}, lambda: encoder(x))
        cache.run(layer, "forward", (x,), {}, lambda: x.clone())
        if operation == "set":
            pipeline.set_lora("adapter", strength=0.5, merge_mode="dynamic")
        elif operation == "merge":
            pipeline.merge_lora_weights()
        elif operation == "unmerge":
            pipeline.unmerge_lora_weights()
        else:
            pipeline.deactivate_lora_weights()
        cached = cache.run(encoder, "forward", (x,), {}, lambda: encoder(x))
        cache.run(layer, "forward", (x,), {}, lambda: x.clone())
    torch.testing.assert_close(cached, encoder(x), rtol=0, atol=0)
    assert cache.hits == 1
    assert cache.misses == 3


def test_merge_cache_only_accepts_cpu_backed_weights(tmp_path):
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"x")
    pipeline = _make_pipeline(_make_layer())
    cpu_cache = pipeline._merge_cache_for(
        "transformer",
        pipeline.lora_layers,
        [str(adapter)],
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
        [str(adapter)],
        [1.0],
        enabled=True,
    )
    assert resident_cache is None


def test_zero_copy_snapshot_is_limited_to_cpu_backed_layers():
    assert not _use_owned_base_snapshot(False, "cpu")
    assert not _use_owned_base_snapshot(False, "meta")
    assert _use_owned_base_snapshot(False, "cuda")
    assert not _use_owned_base_snapshot(False, "cuda", numel=1)
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("offloaded", [False, True])
def test_view_merge_restores_exact_base_after_strength_changes(dtype, offloaded):
    torch.manual_seed(0)
    base = torch.nn.Linear(4, 4, bias=False, dtype=dtype)
    original = base.weight.detach().clone()
    layer = wrap_with_lora_layer(base, lora_rank=2, lora_alpha=2, snapshot_base=False)
    assert layer is not None
    assert layer._base_is_view
    stored = base.weight.detach()
    if offloaded:
        base._packed_weight_cpu = stored
        base.weight.data = torch.empty(1, dtype=dtype)
    A = torch.randn(2, 4)
    B = torch.randn(4, 2)
    layer.set_lora_weights(A, B, clear_existing=True, merge_weights=False)
    assert layer._base_is_view
    for strength in (1.0, 0.0, 0.5, 1.0):
        layer.set_lora_weights(
            A, B, strength=strength, clear_existing=True, merge_weights=True
        )
        assert layer.merged
        assert not layer._base_is_view
        torch.testing.assert_close(layer.cpu_weight, original, rtol=0, atol=0)
        if strength == 0:
            torch.testing.assert_close(stored, original, rtol=0, atol=0)
        else:
            assert not torch.equal(stored, original)
    layer.unmerge_lora_weights()
    torch.testing.assert_close(stored, original, rtol=0, atol=0)
    assert not layer.merged


def test_cached_merge_restores_untouched_base_view():
    base = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    original = base.weight.detach().clone()
    layer = wrap_with_lora_layer(base, lora_rank=2, lora_alpha=2, snapshot_base=False)
    layer.set_lora_weights(torch.ones(2, 4), torch.ones(4, 2), merge_weights=False)
    layer.install_merged_weight(original + 2, base.weight.detach())
    assert layer._base_is_view
    torch.testing.assert_close(layer.cpu_weight, original, rtol=0, atol=0)
    layer.unmerge_lora_weights()
    torch.testing.assert_close(base.weight, original, rtol=0, atol=0)
    layer.merge_lora_weights(strength=0.5)
    layer.unmerge_lora_weights()
    torch.testing.assert_close(base.weight, original, rtol=0, atol=0)


def test_switching_adapters_unmerges_layers_the_new_adapter_skips():
    layer_x, layer_y = _make_layer(), _make_layer()
    base_x = layer_x.base_layer.weight.detach().clone()
    pipeline = _make_pipeline(layer_x)
    pipeline.lora_layers = {"x": layer_x, "y": layer_y}
    pipeline.lora_adapters["a"]["x.lora_A"] = torch.ones(1, 2)
    pipeline.lora_adapters["a"]["x.lora_B"] = torch.ones(2, 1)
    pipeline.lora_adapters["b"]["y.lora_A"] = torch.ones(1, 2)
    pipeline.lora_adapters["b"]["y.lora_B"] = torch.ones(2, 1)

    with patch(_RANK_PATCH, return_value=0):
        pipeline._apply_lora_to_layers(
            pipeline.lora_layers, ["a"], ["/a"], 0, [1.0], clear_existing=True
        )
        assert not torch.equal(layer_x.base_layer.weight, base_x)
        pipeline._apply_lora_to_layers(
            pipeline.lora_layers, ["b"], ["/b"], 0, [1.0], clear_existing=True
        )

    assert not layer_x.merged
    assert layer_x.disable_lora
    assert layer_x.lora_weights_list == []
    torch.testing.assert_close(layer_x.base_layer.weight, base_x)
    x = torch.randn(3, 2)
    torch.testing.assert_close(layer_x(x), x @ base_x.T)


def test_unmerge_restores_base_even_when_lora_disabled():
    layer = _make_layer()
    base = layer.base_layer.weight.detach().clone()
    layer.set_lora_weights(torch.ones(1, 2), torch.ones(2, 1), merge_weights=True)
    layer.disable_lora = True

    layer.unmerge_lora_weights()

    assert not layer.merged
    torch.testing.assert_close(layer.base_layer.weight, base)


def _two_layer_pipeline():
    layers = {
        "attn": wrap_with_lora_layer(
            torch.nn.Linear(4, 4, bias=False), lora_rank=1, lora_alpha=1
        ),
        "ffn": wrap_with_lora_layer(
            torch.nn.Linear(4, 4, bias=False), lora_rank=1, lora_alpha=1
        ),
    }
    pipeline = _make_pipeline(layers["attn"])
    pipeline.lora_layers = layers
    pipeline.loaded_adapter_alphas = {"a": None, "b": None, "c": None}
    ad = pipeline.lora_adapters
    # "a" covers attn + ffn, "b" covers attn only, "c" covers ffn only.
    for nick, names in (("a", ["attn", "ffn"]), ("b", ["attn"]), ("c", ["ffn"])):
        for n in names:
            ad[nick][f"{n}.lora_A"] = torch.ones(1, 4)
            ad[nick][f"{n}.lora_B"] = torch.ones(4, 1)
    return pipeline, layers


def _apply(pipeline, nicks, **kw):
    with patch(_RANK_PATCH, return_value=0):
        return pipeline._apply_lora_to_layers(
            pipeline.lora_layers,
            nicks,
            [f"/{n}" for n in nicks],
            0,
            [1.0] * len(nicks),
            **kw,
        )


def test_clear_existing_follows_first_covering_adapter_not_index_zero():
    pipeline, layers = _two_layer_pipeline()
    _apply(pipeline, ["a"], clear_existing=True, merge_weights=False)
    # "b" misses ffn but "c" covers it: old "a" entry must not survive.
    _apply(pipeline, ["b", "c"], clear_existing=True, merge_weights=False)
    assert [e[2] for e in layers["ffn"].lora_weights_list] == ["/c"]


def test_fractional_alpha_is_not_truncated():
    pipeline, layers = _two_layer_pipeline()
    pipeline.lora_adapters["a"]["attn.alpha"] = torch.tensor(0.5)
    _apply(pipeline, ["a"], clear_existing=True, merge_weights=False)
    assert layers["attn"].lora_alpha == 0.5


def test_failed_adapter_load_keeps_previous_adapter():
    from sglang.multimodal_gen.test.unit.test_fused_lora_compose import (
        _load_adapter,
        _make_loader_pipeline,
    )

    pipeline = _make_loader_pipeline()
    pipeline.loaded_adapter_files = {}
    good = {
        "mlp.gate.lora_A.weight": torch.randn(2, 4),
        "mlp.gate.lora_B.weight": torch.randn(4, 2),
        "mlp.up.lora_A.weight": torch.randn(2, 4),
        "mlp.up.lora_B.weight": torch.randn(4, 2),
    }
    _load_adapter(pipeline, good)
    before = dict(pipeline.lora_adapters["adapter"])
    assert before
    # Fail after parsing has started; the previous adapter must survive.
    with (
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline."
            "_store_fused_lora_groups",
            side_effect=RuntimeError("boom"),
        ),
        pytest.raises(RuntimeError),
    ):
        _load_adapter(pipeline, good)
    assert pipeline.lora_adapters["adapter"].keys() == before.keys()


def test_cache_merge_covers_layer_the_last_adapter_misses(tmp_path):
    pipeline, layers = _two_layer_pipeline()
    merged = []

    class _Cache:
        def get(self, name, shape, dtype):
            return None

        def put(self, name, tensor):
            merged.append((name, len(layers[name].lora_weights_list)))
            return None

    pipeline._merge_via_cache = lambda name, layer, cache: merged.append(
        (name, len(layer.lora_weights_list))
    )
    # "b" covers attn only, "c" ffn only, "a" both; last adapter is "b".
    _apply(pipeline, ["a", "c", "b"], clear_existing=True, merge_cache=_Cache())
    assert sorted(merged) == [("attn", 2), ("ffn", 2)]


def test_critic_offload_lookup_uses_fake_score_transformer():
    pipeline, _ = _two_layer_pipeline()
    critic = torch.nn.Module()
    pipeline.modules["fake_score_transformer"] = critic
    assert pipeline._lora_module("critic") is critic
    assert pipeline._lora_module("transformer") is pipeline.modules["transformer"]
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline."
        "is_layerwise_offloaded_module",
        side_effect=lambda module: module is critic,
    ):
        assert pipeline._has_layerwise_offload([("critic", {})])


def test_set_lora_skip_requires_every_target_module_to_match():
    pipeline, layers = _two_layer_pipeline()
    pipeline.lora_layers_transformer_2 = {"attn": layers["attn"]}
    pipeline.modules["transformer_2"] = torch.nn.Module()
    pipeline.lora_adapters["hi"]["attn.lora_A"] = torch.ones(1, 4)
    pipeline.lora_adapters["hi"]["attn.lora_B"] = torch.ones(4, 1)
    pipeline.lora_adapters["lo"] = dict(pipeline.lora_adapters["hi"])
    pipeline.loaded_adapter_paths.update({"hi": "/hi", "lo": "/lo"})
    pipeline.loaded_adapter_alphas.update({"hi": None, "lo": None})
    pipeline._temporarily_disable_offload = lambda *a, **k: nullcontext([])
    pipeline.server_args.lora_merge_mode = "dynamic"
    pipeline.lora_layers_transformer_2 = {
        "attn": wrap_with_lora_layer(
            torch.nn.Linear(4, 4, bias=False), lora_rank=1, lora_alpha=1
        )
    }
    with patch(_RANK_PATCH, return_value=0):
        pipeline.set_lora("hi", "/hi", target="transformer", merge_mode="dynamic")
        pipeline.set_lora("lo", "/lo", target="transformer_2", merge_mode="dynamic")
        pipeline.set_lora("hi", "/hi", target="all", merge_mode="dynamic")
    assert pipeline.cur_adapter_name["transformer_2"] == "hi"


def test_merge_without_strength_keeps_per_adapter_strengths():
    pipeline, layers = _two_layer_pipeline()
    with patch(_RANK_PATCH, return_value=0):
        pipeline._apply_lora_to_layers(
            pipeline.lora_layers,
            ["a", "b"],
            ["/a", "/b"],
            0,
            [0.25, 0.75],
            clear_existing=True,
            merge_weights=False,
        )
    pipeline.is_lora_merged["transformer"] = False
    pipeline.cur_adapter_config["transformer"] = (["a", "b"], [0.25, 0.75])
    pipeline._temporarily_disable_offload = lambda *a, **k: nullcontext([])
    pipeline.merge_lora_weights("transformer")
    assert [e[3] for e in layers["attn"].lora_weights_list] == [0.25, 0.75]
    assert pipeline.cur_adapter_config["transformer"][1] == [0.25, 0.75]
    pipeline.merge_lora_weights("transformer", strength=0.5)
    assert pipeline.cur_adapter_config["transformer"][1] == [0.5, 0.5]


def test_deactivate_clears_adapter_identity():
    pipeline, _ = _two_layer_pipeline()
    pipeline._temporarily_disable_offload = lambda *a, **k: nullcontext([])
    with patch(_RANK_PATCH, return_value=0):
        pipeline.set_lora("a", merge_mode="dynamic")
        assert pipeline.is_lora_set()
        pipeline.deactivate_lora_weights()
    assert not pipeline.is_lora_set()


def test_unmerge_enters_weight_update_context_once():
    pipeline, layers = _two_layer_pipeline()
    pipeline.lora_layers_transformer_2 = {"x": layers["ffn"]}
    pipeline.modules["transformer_2"] = torch.nn.Module()
    pipeline.is_lora_merged = {"transformer": True, "transformer_2": True}
    for layer in layers.values():
        layer.merged = True
        layer.disable_lora = False
    calls = []

    @contextmanager
    def counted(target_modules=None):
        calls.append(len(target_modules))
        yield

    pipeline._weight_update_context = counted
    with patch.object(BaseLayerWithLoRA, "unmerge_lora_weights", lambda self: None):
        pipeline.unmerge_lora_weights("all")
    assert calls == [2]


def test_reactivation_fast_path_survives_uncovered_layers():
    pipeline, layers = _two_layer_pipeline()
    _apply(pipeline, ["b"], clear_existing=True, merge_weights=False)
    assert not layers["ffn"].lora_weights_list
    count = pipeline._reactivate_cached_dynamic_lora_layers(
        pipeline.lora_layers, ["b"], ["/b"], [1.0]
    )
    assert count == 1
