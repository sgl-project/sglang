import pytest
import torch

from sglang.multimodal_gen.runtime.layers.lora.lokr import LoKrAdapter, LoKrTerm


@pytest.mark.parametrize("shape", [(7, 8), (2, 3, 8), (0, 8)])
def test_compact_lokr_matches_dense_kronecker(shape):
    generator = torch.Generator().manual_seed(734)
    w1 = torch.randn(3, 2, generator=generator, dtype=torch.float64)
    w2 = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    x = torch.randn(shape, generator=generator, dtype=torch.float64)
    adapter = LoKrAdapter([LoKrTerm(w1, w2)], 15)
    torch.testing.assert_close(adapter(x), x @ torch.kron(w1, w2).T)
    assert sum(b.numel() for b in adapter.buffers()) == w1.numel() + w2.numel()


def test_fused_lokr_leaves_missing_projections_zero():
    w1, w2 = torch.randn(2, 2), torch.randn(3, 4)
    x = torch.randn(2, 5, 8)
    adapter = LoKrAdapter([LoKrTerm(w1, w2, 0), LoKrTerm(w1, -w2, 12)], 18)
    expected = x @ torch.kron(w1, w2).T
    torch.testing.assert_close(
        adapter(x), torch.cat([expected, torch.zeros_like(expected), -expected], -1)
    )


def test_lokr_rejects_overlapping_slices():
    term = LoKrTerm(torch.ones(2, 2), torch.ones(3, 4))
    with pytest.raises(ValueError, match="Overlapping"):
        LoKrAdapter([term, term], 12)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["fp32", "fp64", "fp16", "bf16", "amp_bf16"])
@torch.no_grad()
def test_full_matrix_lokr_matches_lycoris(mode, device):
    # Optional reference dependency, required by the acceptance runner.
    lokr = pytest.importorskip("lycoris.modules.lokr")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dtype = {
        "fp32": torch.float32,
        "fp64": torch.float64,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "amp_bf16": torch.float32,
    }[mode]
    torch.manual_seed(734)
    base = torch.nn.Linear(16, 24, bias=False).to(device=device, dtype=dtype)
    reference = lokr.LokrModule("reference", base, lora_dim=10000, factor=2)
    reference = reference.to(device=device, dtype=dtype).eval()
    reference.lokr_w1.normal_()
    reference.lokr_w2.normal_()
    adapter = LoKrAdapter(
        [LoKrTerm(reference.lokr_w1.detach(), reference.lokr_w2.detach())], 24
    )
    x = torch.randn(3, 16, device=device, dtype=dtype)
    with torch.autocast(
        device_type=device, dtype=torch.bfloat16, enabled=mode == "amp_bf16"
    ):
        actual = adapter(x)
        expected = reference.bypass_forward_diff(x)
    torch.testing.assert_close(actual, expected)
    dense = torch.kron(reference.lokr_w1.double(), reference.lokr_w2.double())
    ideal = x.double() @ dense.T
    # A normwise bound avoids making near-zero coordinates dominate low-precision checks.
    error = (actual.double() - ideal).norm() / ideal.norm()
    assert error <= 8 * torch.finfo(actual.dtype).eps


@pytest.mark.parametrize(
    "w1,w2",
    [
        (torch.ones(0, 2), torch.ones(3, 4)),
        (torch.ones(2, 2, 2), torch.ones(3, 4)),
        (torch.ones(2, 2, dtype=torch.int64), torch.ones(3, 4, dtype=torch.int64)),
        (torch.ones(2, 2), torch.ones(3, 4, dtype=torch.float64)),
    ],
)
def test_lokr_rejects_invalid_factors(w1, w2):
    with pytest.raises(ValueError):
        LoKrTerm(w1, w2)


def test_lokr_loader_maps_partial_zimage_fusions_and_rejects_unknown_keys():
    from types import SimpleNamespace

    from sglang.multimodal_gen.configs.models.dits.zimage import ZImageArchConfig
    from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
    from sglang.multimodal_gen.runtime.pipelines_core.lora.lokr_adapter import (
        load_lokr_adapter,
    )

    layers = {
        "layers.0.attention.to_qkv": SimpleNamespace(
            base_layer=SimpleNamespace(
                input_size=8, output_size=18, output_sizes=[6] * 3
            )
        ),
        "layers.0.feed_forward.w13": SimpleNamespace(
            base_layer=SimpleNamespace(
                input_size=8, output_size=12, output_sizes=[6] * 2
            )
        ),
    }
    mapping = get_param_names_mapping(ZImageArchConfig().param_names_mapping)
    state = {}
    for name in ["attention.to_q", "attention.to_v", "feed_forward.w3"]:
        prefix = "diffusion_model.layers.0." + name
        state[prefix + ".lokr_w1"] = torch.randn(2, 2)
        state[prefix + ".lokr_w2"] = torch.randn(3, 4)
        # Full-matrix LoKr has scale=1, not alpha/rank.
        state[prefix + ".alpha"] = torch.tensor(10000.0)
    result = load_lokr_adapter(state, layers, mapping, "cpu")
    x = torch.randn(2, 8)
    qkv = result["layers.0.attention.to_qkv.lokr"](x)
    for name, start in [("to_q", 0), ("to_v", 12)]:
        prefix = "diffusion_model.layers.0.attention." + name
        reference = (
            x @ torch.kron(state[prefix + ".lokr_w1"], state[prefix + ".lokr_w2"]).T
        )
        torch.testing.assert_close(qkv[:, start : start + 6], reference)
    assert torch.count_nonzero(qkv[:, 6:12]) == 0
    assert torch.count_nonzero(result["layers.0.feed_forward.w13.lokr"](x)[:, :6]) == 0
    for suffix in ["lokr_t2", "lokr_w2_a", "dora_scale", "lora_A.weight"]:
        with pytest.raises(ValueError, match="Unsupported LoKr tensor"):
            load_lokr_adapter(
                {**state, "bad." + suffix: torch.ones(1)}, layers, mapping, "cpu"
            )
    with pytest.raises(ValueError, match="does not match"):
        load_lokr_adapter(
            {k.replace("layers.0", "layers.99"): v for k, v in state.items()},
            layers,
            mapping,
            "cpu",
        )


@torch.no_grad()
def test_lokr_layer_switches_to_lora_and_restores_base():
    from sglang.multimodal_gen.runtime.layers.lora.linear import LinearWithLoRA

    base = torch.nn.Linear(8, 6)
    layer = LinearWithLoRA(base)
    x = torch.randn(3, 8)
    original = base.weight.clone()
    baseline = base(x)
    adapter = LoKrAdapter([LoKrTerm(torch.randn(2, 2), torch.randn(3, 4))], 6)
    layer.set_lokr_weights(adapter, "/lokr", 0.7)
    assert layer.lora_active
    torch.testing.assert_close(layer(x), baseline + adapter(x) * 0.7)
    with pytest.raises(ValueError, match="dynamic mode"):
        layer.merge_lora_weights()
    layer.disable_lora = True
    assert not layer.lora_active
    assert torch.equal(layer(x), baseline)
    layer.set_lokr_weights(adapter, "/lokr", -0.3)
    torch.testing.assert_close(layer(x), baseline + adapter(x) * -0.3)
    a, b = torch.randn(2, 8), torch.randn(6, 2)
    layer.lora_rank = layer.lora_alpha = 2
    layer.set_lora_weights(a, b, clear_existing=True, merge_weights=False)
    assert layer.lokr_adapter is None
    torch.testing.assert_close(layer(x), baseline + x @ a.T @ b.T)
    layer.set_lokr_weights(adapter, "/lokr", 1.0)
    layer.disable_lora = True
    assert torch.equal(layer(x), baseline)
    assert torch.equal(base.weight, original)


@torch.no_grad()
def test_lokr_pipeline_load_switch_disable_and_failed_reload(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch

    from safetensors.torch import save_file

    from sglang.multimodal_gen.configs.models.dits.zimage import ZImageArchConfig
    from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
    from sglang.multimodal_gen.runtime.layers.lora.linear import BaseLayerWithLoRA
    from sglang.multimodal_gen.test.unit.test_lora_pipeline import _make_pipeline

    base = ReplicatedLinear(8, 6, bias=True)
    base.weight.normal_()
    base.bias.zero_()
    layer = BaseLayerWithLoRA(base)
    pipeline = _make_pipeline(layer)
    pipeline.modules["transformer"].add_module("linear", layer)
    pipeline.modules["transformer"].lora_param_names_mapping = {}
    pipeline.device = "cpu"
    pipeline.server_args.num_gpus = pipeline.server_args.tp_size = 1
    pipeline.server_args.lora_path = None
    pipeline.server_args.pipeline_config = SimpleNamespace(
        dit_config=SimpleNamespace(arch_config=ZImageArchConfig())
    )
    state = {
        "diffusion_model.linear.lokr_w1": torch.randn(2, 2),
        "diffusion_model.linear.lokr_w2": torch.randn(3, 4),
        "diffusion_model.linear.alpha": torch.tensor(10000.0),
    }
    adapter_file = str(tmp_path / "adapter.safetensors")
    save_file(state, adapter_file)
    x = torch.randn(3, 8)
    baseline, _ = base(x)
    original = base.weight.clone()
    delta = (
        x
        @ torch.kron(
            state["diffusion_model.linear.lokr_w1"],
            state["diffusion_model.linear.lokr_w2"],
        ).T
    )
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline.dist.get_rank",
        return_value=0,
    ):
        pipeline.set_lora("lokr", adapter_file, merge_mode="auto", strength=0.7)
        torch.testing.assert_close(layer(x)[0], baseline + 0.7 * delta)
        assert pipeline.is_lora_effective()
        assert not pipeline.is_lora_merged["transformer"]
        pipeline.unmerge_lora_weights()
        assert torch.equal(layer(x)[0], baseline)
        pipeline.set_lora("lokr", merge_mode="auto", strength=-0.5)
        torch.testing.assert_close(layer(x)[0], baseline - 0.5 * delta)
        for operation in [
            lambda: pipeline.set_lora("lokr", merge_mode="merge"),
            lambda: pipeline.merge_lora_weights(),
            lambda: pipeline.set_lora(
                ["lokr", "adapter"], [None, None], merge_mode="dynamic"
            ),
        ]:
            with pytest.raises(ValueError, match="LoKr"):
                operation()
            torch.testing.assert_close(layer(x)[0], baseline - 0.5 * delta)
        for argument, value in [("num_gpus", 2), ("tp_size", 2)]:
            setattr(pipeline.server_args, argument, value)
            with pytest.raises(ValueError, match="single-GPU"):
                pipeline.load_lora_adapter(adapter_file, "unsupported", 0)
            setattr(pipeline.server_args, argument, 1)
            assert "unsupported" not in pipeline.lora_adapters
        with pytest.raises(ValueError, match="alpha override"):
            pipeline.load_lora_adapter(adapter_file, "unsupported", 0, lora_alpha=8)
        save_file({**state, "bad.dora_scale": torch.ones(1)}, adapter_file)
        with pytest.raises(ValueError, match="Unsupported LoKr tensor"):
            pipeline.load_lora_adapter(adapter_file, "lokr", 0)
        torch.testing.assert_close(layer(x)[0], baseline - 0.5 * delta)
        pipeline.lora_adapters["adapter"] = {
            "linear.lora_A": torch.randn(2, 8),
            "linear.lora_B": torch.randn(6, 2),
        }
        pipeline.set_lora("adapter", merge_mode="dynamic")
        assert layer.lokr_adapter is None
        pipeline.set_lora("adapter", merge_mode="merge")
        assert layer.merged
        pipeline.set_lora("lokr", merge_mode="auto")
        assert not layer.merged
        torch.testing.assert_close(layer(x)[0], baseline + delta)
        pipeline.deactivate_lora_weights()
        assert torch.equal(layer(x)[0], baseline)
        assert torch.equal(base.weight, original)


@pytest.mark.parametrize("kind", ["column", "row", "merged_column"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_lokr_parallel_linear_at_tp1_and_runtime_scale(kind, device):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        RowParallelLinear,
    )
    from sglang.multimodal_gen.runtime.layers.lora.linear import wrap_with_lora_layer
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    group = SimpleNamespace(world_size=1, rank_in_group=0)
    constructors = {
        "column": lambda: ColumnParallelLinear(8, 12, tp_group=group),
        "row": lambda: RowParallelLinear(8, 12, tp_group=group),
        "merged_column": lambda: MergedColumnParallelLinear(8, [6, 6], tp_group=group),
    }
    base = constructors[kind]().to(device)
    base.weight.normal_()
    base.bias.normal_()
    layer = wrap_with_lora_layer(base)
    w1, w2 = torch.randn(2, 2, device=device), torch.randn(3, 4, device=device)
    adapter = LoKrAdapter([LoKrTerm(w1, w2, 6)], 12)
    layer.set_lokr_weights(adapter, None, 0.7)
    x = torch.randn(2, 3, 8, device=device)
    baseline = base(x)[0]
    delta = torch.cat(
        [torch.zeros(2, 3, 6, device=device), x @ torch.kron(w1, w2).T], dim=-1
    )
    for scale in [0.0, 0.5, 1.0, -0.5]:
        with set_forward_context(
            0, None, forward_batch=SimpleNamespace(runtime_lora_scale=scale)
        ):
            torch.testing.assert_close(layer(x)[0], baseline + delta * 0.7 * scale)
    with pytest.raises(ValueError, match="Cannot combine"):
        layer.set_lora_weights(torch.ones(1, 8), torch.ones(12, 1), merge_weights=False)
    layer.disable_lora = True
    assert torch.equal(layer(x)[0], baseline)


@pytest.mark.parametrize("invalid_request", ["merge", "target", "multi"])
@torch.no_grad()
def test_rejected_lokr_replacement_preserves_cached_adapter(tmp_path, invalid_request):
    from types import SimpleNamespace
    from unittest.mock import patch

    from safetensors.torch import save_file

    from sglang.multimodal_gen.configs.models.dits.zimage import ZImageArchConfig
    from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
    from sglang.multimodal_gen.runtime.layers.lora.linear import BaseLayerWithLoRA
    from sglang.multimodal_gen.test.unit.test_lora_pipeline import _make_pipeline

    layer = BaseLayerWithLoRA(ReplicatedLinear(8, 6, bias=False))
    layer.base_layer.weight.zero_()
    pipeline = _make_pipeline(layer)
    pipeline.modules["transformer"].lora_param_names_mapping = {}
    pipeline.device = "cpu"
    pipeline.server_args.num_gpus = pipeline.server_args.tp_size = 1
    pipeline.server_args.lora_path = None
    pipeline.server_args.pipeline_config = SimpleNamespace(
        dit_config=SimpleNamespace(arch_config=ZImageArchConfig())
    )
    state = {"linear.lokr_w1": torch.ones(2, 2), "linear.lokr_w2": torch.ones(3, 4)}
    original, replacement = (
        str(tmp_path / "a.safetensors"),
        str(tmp_path / "b.safetensors"),
    )
    save_file(state, original)
    save_file({key: value * 2 for key, value in state.items()}, replacement)
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline.dist.get_rank",
        return_value=0,
    ):
        pipeline.set_lora("lokr", original, merge_mode="dynamic")
        cached = pipeline.lora_adapters["lokr"]
        x = torch.ones(1, 8)
        expected = layer(x)[0]
        with pytest.raises(ValueError, match="LoKr"):
            if invalid_request == "merge":
                pipeline.set_lora("lokr", replacement, merge_mode="merge")
            elif invalid_request == "target":
                pipeline.set_lora(
                    "lokr", replacement, target="critic", merge_mode="dynamic"
                )
            else:
                pipeline.set_lora(
                    ["lokr", "adapter"], [replacement, None], merge_mode="dynamic"
                )
        assert pipeline.lora_adapters["lokr"] is cached
        assert pipeline.loaded_adapter_paths["lokr"] == original
        assert torch.equal(layer(x)[0], expected)
        pipeline.set_lora("lokr", replacement, merge_mode="dynamic")
        torch.testing.assert_close(layer(x)[0], expected * 4)
