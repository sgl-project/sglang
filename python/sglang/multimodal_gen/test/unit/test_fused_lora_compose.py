# SPDX-License-Identifier: Apache-2.0
"""Fused-layer LoRA groups: stack-vs-compose decision, loader wiring, IPC resolution."""

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.lora.linear import (
    MergedColumnParallelLinearWithLoRA,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import (
    LoRAPipeline,
    _store_fused_lora_groups,
    stack_or_compose_fused_lora,
)
from sglang.multimodal_gen.runtime.post_training.weights_updater import (
    _resolve_lora_ipc_layer_dict_key,
)

_TP_RANK_PATCH = "sglang.multimodal_gen.runtime.layers.lora.linear.get_tp_rank"
_LOCAL_DEVICE_PATCH = (
    "sglang.multimodal_gen.runtime.layers.lora.linear.get_local_torch_device"
)

# GQA-like fused layout: unequal q/k/v sections.
OUTPUT_SIZES = [8, 2, 2]
IN_DIM = 4


def _make_ab_lists(ranks: list[int]):
    a_list, b_list = [], []
    for index, rank in enumerate(ranks):
        torch.manual_seed(100 + index * 10 + rank)
        a_list.append(torch.randn(rank, IN_DIM))
        b_list.append(torch.randn(OUTPUT_SIZES[index], rank))
    return a_list, b_list


def _reference_delta(x, a_list, b_list, adapter_alpha):
    out = torch.zeros(*x.shape[:-1], sum(OUTPUT_SIZES))
    row = 0
    for a, b in zip(a_list, b_list):
        scale = 1.0 if adapter_alpha is None else adapter_alpha / a.shape[0]
        out[..., row : row + b.shape[0]] = (x @ a.T @ b.T) * scale
        row += b.shape[0]
    return out


def test_compose_unequal_sections_matches_reference():
    a_list, b_list = _make_ab_lists([2, 3, 1])
    a_2d, b_2d, fused_alpha = stack_or_compose_fused_lora(a_list, b_list, 4)
    assert a_2d.shape == (6, IN_DIM)
    assert b_2d.shape == (sum(OUTPUT_SIZES), 6)
    assert fused_alpha == 6

    x = torch.randn(5, IN_DIM)
    torch.testing.assert_close(
        x @ a_2d.T @ b_2d.T,
        _reference_delta(x, a_list, b_list, 4),
        rtol=1e-5,
        atol=1e-5,
    )


def test_fused_sections_preserve_per_layer_alpha():
    a_list, b_list = _make_ab_lists([2, 3, 1])
    alphas = [2, 6, 1]
    pending = defaultdict(dict)
    for index, (lora_a, lora_b, alpha) in enumerate(zip(a_list, b_list, alphas)):
        pending["attn.qkv.lora_A"][index] = lora_a
        pending["attn.qkv.lora_B"][index] = lora_b
        pending["attn.qkv.alpha"][index] = torch.tensor(alpha)

    adapter = {}
    _store_fused_lora_groups(adapter, pending, adapter_alpha=4, device="cpu")

    x = torch.randn(5, IN_DIM)
    actual = x @ adapter["attn.qkv.lora_A"].T @ adapter["attn.qkv.lora_B"].T
    expected = torch.cat(
        [
            (x @ lora_a.T @ lora_b.T) * (alpha / lora_a.shape[0])
            for lora_a, lora_b, alpha in zip(a_list, b_list, alphas)
        ],
        dim=-1,
    )
    assert adapter["attn.qkv.alpha"].item() == sum(a.shape[0] for a in a_list)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_stack_kept_for_equal_sections():
    torch.manual_seed(0)
    a_list = [torch.randn(2, IN_DIM) for _ in range(2)]
    b_list = [torch.randn(4, 2) for _ in range(2)]
    a, b, fused_alpha = stack_or_compose_fused_lora(a_list, b_list, 4)
    assert a.shape == (2, 2, IN_DIM)
    assert b.shape == (2, 4, 2)
    assert fused_alpha is None
    torch.testing.assert_close(a[0], a_list[0])
    torch.testing.assert_close(b[1], b_list[1])


class _FakeMergedLinear(torch.nn.Module):
    def __init__(
        self,
        output_sizes: list[int],
        in_dim: int,
        weight: torch.Tensor | None = None,
        output_partition_sizes: list[int] | None = None,
    ):
        super().__init__()
        self.output_sizes = output_sizes
        self.output_partition_sizes = (
            output_partition_sizes
            if output_partition_sizes is not None
            else output_sizes
        )
        if weight is None:
            weight = torch.randn(sum(output_sizes), in_dim)
        self.weight = torch.nn.Parameter(weight)
        self.bias = None
        self.skip_bias_add = False
        self.gather_output = False
        self.quant_method = SimpleNamespace(
            apply=lambda layer, x, bias=None: F.linear(x, layer.weight, bias)
        )


def _make_layer() -> MergedColumnParallelLinearWithLoRA:
    torch.manual_seed(0)
    return MergedColumnParallelLinearWithLoRA(_FakeMergedLinear(OUTPUT_SIZES, IN_DIM))


def _set_composed(layer, a_list, b_list, adapter_alpha, merge_weights=False):
    a_2d, b_2d, fused_alpha = stack_or_compose_fused_lora(a_list, b_list, adapter_alpha)
    layer.lora_rank = fused_alpha
    layer.lora_alpha = fused_alpha
    layer.set_lora_weights(a_2d, b_2d, merge_weights=merge_weights)


def test_unmerged_forward_applies_composed_pair():
    layer = _make_layer()
    base_weight = layer.base_layer.weight.detach().clone()
    a_list, b_list = _make_ab_lists([2, 3, 1])
    _set_composed(layer, a_list, b_list, 4)

    assert not layer.merged
    assert not layer.disable_lora

    x = torch.randn(5, IN_DIM)
    with patch(_TP_RANK_PATCH, return_value=0):
        out, _ = layer.forward(x)
    expected = x @ base_weight.T + _reference_delta(x, a_list, b_list, 4)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_merge_writes_section_rows_and_unmerge_restores():
    layer = _make_layer()
    base_weight = layer.base_layer.weight.detach().clone()
    a_list, b_list = _make_ab_lists([2, 2, 2])
    with (
        patch(_TP_RANK_PATCH, return_value=0),
        patch(_LOCAL_DEVICE_PATCH, return_value=torch.device("cpu")),
    ):
        _set_composed(layer, a_list, b_list, 4, merge_weights=True)

    assert layer.merged
    merged = layer.base_layer.weight.detach().cpu()
    row = 0
    for a, b in zip(a_list, b_list):
        expected_rows = base_weight[row : row + b.shape[0]] + (4 / a.shape[0]) * (b @ a)
        torch.testing.assert_close(
            merged[row : row + b.shape[0]], expected_rows, rtol=1e-5, atol=1e-5
        )
        row += b.shape[0]

    with patch(_TP_RANK_PATCH, return_value=0):
        layer.unmerge_lora_weights()
    torch.testing.assert_close(layer.base_layer.weight.detach().cpu(), base_weight)


def test_composed_pair_shards_correctly_under_mock_tp():
    tp_size = 2
    a_list, b_list = _make_ab_lists([2, 3, 1])
    a_2d, b_2d, fused_alpha = stack_or_compose_fused_lora(a_list, b_list, 4)
    torch.manual_seed(0)
    full_weight = torch.randn(sum(OUTPUT_SIZES), IN_DIM)
    x = torch.randn(5, IN_DIM)
    expected = x @ full_weight.T + _reference_delta(x, a_list, b_list, 4)

    part_sizes = [size // tp_size for size in OUTPUT_SIZES]
    rank_outputs = []
    for tp_rank in range(tp_size):
        local_rows = []
        for index, (size, part) in enumerate(zip(OUTPUT_SIZES, part_sizes)):
            row = sum(OUTPUT_SIZES[:index]) + tp_rank * part
            local_rows.append(full_weight[row : row + part])
        layer = MergedColumnParallelLinearWithLoRA(
            _FakeMergedLinear(
                OUTPUT_SIZES,
                IN_DIM,
                weight=torch.cat(local_rows, dim=0),
                output_partition_sizes=part_sizes,
            )
        )
        layer.lora_rank = fused_alpha
        layer.lora_alpha = fused_alpha
        layer.set_lora_weights(a_2d, b_2d, merge_weights=False)
        with patch(_TP_RANK_PATCH, return_value=tp_rank):
            out, _ = layer.forward(x)
        rank_outputs.append(out)

    # All-gather equivalent: stitch each section's rank slices back together.
    stitched = []
    for index, part in enumerate(part_sizes):
        local_col = sum(part_sizes[:index])
        stitched.append(
            torch.cat(
                [out[..., local_col : local_col + part] for out in rank_outputs],
                dim=-1,
            )
        )
    torch.testing.assert_close(
        torch.cat(stitched, dim=-1), expected, rtol=1e-5, atol=1e-5
    )


def test_ipc_key_resolution_returns_fused_merge_index():
    module = torch.nn.Module()
    module.param_names_mapping = {
        r"^attn\.q\.(.*)$": (r"attn.to_qkv.\1", 0, 3),
        r"^attn\.k\.(.*)$": (r"attn.to_qkv.\1", 1, 3),
        r"^attn\.v\.(.*)$": (r"attn.to_qkv.\1", 2, 3),
        r"^proj\.(.*)$": r"out_proj.\1",
    }
    sentinel = object()
    layer_dict = {"attn.to_qkv": sentinel, "out_proj": sentinel}

    layer, key, merge_index = _resolve_lora_ipc_layer_dict_key(
        "attn.k", layer_dict, module
    )
    assert layer is sentinel
    assert key == "attn.to_qkv"
    assert merge_index == 1

    layer, key, merge_index = _resolve_lora_ipc_layer_dict_key(
        "proj", layer_dict, module
    )
    assert layer is sentinel
    assert key == "out_proj"
    assert merge_index is None

    layer, key, merge_index = _resolve_lora_ipc_layer_dict_key(
        "attn.to_qkv", layer_dict, module
    )
    assert layer is sentinel
    assert merge_index is None


class _TestLoRAPipeline(LoRAPipeline):
    def create_pipeline_stages(self, server_args):
        return None


_QKV_MAPPING = {
    r"^attn\.q\.(.*)$": (r"attn.to_qkv.\1", 0, 3),
    r"^attn\.k\.(.*)$": (r"attn.to_qkv.\1", 1, 3),
    r"^attn\.v\.(.*)$": (r"attn.to_qkv.\1", 2, 3),
    r"^mlp\.gate\.(.*)$": (r"mlp.gate_up.\1", 0, 2),
    r"^mlp\.up\.(.*)$": (r"mlp.gate_up.\1", 1, 2),
}


def _make_loader_pipeline() -> _TestLoRAPipeline:
    pipeline = object.__new__(_TestLoRAPipeline)
    pipeline.lora_adapters = defaultdict(dict)
    pipeline.loaded_adapter_paths = {}
    pipeline.loaded_adapter_alphas = {}
    pipeline.device = "cpu"
    pipeline.modules = {"transformer": torch.nn.Module()}
    pipeline.server_args = SimpleNamespace(
        lora_path=None,
        lora_weight_name=None,
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(
                arch_config=SimpleNamespace(
                    param_names_mapping=_QKV_MAPPING,
                    lora_param_names_mapping={
                        r"^(.*\.lora_[AB])\.default$": r"\1",
                    },
                )
            )
        ),
    )
    return pipeline


def _load_adapter(pipeline, state_dict, lora_alpha=None):
    loader_mod = "sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline"
    with (
        patch(f"{loader_mod}.maybe_download_lora", return_value="/adapter"),
        patch(f"{loader_mod}.load_file", return_value=state_dict),
        patch(f"{loader_mod}.dist.is_initialized", return_value=False),
    ):
        pipeline.load_lora_adapter("/adapter", "adapter", rank=0, lora_alpha=lora_alpha)


def _qkv_state_dict(rank=2):
    torch.manual_seed(3)
    state = {}
    for name, rows in (("q", 8), ("k", 2), ("v", 2)):
        state[f"attn.{name}.lora_A.default.weight"] = torch.randn(rank, IN_DIM)
        state[f"attn.{name}.lora_B.default.weight"] = torch.randn(rows, rank)
    return state


def test_loader_composes_unequal_fused_sections():
    pipeline = _make_loader_pipeline()
    _load_adapter(pipeline, _qkv_state_dict(), lora_alpha=4)

    adapter = pipeline.lora_adapters["adapter"]
    assert adapter["attn.to_qkv.lora_A"].shape == (6, IN_DIM)
    assert adapter["attn.to_qkv.lora_B"].shape == (sum(OUTPUT_SIZES), 6)
    assert adapter["attn.to_qkv.alpha"].item() == 6.0
    assert set(adapter) == {
        "attn.to_qkv.lora_A",
        "attn.to_qkv.lora_B",
        "attn.to_qkv.alpha",
    }


def test_loader_stacks_equal_fused_sections():
    pipeline = _make_loader_pipeline()
    state_dict = {
        "mlp.gate.lora_A.weight": torch.randn(2, IN_DIM),
        "mlp.gate.lora_B.weight": torch.randn(4, 2),
        "mlp.up.lora_A.weight": torch.randn(2, IN_DIM),
        "mlp.up.lora_B.weight": torch.randn(4, 2),
    }
    _load_adapter(pipeline, state_dict)

    adapter = pipeline.lora_adapters["adapter"]
    assert adapter["mlp.gate_up.lora_A"].shape == (2, 2, IN_DIM)
    assert adapter["mlp.gate_up.lora_B"].shape == (2, 4, 2)
    assert "mlp.gate_up.alpha" not in adapter


def test_loader_drops_incomplete_fused_groups():
    pipeline = _make_loader_pipeline()
    state_dict = _qkv_state_dict()
    for key in list(state_dict):
        if ".k." in key:
            del state_dict[key]
    _load_adapter(pipeline, state_dict)

    assert pipeline.lora_adapters["adapter"] == {}


def test_apply_composed_adapter_end_to_end():
    layer = _make_layer()
    base_weight = layer.base_layer.weight.detach().clone()
    pipeline = _make_loader_pipeline()

    rank = 2
    adapter_alpha = 4
    state_dict = _qkv_state_dict(rank=rank)
    _load_adapter(pipeline, state_dict, lora_alpha=adapter_alpha)

    strength = 2.0
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline.dist.get_rank",
        return_value=0,
    ):
        applied = pipeline._apply_lora_to_layers(
            {"attn.to_qkv": layer},
            ["adapter"],
            ["/adapter"],
            rank=0,
            strengths=[strength],
            merge_weights=False,
        )

    assert applied == 1
    a_list = [
        state_dict[f"attn.{name}.lora_A.default.weight"] for name in ("q", "k", "v")
    ]
    b_list = [
        state_dict[f"attn.{name}.lora_B.default.weight"] for name in ("q", "k", "v")
    ]
    x = torch.randn(3, IN_DIM)
    with patch(_TP_RANK_PATCH, return_value=0):
        out, _ = layer.forward(x)
    expected = (
        x @ base_weight.T
        + _reference_delta(x, a_list, b_list, adapter_alpha) * strength
    )
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
