from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    _minimax_h3_override_qkv_layout,
)
from sglang.multimodal_gen.runtime.loader.minimax_h3_weights import (
    detect_minimax_h3_qkv_layout,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    _reorder_grouped_qkv_to_qkv,
)

HEADS, HEAD_DIM, HIDDEN = 4, 8, 16


def _native_qkv(scales=(1.0, 3.0, 0.3)) -> torch.Tensor:
    torch.manual_seed(0)
    parts = [
        torch.randn(HEADS * HEAD_DIM, HIDDEN) * scale for scale in scales
    ]  # [q_all, k_all, v_all]
    return torch.cat(parts).to(torch.bfloat16)


def _interleaved(native: torch.Tensor) -> torch.Tensor:
    q, k, v = native.reshape(3, HEADS, HEAD_DIM, HIDDEN).unbind(0)
    grouped = torch.stack([q, k, v], dim=1).reshape(-1, HIDDEN)  # per head [q, k, v]
    restored = _reorder_grouped_qkv_to_qkv(
        grouped, num_query_groups=HEADS, heads_per_group=1, head_dim=HEAD_DIM
    )
    assert torch.equal(restored, native)
    return grouped


def _write(tmp_path, name, qkv, key="blocks.0.attn.qkv_proj.weight"):
    path = tmp_path / name
    save_file(
        {key: qkv.contiguous(), "blocks.0.attn.out_proj.weight": qkv[:HIDDEN].clone()},
        str(path),
    )
    return str(path)


def test_detects_native_and_interleaved_layouts(tmp_path):
    native = _native_qkv()
    assert (
        detect_minimax_h3_qkv_layout(
            _write(tmp_path, "native.safetensors", native), HEADS, HEAD_DIM
        )
        == "native"
    )
    assert (
        detect_minimax_h3_qkv_layout(
            _write(tmp_path, "grouped.safetensors", _interleaved(native)),
            HEADS,
            HEAD_DIM,
        )
        == "interleaved"
    )


def test_detection_follows_param_name_mapping(tmp_path):
    path = _write(
        tmp_path,
        "hybrid.safetensors",
        _native_qkv(),
        key="model.diffusion_model.blocks.0.attn.qkv_proj.weight",
    )
    assert detect_minimax_h3_qkv_layout(path, HEADS, HEAD_DIM) == "native"


def test_ambiguous_or_quantized_weights_are_not_guessed(tmp_path):
    flat = torch.ones(3 * HEADS * HEAD_DIM, HIDDEN, dtype=torch.bfloat16)
    same_scale = _write(tmp_path, "flat.safetensors", flat)
    assert detect_minimax_h3_qkv_layout(same_scale, HEADS, HEAD_DIM) is None
    int8 = _write(tmp_path, "int8.safetensors", _native_qkv().to(torch.int8))
    assert detect_minimax_h3_qkv_layout(int8, HEADS, HEAD_DIM) is None


def _arch(layout=None):
    return SimpleNamespace(
        checkpoint_qkv_layout=layout,
        num_attention_heads=HEADS,
        attention_head_dim=HEAD_DIM,
    )


def test_unquantized_single_file_override_uses_detected_layout(tmp_path):
    native = _native_qkv()
    comfy = _write(tmp_path, "comfy.safetensors", native)
    official = _write(tmp_path, "official.safetensors", _interleaved(native))
    assert _minimax_h3_override_qkv_layout(_arch(), [comfy], None, None) == "native"
    assert (
        _minimax_h3_override_qkv_layout(_arch(), [official], None, None)
        == "interleaved"
    )


@pytest.mark.parametrize(
    "layout, n_files, config_path, quant",
    [
        ("interleaved", 1, None, None),  # explicit layout wins
        (None, 2, None, None),  # sharded override
        (None, 1, "/repo/transformer/config.json", None),  # component with config
        (None, 1, None, SimpleNamespace()),  # quantized file reports its layout
    ],
)
def test_other_overrides_keep_current_layout(
    tmp_path, layout, n_files, config_path, quant
):
    path = _write(tmp_path, "comfy.safetensors", _native_qkv())
    assert (
        _minimax_h3_override_qkv_layout(
            _arch(layout), [path] * n_files, config_path, quant
        )
        == layout
    )
