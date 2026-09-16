import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.loader import ideogram_weights
from sglang.multimodal_gen.runtime.loader.ideogram_weights import (
    inspect_ideogram_safetensors,
)


def _write(tmp_path, tensors, name="model.safetensors"):
    path = tmp_path / name
    save_file(tensors, str(path))
    return str(path)


@pytest.mark.parametrize(
    "prefix", ["input_proj", "language_model.layers.0.self_attn.q_proj"]
)
@pytest.mark.parametrize("sharded", [False, True])
def test_fp8_with_scales(tmp_path, prefix, sharded):
    weights = {
        f"{prefix}.weight": torch.zeros(4, 8).to(torch.float8_e4m3fn),
        "norm.weight": torch.ones(8, dtype=torch.bfloat16),
    }
    scales = {f"{prefix}.weight_scale": torch.ones(4)}
    if sharded:
        paths = [
            _write(tmp_path, weights, "weights.safetensors"),
            _write(tmp_path, scales, "scales.safetensors"),
        ]
    else:
        paths = [_write(tmp_path, weights | scales)]
    assert inspect_ideogram_safetensors(paths) == "fp8"


def test_bf16(tmp_path):
    path = _write(
        tmp_path,
        {
            "input_proj.weight": torch.zeros(4, 8, dtype=torch.bfloat16),
            "input_proj.bias": torch.zeros(4, dtype=torch.bfloat16),
        },
    )
    assert inspect_ideogram_safetensors([path]) == "bf16"


def test_missing_scale(tmp_path):
    path = _write(
        tmp_path, {"input_proj.weight": torch.zeros(4, 8).to(torch.float8_e4m3fn)}
    )
    with pytest.raises(ValueError, match="input_proj.weight.*missing scale"):
        inspect_ideogram_safetensors([path])


@pytest.mark.parametrize("shape", [(1,), (4, 1), (8,)])
def test_invalid_scale_shape(tmp_path, shape):
    path = _write(
        tmp_path,
        {
            "input_proj.weight": torch.zeros(4, 8).to(torch.float8_e4m3fn),
            "input_proj.weight_scale": torch.ones(shape),
        },
    )
    with pytest.raises(ValueError, match="Invalid row-wise scale.*input_proj.weight"):
        inspect_ideogram_safetensors([path])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float8_e5m2])
def test_unsupported_layout(tmp_path, dtype):
    path = _write(tmp_path, {"input_proj.weight": torch.zeros(4, 8).to(dtype)})
    with pytest.raises(ValueError, match="Ideogram"):
        inspect_ideogram_safetensors([path])


def test_empty_checkpoint():
    with pytest.raises(ValueError, match="No tensors"):
        inspect_ideogram_safetensors([])


def test_duplicate_tensor(tmp_path):
    path = _write(tmp_path, {"input_proj.weight": torch.zeros(4, 8)})
    with pytest.raises(ValueError, match="Duplicate.*input_proj.weight"):
        inspect_ideogram_safetensors([path, path])


def test_only_reads_headers(tmp_path, monkeypatch):
    path = _write(
        tmp_path, {"input_proj.weight": torch.zeros(4, 8, dtype=torch.bfloat16)}
    )
    safe_open = ideogram_weights.safe_open

    class HeaderOnly:
        def __init__(self, *args, **kwargs):
            self.checkpoint = safe_open(*args, **kwargs)

        def __enter__(self):
            self.checkpoint.__enter__()
            return self

        def __exit__(self, *args):
            return self.checkpoint.__exit__(*args)

        def keys(self):
            return self.checkpoint.keys()

        def get_slice(self, name):
            return self.checkpoint.get_slice(name)

        def get_tensor(self, name):
            pytest.fail("Checkpoint inspection must not load tensors")

    monkeypatch.setattr(ideogram_weights, "safe_open", HeaderOnly)
    assert inspect_ideogram_safetensors([path]) == "bf16"
