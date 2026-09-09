import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.runtime.loader.readonly_safetensors import (
    iter_safetensors_readonly,
    load_safetensors_readonly,
    safetensors_keys,
)


def _write(tmp_path):
    tensors = {
        "a.weight": torch.randn(64, 32, dtype=torch.bfloat16),
        "b.bias": torch.arange(17, dtype=torch.float32),
        "c.empty": torch.empty(0, 4, dtype=torch.float16),
        "d.flag": torch.tensor([True, False]),
        "e.int": torch.arange(6, dtype=torch.int64).view(2, 3),
    }
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    return path, tensors


def test_readonly_load_matches_safetensors(tmp_path):
    path, tensors = _write(tmp_path)
    ours = load_safetensors_readonly(str(path))
    theirs = load_file(str(path))
    assert set(ours) == set(theirs) == set(tensors)
    for name in tensors:
        assert ours[name].dtype == theirs[name].dtype
        assert ours[name].shape == theirs[name].shape
        assert torch.equal(ours[name], theirs[name])
    assert safetensors_keys(str(path)) == list(theirs)


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/self/maps")
def test_readonly_mapping_has_no_write_permission(tmp_path):
    path, _ = _write(tmp_path)
    tensor = dict(iter_safetensors_readonly(str(path)))["a.weight"]
    ptr = tensor.data_ptr()
    perms = None
    for line in open("/proc/self/maps"):
        fields = line.split()
        low, high = (int(x, 16) for x in fields[0].split("-"))
        if low <= ptr < high:
            perms = fields[1]
            break
    assert perms is not None and perms.startswith("r--"), perms


def test_mmap_reader_maps_read_only_where_host_copies_are_redundant(
    tmp_path, monkeypatch
):
    from sglang.multimodal_gen.runtime.loader.weight_readers import safetensors_mmap

    path, tensors = _write(tmp_path)
    monkeypatch.setattr(safetensors_mmap, "host_copies_are_redundant", lambda: True)
    reader = safetensors_mmap.SafetensorsMmapReader()
    got = dict(
        reader.iter_weights(
            [str(path)],
            device="cpu",
            to_cpu=True,
            key_filter=lambda name: name != "b.bias",
            show_progress=False,
        )
    )
    assert set(got) == set(tensors) - {"b.bias"}
    assert torch.equal(got["a.weight"], tensors["a.weight"])
    if sys.platform == "linux":
        ptr = got["a.weight"].data_ptr()
        perms = next(
            line.split()[1]
            for line in open("/proc/self/maps")
            if int(line.split()[0].split("-")[0], 16)
            <= ptr
            < int(line.split()[0].split("-")[1], 16)
        )
        assert perms.startswith("r--"), perms
