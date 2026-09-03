import os
import sys

import pytest
import torch

from sglang.multimodal_gen.runtime.loader import host_spill as host_spill_module
from sglang.multimodal_gen.runtime.loader.host_spill import (
    HostSpill,
    checkpoint_fingerprint,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    MappedRegions,
    hf_to_custom_state_dict,
)


@pytest.fixture(autouse=True)
def _small_spill_threshold(monkeypatch):
    monkeypatch.setattr(host_spill_module, "MIN_SPILL_BYTES", 0)
    monkeypatch.setattr(host_spill_module, "SPILL_DISK_RESERVE_BYTES", 0)


def test_spilled_tensor_is_file_backed_and_reused_after_sealing(tmp_path):
    spill = HostSpill(tmp_path, "ckpt")
    shape = torch.Size([4, 8])
    tensor, filled = spill.tensor("blocks.0.qkv", shape, torch.bfloat16)
    assert not filled
    tensor.copy_(torch.arange(32, dtype=torch.bfloat16).view(4, 8))
    if sys.platform == "linux":
        assert MappedRegions().holds(tensor)
    # unsealed: the next start must not trust it
    again, filled = HostSpill(tmp_path, "ckpt").tensor(
        "blocks.0.qkv", shape, torch.bfloat16
    )
    assert not filled
    spill.seal("blocks.0.qkv", shape, torch.bfloat16)
    reused, filled = HostSpill(tmp_path, "ckpt").tensor(
        "blocks.0.qkv", shape, torch.bfloat16
    )
    assert filled
    assert torch.equal(reused.float(), torch.arange(32, dtype=torch.float32).view(4, 8))
    # a different dtype or shape is a different file
    other, filled = HostSpill(tmp_path, "ckpt").tensor(
        "blocks.0.qkv", shape, torch.float16
    )
    assert not filled
    del again, other


def test_spill_disables_itself_when_the_disk_is_full(tmp_path, monkeypatch):
    spill = HostSpill(tmp_path, "ckpt")

    class _Usage:
        free = 0

    monkeypatch.setattr(host_spill_module.shutil, "disk_usage", lambda _p: _Usage())
    monkeypatch.setattr(host_spill_module, "SPILL_DISK_RESERVE_BYTES", 1 << 30)
    assert spill.tensor("w", torch.Size([2, 2]), torch.float32) is None
    assert spill.tensor("w2", torch.Size([2, 2]), torch.float32) is None
    assert spill.count_written == 0


def test_fused_weights_are_concatenated_into_the_provided_tensor(tmp_path):
    spill = HostSpill(tmp_path, "ckpt")
    q = torch.full((2, 3), 1.0)
    k = torch.full((2, 3), 2.0)
    v = torch.full((2, 3), 3.0)

    def mapping(name):
        prefix, _, which = name.rpartition(".")
        return f"{prefix}.qkv", {"q": 0, "k": 1, "v": 2}[which], 3

    weights = [("blocks.0.q", q), ("blocks.0.k", k), ("blocks.0.v", v)]
    merged, _ = hf_to_custom_state_dict(
        iter(weights), mapping, fused_tensor_factory=spill.tensor
    )
    fused = merged["blocks.0.qkv"]
    assert torch.equal(fused, torch.cat([q, k, v], dim=0))
    if sys.platform == "linux":
        assert MappedRegions().holds(fused)
    assert spill.count_written == 1
    spill.seal("blocks.0.qkv", fused.shape, fused.dtype)

    # the next start reuses the sealed file without reading the pieces
    reused_spill = HostSpill(tmp_path, "ckpt")
    merged_again, _ = hf_to_custom_state_dict(
        iter([(n, torch.zeros_like(t)) for n, t in weights]),
        mapping,
        fused_tensor_factory=reused_spill.tensor,
    )
    assert torch.equal(merged_again["blocks.0.qkv"], torch.cat([q, k, v], dim=0))
    assert reused_spill.count_reused == 1


def test_fingerprint_changes_with_the_checkpoint_files(tmp_path):
    shard = tmp_path / "model-00001-of-00002.safetensors"
    shard.write_bytes(b"a" * 16)
    before = checkpoint_fingerprint([str(tmp_path)])
    shard.write_bytes(b"b" * 32)
    os.utime(shard, ns=(1, 1))
    assert checkpoint_fingerprint([str(tmp_path)]) != before
