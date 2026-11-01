"""The safetensors iterators fault mmap-backed views in on the CPU on integrated GPUs.

CPU-only: the integrated-GPU gate is forced on, so this is the only coverage of
the touch path (CI has no integrated GPU)."""

import os
import re

import pytest
import safetensors.torch
import torch

from sglang.srt.model_loader import weight_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _mapped_rss_kib(path: str) -> int:
    """Resident KiB of the file mapping of `path` in this process (from /proc/self/smaps)."""
    real = os.path.realpath(path)
    total, current = 0, False
    with open("/proc/self/smaps") as f:
        for line in f:
            if re.match(r"^[0-9a-f]+-[0-9a-f]+ ", line):
                current = line.rstrip("\n").endswith(" " + real)
            elif current and line.startswith("Rss:"):
                total += int(line.split()[1])
    return total


@pytest.fixture
def shard(tmp_path):
    tensors = {
        f"layer.{i}.weight": torch.arange(
            i * 1000, i * 1000 + 256 * 1024, dtype=torch.int32
        ).view(256, 1024)
        for i in range(4)
    }
    p = tmp_path / "model.safetensors"
    safetensors.torch.save_file(tensors, str(p))
    return str(p), tensors


@pytest.mark.parametrize("iterator", ["single", "multi"])
def test_views_are_faulted_in_on_integrated_gpus(monkeypatch, shard, iterator):
    path, expected = shard
    monkeypatch.setattr(weight_utils, "_integrated_gpu", lambda: True)
    if iterator == "single":
        it = weight_utils.safetensors_weights_iterator([path])
    else:
        it = weight_utils.buffered_multi_thread_safetensors_weights_iterator(
            [path], max_workers=1
        )
    seen = {}
    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        for name, t in it:
            seen[name] = t
            # every page of the view is resident while the mapping is alive
            assert _mapped_rss_kib(path) >= t.numel() * t.element_size() // 1024 - 8
    assert set(seen) == set(expected)
    for name, t in seen.items():
        assert torch.equal(t, expected[name])


def test_touch_is_a_no_op_off_integrated_gpus(monkeypatch):
    monkeypatch.setattr(weight_utils, "_integrated_gpu", lambda: False)
    t = torch.zeros(3, 5)
    assert weight_utils._fault_in_mmap_view(t) is t


def test_touch_accepts_edge_cases(monkeypatch):
    monkeypatch.setattr(weight_utils, "_integrated_gpu", lambda: True)
    for t in (
        torch.zeros(()),
        torch.zeros(0),
        torch.zeros(5, 7)[:, 1:3],
        torch.zeros(9, dtype=torch.bfloat16),
    ):
        assert weight_utils._fault_in_mmap_view(t) is t
