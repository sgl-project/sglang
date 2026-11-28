"""The loader reports where a component's weights are, not a zero delta."""

import pathlib

import pytest
import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.loader.utils import (
    component_residency_bytes,
    format_component_residency,
)


class _FakeOffloadManager:
    """Stands in for LayerwiseOffloadManager's host-side weight store."""

    def __init__(self, tensors):
        self._tensors = tensors

    def iter_cpu_weights(self):
        for index, tensor in enumerate(self._tensors):
            yield f"w{index}", tensor


class _Streamed(nn.Module):
    """A module whose real weights live in its managers, not its parameters."""

    def __init__(self, managers):
        super().__init__()
        # layerwise offload leaves (1,) placeholders behind
        self.placeholder = nn.Parameter(torch.empty(0), requires_grad=False)
        self.layerwise_offload_managers = managers


class TestComponentResidencyBytes:
    def test_host_weights_are_counted(self):
        buffer = torch.empty(1024, dtype=torch.float32)
        module = _Streamed([_FakeOffloadManager([buffer])])
        totals = component_residency_bytes(module)
        assert totals["host"] == 4096
        assert totals["vram"] == 0
        assert totals["host_pinned"] == 0

    def test_slices_of_one_buffer_are_counted_once(self):
        # this is the layerwise layout: one flat buffer, many logical weights
        buffer = torch.empty(1024, dtype=torch.float32)
        views = [buffer[0:256], buffer[256:512], buffer[512:1024]]
        module = _Streamed([_FakeOffloadManager(views)])
        assert component_residency_bytes(module)["host"] == 4096

    def test_empty_placeholders_are_not_counted(self):
        module = _Streamed([])
        assert component_residency_bytes(module) == {
            "vram": 0,
            "host_pinned": 0,
            "host_mapped": 0,
            "host": 0,
        }

    def test_a_file_backed_tensor_is_separated_from_anonymous(self, tmp_path):
        if not pathlib.Path("/proc/self/maps").exists():
            pytest.skip("needs /proc to tell a mapping from anonymous memory")
        backing = tmp_path / "weights.bin"
        backing.write_bytes(b"\0" * 4096)
        mapped = torch.from_file(
            str(backing), shared=True, size=1024, dtype=torch.float32
        )
        module = _Streamed([_FakeOffloadManager([mapped])])
        totals = component_residency_bytes(module)
        assert totals["host_mapped"] == 4096
        assert totals["host"] == 0

    def test_pinned_wins_over_the_file_backed_check(self):
        if not torch.cuda.is_available():
            pytest.skip("pinning needs CUDA")
        # CUDA's host allocator sits behind a named mapping, so the
        # file-backed check alone would call this one mapped
        pinned = torch.empty(1024, dtype=torch.float32, pin_memory=True)
        module = _Streamed([_FakeOffloadManager([pinned])])
        totals = component_residency_bytes(module)
        assert totals["host_pinned"] == 4096
        assert totals["host_mapped"] == 0

    def test_resident_parameters_land_in_vram(self):
        if not torch.cuda.is_available():
            pytest.skip("needs a device to report device residency")
        module = nn.Linear(64, 64, bias=False).cuda()
        totals = component_residency_bytes(module)
        assert totals["vram"] == 64 * 64 * module.weight.element_size()
        assert totals["host"] == 0

    def test_pinned_host_weights_are_separated(self):
        if not torch.cuda.is_available():
            pytest.skip("pinning needs CUDA")
        pinned = torch.empty(512, dtype=torch.float32, pin_memory=True)
        module = _Streamed([_FakeOffloadManager([pinned])])
        totals = component_residency_bytes(module)
        assert totals["host_pinned"] == 2048
        assert totals["host"] == 0

    def test_non_module_reports_nothing(self):
        assert component_residency_bytes(object()) == {}


class TestFormatComponentResidency:
    def test_only_non_zero_places_are_named(self):
        buffer = torch.empty(int(0.5 * 1024**3 / 4), dtype=torch.float32)
        module = _Streamed([_FakeOffloadManager([buffer])])
        assert format_component_residency(module) == "host pageable: 0.50 GB"

    def test_a_component_without_weights_says_so(self):
        assert format_component_residency(_Streamed([])) == "weights: none"
