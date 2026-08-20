"""Pinned host memory is planned against the cgroup cap, not the whole machine."""

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.managers.memory_managers import host_memory_budget
from sglang.multimodal_gen.runtime.managers.memory_managers.host_memory_budget import (
    GIB_BYTES,
    HostPinBudget,
    cgroup_memory_limit_bytes,
    host_memory_available_bytes,
    module_weight_bytes,
    pin_benefit_bytes,
)


def _point_at(monkeypatch, tmp_path, *, v2=None, v1=None):
    """Redirect the cgroup lookups at files under tmp_path."""

    def write(name, value):
        path = tmp_path / name
        path.write_text(str(value))
        return str(path)

    missing = str(tmp_path / "absent")
    v2_paths = (
        (write("memory.max", v2[0]), write("memory.current", v2[1]))
        if v2
        else (missing, missing)
    )
    v1_paths = (
        (write("limit_in_bytes", v1[0]), write("usage_in_bytes", v1[1]))
        if v1
        else (missing, missing)
    )
    monkeypatch.setattr(host_memory_budget, "_CGROUP_V2", v2_paths)
    monkeypatch.setattr(host_memory_budget, "_CGROUP_V1", v1_paths)


class TestCgroupLimit:
    def test_v2_cap_is_read(self, monkeypatch, tmp_path):
        _point_at(monkeypatch, tmp_path, v2=(32 * GIB_BYTES, 4 * GIB_BYTES))
        assert cgroup_memory_limit_bytes() == (32 * GIB_BYTES, 4 * GIB_BYTES)

    def test_v1_cap_is_read_when_v2_is_absent(self, monkeypatch, tmp_path):
        _point_at(monkeypatch, tmp_path, v1=(64 * GIB_BYTES, 8 * GIB_BYTES))
        assert cgroup_memory_limit_bytes() == (64 * GIB_BYTES, 8 * GIB_BYTES)

    def test_no_cgroup_reports_uncapped(self, monkeypatch, tmp_path):
        _point_at(monkeypatch, tmp_path)
        assert cgroup_memory_limit_bytes() is None

    def test_v2_max_keyword_is_uncapped(self, monkeypatch, tmp_path):
        _point_at(monkeypatch, tmp_path, v2=("max", 4 * GIB_BYTES))
        assert cgroup_memory_limit_bytes() is None

    def test_v1_sentinel_is_uncapped(self, monkeypatch, tmp_path):
        # an unlimited v1 cgroup reports a number near 2**63 rather than "max"
        _point_at(monkeypatch, tmp_path, v1=(2**63 - 4096, 8 * GIB_BYTES))
        assert cgroup_memory_limit_bytes() is None

    def test_the_cap_wins_over_what_the_kernel_reports_free(
        self, monkeypatch, tmp_path
    ):
        # the case measured on a rented box: psutil sees the whole machine
        _point_at(monkeypatch, tmp_path, v2=(32 * GIB_BYTES, 8 * GIB_BYTES))
        monkeypatch.setattr(
            host_memory_budget.psutil,
            "virtual_memory",
            lambda: type("VM", (), {"available": 900 * GIB_BYTES})(),
        )
        assert host_memory_available_bytes() == 24 * GIB_BYTES

    def test_free_memory_wins_when_it_is_the_smaller_number(
        self, monkeypatch, tmp_path
    ):
        _point_at(monkeypatch, tmp_path, v2=(900 * GIB_BYTES, 0))
        monkeypatch.setattr(
            host_memory_budget.psutil,
            "virtual_memory",
            lambda: type("VM", (), {"available": 12 * GIB_BYTES})(),
        )
        assert host_memory_available_bytes() == 12 * GIB_BYTES


class TestHostPinBudget:
    def test_a_component_that_fits_is_granted(self):
        budget = HostPinBudget(available_bytes=40 * GIB_BYTES)
        assert budget.request(component_name="dit", weight_bytes=20 * GIB_BYTES)

    def test_the_reserve_is_not_spendable(self):
        budget = HostPinBudget(available_bytes=40 * GIB_BYTES)
        # 5% of 40 GiB is 2 GiB, so 38 GiB is spendable and 39 GiB is not
        assert not budget.request(component_name="dit", weight_bytes=39 * GIB_BYTES)
        assert budget.request(component_name="dit", weight_bytes=38 * GIB_BYTES)

    def test_the_reserve_has_a_floor_on_a_small_host(self):
        budget = HostPinBudget(available_bytes=8 * GIB_BYTES)
        # 5% of 8 GiB is well under the 2 GiB floor
        assert budget.reserve_bytes == 2 * GIB_BYTES

    def test_a_later_component_is_denied_once_the_budget_is_spent(self):
        budget = HostPinBudget(available_bytes=40 * GIB_BYTES)
        assert budget.request(component_name="dit", weight_bytes=30 * GIB_BYTES)
        assert not budget.request(
            component_name="text_encoder", weight_bytes=20 * GIB_BYTES
        )

    def test_the_cap_binds_even_for_the_first_component(self):
        # priority comes from asking first, not from being allowed to overrun
        budget = HostPinBudget(available_bytes=8 * GIB_BYTES)
        assert not budget.request(component_name="dit", weight_bytes=20 * GIB_BYTES)

    def test_a_weightless_component_needs_no_budget(self):
        budget = HostPinBudget(available_bytes=0)
        assert budget.request(component_name="scheduler", weight_bytes=0)


class TestModuleWeightBytes:
    def test_parameters_and_buffers_are_counted(self):
        module = nn.Linear(64, 64, bias=False)
        assert module_weight_bytes(module) == 64 * 64 * module.weight.element_size()

    def test_shared_storage_is_counted_once(self):
        module = nn.Module()
        backing = torch.empty(1024, dtype=torch.float32)
        module.register_buffer("a", backing[:512])
        module.register_buffer("b", backing[512:])
        assert module_weight_bytes(module) == 4096


class TestPinBenefit:
    def test_a_stepped_component_counts_every_step(self):
        assert pin_benefit_bytes(weight_bytes=1000, uses_per_request=50) == 50_000

    def test_a_one_shot_component_counts_once(self):
        assert pin_benefit_bytes(weight_bytes=1000, uses_per_request=1) == 1000

    def test_a_few_step_model_inverts_the_obvious_order(self):
        # 1 GB DiT over 4 steps against a 20 GB one-shot text encoder: ranking
        # by "is it the DiT" would hand the budget to the wrong one
        dit = pin_benefit_bytes(weight_bytes=1 * GIB_BYTES, uses_per_request=4)
        text_encoder = pin_benefit_bytes(
            weight_bytes=20 * GIB_BYTES, uses_per_request=1
        )
        assert text_encoder > dit

    def test_a_many_step_model_keeps_the_dit_first(self):
        dit = pin_benefit_bytes(weight_bytes=3 * GIB_BYTES, uses_per_request=50)
        text_encoder = pin_benefit_bytes(
            weight_bytes=21 * GIB_BYTES, uses_per_request=1
        )
        assert dit > text_encoder

    def test_missing_step_count_is_treated_as_one_use(self):
        assert pin_benefit_bytes(weight_bytes=1000, uses_per_request=0) == 1000
