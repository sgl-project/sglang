"""CPU tests for ``graph_serialization.kernels``: ``KernelResolver`` ordering,
fall-through and error reporting with fake providers, the provider skeletons,
and the guarded ``apply_func_attrs`` against a fake driver binding.
"""

import sys
from enum import IntEnum

import pytest

from sglang.srt.model_executor.graph_serialization import kernels
from sglang.srt.model_executor.graph_serialization.format import KernelIdentity
from sglang.srt.model_executor.graph_serialization.kernels import (
    RESOLUTION_ORDER,
    SETTABLE_FUNC_ATTR_NAMES,
    CapturedBytesProvider,
    ElfSectionProvider,
    JitCacheProvider,
    KernelResolver,
    KernelUnresolved,
    LiveHarvestProvider,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeProvider:
    def __init__(self, name, *, ident=None, handles=None):
        self.name = name
        self._ident = ident
        self._handles = handles
        self.identify_calls = []
        self.resolve_calls = []

    def identify(self, func, kern):
        self.identify_calls.append((func, kern))
        return self._ident

    def resolve(self, ident, images):
        self.resolve_calls.append((ident.name, dict(images)))
        return self._handles


IDENT = KernelIdentity(name="nvjet_tst_128x64", image_sha256="ab" * 32)
LIVE_ONLY = KernelIdentity(name="nvjet_tst_128x64", module_names_digest="d1")


# --- resolve -----------------------------------------------------------------


def test_resolve_follows_resolution_order_not_construction_order():
    harvest = FakeProvider("live_harvest", handles=(1, 0))
    elf = FakeProvider("elf_section", handles=(2, 3))
    resolver = KernelResolver([harvest, elf])

    assert resolver.resolve(IDENT) == (2, 3)
    assert harvest.resolve_calls == []
    assert [p.name for p in resolver.ordered_providers()] == [
        "elf_section",
        "live_harvest",
    ]


def test_resolve_falls_through_to_the_next_tier():
    captured = FakeProvider("captured_bytes", handles=None)
    elf = FakeProvider("elf_section", handles=None)
    jit = FakeProvider("jit_cache", handles=(5, 6))
    harvest = FakeProvider("live_harvest", handles=(9, 9))
    resolver = KernelResolver([jit, harvest, elf, captured])

    assert resolver.resolve(IDENT) == (5, 6)
    assert len(captured.resolve_calls) == 1
    assert len(elf.resolve_calls) == 1
    assert harvest.resolve_calls == []


def test_resolve_tries_unknown_provider_names_last_in_construction_order():
    custom_a = FakeProvider("custom_a", handles=(7, 7))
    custom_b = FakeProvider("custom_b", handles=(8, 8))
    harvest = FakeProvider("live_harvest", handles=None)
    resolver = KernelResolver([custom_b, custom_a, harvest])

    assert [p.name for p in resolver.ordered_providers()] == [
        "live_harvest",
        "custom_b",
        "custom_a",
    ]
    assert resolver.resolve(IDENT) == (8, 8)
    assert custom_a.resolve_calls == []


class IntHandle:
    """Stands in for a cuda.bindings handle object (``int()`` convertible)."""

    def __init__(self, value):
        self.value = value

    def __int__(self):
        return self.value


def test_resolve_coerces_handles_to_int():
    provider = FakeProvider("elf_section", handles=(IntHandle(42), IntHandle(0)))

    assert KernelResolver([provider]).resolve(IDENT) == (42, 0)


def test_resolve_raises_kernel_unresolved_naming_identity_and_providers():
    providers = [
        FakeProvider("live_harvest"),
        FakeProvider("elf_section"),
        FakeProvider("custom"),
    ]
    resolver = KernelResolver(providers)

    with pytest.raises(KernelUnresolved) as info:
        resolver.resolve(LIVE_ONLY)

    message = str(info.value)
    assert "nvjet_tst_128x64" in message
    assert "d1" in message
    assert "elf_section, live_harvest, custom" in message
    assert all(len(p.resolve_calls) == 1 for p in providers)


def test_resolve_with_no_providers_raises():
    with pytest.raises(KernelUnresolved, match="<no providers>"):
        KernelResolver([]).resolve(IDENT)


def test_resolver_passes_its_image_store_to_providers():
    provider = FakeProvider("elf_section", handles=(1, 1))
    images = {"ab" * 32: b"fatbin"}
    resolver = KernelResolver([provider], images=images)
    images["cd" * 32] = b"later"  # the resolver keeps its own copy

    resolver.resolve(IDENT)

    assert provider.resolve_calls == [("nvjet_tst_128x64", {"ab" * 32: b"fatbin"})]
    assert dict(resolver.images) == {"ab" * 32: b"fatbin"}
    assert resolver.providers == (provider,)


# --- identify ------------------------------------------------------------------


def test_identify_returns_first_answer_in_construction_order():
    first = FakeProvider("live_harvest", ident=None)
    second = FakeProvider("elf_section", ident=IDENT)
    third = FakeProvider("jit_cache", ident=LIVE_ONLY)
    resolver = KernelResolver([first, second, third])

    assert resolver.identify(0x100, 0) is IDENT
    assert first.identify_calls == [(0x100, 0)]
    assert second.identify_calls == [(0x100, 0)]
    assert third.identify_calls == []


def test_identify_raises_when_no_provider_answers():
    resolver = KernelResolver([FakeProvider("elf_section"), FakeProvider("x")])

    with pytest.raises(KernelUnresolved, match="0x100.*elf_section, x"):
        resolver.identify(0x100, 0x200)


# --- provider skeletons ----------------------------------------------------------


@pytest.mark.parametrize(
    "provider",
    [
        ElfSectionProvider(index_cache_dir="/tmp/idx"),
        JitCacheProvider(recorded_paths={"k": "/tmp/k.cubin"}, cache_dirs=("/c",)),
        CapturedBytesProvider(profiling_enabled=True),
        LiveHarvestProvider(harvest_capture=lambda shape: 0),
    ],
    ids=lambda p: p.name,
)
def test_provider_skeletons_raise_not_implemented(provider):
    assert provider.name in RESOLUTION_ORDER
    with pytest.raises(NotImplementedError, match="section 6.5"):
        provider.identify(1, 2)
    with pytest.raises(NotImplementedError, match="section 6.5"):
        provider.resolve(IDENT, {})


def test_provider_names_cover_the_resolution_order():
    names = {
        ElfSectionProvider.name,
        JitCacheProvider.name,
        CapturedBytesProvider.name,
        LiveHarvestProvider.name,
    }
    assert names == set(RESOLUTION_ORDER)
    assert RESOLUTION_ORDER[0] == "captured_bytes"
    assert RESOLUTION_ORDER[-1] == "live_harvest"


def test_captured_bytes_provider_arm_is_a_stub():
    provider = CapturedBytesProvider()
    assert provider.armed is False and provider.store == {}
    with pytest.raises(NotImplementedError, match="CapturedBytesProvider.arm"):
        provider.arm()


def test_live_harvest_provider_starts_with_no_forwards():
    provider = LiveHarvestProvider()
    assert provider.forwards_used == 0
    assert provider.registry == {}
    assert provider.harvest_capture is None


# --- apply_func_attrs ------------------------------------------------------------


class FakeFuncAttr(IntEnum):
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH = 11
    CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14


class FakeDriver:
    CUfunction_attribute = FakeFuncAttr

    def __init__(self, current):
        self.current = dict(current)
        self.sets = []

    def cuFuncGetAttribute(self, attrib, func):
        return (0, self.current[int(attrib)])

    def cuFuncSetAttribute(self, func, attrib, value):
        self.sets.append((func, int(attrib), value))
        self.current[int(attrib)] = value
        return (0,)


def fake_check(result):
    if result[0] != 0:
        raise RuntimeError(f"CUDA error {result[0]}")
    return None if len(result) == 1 else result[1]


def test_apply_func_attrs_sets_only_changed_settable_attributes(monkeypatch):
    driver = FakeDriver(current={8: 0, 9: -1, 11: 0, 14: 0})
    monkeypatch.setattr(kernels, "cuda_drv", driver)
    monkeypatch.setattr(kernels, "checkCudaErrors", fake_check)
    ident = KernelIdentity(
        name="k",
        image_sha256="00" * 32,
        func_attrs=((4, 96), (8, 100_000), (9, -1), (11, 2), (14, 0)),
    )

    KernelResolver([]).apply_func_attrs(0xF00, ident)

    # NUM_REGS is read-only and skipped; carveout and cluster-size flag are
    # already at their saved values; the two changed ones are written.
    assert driver.sets == [(0xF00, 8, 100_000), (0xF00, 11, 2)]


def test_apply_func_attrs_ignores_attributes_the_binding_does_not_know(
    monkeypatch,
):
    driver = FakeDriver(current={8: 0})
    monkeypatch.setattr(kernels, "cuda_drv", driver)
    monkeypatch.setattr(kernels, "checkCudaErrors", fake_check)
    # Attribute 15 (cluster scheduling policy) is not in FakeFuncAttr.
    ident = KernelIdentity(name="k", func_attrs=((15, 1), (8, 64)))

    KernelResolver([]).apply_func_attrs(1, ident)

    assert driver.sets == [(1, 8, 64)]


def test_apply_func_attrs_surfaces_driver_errors(monkeypatch):
    class FailingDriver(FakeDriver):
        def cuFuncSetAttribute(self, func, attrib, value):
            return (1,)

    monkeypatch.setattr(kernels, "cuda_drv", FailingDriver(current={8: 0}))
    monkeypatch.setattr(kernels, "checkCudaErrors", fake_check)

    with pytest.raises(RuntimeError, match="CUDA error 1"):
        KernelResolver([]).apply_func_attrs(
            1, KernelIdentity(name="k", func_attrs=((8, 1),))
        )


def test_apply_func_attrs_without_binding_raises_not_implemented(monkeypatch):
    monkeypatch.setattr(kernels, "cuda_drv", None)

    with pytest.raises(NotImplementedError, match="apply_func_attrs.*section 6.5"):
        KernelResolver([]).apply_func_attrs(1, IDENT)


def test_settable_attribute_names_exclude_read_only_ones():
    assert "CU_FUNC_ATTRIBUTE_NUM_REGS" not in SETTABLE_FUNC_ATTR_NAMES
    assert "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES" in SETTABLE_FUNC_ATTR_NAMES


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
