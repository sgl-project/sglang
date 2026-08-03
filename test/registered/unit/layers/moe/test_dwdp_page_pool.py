import pytest

from sglang.srt.layers.moe.dwdp.page_pool import PagePool
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-c-test-cpu")

_MIB = 1024 * 1024


class _FakeVmmOps:
    ACCESS_IS_ALLOCATION_SCOPED = True

    def __init__(self, fail_access_at=None):
        self.fail_access_at = fail_access_at
        self.next_handle = 1
        self.created = []
        self.mapped = []
        self.accessed = []
        self.unmapped = []
        self.released = []

    def get_allocation_granularity(self, device_id):
        return 4096

    def create_local_handle(self, size, device_id):
        handle = self.next_handle
        self.next_handle += 1
        self.created.append((handle, size, device_id))
        return handle

    def map_handle(self, va, size, handle, offset=0):
        self.mapped.append((va, size, handle, offset))

    def set_access(self, va, size, device_id):
        self.accessed.append((va, size, device_id))
        if self.fail_access_at == len(self.accessed):
            raise RuntimeError("injected access failure")

    def unmap_va(self, va, size):
        self.unmapped.append((va, size))

    def release_handle(self, handle):
        self.released.append(handle)


def test_region_pool_maps_one_coarse_allocation():
    ops = _FakeVmmOps()
    pool = PagePool(
        [64 * _MIB, 64 * _MIB],
        device_id=0,
        page_size=32 * _MIB,
        vmm_ops=ops,
    )

    mappings = pool.map_pages(
        slot=0,
        va_start=0x10E00000,
        size=64 * _MIB,
    )

    assert mappings == [(0x10E00000, 64 * _MIB)]
    assert ops.created == [(1, 64 * _MIB, 0)]
    assert ops.accessed == [(0x10E00000, 64 * _MIB, 0)]


def test_region_pool_rolls_back_partial_mapping_failure():
    ops = _FakeVmmOps(fail_access_at=1)
    pool = PagePool(
        [64 * _MIB, 64 * _MIB],
        device_id=0,
        page_size=32 * _MIB,
        vmm_ops=ops,
    )

    with pytest.raises(RuntimeError, match="region-pool VMM access"):
        pool.map_pages(
            slot=0,
            va_start=0x10E00000,
            size=64 * _MIB,
        )

    assert ops.unmapped == [(0x10E00000, 64 * _MIB)]
    assert ops.released == [1]
